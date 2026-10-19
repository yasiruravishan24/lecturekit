"""Claude API wrapper: fixed temperature, JSON output, on-disk cache, token log.

The cache means re-running the same prompt costs nothing, so a debugging loop does
not burn credit and the evaluation is reproducible (one run per configuration).
"""
import hashlib
import json
import re
import time
from pathlib import Path

from . import config, log
from .errors import LLMError, MissingInputError

logger = log.get("llm")
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def parse_json(text: str) -> dict:
    """Extract a JSON object from a model reply (tolerates ```json fences)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in reply: {text[:200]!r}")
    return json.loads(text[start:end + 1])


class WebSearchUnavailable(LLMError):
    """The account or model does not allow the web search tool (the caller can go on without it)."""
    code = "web_search_unavailable"


class LLM:
    def __init__(self, model: str | None = None, max_tokens: int = 8192,
                 use_cache: bool = True, usage_log: Path | None = None):
        self.model = model or config.MODEL_DEV
        self.max_tokens = max_tokens
        self.use_cache = use_cache
        self.usage_log = Path(usage_log or config.CACHE_DIR / "usage.jsonl")
        self.calls: list[dict] = []          # per-call stats of this object
        self._client = None
        self.temperature_set = None

    # -- internals ---------------------------------------------------------
    def _key(self, system: str, user: str, images: list[bytes] | None = None) -> str:
        parts = [self.model, config.TEMPERATURE, system, user]
        if images:                              # text-only prompts keep the key they always had
            parts.append(hashlib.sha256(b"".join(images)).hexdigest())
        return hashlib.sha256(json.dumps(parts).encode()).hexdigest()

    def _call_api(self, system: str, user: str, images: list[bytes] | None = None) -> tuple[str, int, int]:
        if self._client is None:
            import os
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise MissingInputError("ANTHROPIC_API_KEY is not set (add it to .env)")
            import anthropic
            self._client = anthropic.Anthropic(timeout=300.0, max_retries=2)    # reads ANTHROPIC_API_KEY; a stalled call must not block the queue for ten minutes
        content = user
        if images:                              # pictures first (JPEG bytes), then the question
            import base64
            content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.b64encode(img).decode()}} for img in images]
            content.append({"type": "text", "text": user})
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=[{"role": "user", "content": content}])
        try:                                   # fixed temperature where the SDK/model allows it
            msg = self._client.messages.create(temperature=config.TEMPERATURE, **kwargs)
            self.temperature_set = True
        except TypeError:                      # newer SDK/model: setting not accepted
            msg = self._client.messages.create(**kwargs)
            self.temperature_set = False
        if getattr(msg, "stop_reason", None) == "max_tokens":
            raise LLMError(f"Claude reply was cut off at max_tokens={self.max_tokens}; "
                           "raise max_tokens or shorten the request")
        text = "".join(b.text for b in msg.content if b.type == "text")
        return text, msg.usage.input_tokens, msg.usage.output_tokens

    def _call_with_retry(self, system: str, user: str, tag: str, attempts: int = 3, images: list[bytes] | None = None):
        for i in range(attempts):
            try:
                return self._call_api(system, user, images) if images else self._call_api(system, user)
            except (MissingInputError, LLMError):
                raise
            except Exception as e:                       # SDK errors differ between versions
                status = getattr(e, "status_code", None)
                retry = status in RETRY_STATUS or type(e).__name__ in (
                    "APIConnectionError", "APITimeoutError", "RateLimitError")
                logger.warning("LLM call '%s' failed (%s %s)%s", tag, type(e).__name__, status,
                               ", retrying" if retry and i < attempts - 1 else "")
                if not retry or i == attempts - 1:
                    raise LLMError(f"Claude API call '{tag}' failed: {type(e).__name__}: "
                                   f"{str(e)[:300]}") from e
                time.sleep(2 ** (i + 1))

    # -- public ------------------------------------------------------------
    def complete_json(self, system: str, user: str, tag: str = "", json_attempts: int = 2,
                      images: list[bytes] | None = None) -> dict:
        """Ask for a JSON reply (optionally about pictures). An invalid reply is dropped from the cache and asked for once more."""
        key = self._key(system, user, images)
        cache_file = config.CACHE_DIR / "llm" / f"{key}.json"
        for attempt in range(1, json_attempts + 1):
            t0 = time.time()
            cached = False
            if self.use_cache and cache_file.exists():
                rec = json.loads(cache_file.read_text())
                text, tin, tout = rec["text"], rec["in"], rec["out"]
                cached = True
            else:
                text, tin, tout = self._call_with_retry(system, user, tag, images=images)
                if self.use_cache:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps({"text": text, "in": tin, "out": tout}))
            stat = {"tag": tag, "model": self.model, "tokens_in": tin, "tokens_out": tout,
                    "seconds": round(time.time() - t0, 2), "cached": cached}
            self.calls.append(stat)
            logger.debug("llm %s: %s in/%s out, %s s%s", tag, tin, tout, stat["seconds"],
                         " (cached)" if cached else "")
            self.usage_log.parent.mkdir(parents=True, exist_ok=True)
            with self.usage_log.open("a") as f:
                f.write(json.dumps(stat) + "\n")
            try:
                return parse_json(text)
            except ValueError as e:
                if cache_file.exists():          # do not keep a bad reply in the cache
                    cache_file.unlink()
                if attempt == json_attempts:
                    raise LLMError(f"Claude reply for '{tag}' was not valid JSON: {e}") from e
                logger.warning("LLM reply for '%s' was not valid JSON (%s), asking again", tag, e)

    def complete_json_web(self, system: str, user: str, tag: str = "", max_searches: int = 4, json_attempts: int = 2) -> tuple[dict, list[dict]]:
        """Like complete_json, but Claude may search the web first (Anthropic's server-side search tool).

        Returns (the JSON reply, the pages it found: [{"url", "title"}]). The pages come from the search results themselves, so a link
        in the reply can be checked against them. Raises WebSearchUnavailable when the tool is not allowed for this account or model."""
        key = hashlib.sha256(json.dumps([self.model, config.TEMPERATURE, system, user, "web", max_searches]).encode()).hexdigest()
        cache_file = config.CACHE_DIR / "llm" / f"{key}.json"
        for attempt in range(1, json_attempts + 1):
            t0 = time.time()
            cached = False
            if self.use_cache and cache_file.exists():
                rec = json.loads(cache_file.read_text())
                text, sources, tin, tout = rec["text"], rec["sources"], rec["in"], rec["out"]
                cached = True
            else:
                text, sources, tin, tout = self._web_with_retry(system, user, tag, max_searches)
                if self.use_cache:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps({"text": text, "sources": sources, "in": tin, "out": tout}))
            stat = {"tag": tag, "model": self.model, "tokens_in": tin, "tokens_out": tout, "seconds": round(time.time() - t0, 2),
                    "cached": cached, "searches": len(sources)}
            self.calls.append(stat)
            self.usage_log.parent.mkdir(parents=True, exist_ok=True)
            with self.usage_log.open("a") as f:
                f.write(json.dumps(stat) + "\n")
            try:
                return parse_json(text), sources
            except ValueError as e:
                if cache_file.exists():
                    cache_file.unlink()
                if attempt == json_attempts:
                    raise LLMError(f"Claude reply for '{tag}' was not valid JSON: {e}") from e
                logger.warning("LLM reply for '%s' was not valid JSON (%s), asking again", tag, e)

    def _web_with_retry(self, system: str, user: str, tag: str, max_searches: int, attempts: int = 3):
        for i in range(attempts):
            try:
                return self._call_web(system, user, max_searches)
            except (MissingInputError, LLMError):
                raise
            except Exception as e:
                status = getattr(e, "status_code", None)
                message = str(e).lower()
                if status in (400, 403) and ("web_search" in message or "web search" in message or "tool" in message):
                    raise WebSearchUnavailable(f"web search is not available: {str(e)[:200]}") from e
                retry = status in RETRY_STATUS or type(e).__name__ in ("APIConnectionError", "APITimeoutError", "RateLimitError")
                logger.warning("LLM web call '%s' failed (%s %s)%s", tag, type(e).__name__, status, ", retrying" if retry and i < attempts - 1 else "")
                if not retry or i == attempts - 1:
                    raise LLMError(f"Claude API call '{tag}' failed: {type(e).__name__}: {str(e)[:300]}") from e
                time.sleep(2 ** (i + 1))

    def _call_web(self, system: str, user: str, max_searches: int):
        if self._client is None:
            import os
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise MissingInputError("ANTHROPIC_API_KEY is not set (add it to .env)")
            import anthropic
            self._client = anthropic.Anthropic(timeout=300.0, max_retries=2)
        tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": max_searches}]
        messages = [{"role": "user", "content": user}]
        sources: dict[str, str] = {}
        tin = tout = 0
        for _ in range(6):                               # a long search turn can pause: send it back to carry on
            msg = self._client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system, tools=tools, messages=messages)
            tin += msg.usage.input_tokens
            tout += msg.usage.output_tokens
            for block in msg.content:
                if block.type == "web_search_tool_result" and isinstance(block.content, list):
                    for r in block.content:
                        if getattr(r, "url", None):
                            sources.setdefault(r.url, getattr(r, "title", "") or r.url)
            if msg.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": msg.content})
                continue
            break
        if getattr(msg, "stop_reason", None) == "max_tokens":
            raise LLMError(f"Claude reply was cut off at max_tokens={self.max_tokens}; raise max_tokens or shorten the request")
        text = "".join(b.text for b in msg.content if b.type == "text")
        return text, [{"url": u, "title": n} for u, n in sources.items()], tin, tout

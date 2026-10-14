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
    def _key(self, system: str, user: str) -> str:
        raw = json.dumps([self.model, config.TEMPERATURE, system, user])
        return hashlib.sha256(raw.encode()).hexdigest()

    def _call_api(self, system: str, user: str) -> tuple[str, int, int]:
        if self._client is None:
            import os
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise MissingInputError("ANTHROPIC_API_KEY is not set (add it to .env)")
            import anthropic
            self._client = anthropic.Anthropic()    # reads ANTHROPIC_API_KEY
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=[{"role": "user", "content": user}])
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

    def _call_with_retry(self, system: str, user: str, tag: str, attempts: int = 3):
        for i in range(attempts):
            try:
                return self._call_api(system, user)
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
    def complete_json(self, system: str, user: str, tag: str = "", json_attempts: int = 2) -> dict:
        """Ask for a JSON reply. An invalid reply is dropped from the cache and asked for once more."""
        key = self._key(system, user)
        cache_file = config.CACHE_DIR / "llm" / f"{key}.json"
        for attempt in range(1, json_attempts + 1):
            t0 = time.time()
            cached = False
            if self.use_cache and cache_file.exists():
                rec = json.loads(cache_file.read_text())
                text, tin, tout = rec["text"], rec["in"], rec["out"]
                cached = True
            else:
                text, tin, tout = self._call_with_retry(system, user, tag)
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

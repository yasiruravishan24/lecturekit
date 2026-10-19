"""Going beyond the lecture (optional, never part of the evaluated configurations A, B and C):

* deeper():        for every topic of the notes: missing background, a worked example, common mistakes, things in the notes to check, and
                   a few real pages to read (found with Claude's web search, so the links exist).
* review_quiz():   a second look at a finished quiz: drops questions that are unclear or have no clear answer, corrects a wrong answer mark.
* make_extra_quiz(): extra questions on the material of deeper().
"""
import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional
from urllib.parse import urlparse

from . import log
from .llm import LLM, WebSearchUnavailable

logger = log.get("enrich")

DEEPER_SYSTEM = """You help a student who attended a university lecture. You get the notes of that lecture, numbered by topic. The lecture
may have skipped background, been too short, or contained mistakes.
For facts you are not sure about, and to find good pages to learn from (official documentation, university course pages, reputable
tutorials), search the web when you can. Write everything in your own words; never copy text from a page. Do not repeat the notes.
Reply ONLY with JSON:
{"topics": [{"index": int (the topic number),
  "background": str (2-4 sentences: what the lecture took for granted or skipped, so a beginner can follow),
  "example": str (a short worked example; code goes in a fenced block with real line breaks),
  "mistakes": [str] (2-3 common mistakes students make with this topic),
  "check": [{"claim": str (what the notes say), "why": str (what looks wrong or unclear, and the correct fact)}],
  "links": [{"title": str, "url": str}] (when you searched, give the 1-3 best pages for this topic from your search results: a short title and the URL copied exactly; an empty list only when you could not search)}],
 "not_covered": [{"title": str (a subtopic usually taught together with these topics but missing here),
                  "explanation": str (3-4 sentences teaching the idea simply, with a tiny example if it helps),
                  "why": str (1-2 sentences: why a student needs it and where it will come up)}] (up to 5)}
"check" must be empty when nothing is doubtful: never invent doubts. If you are unsure of something, say so instead of guessing."""

PICK_SYSTEM = """You choose pages for a student to read. You get the topics of a lecture and a numbered list of web pages (title and address) that
were found while researching them. For each topic pick the 1-3 most useful pages for a learner: official documentation first, then
university course pages, then well-known tutorials. Avoid pages that are only loosely related, forums, videos and paywalled pages.
Reply ONLY with JSON: {"topics": [{"index": int (the topic number), "pages": [int] (numbers from the list)}]}"""

NO_WEB = "\nYou cannot search the web in this run: answer from your own knowledge and leave every \"links\" list empty."

EXTRA_QUIZ_SYSTEM = """You write extra multiple-choice questions for a student, about material that goes BEYOND the lecture (given below as background,
examples, common mistakes and corrections). You also get the questions that already exist: do not repeat them. Reply ONLY with JSON:
{"questions": [{"question": str, "options": [4 strings], "answer_index": 0-3,
 "bloom_level": "remember"|"understand"|"apply"|"analyze", "explanation": str}]}
Exactly one correct option, answerable from the given material. Code goes in a fenced block (```python ... ```) with real line breaks."""

REVIEW_SYSTEM = """You check multiple-choice questions for a university course. For each numbered question decide:
"ok"           the question is clear and the marked answer is the single correct one;
"wrong_answer" the marked answer is wrong, but exactly one other option is clearly correct (give its index);
"bad"          ambiguous, two options are correct, none is correct, it needs facts that were not given, or it is confusing.
A question is fine even if the notes do not mention it, when it is correct and clear. Be strict only about real faults.
Reply ONLY with JSON: {"checks": [{"index": int (the question number), "verdict": "ok"|"wrong_answer"|"bad", "correct_index": int or null, "reason": str}]}"""


def _notes_text(notes: dict) -> str:
    lines = []
    for i, t in enumerate(notes.get("full_notes", []), 1):
        points = "; ".join(t.get("key_points", []))
        lines.append(f"Topic {i}: {t.get('heading', '')}\nSummary: {t.get('summary', '')}\nKey points: {points}")
    lines.append("Short notes: " + " | ".join(notes.get("short_notes", [])))
    return "\n\n".join(lines)


def _clean(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip() if not isinstance(s, str) or "```" not in s else s.strip()


def _norm(url: str) -> str:
    return url.strip().rstrip("/").lower()


def deeper(llm: LLM, notes: dict, max_searches: int = 4, progress: Optional[Callable[[str, int], None]] = None) -> dict:
    """{"web": bool, "topics": [{index, background, example, mistakes, check, links}], "not_covered": [str]}."""
    p = progress or (lambda stage, pct: None)
    n_topics = len(notes.get("full_notes", []))
    user = f"Lecture notes ({n_topics} topics):\n\n{_notes_text(notes)}"
    p("search", 10)
    web = True
    try:
        reply, sources = llm.complete_json_web(DEEPER_SYSTEM, user, tag="deeper", max_searches=max_searches)
    except WebSearchUnavailable as e:
        logger.warning("no web search (%s): going on from the model's own knowledge", e)
        web = False
        reply, sources = llm.complete_json(DEEPER_SYSTEM + NO_WEB, user, tag="deeper"), []
    found = {_norm(s["url"]) for s in sources}
    picked = _pick_pages(llm, notes, sources) if web and sources and not any(isinstance(t, dict) and t.get("links") for t in reply.get("topics", [])) else {}
    topics = []
    for t in reply.get("topics", []):
        if not isinstance(t, dict) or not isinstance(t.get("index"), int) or not 1 <= t["index"] <= n_topics:
            continue
        links = []
        for link in (t.get("links") or picked.get(t["index"], [])) if web else []:                       # only pages that really came out of the search
            url = str(link.get("url", "")) if isinstance(link, dict) else ""
            if urlparse(url).scheme in ("http", "https") and _norm(url) in found:
                links.append({"title": _clean(link.get("title")) or urlparse(url).netloc, "url": url})
        checks = [{"claim": _clean(c.get("claim")), "why": _clean(c.get("why"))} for c in t.get("check", [])
                  if isinstance(c, dict) and _clean(c.get("claim")) and _clean(c.get("why"))]
        topics.append({"index": t["index"], "heading": str(notes["full_notes"][t["index"] - 1].get("heading", "")),
                       "background": t.get("background", "").strip() if isinstance(t.get("background"), str) else "",
                       "example": t.get("example", "").strip() if isinstance(t.get("example"), str) else "",
                       "mistakes": [_clean(m) for m in t.get("mistakes", []) if _clean(m)][:4], "check": checks[:4], "links": links[:3]})
    topics.sort(key=lambda t: t["index"])
    if web:
        _only_working(topics)
    not_covered = []
    for x in reply.get("not_covered", []):
        if isinstance(x, str) and _clean(x):                                      # an older or simpler reply: just the name
            not_covered.append({"title": _clean(x), "explanation": "", "why": ""})
        elif isinstance(x, dict) and _clean(x.get("title")):
            not_covered.append({"title": _clean(x.get("title")), "explanation": _clean(x.get("explanation")), "why": _clean(x.get("why"))})
    not_covered = not_covered[:5]
    return {"web": web, "topics": topics, "not_covered": not_covered}


def _alive(url: str, timeout: float = 8.0) -> bool:
    """Does the page open? Pages that are gone (404, 410), a server that fails (5xx), a host that does not exist or a timeout say no.
    401, 403 and 429 are kept: many good sites turn a program away while a person gets in, so those cannot be judged."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LectureAssistant link check)", "Accept": "text/html,*/*;q=0.5"}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            return r.status < 400
    except urllib.error.HTTPError as e:
        return e.code in (401, 403, 429)
    except Exception:
        return False


def _only_working(topics: list[dict]) -> None:
    """Remove the links that do not open (checked all at once, a few seconds in all)."""
    urls = sorted({link["url"] for t in topics for link in t["links"]})
    if not urls:
        return
    with ThreadPoolExecutor(max_workers=8) as pool:
        ok = dict(zip(urls, pool.map(_alive, urls)))
    for t in topics:
        t["links"] = [link for link in t["links"] if ok.get(link["url"])]


def _pick_pages(llm: LLM, notes: dict, sources: list[dict]) -> dict:
    """Claude often searches but leaves the links out of its answer: ask which of the pages it found suit which topic.
    Returns {topic index: [{"title", "url"}]}; the pages are always ones from the search results."""
    topics = "\n".join(f"{i}. {t.get('heading', '')}" for i, t in enumerate(notes.get("full_notes", []), 1))
    pages = "\n".join(f"{i}. {s['title']} ({s['url']})" for i, s in enumerate(sources[:40], 1))
    try:
        reply = llm.complete_json(PICK_SYSTEM, f"Topics:\n{topics}\n\nPages:\n{pages}", tag="deeper_links")
    except Exception as e:
        logger.warning("could not choose pages: %s", e)
        return {}
    out = {}
    for t in reply.get("topics", []):
        if isinstance(t, dict) and isinstance(t.get("index"), int):
            out[t["index"]] = [{"title": sources[n - 1]["title"], "url": sources[n - 1]["url"]} for n in t.get("pages", [])
                               if isinstance(n, int) and 1 <= n <= min(len(sources), 40)]
    return out


def _valid(q) -> bool:
    opts = q.get("options") if isinstance(q, dict) else None
    return (isinstance(q, dict) and isinstance(q.get("question"), str) and q["question"].strip() and isinstance(opts, list) and len(opts) == 4
            and all(isinstance(o, str) and o.strip() for o in opts) and isinstance(q.get("answer_index"), int) and 0 <= q["answer_index"] < 4)


def apply_review(quiz: list[dict], checks: list[dict], min_keep: int = 5, max_drop_share: float = 0.3) -> tuple[list[dict], dict]:
    """Drop the questions the reviewer calls bad (never more than max_drop_share of them, never below min_keep) and correct wrong marks."""
    verdicts = {c["index"]: c for c in checks if isinstance(c, dict) and isinstance(c.get("index"), int)}
    allowed_drops = max(0, min(int(len(quiz) * max_drop_share), len(quiz) - min_keep))
    out, dropped, fixed = [], [], []
    for i, q in enumerate(quiz, 1):
        v = verdicts.get(i, {})
        if v.get("verdict") == "bad" and len(dropped) < allowed_drops:
            dropped.append({"question": q["question"], "reason": str(v.get("reason", ""))[:200]})
            continue
        if v.get("verdict") == "wrong_answer" and isinstance(v.get("correct_index"), int) and 0 <= v["correct_index"] < 4 and v["correct_index"] != q["answer_index"]:
            q = {**q, "answer_index": v["correct_index"]}
            fixed.append({"question": q["question"], "reason": str(v.get("reason", ""))[:200]})
        out.append(q)
    return out, {"dropped": dropped, "fixed": fixed}


def review_quiz(llm: LLM, material: str, quiz: list[dict], tag: str = "quiz_review") -> tuple[list[dict], dict]:
    """The quiz after a second look: (new quiz, {"dropped": [...], "fixed": [...]}). Any failure of the review leaves the quiz as it is."""
    if not quiz:
        return quiz, {"dropped": [], "fixed": []}
    listing = "\n\n".join(f"Question {i}: {q['question']}\n" + "\n".join(f"  {'ABCD'[j]}. {o}" for j, o in enumerate(q["options"]))
                          + f"\n  Marked answer: {'ABCD'[q['answer_index']]}" for i, q in enumerate(quiz, 1))
    try:
        reply = llm.complete_json(REVIEW_SYSTEM, f"Course material:\n{material}\n\nQuestions:\n{listing}", tag=tag)
        return apply_review(quiz, reply.get("checks", []))
    except Exception as e:                                           # a second look must never lose the quiz
        logger.warning("quiz review skipped: %s", e)
        return quiz, {"dropped": [], "fixed": []}


def make_extra_quiz(llm: LLM, deeper_result: dict, existing: list[str], n: int = 3) -> list[dict]:
    """Up to n questions about the material of deeper(), checked by review_quiz before they are returned, each with "source": "extra"."""
    material = []
    for t in deeper_result.get("topics", []):
        parts = [f"Topic {t['index']} background: {t['background']}"] if t.get("background") else []
        parts += [f"Common mistake: {m}" for m in t.get("mistakes", [])]
        parts += [f"Correction: {c['claim']} -> {c['why']}" for c in t.get("check", [])]
        if t.get("example"):
            parts.append(f"Example: {t['example']}")
        material.extend(parts)
    if n <= 0 or not material:
        return []
    listing = "\n".join(f"- {q}" for q in existing) or "(none)"
    reply = llm.complete_json(EXTRA_QUIZ_SYSTEM, f"Write at most {n} questions.\n\nExisting questions:\n{listing}\n\nMaterial beyond the lecture:\n" + "\n".join(material),
                              tag="quiz_extra")
    seen, new = {q.strip().lower() for q in existing}, []
    for q in reply.get("questions", []):
        if _valid(q) and q["question"].strip().lower() not in seen:
            seen.add(q["question"].strip().lower())
            new.append({"question": q["question"].strip(), "options": [o.strip() for o in q["options"]], "answer_index": q["answer_index"],
                        "bloom_level": q.get("bloom_level") if q.get("bloom_level") in ("remember", "understand", "apply", "analyze") else "understand",
                        "explanation": str(q.get("explanation", "")).strip(), "source": "extra"})
    new = new[:n]
    reviewed, _ = review_quiz(llm, "\n".join(material), new, tag="quiz_extra_review")
    return [{**q, "source": "extra"} for q in reviewed]

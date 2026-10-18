"""Configurations A, B, C (Chapter 5.2.2).

A  speech only
B  speech + slides (slide-aligned context, slide-term correction of the transcript)
C  B + off-topic filtering
"""
import json
import time
from pathlib import Path

from . import align, generate, log, offtopic
from . import config as settings          # the name `config` is a lecture configuration (A, B or C) in this file
from .errors import MissingInputError, ValidationError
from .llm import LLM

logger = log.get("pipeline")

CONFIGS = ("A", "B", "C")


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def load_meta(ldir: Path) -> dict:
    f = ldir / "meta.json"
    if not f.exists():
        raise MissingInputError(f"{f} not found")
    return json.loads(f.read_text())


def clip_segments(ldir: Path) -> list[dict]:
    meta = load_meta(ldir)
    tf = ldir / "transcript.json"
    if not tf.exists():
        raise MissingInputError(f"{tf} not found (run transcribe first)")
    tr = json.loads(tf.read_text())
    a, b = meta.get("clip_start", 0), meta.get("clip_end", 1e9)
    return [g for g in tr["segments"] if a <= (g["start"] + g["end"]) / 2 <= b]


def build_context(config: str, segments: list[dict], slides: list[dict],
                  slide_of: list[int], off: set[int]) -> str:
    if config == "A":
        return "Lecture transcript (speech only):\n" + "\n".join(
            f"[{mmss(g['start'])}] {g['text']}" for g in segments)
    kept = [(g, s) for g, s in zip(segments, slide_of) if g["id"] not in off]
    parts = ["Lecture material: slide text with the speech spoken while it was shown."]
    for si in sorted({s for _, s in kept}):
        text = slides[si]["text"].strip() if si < len(slides) else ""
        speech = " ".join(g["text"] for g, s in kept if s == si)
        parts.append(f"=== SLIDE {si + 1} ===\nSlide text:\n{text}\nSpeech:\n{speech}")
    return "\n\n".join(parts)


def run(ldir: Path, config: str, llm: LLM, n_questions: int = 10, progress=None) -> dict:
    """Run one configuration on a prepared lecture folder.

    progress: optional callback progress(stage, message) with stage in
    align | filter | notes | quiz | write (used by the backend for its progress bar).
    """
    if config not in CONFIGS:
        raise ValidationError(f"unknown configuration {config!r}; use one of {CONFIGS}")
    _p = progress or (lambda stage, message="": None)
    meta = load_meta(ldir)
    segments = clip_segments(ldir)
    slides = []
    if config != "A":
        sf = ldir / "slides.json"
        if not sf.exists():
            raise MissingInputError(f"{sf} not found (config {config} needs slides; run slides first)")
        slides = json.loads(sf.read_text())["slides"]
    logger.info("%s config %s: %d segments, %d slides", ldir.name, config, len(segments), len(slides))
    _p("align", "aligning speech to slides" if config != "A" else "no slides (config A)")
    times, changes, off = {}, [], set()
    slide_of = [0] * len(segments)

    t = time.time()
    if config != "A":
        segments, changes = align.correct_with_slides(segments, slides)
        slide_of = align.align(segments, slides)
    times["align"] = round(time.time() - t, 2)

    t = time.time()
    _p("filter", "filtering off-topic speech" if config == "C" else "filter skipped")
    if config == "C":
        titles = [s["text"].splitlines()[0] if s["text"] else "" for s in slides]
        off = offtopic.find_off_topic(llm, segments, titles)
    times["filter"] = round(time.time() - t, 2)

    context = build_context(config, segments, slides, slide_of, off)
    start = len(llm.calls)
    _p("notes", "writing notes")
    t = time.time()
    notes = generate.make_notes(llm, context)
    times["notes"] = round(time.time() - t, 2)
    _p("quiz", "writing quiz")
    t = time.time()
    quiz = generate.make_quiz(llm, context, n_questions, code_blocks=settings.QUIZ_CODE_BLOCKS)
    times["quiz"] = round(time.time() - t, 2)

    asr_s = json.loads((ldir / "transcript.json").read_text()).get("seconds", 0)
    ratio_total = (asr_s + sum(times.values()))
    duration = meta.get("clip_end", 0) - meta.get("clip_start", 0)
    calls = llm.calls[start:]
    result = {
        "lecture": ldir.name, "config": config, "model": llm.model,
        "notes": notes, "quiz": quiz,
        "off_topic_ids": sorted(off), "slide_corrections": changes,
        "stats": {"asr_seconds": asr_s, **times,
                  "total_seconds": round(ratio_total, 1), "clip_seconds": duration,
                  "ratio": round(ratio_total / duration, 2) if duration else None,
                  "temperature_fixed": llm.temperature_set,
                  "tokens_in": sum(c["tokens_in"] for c in calls),
                  "tokens_out": sum(c["tokens_out"] for c in calls)},
    }
    _p("write", "saving results")
    out = ldir / "out"
    out.mkdir(exist_ok=True)
    (out / f"{config}.json").write_text(json.dumps(result, indent=1))
    (out / f"{config}.md").write_text(render_markdown(result))
    # corrected transcript (for WER with slide terms)
    if config != "A":
        (out / f"transcript_{config}.json").write_text(json.dumps(segments, indent=1))
    logger.info("%s %s done: %s s, %d off-topic segments removed, %d slide corrections",
                ldir.name, config, result["stats"]["total_seconds"], len(off), len(changes))
    log.event("run_done", lecture=ldir.name, config=config, model=llm.model,
              seconds=result["stats"]["total_seconds"], tokens_in=result["stats"]["tokens_in"],
              tokens_out=result["stats"]["tokens_out"], questions=len(quiz), off_topic=len(off))
    return result


def render_markdown(r: dict) -> str:
    n = r["notes"]
    md = [f"# {n.get('title', r['lecture'])}", "", "## Notes"]
    for t in n.get("full_notes", []):
        md += [f"### {t['heading']}", t["summary"], ""] + [f"- {k}" for k in t.get("key_points", [])] + [""]
    md += ["## Short notes"] + [f"- {s}" for s in n.get("short_notes", [])] + ["", "## Quiz"]
    for i, q in enumerate(r["quiz"], 1):
        md.append(f"{i}. ({q.get('bloom_level', '?')}) {q['question']}")
        md += [f"   {'ABCD'[j]}. {o}" for j, o in enumerate(q["options"])]
        md.append(f"   Answer: {'ABCD'[q['answer_index']]} - {q.get('explanation', '')}")
    return "\n".join(md) + "\n"

"""Evaluation measures (Chapter 3.9 / 5.3): WER, slide-text accuracy, filter P/R/F1."""
import json
import re
from pathlib import Path

from .pipeline import clip_segments, load_meta


def norm(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def wer(ref: str, hyp: str) -> float:
    import jiwer
    return jiwer.wer(norm(ref), norm(hyp))


def wer_details(ref: str, hyp: str) -> dict:
    """Word counts and error types, to sanity-check a WER value."""
    import jiwer
    o = jiwer.process_words(norm(ref), norm(hyp))
    return {"ref_words": len(norm(ref).split()), "hyp_words": len(norm(hyp).split()),
            "substitutions": o.substitutions, "deletions": o.deletions,
            "insertions": o.insertions}


def ref_window(ldir: Path) -> tuple[float, float]:
    m = load_meta(ldir)
    return m["ref_start"], m["ref_end"]


def hyp_text(segments: list[dict], a: float, b: float) -> str:
    return " ".join(g["text"] for g in segments if a <= (g["start"] + g["end"]) / 2 <= b)


def wer_lecture(ldir: Path) -> dict:
    """WER of the raw transcript (A) and the slide-corrected transcript (B/C)."""
    a, b = ref_window(ldir)
    ref = re.sub(r"\[[^\]]*\]", " ", (ldir / "ref_transcript.txt").read_text())  # ignore [mm:ss] tags
    raw = json.loads((ldir / "transcript.json").read_text())["segments"]
    raw_hyp = hyp_text(raw, a, b)
    res = {"wer_without_slides": wer(ref, raw_hyp), "details": wer_details(ref, raw_hyp)}
    corr = ldir / "out" / "transcript_B.json"
    if corr.exists():
        res["wer_with_slides"] = wer(ref, hyp_text(json.loads(corr.read_text()), a, b))
    return res


def slide_accuracy(ldir: Path) -> float:
    """Word accuracy of OCR slide text = 1 - WER against hand-typed slide text."""
    ref = (ldir / "ref_slides.txt").read_text()
    hyp = " ".join(s["text"] for s in json.loads((ldir / "slides.json").read_text())["slides"])
    return max(0.0, 1.0 - wer(ref, hyp))


def _gold_pred(ldir: Path, drop_before: float | None = None):
    spans = json.loads((ldir / "off_topic_spans.json").read_text())
    pred = set(json.loads((ldir / "out" / "C.json").read_text())["off_topic_ids"])
    segs = clip_segments(ldir)
    if drop_before is not None:                        # e.g. ignore the OCW licence notice
        segs = [g for g in segs if (g["start"] + g["end"]) / 2 > drop_before]
    gold = {g["id"] for g in segs
            if any(s <= (g["start"] + g["end"]) / 2 <= e for s, e in spans)}
    return segs, gold, pred & {g["id"] for g in segs}


def filter_prf(ldir: Path, drop_before: float | None = None) -> dict:
    """Precision/recall/F1 of the off-topic filter on the clip's segments.

    off_topic_spans.json = researcher labels, a list of [start, end] seconds.
    drop_before: ignore segments whose midpoint is before this time (licence notice).
    """
    segs, gold, pred = _gold_pred(ldir, drop_before)
    tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"segments": len(segs), "gold_off_topic": len(gold), "predicted": len(pred),
            "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def slide_word_scores(ldir: Path) -> dict:
    """Order-independent OCR quality: overlap of words (multiset) between OCR and reference.

    precision = share of OCR words found in the reference; recall = share of reference
    words found in the OCR text. Complements slide_accuracy (which also penalises a
    different reading order, e.g. code boxes or columns).
    """
    from collections import Counter
    ref = Counter(norm((ldir / "ref_slides.txt").read_text()).split())
    hyp = Counter(norm(" ".join(s["text"] for s in
                  json.loads((ldir / "slides.json").read_text())["slides"])).split())
    overlap = sum((ref & hyp).values())
    p = overlap / max(1, sum(hyp.values()))
    r = overlap / max(1, sum(ref.values()))
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def filter_prf_without_licence(ldir: Path) -> dict | None:
    """Same as filter_prf but ignores the intro up to meta['licence_end'] (None if not set)."""
    end = load_meta(ldir).get("licence_end")
    return None if end is None else filter_prf(ldir, drop_before=end)


def filter_diff(ldir: Path) -> list[dict]:
    """Segments where the filter and the labels disagree: FP (removed, not labelled) / FN (kept, labelled)."""
    segs, gold, pred = _gold_pred(ldir)
    rows = []
    for g in segs:
        if g["id"] in pred and g["id"] not in gold:
            rows.append({"type": "FP", **g})
        elif g["id"] in gold and g["id"] not in pred:
            rows.append({"type": "FN", **g})
    return rows

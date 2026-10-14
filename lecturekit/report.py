"""Collect results into the tables used in Chapter 5 (Tables 7-11)."""
import json
from pathlib import Path

from . import evaluate, rating
from .pipeline import CONFIGS, load_meta


def build(data_dir: Path, lectures: list[str], pack_dir: Path | None = None) -> str:
    md = ["# Results (generated; copy into Chapter 5)\n"]
    md += ["## Table 7 - System performance (configuration C)",
           "| Lecture | Clip (s) | ASR (s) | Align | Filter | Notes | Quiz | Total (s) | Ratio | Tokens in/out |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    ratios = []
    for lec in lectures:
        p = data_dir / lec / "out" / "C.json"
        if not p.exists(): continue
        s = json.loads(p.read_text())["stats"]
        ratios.append(s["ratio"] or 0)
        md.append(f"| {lec} | {s['clip_seconds']:.0f} | {s['asr_seconds']} | {s['align']} | {s['filter']} "
                  f"| {s['notes']} | {s['quiz']} | {s['total_seconds']} | {s['ratio']} | {s['tokens_in']}/{s['tokens_out']} |")
    if ratios: md.append(f"\nMean ratio: {sum(ratios) / len(ratios):.2f}\n")

    md += ["## Table 8 - WER (reference window only)", "| Lecture | Without slides | With slides | Change (points) |", "|---|---|---|---|"]
    w0, w1 = [], []
    for lec in lectures:
        try: r = evaluate.wer_lecture(data_dir / lec)
        except FileNotFoundError: continue
        a, b = r["wer_without_slides"] * 100, r.get("wer_with_slides", float("nan")) * 100
        w0.append(a); w1.append(b)
        md.append(f"| {lec} | {a:.1f} | {b:.1f} | {b - a:+.1f} |")
    if w0: md.append(f"| Mean | {sum(w0)/len(w0):.1f} | {sum(w1)/len(w1):.1f} | {(sum(w1)-sum(w0))/len(w0):+.1f} |\n")

    md += ["## Slide text accuracy (OCR)", "| Lecture | Word accuracy |", "|---|---|"]
    for lec in lectures:
        try: md.append(f"| {lec} | {evaluate.slide_accuracy(data_dir / lec) * 100:.1f}% |")
        except FileNotFoundError: pass

    md += ["\n## Table 9 - Off-topic filter", "| Lecture | Segments | Gold off-topic | Predicted | P | R | F1 |", "|---|---|---|---|---|---|---|"]
    tp = fp = fn = 0
    for lec in lectures:
        try: r = evaluate.filter_prf(data_dir / lec)
        except FileNotFoundError: continue
        tp += r["tp"]; fp += r["fp"]; fn += r["fn"]
        md.append(f"| {lec} | {r['segments']} | {r['gold_off_topic']} | {r['predicted']} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1']:.2f} |")
    if tp + fp and tp + fn:
        p, r_ = tp / (tp + fp), tp / (tp + fn)
        md.append(f"| Overall | | | | {p:.2f} | {r_:.2f} | {2*p*r_/(p+r_) if p+r_ else 0:.2f} |\n")

    if pack_dir and (pack_dir / "notes_rater1.csv").exists():
        for kind, crit in (("notes", rating.NOTE_CRITERIA), ("questions", rating.Q_CRITERIA)):
            files = {r: pack_dir / f"{kind}_{r}.csv" for r in ("rater1", "rater2")}
            s = rating.summarise(pack_dir, files, kind)
            md += [f"## Table {10 if kind == 'notes' else 11} - {kind} ratings (mean of both raters)",
                   "| Config | " + " | ".join(crit) + " | overall |", "|---|" + "---|" * (len(crit) + 1)]
            for c in CONFIGS:
                if c in s: md.append(f"| {c} | " + " | ".join(str(s[c].get(k, "")) for k in crit) + f" | {s[c]['overall']} |")
            k, n = rating.weighted_kappa(rating.read_ratings(files["rater1"]), rating.read_ratings(files["rater2"]))
            md.append(f"\nWeighted kappa ({kind}): {k:.2f} ({rating.kappa_band(k)}), n = {n}\n")
    return "\n".join(md)

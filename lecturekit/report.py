"""Collect results into the tables used in Chapter 5 (Tables 7-12, effects, objective outcome words).

Everything here is computed from files under data/<ID>/ and the rating pack; nothing is typed by hand.
"""
import json
from pathlib import Path

from . import evaluate, rating
from .pipeline import CONFIGS

# Decision 4 (TESTING.md section 11): fixed thresholds for the objective outcome words.
# Written before the final runs and applied mechanically; do not change them after seeing results.
ACHIEVED, LARGELY, PARTIAL, NOT_COMPUTED = "Achieved", "Largely achieved", "Partially achieved", "not computed"
RATING_ACHIEVED, RATING_LARGELY = 4.0, 3.5          # mean rating (objectives 2 and 3)
SHARE_ACHIEVED, SHARE_LARGELY = 75.0, 60.0          # share clear and answerable, percent (objective 3)


def outcome_pipeline(n_complete: int, total: int) -> str:
    """Objective 1: pipeline completes (A, B and C outputs exist) on all / at least 2 of 3 lectures."""
    if total == 0:
        return NOT_COMPUTED
    if n_complete == total:
        return ACHIEVED
    return LARGELY if n_complete >= total * 2 / 3 else PARTIAL


def outcome_notes(mean_c) -> str:
    """Objective 2: mean note rating of configuration C (both raters)."""
    if mean_c is None:
        return NOT_COMPUTED
    return ACHIEVED if mean_c >= RATING_ACHIEVED else LARGELY if mean_c >= RATING_LARGELY else PARTIAL


def outcome_questions(mean_c, share) -> str:
    """Objective 3: mean question rating of C and share clear and answerable."""
    if mean_c is None or share is None:
        return NOT_COMPUTED
    if mean_c >= RATING_ACHIEVED and share >= SHARE_ACHIEVED:
        return ACHIEVED
    return LARGELY if mean_c >= RATING_LARGELY or share >= SHARE_LARGELY else PARTIAL


def outcome_evaluation(metrics_missing: int, raters_incomplete: int) -> str:
    """Objective 4: all metrics computed and both raters complete / one gap / otherwise."""
    if metrics_missing == 0 and raters_incomplete == 0:
        return ACHIEVED
    return LARGELY if metrics_missing + raters_incomplete == 1 else PARTIAL


def _f(x, nd=2, suffix=""):
    return "n/a" if x is None or x != x else f"{x:.{nd}f}{suffix}"


def _delta(a, b):
    return None if a is None or b is None else round(b - a, 2)


def _rater_files(pack_dir: Path, kind: str) -> dict:
    return {r: pack_dir / f"{kind}_{r}.csv" for r in ("rater1", "rater2")}


def _pooled(rows: list) -> dict | None:
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    if not rows or not (tp + fp and tp + fn):
        return None
    p, r = tp / (tp + fp), tp / (tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def build(data_dir: Path, lectures: list[str], pack_dir: Path | None = None) -> str:
    md = ["# Results (generated from data/<ID>/ and the rating pack; copy into Chapter 5)\n"]
    metrics_missing = 0

    # ---- Table 7: system performance --------------------------------------------------------------
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

    # ---- Table 8: WER ------------------------------------------------------------------------------
    md += ["## Table 8 - WER (reference window only)", "| Lecture | Without slides | With slides | Change (points) |", "|---|---|---|---|"]
    w0, w1 = [], []
    for lec in lectures:
        try: r = evaluate.wer_lecture(data_dir / lec)
        except FileNotFoundError: continue
        a, b = r["wer_without_slides"] * 100, r.get("wer_with_slides", float("nan")) * 100
        w0.append(a); w1.append(b)
        md.append(f"| {lec} | {a:.1f} | {b:.1f} | {b - a:+.1f} |")
    if w0: md.append(f"| Mean | {sum(w0)/len(w0):.1f} | {sum(w1)/len(w1):.1f} | {(sum(w1)-sum(w0))/len(w0):+.1f} |\n")
    else: metrics_missing += 1
    wer_a = sum(w0) / len(w0) if w0 else None
    wer_b = sum(w1) / len(w1) if w1 else None

    # ---- slide text accuracy ----------------------------------------------------------------------
    md += ["## Slide text accuracy (OCR)", "| Lecture | Word accuracy (order-sensitive) | Word overlap P / R / F1 |", "|---|---|---|"]
    n_slide = 0
    for lec in lectures:
        try:
            sw = evaluate.slide_word_scores(data_dir / lec)
            md.append(f"| {lec} | {evaluate.slide_accuracy(data_dir / lec) * 100:.1f}% | "
                      f"{sw['precision']:.2f} / {sw['recall']:.2f} / {sw['f1']:.2f} |")
            n_slide += 1
        except FileNotFoundError: pass
    metrics_missing += n_slide == 0

    # ---- Table 9: filter --------------------------------------------------------------------------
    md += ["\n## Table 9 - Off-topic filter", "| Lecture | Segments | Gold off-topic | Predicted | P | R | F1 |", "|---|---|---|---|---|---|---|"]
    rows, rows_ex, ex_names = [], [], []
    for lec in lectures:
        try: r = evaluate.filter_prf(data_dir / lec)
        except FileNotFoundError: continue
        rows.append(r)
        md.append(f"| {lec} | {r['segments']} | {r['gold_off_topic']} | {r['predicted']} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1']:.2f} |")
        try: ex = evaluate.filter_prf_without_licence(data_dir / lec)
        except FileNotFoundError: ex = None
        if ex: rows_ex.append(ex); ex_names.append((lec, ex))
    pooled = _pooled(rows)
    if pooled:
        md.append(f"| Overall (pooled counts) | | | | {pooled['precision']:.2f} | {pooled['recall']:.2f} | {pooled['f1']:.2f} |")
    else:
        metrics_missing += 1
    if ex_names:
        md += ["", "Without the OCW licence notice (segments after `licence_end` only):",
               "| Lecture | Segments | Gold off-topic | Predicted | P | R | F1 |", "|---|---|---|---|---|---|---|"]
        md += [f"| {lec} | {r['segments']} | {r['gold_off_topic']} | {r['predicted']} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1']:.2f} |"
               for lec, r in ex_names]
        pe = _pooled(rows_ex)
        if pe: md.append(f"| Overall (pooled counts) | | | | {pe['precision']:.2f} | {pe['recall']:.2f} | {pe['f1']:.2f} |")
    md.append("")

    # ---- ratings: Tables 10-12, effects, objectives -----------------------------------------------
    sn = sq = shares = None
    raters_incomplete = 0
    have_ratings = bool(pack_dir and (pack_dir / "notes_rater1.csv").exists())
    if have_ratings:
        key = json.loads((pack_dir / "KEY_DO_NOT_SHARE.json").read_text())
        for kind, crit in (("notes", rating.NOTE_CRITERIA), ("questions", rating.Q_CRITERIA)):
            files = _rater_files(pack_dir, kind)
            s = rating.summarise(pack_dir, files, kind)
            if kind == "notes": sn = s
            else: sq = s
            md += [f"## Table {10 if kind == 'notes' else 11} - {kind} ratings (mean of both raters)",
                   "| Config | " + " | ".join(crit) + " | overall |", "|---|" + "---|" * (len(crit) + 1)]
            for c in CONFIGS:
                if c in s: md.append(f"| {c} | " + " | ".join(str(s[c].get(k, "")) for k in crit) + f" | {s[c]['overall']} |")
            k, n = rating.weighted_kappa(rating.read_ratings(files["rater1"]), rating.read_ratings(files["rater2"]))
            md.append(f"\nWeighted kappa ({kind}): {_f(k)} ({rating.kappa_band(k)}), n = {n}\n")
            for r, path in files.items():
                comp = rating.completeness(path, key)
                if comp["missing"]:
                    raters_incomplete += 1
                    md.append(f"*{r}: {comp['missing']} of {comp['total']} {kind} cells still empty.*\n")
        shares = rating.question_shares(pack_dir, _rater_files(pack_dir, "questions"))
        md += ["## Table 11b - Question quality shares (first questions of each set, both raters)",
               f"Clear and answerable = clarity >= {rating.CLEAR_MIN} and answerability >= {rating.CLEAR_MIN}; "
               f"Bloom-matched = Bloom alignment >= {rating.CLEAR_MIN}. One judgement = one question rated by one rater.", "",
               "| Config | Questions rated | Judgements | Clear and answerable | Bloom-matched |", "|---|---|---|---|---|"]
        for c in CONFIGS:
            v = shares[c]
            md.append(f"| {c} | {v['questions_rated']} | {v['judgements']} | {_f(v['clear_answerable_pct'], 0, '%')} | {_f(v['bloom_matched_pct'], 0, '%')} |")

        use_files = {r: pack_dir / f"usefulness_{r}.csv" for r in ("rater1", "rater2")}
        if all(p.exists() for p in use_files.values()):
            u = rating.usefulness(use_files)
            md += ["", "## Table 12 - Overall usefulness (1-5)", "| Statement | Rater 1 | Rater 2 | Mean |", "|---|---|---|---|"]
            for uid, text in rating.USEFUL_STATEMENTS.items():
                v = u["statements"][uid]
                md.append(f"| {text} | {v['rater1'] if v['rater1'] is not None else ''} | {v['rater2'] if v['rater2'] is not None else ''} | {_f(v['mean'])} |")
            md.append(f"| **Overall mean** | | | {_f(u['overall'])} |\n")

        # ---- effects of slide grounding (A to B) and filtering (B to C) --------------------------
        md += ["## Effects between configurations (differences of the means above)",
               "| Measure | A | B | C | A to B (slides) | B to C (filter) |", "|---|---|---|---|---|---|"]
        for label, s, crit in (("Notes", sn, rating.NOTE_CRITERIA), ("Questions", sq, rating.Q_CRITERIA)):
            for k in crit + ["overall"]:
                v = {c: s.get(c, {}).get(k) for c in CONFIGS}
                md.append(f"| {label}: {k} | {_f(v['A'])} | {_f(v['B'])} | {_f(v['C'])} | "
                          f"{_f(_delta(v['A'], v['B']))} | {_f(_delta(v['B'], v['C']))} |")
        md.append(f"| WER (%) | {_f(wer_a, 1)} | {_f(wer_b, 1)} | {_f(wer_b, 1)} | {_f(_delta(wer_a, wer_b), 1)} | n/a (the filter does not change the transcript) |")
        for label, s in (("notes", sn), ("questions", sq)):
            ab = _delta(s.get("A", {}).get("overall"), s.get("B", {}).get("overall"))
            bc = _delta(s.get("B", {}).get("overall"), s.get("C", {}).get("overall"))
            if ab is not None and bc is not None:
                bigger = "slide grounding (A to B)" if abs(ab) >= abs(bc) else "filtering (B to C)"
                md.append(f"\nLargest change in overall {label} rating: {bigger} ({ab:+.2f} vs {bc:+.2f}).")
        md.append("")

    # ---- objectives (Tables 14 and 15) --------------------------------------------------------------
    n_complete = sum(all((data_dir / lec / "out" / f"{c}.json").exists() for c in CONFIGS) for lec in lectures)
    mean_notes_c = (sn or {}).get("C", {}).get("overall")
    mean_q_c = (sq or {}).get("C", {}).get("overall")
    share_c = (shares or {}).get("C", {}).get("clear_answerable_pct")
    if not have_ratings:
        metrics_missing += 3                       # note ratings, question ratings, kappa
    md += ["## Objective outcome words (fixed thresholds, TESTING.md section 11 decision 4)",
           "| Objective | Evidence | Outcome |", "|---|---|---|",
           f"| 1 Pipeline for capture and processing | all of A, B, C produced for {n_complete} of {len(lectures)} lectures | {outcome_pipeline(n_complete, len(lectures))} |",
           f"| 2 Note generation | mean note rating, configuration C: {_f(mean_notes_c)} (Achieved >= {RATING_ACHIEVED}, Largely >= {RATING_LARGELY}) | {outcome_notes(mean_notes_c)} |",
           f"| 3 Knowledge assessment | mean question rating C: {_f(mean_q_c)}; clear and answerable: {_f(share_c, 0, '%')} (Achieved: >= {RATING_ACHIEVED} and >= {SHARE_ACHIEVED:.0f}%; Largely: mean >= {RATING_LARGELY} or share >= {SHARE_LARGELY:.0f}%) | {outcome_questions(mean_q_c, share_c)} |",
           f"| 4 Technical and rater evaluation | metrics missing: {metrics_missing}; incomplete rater files: {raters_incomplete} | {outcome_evaluation(metrics_missing, raters_incomplete)} |",
           ""]
    return "\n".join(md)


def figure_note_ratings_svg(pack_dir: Path) -> str | None:
    """Figure 21: grouped bars of the mean note rating per criterion and configuration (None if no ratings)."""
    if not (pack_dir / "notes_rater1.csv").exists():
        return None
    s = rating.summarise(pack_dir, _rater_files(pack_dir, "notes"), "notes")
    if not s:
        return None
    cats = rating.NOTE_CRITERIA + ["overall"]
    colours = {"A": "#3B6FB6", "B": "#E0A030", "C": "#4C9F8A"}
    W, H, L, T, B = 640, 360, 50, 30, 60
    pw, ph = W - L - 20, H - T - B
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{W / 2}" y="18" text-anchor="middle" font-size="14">Mean note rating by criterion and configuration (1-5)</text>']
    for v in range(0, 6):
        y = T + ph - ph * v / 5
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - 20}" y2="{y:.1f}" stroke="#ddd"/>'
                   f'<text x="{L - 6}" y="{y + 4:.1f}" text-anchor="end">{v}</text>')
    gw = pw / len(cats)
    bw = gw * 0.7 / len(CONFIGS)
    for i, k in enumerate(cats):
        for j, c in enumerate(CONFIGS):
            val = s.get(c, {}).get(k)
            if val is None: continue
            x = L + i * gw + gw * 0.15 + j * bw
            h = ph * val / 5
            out.append(f'<rect x="{x:.1f}" y="{T + ph - h:.1f}" width="{bw - 2:.1f}" height="{h:.1f}" fill="{colours[c]}"/>'
                       f'<text x="{x + bw / 2 - 1:.1f}" y="{T + ph - h - 3:.1f}" text-anchor="middle" font-size="10">{val}</text>')
        out.append(f'<text x="{L + i * gw + gw / 2:.1f}" y="{H - B + 18}" text-anchor="middle">{k}</text>')
    for j, c in enumerate(CONFIGS):
        x = L + j * 90
        out.append(f'<rect x="{x}" y="{H - 22}" width="12" height="12" fill="{colours[c]}"/>'
                   f'<text x="{x + 18}" y="{H - 12}">Configuration {c}</text>')
    out.append("</svg>")
    return "\n".join(out)

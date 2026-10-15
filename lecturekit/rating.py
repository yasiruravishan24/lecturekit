"""Blind rating packs for the two raters, agreement (weighted kappa) and summaries."""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from .errors import MissingInputError, ValidationError
from .pipeline import CONFIGS, load_meta

NOTE_CRITERIA = ["accuracy", "completeness", "structure", "relevance"]
Q_CRITERIA = ["clarity", "relevance", "difficulty", "bloom", "answerability"]
RATED_QUESTIONS = 4        # decision 1 (TESTING.md section 11): the first 4 of the 10 questions of a set are rated
# Table 12: overall usefulness, rated once per rater (1-5), the five Appendix B statements
USEFUL_STATEMENTS = {
    "U1": "The notes are easy to read.",
    "U2": "Using the notes and questions would save time compared with watching the lecture again.",
    "U3": "The questions help check understanding.",
    "U4": "The notes cover the main points of the lecture.",
    "U5": "I would recommend this tool to a student.",
}
CLEAR_MIN = 4              # decision 2: clear and answerable = clarity >= 4 and answerability >= 4; Bloom-matched = bloom >= 4


def make_packs(data_dir: Path, lectures: list[str], out_dir: Path, seed: int = 2026,
               rated_questions: int = RATED_QUESTIONS) -> None:
    """One anonymous set per (lecture, configuration); configuration hidden, order shuffled.

    Only the first `rated_questions` questions of each set are listed and get CSV rows (all questions
    stay in out/<config>.json). Also writes the usefulness CSV (Table 12) for each rater.
    """
    items = [(lec, c) for lec in lectures for c in CONFIGS]
    random.Random(seed).shuffle(items)
    out_dir.mkdir(parents=True, exist_ok=True)
    key, md = {}, ["# Rating pack\n",
                   "Rate every set on the 1-5 scale in Appendix B. Do not try to guess how a set "
                   "was made. Fill the CSV files: notes_<rater>.csv one row per set, questions_<rater>.csv one row per "
                   f"question (the first {rated_questions} questions of each set are shown and rated), and "
                   "usefulness_<rater>.csv once, after you have read all sets.\n"]
    for i, (lec, cfg) in enumerate(items, 1):
        sid = f"S{i:02d}"
        key[sid] = {"lecture": lec, "config": cfg}
        res = json.loads((data_dir / lec / "out" / f"{cfg}.json").read_text())
        n = res["notes"]
        md += [f"\n---\n## Set {sid}  (lecture: {load_meta(data_dir / lec).get('title', lec)})\n",
               f"### Notes: {n.get('title', '')}"]
        for t in n.get("full_notes", []):
            md += [f"**{t['heading']}** - {t['summary']}"] + [f"- {k}" for k in t.get("key_points", [])]
        md += ["", "### Short notes"] + [f"- {s}" for s in n.get("short_notes", [])]
        md += ["", "### Questions"]
        for q_no, q in enumerate(res["quiz"][:rated_questions], 1):
            md.append(f"Q{q_no}. {q['question']}")
            md += [f"   {'ABCD'[j]}. {o}" for j, o in enumerate(q["options"])]
            md.append(f"   Intended Bloom level: {q.get('bloom_level', '?')}")
    md += ["\n---\n# Overall usefulness (rate once, 1-5, after reading all sets)\n"]
    md += [f"- {sid}. {text}" for sid, text in USEFUL_STATEMENTS.items()]
    # source material per lecture so raters can judge accuracy / completeness
    md += ["\n---\n# Source material (same for all sets of a lecture)\n"]
    for lec in lectures:
        ref = data_dir / lec / "ref_transcript.txt"
        md += [f"## {lec}", ref.read_text() if ref.exists() else "(add the reference transcript)", ""]
    (out_dir / "rating_pack.md").write_text("\n".join(md))
    (out_dir / "KEY_DO_NOT_SHARE.json").write_text(json.dumps(key, indent=1))
    for rater in ("rater1", "rater2"):
        with (out_dir / f"notes_{rater}.csv").open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["set_id"] + NOTE_CRITERIA)
            for sid in key: w.writerow([sid] + [""] * len(NOTE_CRITERIA))
        with (out_dir / f"questions_{rater}.csv").open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["set_id", "q_no"] + Q_CRITERIA)
            for sid, v in key.items():
                nq = len(json.loads((data_dir / v["lecture"] / "out" / f"{v['config']}.json").read_text())["quiz"])
                for q_no in range(1, min(nq, rated_questions) + 1): w.writerow([sid, q_no] + [""] * len(Q_CRITERIA))
        with (out_dir / f"usefulness_{rater}.csv").open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["set_id", "score"])
            for uid in USEFUL_STATEMENTS: w.writerow([uid, ""])


# Simplified layouts the raters used instead of the per-criterion template (CHANGELOG.md decision 6):
# one overall score per set / per question / per usefulness statement.
OVERALL_LAYOUTS = {("Set", "Rating"): (("Set",), "overall"),
                   ("Set", "Question", "Rating"): (("Set", "Question"), "overall"),
                   ("Criteria", "Rating"): (("Criteria",), "score")}


def overall_layout(fieldnames) -> tuple | None:
    """(id columns, criterion name) if the CSV is one of the simplified overall-score layouts, else None."""
    return OVERALL_LAYOUTS.get(tuple(fieldnames or ()))


def read_ratings(path: Path, key: dict | None = None) -> dict[tuple, int]:
    """Read a filled rating CSV. Raises ValidationError with the file, row and column on bad data.

    key: optional KEY_DO_NOT_SHARE.json content; if given, unknown set ids are rejected.
    Also reads the simplified overall-score layouts: keys ("S01", "overall"), ("S01", "Q1", "overall"), ("U1", "score").
    """
    if not Path(path).exists():
        raise MissingInputError(f"rating file not found: {path}")
    out = {}
    with Path(path).open(newline="") as f:
        rd = csv.DictReader(f)
        lay = overall_layout(rd.fieldnames)
        if lay is not None:
            id_cols, crit = lay
            for n, row in enumerate(rd, 2):
                if key is not None and "Set" in id_cols and row["Set"] not in key:
                    raise ValidationError(f"{path} line {n}: unknown set {row['Set']!r}")
                v = (row["Rating"] or "").strip()
                if not v:
                    continue
                try:
                    val = int(v)
                except ValueError:
                    raise ValidationError(f"{path} line {n}, column 'Rating': {v!r} is not a whole number") from None
                if not 1 <= val <= 5:
                    raise ValidationError(f"{path} line {n}, column 'Rating': {val} is outside 1-5")
                out[tuple(row[c] for c in id_cols) + (crit,)] = val
            return out
        if not rd.fieldnames or "set_id" not in rd.fieldnames:
            raise ValidationError(f"{path}: first column must be 'set_id'")
        for n, row in enumerate(rd, 2):
            if key is not None and row["set_id"] not in key:
                raise ValidationError(f"{path} line {n}: unknown set_id {row['set_id']!r}")
            ids = tuple(row[k] for k in ("set_id", "q_no") if k in row)
            for k, v in row.items():
                if k in ("set_id", "q_no") or v is None or not v.strip():
                    continue
                try:
                    val = int(v)
                except ValueError:
                    raise ValidationError(f"{path} line {n}, column '{k}': {v!r} is not a whole number") from None
                if not 1 <= val <= 5:
                    raise ValidationError(f"{path} line {n}, column '{k}': {val} is outside 1-5")
                out[ids + (k,)] = val
    return out


def completeness(path: Path, key: dict) -> dict:
    """How many cells are still empty in a rating CSV (to chase a rater)."""
    filled = len(read_ratings(path, key))
    with Path(path).open(newline="") as f:
        rd = csv.DictReader(f)
        lay = overall_layout(rd.fieldnames)
        rows = list(rd)
    total = len(rows) if lay is not None else sum(1 for r in rows for k in r if k not in ("set_id", "q_no"))
    return {"filled": filled, "total": total, "missing": total - filled}


def weighted_kappa(a: dict, b: dict) -> tuple[float, int]:
    from sklearn.metrics import cohen_kappa_score
    keys = sorted(set(a) & set(b))
    x, y = [a[k] for k in keys], [b[k] for k in keys]
    if len(set(x) | set(y)) < 2:
        return float("nan"), len(keys)
    return float(cohen_kappa_score(x, y, weights="quadratic", labels=[1, 2, 3, 4, 5])), len(keys)


def kappa_band(k: float) -> str:
    if k != k:                      # NaN: not enough variation to compute kappa
        return "n/a"
    for hi, name in [(0, "poor"), (0.20, "slight"), (0.40, "fair"), (0.60, "moderate"),
                     (0.80, "substantial"), (1.01, "almost perfect")]:
        if k <= hi: return name
    return "almost perfect"


def summarise(pack_dir: Path, rater_files: dict[str, Path], kind: str) -> dict:
    """Mean rating per configuration and criterion (both raters pooled)."""
    key = json.loads((pack_dir / "KEY_DO_NOT_SHARE.json").read_text())
    acc = defaultdict(list)
    for path in rater_files.values():
        for k, v in read_ratings(path, key).items():
            acc[(key[k[0]]["config"], k[-1])].append(v)
    out = defaultdict(dict)
    for (cfg, crit), vals in acc.items():
        out[cfg][crit] = round(sum(vals) / len(vals), 2)
    for cfg in out:
        out[cfg]["overall"] = round(sum(out[cfg].values()) / len(out[cfg]), 2)
    return dict(out)


def question_shares(pack_dir: Path, rater_files: dict[str, Path]) -> dict:
    """Per configuration: questions rated, share clear and answerable, share Bloom-matched (decision 2).

    One judgement = one question rated by one rater. Clear and answerable = clarity >= 4 and
    answerability >= 4; Bloom-matched = bloom >= 4. Only judgements with the needed cells filled count.
    """
    key = json.loads((pack_dir / "KEY_DO_NOT_SHARE.json").read_text())
    cells = defaultdict(dict)                       # (config, rater, set, q_no) -> {criterion: value}
    for rater, path in rater_files.items():
        for (sid, q_no, crit), v in read_ratings(path, key).items():
            cells[(key[sid]["config"], rater, sid, q_no)][crit] = v
    out = {}
    for cfg in CONFIGS:
        rows = [c for (cf, *_), c in cells.items() if cf == cfg]
        ca = [c for c in rows if "clarity" in c and "answerability" in c]
        bm = [c for c in rows if "bloom" in c]
        out[cfg] = {
            "questions_rated": len({(sid, q) for (cf, _, sid, q) in cells if cf == cfg}),
            "judgements": len(rows),
            "clear_answerable_pct": (100 * sum(c["clarity"] >= CLEAR_MIN and c["answerability"] >= CLEAR_MIN
                                               for c in ca) / len(ca)) if ca else None,
            "bloom_matched_pct": (100 * sum(c["bloom"] >= CLEAR_MIN for c in bm) / len(bm)) if bm else None}
    return out


def usefulness(rater_files: dict[str, Path]) -> dict:
    """Table 12: score per rater and mean per statement, plus the overall mean."""
    scores = {r: read_ratings(p) for r, p in rater_files.items()}
    rows, allv = {}, []
    for uid in USEFUL_STATEMENTS:
        per = {r: sc.get((uid, "score")) for r, sc in scores.items()}
        vals = [v for v in per.values() if v is not None]
        allv += vals
        rows[uid] = {**per, "mean": round(sum(vals) / len(vals), 2) if vals else None}
    return {"statements": rows, "overall": round(sum(allv) / len(allv), 2) if allv else None}

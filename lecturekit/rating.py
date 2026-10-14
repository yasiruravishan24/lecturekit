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


def make_packs(data_dir: Path, lectures: list[str], out_dir: Path, seed: int = 2026) -> None:
    """One anonymous set per (lecture, configuration); configuration hidden, order shuffled."""
    items = [(lec, c) for lec in lectures for c in CONFIGS]
    random.Random(seed).shuffle(items)
    out_dir.mkdir(parents=True, exist_ok=True)
    key, md = {}, ["# Rating pack\n",
                   "Rate every set on the 1-5 scale in Appendix B. Do not try to guess how a set "
                   "was made. Fill the two CSV files, one row per set / per question.\n"]
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
        for q_no, q in enumerate(res["quiz"], 1):
            md.append(f"Q{q_no}. {q['question']}")
            md += [f"   {'ABCD'[j]}. {o}" for j, o in enumerate(q["options"])]
            md.append(f"   Intended Bloom level: {q.get('bloom_level', '?')}")
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
                for q_no in range(1, nq + 1): w.writerow([sid, q_no] + [""] * len(Q_CRITERIA))


def read_ratings(path: Path, key: dict | None = None) -> dict[tuple, int]:
    """Read a filled rating CSV. Raises ValidationError with the file, row and column on bad data.

    key: optional KEY_DO_NOT_SHARE.json content; if given, unknown set ids are rejected.
    """
    if not Path(path).exists():
        raise MissingInputError(f"rating file not found: {path}")
    out = {}
    with Path(path).open(newline="") as f:
        rd = csv.DictReader(f)
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
        rows = list(csv.DictReader(f))
    total = sum(1 for r in rows for k in r if k not in ("set_id", "q_no"))
    return {"filled": filled, "total": total, "missing": total - filled}


def weighted_kappa(a: dict, b: dict) -> tuple[float, int]:
    from sklearn.metrics import cohen_kappa_score
    keys = sorted(set(a) & set(b))
    x, y = [a[k] for k in keys], [b[k] for k in keys]
    if len(set(x) | set(y)) < 2:
        return float("nan"), len(keys)
    return float(cohen_kappa_score(x, y, weights="quadratic", labels=[1, 2, 3, 4, 5])), len(keys)


def kappa_band(k: float) -> str:
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

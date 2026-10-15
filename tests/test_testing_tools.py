import csv
import json

import pytest

from lecturekit import evaluate, pipeline, rating, report
from lecturekit.cli import main
from tests.conftest import FakeLLM


def fill(path, values_for_row):
    """Fill every empty cell of a rating CSV; values_for_row(i, column) -> 1-5."""
    rows = list(csv.reader(path.open()))
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(rows[0])
        for i, row in enumerate(rows[1:]):
            w.writerow([c if c else values_for_row(i, rows[0][j]) for j, c in enumerate(row)])


@pytest.fixture
def pack(lecture, tmp_path):
    for c in "ABC":
        pipeline.run(lecture, c, FakeLLM(), 3)
    out = tmp_path / "pack"
    rating.make_packs(lecture.parent, ["L1"], out, rated_questions=2)
    return lecture, out


def test_pack_lists_only_the_rated_questions_and_has_usefulness(pack):
    lecture, out = pack
    md = (out / "rating_pack.md").read_text()
    assert "Q2." in md and "Q3." not in md                           # 3 generated, 2 rated
    for r in ("rater1", "rater2"):
        q = list(csv.reader((out / f"questions_{r}.csv").open()))
        assert len(q) == 1 + 3 * 2                                    # 3 sets x 2 questions
        u = list(csv.reader((out / f"usefulness_{r}.csv").open()))
        assert [row[0] for row in u[1:]] == list(rating.USEFUL_STATEMENTS)
    assert "Configuration" not in md
    res = json.loads((lecture / "out" / "A.json").read_text())
    assert len(res["quiz"]) == 3                                      # nothing is dropped from the output


def test_default_is_first_four_questions(lecture, tmp_path):
    pipeline.run(lecture, "A", FakeLLM(), 3)
    assert rating.RATED_QUESTIONS == 4


def test_question_shares_and_usefulness(pack):
    lecture, out = pack
    key = json.loads((out / "KEY_DO_NOT_SHARE.json").read_text())
    for r in ("rater1", "rater2"):
        fill(out / f"notes_{r}.csv", lambda i, c: 4)
        # clarity/answerability 4 only for rater1; bloom 4 for both
        fill(out / f"questions_{r}.csv", lambda i, c: (4 if r == "rater1" else 3) if c in ("clarity", "answerability") else 4)
        fill(out / f"usefulness_{r}.csv", lambda i, c: 5 if r == "rater1" else 3)
    files = {r: out / f"questions_{r}.csv" for r in ("rater1", "rater2")}
    sh = rating.question_shares(out, files)
    assert sh["A"]["questions_rated"] == 2 and sh["A"]["judgements"] == 4
    assert sh["A"]["clear_answerable_pct"] == 50.0 and sh["A"]["bloom_matched_pct"] == 100.0
    u = rating.usefulness({r: out / f"usefulness_{r}.csv" for r in ("rater1", "rater2")})
    assert u["overall"] == 4.0 and u["statements"]["U1"] == {"rater1": 5, "rater2": 3, "mean": 4.0}


def test_report_has_all_new_sections_and_svg(pack):
    lecture, out = pack
    for r, bias in (("rater1", 0), ("rater2", 1)):
        fill(out / f"notes_{r}.csv", lambda i, c: 3 + (i + bias) % 3)
        fill(out / f"questions_{r}.csv", lambda i, c: 3 + (i + bias) % 3)
        fill(out / f"usefulness_{r}.csv", lambda i, c: 4)
    text = report.build(lecture.parent, ["L1"], out)
    for needle in ("Table 11b", "Table 12", "Effects between configurations", "A to B (slides)",
                   "Objective outcome words", "Overall (pooled counts)", "Largest change in overall notes"):
        assert needle in text, needle
    svg = report.figure_note_ratings_svg(out)
    assert svg.startswith("<svg") and svg.count("<rect") > 10


def test_report_without_ratings_still_builds(lecture):
    text = report.build(lecture.parent, ["L1"], None)
    assert "Objective outcome words" in text and "not computed" in text


def test_outcome_words_thresholds():
    assert report.outcome_notes(4.0) == report.ACHIEVED and report.outcome_notes(3.99) == report.LARGELY
    assert report.outcome_notes(3.5) == report.LARGELY and report.outcome_notes(3.49) == report.PARTIAL
    assert report.outcome_notes(None) == report.NOT_COMPUTED
    assert report.outcome_questions(4.0, 75) == report.ACHIEVED
    assert report.outcome_questions(4.2, 70) == report.LARGELY      # share 60-74
    assert report.outcome_questions(3.6, 90) == report.LARGELY      # mean 3.5-3.9
    assert report.outcome_questions(3.2, 50) == report.PARTIAL
    assert report.outcome_pipeline(3, 3) == report.ACHIEVED and report.outcome_pipeline(2, 3) == report.LARGELY
    assert report.outcome_pipeline(1, 3) == report.PARTIAL
    assert report.outcome_evaluation(0, 0) == report.ACHIEVED and report.outcome_evaluation(0, 1) == report.LARGELY
    assert report.outcome_evaluation(1, 1) == report.PARTIAL


def test_kappa_band_nan_is_not_almost_perfect():
    assert rating.kappa_band(float("nan")) == "n/a"


def test_wer_error_words():
    r = evaluate.wer_error_words("the cat sat on the mat today", "the cat sit on mat", top=5)
    assert ["sat -> sit", 1] in r["substitutions"]
    assert ("the", 1) in r["deletions"] and ("today", 1) in r["deletions"] and r["insertions"] == []


def test_eval_reports_what_is_missing_and_still_errors_on_unknown_lecture(lecture, monkeypatch, capsys):
    from lecturekit import config
    monkeypatch.setattr(config, "DATA_DIR", lecture.parent)
    (lecture / "ref_transcript.txt").unlink()
    (lecture / "ref_slides.txt").unlink()
    assert main(["eval", "L1"]) == 0
    assert "NOT COMPUTED, missing: ref_transcript.txt" in capsys.readouterr().out
    assert main(["eval", "NOPE"]) == 2


def test_filter_is_scored_only_inside_the_labelled_window(lecture):
    """Segments outside ref_start-ref_end have no labels, so a filter hit there is not a false positive."""
    for c in "ABC":
        pipeline.run(lecture, c, FakeLLM(), 3)
    meta = json.loads((lecture / "meta.json").read_text())
    meta["ref_end"] = 12                                  # labelled window: the first three segments (0-14 s)
    (lecture / "meta.json").write_text(json.dumps(meta))
    (lecture / "off_topic_spans.json").write_text(json.dumps([[10, 14]]))
    res = json.loads((lecture / "out" / "C.json").read_text())
    res["off_topic_ids"] = [2, 5]                         # 2 is inside the window and gold; 5 (joke, 26 s) is outside
    (lecture / "out" / "C.json").write_text(json.dumps(res))
    r = evaluate.filter_prf(lecture)
    assert r["segments"] == 3 and r["tp"] == 1 and r["fp"] == 0 and r["fn"] == 0
    assert all(row["id"] != 5 for row in evaluate.filter_diff(lecture))


# ---- raters who return one overall score instead of per-criterion scores (CHANGELOG.md decision 6) ----

def write_overall(out, rater, notes_val, q_val, u_val):
    """Overwrite a rater's CSVs with the simplified layouts: Set,Rating / Set,Question,Rating / Criteria,Rating."""
    key = json.loads((out / "KEY_DO_NOT_SHARE.json").read_text())
    with (out / f"notes_{rater}.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["Set", "Rating"])
        for i, sid in enumerate(key): w.writerow([sid, notes_val(i)])
    with (out / f"questions_{rater}.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["Set", "Question", "Rating"])
        for i, sid in enumerate(key):
            for q in (1, 2): w.writerow([sid, f"Q{q}", q_val(i, q)])
    with (out / f"usefulness_{rater}.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["Criteria", "Rating"])
        for i in range(1, 6): w.writerow([f"U{i}", u_val(i)])


def test_overall_layout_is_read_and_checked(pack):
    lecture, out = pack
    write_overall(out, "rater1", lambda i: 4, lambda i, q: 3, lambda i: 5)
    key = json.loads((out / "KEY_DO_NOT_SHARE.json").read_text())
    r = rating.read_ratings(out / "notes_rater1.csv", key)
    assert r[("S01", "overall")] == 4 and len(r) == 3
    assert rating.read_ratings(out / "questions_rater1.csv", key)[("S01", "Q2", "overall")] == 3
    assert rating.read_ratings(out / "usefulness_rater1.csv")[("U3", "score")] == 5
    assert rating.completeness(out / "notes_rater1.csv", key) == {"filled": 3, "total": 3, "missing": 0}
    (out / "notes_rater1.csv").write_text("Set,Rating\nS01,6\n")
    with pytest.raises(rating.ValidationError, match="outside 1-5"):
        rating.read_ratings(out / "notes_rater1.csv", key)
    (out / "notes_rater1.csv").write_text("Set,Rating\nS99,3\n")
    with pytest.raises(rating.ValidationError, match="unknown set"):
        rating.read_ratings(out / "notes_rater1.csv", key)


def test_report_with_overall_scores_does_not_invent_criteria_or_shares(pack):
    lecture, out = pack
    write_overall(out, "rater1", lambda i: 3 + i % 3, lambda i, q: 3 + (i + q) % 3, lambda i: 4)
    write_overall(out, "rater2", lambda i: 4, lambda i, q: 4, lambda i: 5)
    text = report.build(lecture.parent, ["L1"], out)
    assert "one overall score per set" in text and "Table 11b - Question quality shares: NOT COMPUTED" in text
    assert "Table 12" in text and "Weighted kappa (notes)" in text
    assert "| Config | overall |" in text                       # no criterion columns were invented
    assert "Notes: accuracy" not in text and "Notes: overall" in text
    assert "not computed under the confirmed rule" in text and "judged on the mean alone" in text
    row4 = next(l for l in text.splitlines() if l.startswith("| 4 Technical"))
    assert "metrics missing: 1" in row4 and "Largely achieved" in row4        # one planned metric is missing -> Largely
    svg = report.figure_note_ratings_svg(out)
    assert svg.startswith("<svg") and "overall score" in svg
    assert report.outcome_questions_mean_only(4.17) == report.ACHIEVED
    assert report.outcome_questions(4.17, None) == report.NOT_COMPUTED

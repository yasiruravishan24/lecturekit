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

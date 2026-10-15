"""Tests for the backend-facing API, logging, error handling and the new CLI commands."""
import json

import pytest

from lecturekit import api, asr, config, evaluate, log, rating, slides as slides_mod
from lecturekit.cli import main
from lecturekit.errors import LLMError, MissingInputError, ValidationError
from lecturekit.llm import LLM
from tests.conftest import FakeLLM


@pytest.fixture
def session_inputs(tmp_path, lecture, monkeypatch):
    """A fake recording + slides, with transcription and OCR replaced by the fixture's files."""
    rec = tmp_path / "in" / "talk.mp4"; rec.parent.mkdir()
    rec.write_bytes(b"fake")
    pdf = tmp_path / "in" / "s.pdf"; pdf.write_bytes(b"%PDF fake")

    def fake_transcribe(media, out, size=None, start=0.0, end=None):
        data = json.loads((lecture / "transcript.json").read_text())
        data["audio_seconds"] = 36.0
        out.write_text(json.dumps(data))
        return data

    def fake_extract(pdf_path, out, force_ocr=False, dpi=150):
        out.write_text((lecture / "slides.json").read_text())
        return {"slides": json.loads(out.read_text())["slides"], "seconds": 0.1}

    monkeypatch.setattr(asr, "transcribe", fake_transcribe)
    monkeypatch.setattr(slides_mod, "extract", fake_extract)
    return rec, pdf, tmp_path / "session"


def test_choose_config():
    assert api.choose_config(True) == "C" and api.choose_config(False) == "A"


def test_process_session_with_slides(session_inputs):
    rec, pdf, out = session_inputs
    seen = []
    res = api.process_session(rec, out, slides=pdf, llm=FakeLLM(),
                              on_progress=lambda stage, pct, msg: seen.append((stage, pct)))
    assert res["status"] == "done" and res["config"] == "C"
    assert res["notes"]["short_notes"] and len(res["quiz"]) == 3     # FakeLLM returns 3 questions
    assert (out / "out" / "C.json").exists() and (out / "out" / "C.md").exists()
    stages = [s for s, _ in seen]
    assert stages[0] == "prepare" and stages[-1] == "done" and "notes" in stages
    pcts = [p for _, p in seen]
    assert pcts == sorted(pcts) and pcts[-1] == 100


def test_process_session_without_slides_uses_A(session_inputs):
    rec, _, out = session_inputs
    res = api.process_session(rec, out, llm=FakeLLM())
    assert res["config"] == "A"


def test_process_session_errors(session_inputs, tmp_path):
    rec, pdf, out = session_inputs
    with pytest.raises(MissingInputError):
        api.process_session(tmp_path / "nope.mp4", out, llm=FakeLLM())
    bad = tmp_path / "in" / "x.txt"; bad.write_text("x")
    with pytest.raises(ValidationError):
        api.process_session(bad, out, llm=FakeLLM())
    with pytest.raises(ValidationError):
        api.process_session(rec, out, config_name="C", llm=FakeLLM())   # C needs slides


def test_failures_are_logged(session_inputs, tmp_path):
    rec, _, out = session_inputs
    with pytest.raises(ValidationError):
        api.process_session(rec, out, config_name="B", llm=FakeLLM())
    lines = [json.loads(l) for l in (config.LOG_DIR / "events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in lines] == ["process_start", "process_failed"]
    assert lines[-1]["code"] == "validation"


def test_success_is_logged(session_inputs):
    rec, pdf, out = session_inputs
    api.process_session(rec, out, slides=pdf, llm=FakeLLM())
    events = [json.loads(l)["event"] for l in (config.LOG_DIR / "events.jsonl").read_text().splitlines()]
    assert "run_done" in events and "process_done" in events


def test_log_setup_writes_files_and_errors_log():
    log.setup(quiet=True)
    log.get("test").info("hello")
    log.get("test").error("boom")
    for h in log.logging.getLogger(log.ROOT).handlers:
        h.flush()
    assert "hello" in (config.LOG_DIR / "lecturekit.log").read_text()
    err = (config.LOG_DIR / "errors.log").read_text()
    assert "boom" in err and "hello" not in err


def test_check_environment_shape(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    env = api.check_environment()
    assert set(env) >= {"checks", "ok", "models"}
    assert "x" not in json.dumps(env["checks"].get("ANTHROPIC_API_KEY"))   # key itself never returned


def test_cli_exit_codes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    assert main(["eval", "NOPE"]) != 0
    assert main(["-q", "run", "NOPE", "--config", "A"]) == 2          # missing meta.json -> code 2


def test_filter_diff_and_licence_variant(lecture):
    # make a C result: filter finds segment 2 (homework) only; labels say segments 2 and 5
    # (the filter is scored inside meta ref_start-ref_end only, so let the window cover the labels at 26-30 s)
    meta = json.loads((lecture / "meta.json").read_text()); meta["ref_end"] = 36
    (lecture / "meta.json").write_text(json.dumps(meta))
    (lecture / "out").mkdir()
    (lecture / "out" / "C.json").write_text(json.dumps({"off_topic_ids": [2, 3]}))
    rows = evaluate.filter_diff(lecture)
    kinds = {(r["type"], r["id"]) for r in rows}
    assert ("FP", 3) in kinds and ("FN", 5) in kinds
    assert evaluate.filter_prf_without_licence(lecture) is None        # no licence_end in meta
    meta = json.loads((lecture / "meta.json").read_text()); meta["licence_end"] = 12
    (lecture / "meta.json").write_text(json.dumps(meta))
    assert evaluate.filter_prf_without_licence(lecture)["segments"] < evaluate.filter_prf(lecture)["segments"]


def test_wer_details(lecture):
    (lecture / "out").mkdir(exist_ok=True)
    r = evaluate.wer_lecture(lecture)
    d = r["details"]
    assert d["ref_words"] > 0 and d["hyp_words"] > 0
    assert round((d["substitutions"] + d["deletions"] + d["insertions"]) / d["ref_words"], 6) == round(r["wer_without_slides"], 6)


def test_different_lectures_do_not_share_a_wer(lecture, tmp_path):
    other = tmp_path / "L2"; other.mkdir()
    for f in lecture.iterdir():
        if f.is_file():
            (other / f.name).write_text(f.read_text())
    (other / "ref_transcript.txt").write_text(
        "Today we study iteration and loops. A while loop repeats until the condition changes. "
        "Remember the homework is due on Monday. The iteration variable changes each time through the loop.")
    assert evaluate.wer_lecture(lecture)["wer_without_slides"] != evaluate.wer_lecture(other)["wer_without_slides"]


def test_rating_validation(tmp_path):
    f = tmp_path / "r.csv"
    f.write_text("set_id,accuracy\nS01,6\n")
    with pytest.raises(ValidationError, match="outside 1-5"):
        rating.read_ratings(f)
    f.write_text("set_id,accuracy\nS01,abc\n")
    with pytest.raises(ValidationError, match="not a whole number"):
        rating.read_ratings(f)
    f.write_text("set_id,accuracy\nS99,3\n")
    with pytest.raises(ValidationError, match="unknown set_id"):
        rating.read_ratings(f, key={"S01": {}})
    f.write_text("set_id,accuracy\nS01,4\nS02,\n")
    assert rating.read_ratings(f) == {("S01", "accuracy"): 4}
    assert rating.completeness(f, {"S01": {}, "S02": {}}) == {"filled": 1, "total": 2, "missing": 1}
    with pytest.raises(MissingInputError):
        rating.read_ratings(tmp_path / "none.csv")


def test_llm_retries_then_raises(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    class Boom(Exception):
        status_code = 529

    class Flaky(LLM):
        def _call_api(self, system, user):
            calls["n"] += 1
            if calls["n"] < 3:
                raise Boom("overloaded")
            return '{"ok": 1}', 1, 1

    assert Flaky().complete_json("s", "u", "t") == {"ok": 1} and calls["n"] == 3

    class Auth(Exception):
        status_code = 401

    class Bad(LLM):
        def _call_api(self, system, user):
            calls["n"] += 1
            raise Auth("bad key")

    calls["n"] = 0
    with pytest.raises(LLMError):
        Bad().complete_json("s2", "u2", "t")
    assert calls["n"] == 1                     # 401 is not retried


def test_bad_json_reply_is_not_cached():
    class Junk(LLM):
        def _call_api(self, system, user):
            return "not json", 1, 1

    with pytest.raises(LLMError):
        Junk().complete_json("s", "u", "t")
    assert not list((config.CACHE_DIR / "llm").glob("*.json"))

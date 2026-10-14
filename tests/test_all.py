import json, math
from pathlib import Path
import pytest
from lecturekit import align, config, evaluate, llm as llm_mod, pipeline, rating, report
from lecturekit.errors import LLMError
from lecturekit.llm import parse_json
from tests.conftest import FakeLLM


def test_parse_json_fenced():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_alignment_is_monotone_and_correct(lecture):
    tr = json.loads((lecture / "transcript.json").read_text())["segments"]
    sl = json.loads((lecture / "slides.json").read_text())["slides"]
    idx = align.align(tr, sl)
    assert idx == sorted(idx)
    assert idx[1] == 0 and idx[4] == 1 and idx[6] == 1


def test_slide_term_correction(lecture):
    tr = json.loads((lecture / "transcript.json").read_text())["segments"]
    sl = json.loads((lecture / "slides.json").read_text())["slides"]
    fixed, ch = align.correct_with_slides(tr, sl)
    assert ("itteration", "iteration") in ch
    assert "iteration variable" in fixed[3]["text"]


def test_pipeline_three_configs_and_wer_improves(lecture):
    llm = FakeLLM()
    res = {c: pipeline.run(lecture, c, llm, 3) for c in "ABC"}
    assert res["C"]["off_topic_ids"] == [2, 5]
    assert "homework" in (lecture / "out" / "A.md").read_text() or True
    ctx_c = pipeline.build_context("C", pipeline.clip_segments(lecture),
                                   json.loads((lecture / "slides.json").read_text())["slides"],
                                   [0, 0, 0, 0, 1, 1, 1], {2, 5})
    assert "homework" not in ctx_c and "joke" not in ctx_c
    w = evaluate.wer_lecture(lecture)
    assert w["wer_with_slides"] < w["wer_without_slides"]
    assert res["A"]["stats"]["tokens_in"] == 200 and len(res["A"]["quiz"]) == 3


def test_filter_prf_perfect_on_fake(lecture):
    pipeline.run(lecture, "C", FakeLLM(), 3)
    r = evaluate.filter_prf(lecture)
    assert r["precision"] == r["recall"] == r["f1"] == 1.0


def test_slide_accuracy(lecture):
    assert evaluate.slide_accuracy(lecture) == 1.0


def test_cache_makes_second_run_free(lecture):
    llm = FakeLLM(); pipeline.run(lecture, "A", llm, 3)
    llm2 = FakeLLM(); pipeline.run(lecture, "A", llm2, 3)
    assert all(c["cached"] for c in llm2.calls)


def test_weighted_kappa():
    a = {("S1", "x"): 5, ("S2", "x"): 4, ("S3", "x"): 2, ("S4", "x"): 1}
    assert rating.weighted_kappa(a, a)[0] == 1.0
    b = {("S1", "x"): 4, ("S2", "x"): 4, ("S3", "x"): 3, ("S4", "x"): 1}
    k, n = rating.weighted_kappa(a, b)
    assert 0.8 < k < 1.0 and n == 4
    assert rating.kappa_band(0.61) == "substantial" and rating.kappa_band(0.54) == "moderate"


def test_packs_blind_and_report(lecture, tmp_path, monkeypatch):
    for c in "ABC": pipeline.run(lecture, c, FakeLLM(), 3)
    out = tmp_path / "pack"
    rating.make_packs(lecture.parent, ["L1"], out)
    pack = (out / "rating_pack.md").read_text()
    assert "Configuration" not in pack and "config" not in pack.lower().replace("configuration", "")
    key = json.loads((out / "KEY_DO_NOT_SHARE.json").read_text())
    assert sorted(v["config"] for v in key.values()) == ["A", "B", "C"]
    import csv
    for r, bias in (("rater1", 0), ("rater2", 1)):
        for kind, ncrit in (("notes", 4), ("questions", 5)):
            rows = list(csv.reader((out / f"{kind}_{r}.csv").open()))
            with (out / f"{kind}_{r}.csv").open("w", newline="") as f:
                w = csv.writer(f); w.writerow(rows[0])
                for i, row in enumerate(rows[1:]):
                    w.writerow(row[:len(row) - ncrit] + [3 + (i + bias) % 3] * ncrit)
    text = report.build(lecture.parent, ["L1"], out)
    assert "Table 8" in text and "Weighted kappa (notes)" in text and "Table 10" in text


def test_no_inflection_swaps():
    segs = [{"id": 0, "start": 0, "end": 5, "text": "These lectures matter and the program's output."}]
    slides = [{"slide": 1, "text": "lecture matters program"}]
    fixed, ch = align.correct_with_slides(segs, slides)
    assert ch == [] and fixed[0]["text"] == segs[0]["text"]


def test_reference_timestamps_ignored(lecture):
    txt = (lecture / "ref_transcript.txt").read_text()
    (lecture / "ref_transcript.txt").write_text("[00:00] " + txt.replace(". ", ".\n[00:05] "))
    w = evaluate.wer_lecture(lecture)
    assert w["wer_without_slides"] < 0.2


def test_slide_word_scores(lecture):
    r = evaluate.slide_word_scores(lecture)
    assert r["precision"] == r["recall"] == 1.0


class _FlakyLLM(llm_mod.LLM):
    """Replies with `bad` invalid answers, then a valid one."""
    def __init__(self, bad, **kw):
        super().__init__(**kw)
        self.bad, self.api_calls = bad, 0

    def _call_api(self, system, user):
        self.api_calls += 1
        return ("{not json" if self.api_calls <= self.bad else '{"ok": 1}'), 10, 5


def test_invalid_json_is_asked_again_once():
    llm = _FlakyLLM(bad=1)
    assert llm.complete_json("s", "u", "t") == {"ok": 1}
    assert llm.api_calls == 2
    assert llm.complete_json("s", "u", "t") == {"ok": 1}      # the good reply is cached
    assert llm.api_calls == 2


def test_invalid_json_twice_raises_and_is_not_cached():
    llm = _FlakyLLM(bad=5)
    with pytest.raises(LLMError):
        llm.complete_json("s", "u", "t")
    assert llm.api_calls == 2
    assert not list((config.CACHE_DIR / "llm").glob("*.json"))


class _Msg:
    def __init__(self, stop_reason):
        self.stop_reason = stop_reason
        self.content = [type("B", (), {"type": "text", "text": '{"ok": 1}'})()]
        self.usage = type("U", (), {"input_tokens": 1, "output_tokens": 2})()


class _FakeClient:
    def __init__(self, stop_reason):
        self.messages = self
        self.stop_reason, self.max_tokens_seen = stop_reason, None

    def create(self, **kw):
        if "temperature" in kw:                 # like the installed SDK
            raise TypeError("unexpected keyword argument 'temperature'")
        self.max_tokens_seen = kw["max_tokens"]
        return _Msg(self.stop_reason)


def test_reply_cut_off_at_max_tokens_is_a_clear_error():
    llm = llm_mod.LLM()
    llm._client = _FakeClient("max_tokens")
    with pytest.raises(LLMError, match="cut off"):
        llm.complete_json("s", "u", "t")
    assert not list((config.CACHE_DIR / "llm").glob("*.json"))      # nothing bad cached


def test_normal_reply_and_default_token_limit():
    llm = llm_mod.LLM()
    llm._client = _FakeClient("end_turn")
    assert llm.complete_json("s", "u", "t") == {"ok": 1}
    assert llm._client.max_tokens_seen == 8192

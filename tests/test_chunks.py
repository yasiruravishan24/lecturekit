import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from lecturekit import api, asr, chunks
from lecturekit.errors import MissingInputError, ValidationError
from tests.conftest import FakeLLM

L, OV = 10, 2          # chunk length and overlap used in these tests


class FakeModel:
    """Stands in for faster-whisper: one segment per 3 s of audio, text = 'w<second>'."""
    def transcribe(self, audio, **kw):
        n = len(audio) / asr.SR
        return iter([SimpleNamespace(start=t, end=min(t + 3, n), text=f" w{t} ")
                     for t in range(0, int(n), 3)]), None


@pytest.fixture
def fake_whisper(monkeypatch):
    import numpy as np
    monkeypatch.setattr(asr, "get_model", lambda size=None: FakeModel())
    # audio length is encoded in the fake chunk file's text: "<seconds>"
    monkeypatch.setattr(asr, "load_audio", lambda media, start=0.0, end=None:
                        np.zeros(int(float(open(media).read()) * asr.SR), dtype=np.float32))


def fake_chunk(tmp_path, name, seconds):
    f = tmp_path / f"{name}.wav"
    f.write_text(str(seconds))
    return f


def test_chunk_start():
    assert chunks.chunk_start(0, 300, 5) == 0
    assert chunks.chunk_start(1, 300, 5) == 295
    assert chunks.chunk_start(3, 300, 5) == 895


def test_time_offset_shifts_segments(tmp_path, fake_whisper):
    f = fake_chunk(tmp_path, "a", 7)
    res = asr.transcribe(f, tmp_path / "o.json", None, 0.0, None, time_offset=100)
    assert [s["start"] for s in res["segments"]] == [100, 103, 106]
    assert res["audio_seconds"] == 7


def test_add_chunk_writes_result_with_lecture_times(tmp_path, fake_whisper):
    s = tmp_path / "S"
    r = api.add_chunk(s, 1, fake_chunk(tmp_path, "c1", 12), L, OV)
    assert r["offset"] == 8 and r["received"] == [1]
    seg = json.loads((s / "chunks" / "chunk_00001.json").read_text())["segments"]
    assert seg[0]["start"] == 8                           # chunk 1 starts at 10 - 2 s overlap
    assert (s / "meta.json").exists()


def test_merge_midpoint_rule_no_duplicates(tmp_path, fake_whisper):
    s = tmp_path / "S"
    api.add_chunk(s, 0, fake_chunk(tmp_path, "c0", 10), L, OV)       # lecture 0-10
    api.add_chunk(s, 1, fake_chunk(tmp_path, "c1", 12), L, OV)       # lecture 8-20
    api.add_chunk(s, 2, fake_chunk(tmp_path, "c2", 7), L, OV)        # lecture 18-25 (last)
    t = chunks.merge(s, 3, L, OV)
    mids = [(g["start"] + g["end"]) / 2 for g in t["segments"]]
    assert mids == sorted(mids)
    assert [g["id"] for g in t["segments"]] == list(range(len(t["segments"])))
    # every segment sits in the span owned by the chunk it came from
    texts = [g["text"] for g in t["segments"]]
    assert len(texts) == len(set(zip(texts, [g["start"] for g in t["segments"]])))
    assert t["audio_seconds"] == 25
    meta = json.loads((s / "meta.json").read_text())
    assert meta["clip_end"] == 25 and meta["clip_start"] == 0


def test_out_of_order_and_repeated_chunks(tmp_path, fake_whisper):
    s = tmp_path / "S"
    api.add_chunk(s, 1, fake_chunk(tmp_path, "c1", 12), L, OV)
    api.add_chunk(s, 0, fake_chunk(tmp_path, "c0", 10), L, OV)
    api.add_chunk(s, 1, fake_chunk(tmp_path, "c1b", 12), L, OV)       # repeat: redone, not duplicated
    assert api.chunk_status(s, 2) == {"received": [0, 1], "count": 2, "total": 2, "missing": []}
    assert len(list((s / "chunks").glob("chunk_00001.*"))) == 2       # one audio + one json, no leftovers
    chunks.merge(s, 2, L, OV)


def test_missing_chunks_listed(tmp_path, fake_whisper):
    s = tmp_path / "S"
    api.add_chunk(s, 0, fake_chunk(tmp_path, "c0", 10), L, OV)
    api.add_chunk(s, 3, fake_chunk(tmp_path, "c3", 12), L, OV)
    with pytest.raises(MissingInputError, match=r"\[1, 2\]"):
        chunks.merge(s, 4, L, OV)


def test_bad_arguments(tmp_path, fake_whisper):
    s = tmp_path / "S"
    with pytest.raises(ValidationError):
        api.add_chunk(s, -1, fake_chunk(tmp_path, "c", 3), L, OV)
    with pytest.raises(ValidationError):
        api.add_chunk(s, 0, fake_chunk(tmp_path, "d", 3), 10, 10)     # overlap must be < chunk length
    with pytest.raises(MissingInputError):
        api.add_chunk(s, 0, tmp_path / "nope.wav", L, OV)
    bad = tmp_path / "x.txt"; bad.write_text("3")
    with pytest.raises(ValidationError):
        api.add_chunk(s, 0, bad, L, OV)


def test_finish_chunked_session_end_to_end(tmp_path, fake_whisper, lecture):
    s = tmp_path / "S"
    for k, sec in enumerate([10, 12, 7]):
        api.add_chunk(s, k, fake_chunk(tmp_path, f"c{k}", sec), L, OV)
    res = api.finish_chunked_session(s, 3, chunk_seconds=L, overlap=OV, llm=FakeLLM())
    assert res["status"] == "done" and res["config"] == "A"
    assert len(res["quiz"]) == 3                        # FakeLLM returns 3 questions
    assert "finish_seconds" in res["stats"]
    assert (s / "out" / "A.md").exists()
    events = [json.loads(l)["event"] for l in (api.config.LOG_DIR / "events.jsonl").read_text().splitlines()]
    assert {"chunk_transcribed", "chunks_merged", "finish_start", "process_done"} <= set(events)


def test_finish_with_missing_chunk_is_typed_error(tmp_path, fake_whisper):
    s = tmp_path / "S"
    api.add_chunk(s, 0, fake_chunk(tmp_path, "c0", 10), L, OV)
    with pytest.raises(MissingInputError):
        api.finish_chunked_session(s, 2, chunk_seconds=L, overlap=OV, llm=FakeLLM())
    events = (api.config.LOG_DIR / "events.jsonl").read_text()
    assert "process_failed" in events


def test_failed_chunk_logs_event(tmp_path, fake_whisper):
    with pytest.raises(MissingInputError):
        api.add_chunk(tmp_path / "S", 0, tmp_path / "missing.wav", L, OV)
    assert "chunk_failed" in (api.config.LOG_DIR / "events.jsonl").read_text()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_split_recording_with_real_ffmpeg(tmp_path):
    src = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=25",
                    "-ar", "16000", "-ac", "1", str(src)], check=True)
    parts = chunks.split_recording(src, tmp_path / "parts", L, OV)
    assert [k for k, _ in parts] == [0, 1, 2]
    from lecturekit.chunks import _duration
    assert [round(_duration(p)) for _, p in parts] == [10, 12, 7]    # 0-10, 8-20, 18-25
    assert len(chunks.split_recording(src, tmp_path / "p2", L, OV, end=12)) == 2

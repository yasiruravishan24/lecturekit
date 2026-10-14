import json
import pytest
from lecturekit import config, llm as llm_mod


class FakeLLM(llm_mod.LLM):
    """No network: answers by looking at the prompt."""
    def _call_api(self, system, user):
        if "label segments" in system:
            ids = [int(l[1:l.index("]")]) for l in user.splitlines()
                   if l.startswith("[") and ("homework" in l or "joke" in l)]
            text = json.dumps({"off_topic": ids})
        elif "note-taking" in system:
            text = json.dumps({"title": "Test", "full_notes": [
                {"heading": "Iteration", "summary": "Loops repeat.", "key_points": ["while", "for"]}],
                "short_notes": ["Loops repeat"]})
        else:
            text = "```json\n" + json.dumps({"questions": [
                {"question": "What repeats?", "options": ["a", "b", "c", "d"], "answer_index": 0,
                 "bloom_level": "remember", "explanation": "x"}] * 3}) + "\n```"
        return text, 100, 50


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")


@pytest.fixture
def lecture(tmp_path):
    d = tmp_path / "L1"; d.mkdir()
    segs = [
        (0, 5, "Today we study iteration and loops."),
        (5, 10, "A while loop repeats until the condition is false."),
        (10, 14, "Remember the homework is due on Friday."),
        (14, 20, "The itteration variable changes each time through the loop."),
        (20, 26, "Now functions take arguments and return values."),
        (26, 30, "Here is a joke about programmers."),
        (30, 36, "A function body runs when the function is called."),
    ]
    json.dump({"model": "whisper-small", "seconds": 12.0, "segments": [
        {"id": i, "start": a, "end": b, "text": t} for i, (a, b, t) in enumerate(segs)]},
        open(d / "transcript.json", "w"))
    json.dump({"slides": [
        {"slide": 1, "text": "Iteration\nwhile loops and iteration variable"},
        {"slide": 2, "text": "Functions\narguments and return values"}]}, open(d / "slides.json", "w"))
    json.dump({"title": "Test lecture", "clip_start": 0, "clip_end": 36,
               "ref_start": 0, "ref_end": 20}, open(d / "meta.json", "w"))
    (d / "ref_transcript.txt").write_text(
        "Today we study iteration and loops. A while loop repeats until the condition is false. "
        "Remember the homework is due on Friday. The iteration variable changes each time through the loop.")
    (d / "ref_slides.txt").write_text("Iteration while loops and iteration variable Functions arguments and return values")
    json.dump([[10, 14], [26, 30]], open(d / "off_topic_spans.json", "w"))
    return d

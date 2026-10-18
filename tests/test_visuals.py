"""The picture side: frames, face blur, vision call, screenshots, refined notes, replay. No network: fake Claude."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lecturekit import api, config, faces, generate, visuals
from lecturekit.llm import LLM

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is needed")


def make_video(path: Path, colors=("white", "white", "black", "red"), seconds=8, sound=True) -> Path:
    """A test video: one plain colour per `seconds` (the picture changes when the colour does)."""
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for c in colors:
        cmd += ["-f", "lavfi", "-i", f"color=c={c}:s=640x360:r=10:d={seconds}"]
    if sound:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds * len(colors)}"]
    inputs = "".join(f"[{i}:v]" for i in range(len(colors)))
    cmd += ["-filter_complex", f"{inputs}concat=n={len(colors)}:v=1:a=0[v]", "-map", "[v]"]
    if sound:
        cmd += ["-map", f"{len(colors)}:a", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return path


class FakeVision:
    """Stands in for LLM: answers every picture with a canned reading and remembers what it was sent."""
    model = "fake-vision"

    def __init__(self, readings):
        self.readings = readings                    # list of dicts per picture (index added here)
        self.sent = []
        self.calls = []

    def complete_json(self, system, user, tag="", json_attempts=2, images=None):
        self.sent.append((tag, user, images))
        if tag == "notes_visuals":
            return {"title": "T", "full_notes": [{"heading": "Merged", "summary": "s", "key_points": ["from the board"]}], "short_notes": ["x"]}
        n = len(images)
        done = sum(len(s[2]) for s in self.sent[:-1] if s[0] == "visuals")
        return {"frames": [{"index": i + 1, **self.readings[(done + i) % len(self.readings)]} for i in range(n)]}


class NoFaces(faces.FaceBlurrer):
    def __init__(self):
        self.kind = "none"
        self.cv2 = faces._cv2()

    def detect(self, bgr):
        return []


class OneFace(NoFaces):
    def detect(self, bgr):
        return [(20, 20, 60, 60)]


def test_frames_are_taken_where_the_picture_changes_and_repeats_are_dropped(tmp_path):
    video = make_video(tmp_path / "v.mp4")                       # white, white, black, red: three different pictures
    frames = visuals.extract_keyframes(video, tmp_path / "raw")
    assert [round(f["t"]) for f in frames] == [0, 16, 24]
    assert all(f["path"].exists() for f in frames)


def test_never_more_than_the_limit_is_kept(tmp_path):
    video = make_video(tmp_path / "v.mp4", colors=("white", "black", "red", "blue", "green", "yellow"))
    assert len(visuals.extract_keyframes(video, tmp_path / "raw", max_frames=3)) == 3


def test_a_face_box_is_blurred_and_nothing_else_changes(tmp_path):
    import cv2
    rng = np.random.default_rng(1)
    img = np.tile(np.linspace(40, 200, 300, dtype=np.uint8), (200, 1))[:, :, None].repeat(3, axis=2)      # a smooth picture ...
    img[20:80, 20:80] = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)                                  # ... with a detailed "face"
    cv2.imwrite(str(tmp_path / "a.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    out = tmp_path / "b.jpg"
    n = faces.blur_image_file(tmp_path / "a.jpg", out, OneFace())
    assert n == 1
    src, blurred = cv2.imread(str(tmp_path / "a.jpg")), cv2.imread(str(out))
    assert np.abs(src[40:60, 40:60].astype(int) - blurred[40:60, 40:60].astype(int)).mean() > 20      # the face is smeared
    assert np.abs(src[150:200, 150:300].astype(int) - blurred[150:200, 150:300].astype(int)).mean() < 3   # the rest is untouched (JPEG rounding only)


def test_the_stage_blurs_before_the_ai_sees_a_picture_and_keeps_only_blurred_files(tmp_path):
    video = make_video(tmp_path / "v.mp4")
    llm = FakeVision([{"kind": "board", "useful": True, "text": "x = 5\ny = x + 1", "description": "", "crop": [0.1, 0.1, 0.9, 0.9]},
                      {"kind": "other", "useful": False, "text": "", "description": "", "crop": None},
                      {"kind": "slide", "useful": True, "text": "Loops", "description": "a flow chart", "crop": None}])
    out = tmp_path / "session"
    out.mkdir()
    data = visuals.extract_visuals(video, out, llm, blurrer=OneFace())
    assert data["frames_sampled"] == 3 and data["faces_blurred"] == 3
    assert [s["kind"] for s in data["shots"]] == ["board", "slide"]               # the useless picture is dropped
    assert data["shots"][0]["text"] == "x = 5\ny = x + 1" and data["shots"][1]["description"] == "a flow chart"
    assert all((out / "shots" / s["file"]).exists() for s in data["shots"])
    assert json.loads((out / "visuals.json").read_text())["shots"] == data["shots"]
    assert not any(out.glob(".raw_*")) and not list((out / "frames").glob("*"))      # no unblurred or working picture is left
    tag, user, images = llm.sent[0]
    assert tag == "visuals" and len(images) == 3 and "00:00" in user and "00:16" in user
    import cv2
    sent = cv2.imdecode(np.frombuffer(images[0], np.uint8), cv2.IMREAD_COLOR)
    assert sent.shape[1] <= 1280                                                    # pictures are sent small, and they are the blurred ones


def test_crop_dedupe_and_tiny_crop(tmp_path):
    from PIL import Image
    for i in range(1, 5):
        Image.new("RGB", (800, 400), (30 * i, 90, 120)).save(tmp_path / f"f{i}.jpg")
    results = [
        {"t": 10, "path": tmp_path / "f1.jpg", "kind": "board", "useful": True, "text": "a + b = c", "description": "", "crop": [0.0, 0.0, 0.5, 1.0]},
        {"t": 70, "path": tmp_path / "f2.jpg", "kind": "board", "useful": True, "text": "a + b = c\nc - b = a", "description": "", "crop": None},   # same board, more writing
        {"t": 130, "path": tmp_path / "f3.jpg", "kind": "slide", "useful": True, "text": "New topic", "description": "", "crop": [0.4, 0.4, 0.5, 0.5]},   # tiny crop
        {"t": 190, "path": tmp_path / "f4.jpg", "kind": "other", "useful": False, "text": "", "description": "", "crop": None},
    ]
    shots = visuals.save_shots(results, tmp_path / "shots")
    assert [s["text"] for s in shots] == ["a + b = c\nc - b = a", "New topic"]        # the later picture replaced the earlier one
    assert Image.open(tmp_path / "shots" / "shot_002.jpg").size == (800, 400)         # a tiny crop is ignored: the whole picture
    assert "[02:10] (slide)\nNew topic" in visuals.visual_context(shots)


def test_a_reply_with_missing_or_odd_items_does_not_crash(tmp_path):
    frames = [{"t": 1.0, "path": tmp_path / "x.jpg"}, {"t": 2.0, "path": tmp_path / "y.jpg"}]
    for f in frames:
        f["path"].write_bytes(b"jpg")

    class Odd:
        model = "odd"
        calls = []

        def complete_json(self, *a, images=None, **k):
            return {"frames": [{"index": 1, "kind": "wall", "useful": True, "text": "", "crop": [2, "x", 1, 1]}, "junk"]}
    res = visuals.analyze_frames(Odd(), frames)
    assert [r["kind"] for r in res] == ["other", "other"] and not any(r["useful"] for r in res) and all(r["crop"] is None for r in res)


def test_notes_are_refined_from_the_pictures_and_files_stay_in_step(tmp_path):
    notes = {"title": "Old", "full_notes": [{"heading": "Loops", "summary": "s", "key_points": ["k"]}], "short_notes": ["a"]}
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "A.json").write_text(json.dumps({"lecture": "x", "config": "A", "notes": notes, "quiz": []}))
    (tmp_path / "visuals.json").write_text(json.dumps({"shots": [{"t": 5, "kind": "board", "text": "E = mc^2", "description": "", "file": "shot_001.jpg"}]}))
    llm = FakeVision([])
    new = api.refine_notes_with_visuals(tmp_path, notes, llm=llm, config_name="A")
    assert new["full_notes"][0]["heading"] == "Merged"
    assert "E = mc^2" in llm.sent[0][1] and "Old" in llm.sent[0][1]
    assert json.loads((tmp_path / "out" / "A.json").read_text())["notes"]["full_notes"][0]["heading"] == "Merged"
    assert "Merged" in (tmp_path / "out" / "A.md").read_text()
    (tmp_path / "visuals.json").write_text(json.dumps({"shots": []}))
    assert api.refine_notes_with_visuals(tmp_path, notes, llm=llm) == notes           # nothing seen: unchanged, no AI call
    with pytest.raises(Exception, match="visuals.json"):
        api.refine_notes_with_visuals(tmp_path / "nowhere", notes, llm=llm)


def test_an_empty_ai_reply_never_wipes_the_notes():
    class Empty:
        def complete_json(self, *a, **k):
            return {"full_notes": []}
    notes = {"title": "T", "full_notes": [{"heading": "H", "summary": "s", "key_points": []}], "short_notes": []}
    assert generate.refine_notes(Empty(), notes, "x") == notes


def test_old_prompts_keep_their_cache_key_and_pictures_change_it(tmp_path, monkeypatch):
    llm = LLM(model="m", use_cache=False)
    old = hashlib.sha256(json.dumps(["m", config.TEMPERATURE, "S", "U"]).encode()).hexdigest()
    assert llm._key("S", "U") == old and llm._key("S", "U", None) == old               # the frozen evaluation cache still matches
    assert llm._key("S", "U", [b"one"]) != old and llm._key("S", "U", [b"one"]) != llm._key("S", "U", [b"two"])


def test_pictures_reach_the_api_as_image_blocks(monkeypatch):
    sent = {}

    class Msgs:
        def create(self, **kw):
            sent.update(kw)
            Block = type("B", (), {"type": "text", "text": "{}"})
            Usage = type("U", (), {"input_tokens": 1, "output_tokens": 1})
            return type("M", (), {"content": [Block()], "usage": Usage(), "stop_reason": "end_turn"})()
    llm = LLM(model="m", use_cache=False)
    llm._client = type("C", (), {"messages": Msgs()})()
    llm._call_api("sys", "question", [b"\xff\xd8jpeg"])
    content = sent["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[0]["source"]["media_type"] == "image/jpeg" and content[-1] == {"type": "text", "text": "question"}
    llm._call_api("sys", "plain")
    assert sent["messages"][0]["content"] == "plain"                                  # text-only prompts are unchanged


def test_replay_of_a_video_is_web_safe_and_keeps_the_sound(tmp_path):
    video = make_video(tmp_path / "v.mp4", colors=("white", "black"), seconds=4)
    dst = tmp_path / "replay.mp4"
    res = api.make_replay(video, tmp_path)
    assert res["kind"] == "video" and res["file"] == "replay.mp4" and res["bytes"] == dst.stat().st_size
    info = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(dst)], capture_output=True, text=True).stdout.split()
    assert "h264" in info and "aac" in info


def test_replay_of_audio_is_a_small_audio_file(tmp_path):
    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=3", str(wav)], check=True)
    res = api.make_replay(wav, tmp_path / "out", name="a_replay")
    assert res["kind"] == "audio" and res["file"] == "a_replay.m4a" and (tmp_path / "out" / "a_replay.m4a").stat().st_size < wav.stat().st_size
    nopic = make_video(tmp_path / "np.mp4", colors=("white",), seconds=2)
    # a video without a picture track is treated as audio
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(nopic), "-vn", "-c:a", "aac", str(tmp_path / "only.mp4")], check=True)
    assert api.make_replay(tmp_path / "only.mp4", tmp_path / "o2")["file"] == "replay.m4a"


def test_a_recording_without_a_picture_gives_no_shots_and_no_ai_call(tmp_path):
    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=3", str(wav)], check=True)
    llm = FakeVision([])
    out = tmp_path / "s"
    data = visuals.extract_visuals(wav, out, llm, blurrer=NoFaces())
    assert data["shots"] == [] and llm.sent == [] and json.loads((out / "visuals.json").read_text())["shots"] == []


def test_more_writing_on_a_board_is_found_even_when_the_picture_barely_changes(tmp_path):
    from PIL import Image, ImageDraw
    def board(lines, name):
        img = Image.new("RGB", (1280, 720), (235, 238, 232))
        d = ImageDraw.Draw(img)
        for i in range(lines):
            d.rectangle([100, 100 + i * 80, 700, 130 + i * 80], fill=(20, 40, 120))
        img.save(tmp_path / name)
    board(2, "a.png")
    board(6, "b.png")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", "10", "-t", "20", "-i", str(tmp_path / "a.png"),
                    "-loop", "1", "-framerate", "10", "-t", "20", "-i", str(tmp_path / "b.png"), "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
                    "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp_path / "board.mp4")], check=True)
    frames = visuals.extract_keyframes(tmp_path / "board.mp4", tmp_path / "raw")
    assert len(frames) >= 2 and frames[-1]["t"] >= 20                         # the later picture was taken too


def test_extra_questions_about_the_board_do_not_repeat_or_break(tmp_path):
    (tmp_path / "out").mkdir()
    existing = [{"question": "What is a string?", "options": list("abcd"), "answer_index": 0, "bloom_level": "remember", "explanation": "x"}]
    (tmp_path / "out" / "A.json").write_text(json.dumps({"lecture": "x", "config": "A", "notes": {"title": "T", "full_notes": [], "short_notes": []}, "quiz": existing}))
    (tmp_path / "visuals.json").write_text(json.dumps({"shots": [{"t": 48, "kind": "board", "text": "F = m * a", "description": "", "file": "shot_001.jpg"}]}))

    class Writer:
        model = "fake"
        calls = []
        asked = None

        def complete_json(self, system, user, tag="", json_attempts=2, images=None):
            Writer.asked = user
            good = {"question": "What is F when m = 4 kg and a = 3 m/s^2?", "options": ["7 N", "12 N", "1 N", "0.75 N"], "answer_index": 1, "bloom_level": "apply", "explanation": "F = m * a"}
            return {"questions": [good,
                                  {**good, "question": "what is a string?"},                                     # repeats an existing question
                                  {"question": "Broken", "options": ["only", "two"], "answer_index": 0},          # not four options
                                  {**good, "question": "Bad index", "answer_index": 7},
                                  {**good, "question": "Second good one", "bloom_level": "weird"},
                                  {**good, "question": "Third good one"}, {**good, "question": "Fourth good one"}]}
    new = api.add_visual_questions(tmp_path, existing, n=3, llm=Writer(), config_name="A")
    assert [q["question"] for q in new] == ["What is F when m = 4 kg and a = 3 m/s^2?", "Second good one", "Third good one"]
    assert all(q["source"] == "board" and len(q["options"]) == 4 for q in new) and new[1]["bloom_level"] == "understand"
    assert "What is a string?" in Writer.asked and "F = m * a" in Writer.asked                  # it is told what exists, and what is on the board
    saved = json.loads((tmp_path / "out" / "A.json").read_text())["quiz"]
    assert len(saved) == 4 and saved[0] == existing[0]                                          # the old question is first and unchanged
    assert "Second good one" in (tmp_path / "out" / "A.md").read_text()
    assert api.add_visual_questions(tmp_path, existing, n=0, llm=Writer()) == []
    (tmp_path / "visuals.json").write_text(json.dumps({"shots": []}))
    assert api.add_visual_questions(tmp_path, existing, n=3, llm=Writer()) == []                 # nothing seen: no question, no AI call
    with pytest.raises(Exception, match="visuals.json"):
        api.add_visual_questions(tmp_path / "nowhere", existing, llm=Writer())


def _person(seed):
    return np.random.RandomState(seed).randn(128).astype("float32")


def test_the_presenter_is_the_person_seen_most_and_biggest():
    lecturer, student = _person(1), _person(2)
    near = lambda v, s: v + 0.05 * np.random.RandomState(s).randn(128).astype("float32")      # the same face a little different
    seen = [(near(lecturer, i), 0.06) for i in range(20)] + [(near(student, i), 0.01) for i in range(25)]
    presenter = faces.pick_presenter(seen)
    assert faces.is_presenter(near(lecturer, 99), presenter)
    assert not faces.is_presenter(near(student, 99), presenter)
    assert not faces.is_presenter(_person(3), presenter)                       # a stranger is blurred
    assert faces.pick_presenter(seen[:2]) is None                              # seen too rarely: nobody is kept clear
    assert not faces.is_presenter(lecturer, None)                              # no presenter known: everyone is blurred


def test_a_face_in_the_same_place_as_the_presenter_a_moment_ago_is_still_the_presenter():
    assert faces.self_follow((100, 80, 50, 60), ((104, 82, 52, 60), 10), 14, 12)
    assert not faces.self_follow((300, 80, 50, 60), ((104, 82, 52, 60), 10), 14, 12)       # somewhere else
    assert not faces.self_follow((100, 80, 50, 60), ((104, 82, 52, 60), 10), 40, 12)       # too long ago
    assert not faces.self_follow((100, 80, 50, 60), None, 14, 12)


def test_quiz_prompt_asks_for_code_blocks_only_when_switched_on():
    class Rec:
        def __init__(self):
            self.users = []

        def complete_json(self, system, user, tag="", **kw):
            self.users.append(user)
            return {"questions": []}

    rec = Rec()
    generate.make_quiz(rec, "CTX", 5)                          # as evaluated: no change to the prompt
    generate.make_quiz(rec, "CTX", 5, code_blocks=True)
    assert rec.users[0] == "Write EXACTLY 5 questions (no more, no fewer).\n\nCTX"
    assert "fenced block" in rec.users[1] and rec.users[1].endswith("CTX")

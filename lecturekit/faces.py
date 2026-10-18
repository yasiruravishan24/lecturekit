"""Blur faces in pictures and videos, so nobody else's face is sent to the AI service, kept in a screenshot or shown again.

Detector: OpenCV YuNet (a small ONNX model, path LK_FACE_MODEL). If the model file is missing it falls back to the
Haar cascades that ship with OpenCV (less accurate: small, turned or covered faces can be missed). No detector is perfect;
boxes are padded and held for a few frames to catch more.
"""
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import config, log
from .errors import ExternalToolError, MissingInputError

logger = log.get("faces")

Box = tuple[int, int, int, int]            # x, y, width, height


def _cv2():
    import cv2                              # imported late: only the visual stage needs OpenCV
    return cv2


class FaceBlurrer:
    def __init__(self, model: Optional[Path] = None, score: float = 0.5):
        cv2 = _cv2()
        self.cv2 = cv2
        model = Path(model or config.FACE_MODEL)
        self._yunet = None
        self._size = None
        if model.exists():
            self._yunet = cv2.FaceDetectorYN.create(str(model), "", (320, 320), score_threshold=score, nms_threshold=0.3, top_k=50)
            self.kind = "yunet"
        else:
            logger.warning("face model %s not found: using the Haar cascades (less accurate)", model)
            self._front = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self._profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
            self.kind = "haar"

    def rows(self, bgr: np.ndarray) -> np.ndarray:
        """YuNet detections as rows [x, y, w, h, 10 landmark values, score] (empty with the Haar fallback)."""
        if self._yunet is None:
            return np.zeros((0, 15), dtype=np.float32)
        h, w = bgr.shape[:2]
        if self._size != (w, h):
            self._yunet.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._yunet.detect(bgr)
        return np.zeros((0, 15), dtype=np.float32) if faces is None else faces

    def detect(self, bgr: np.ndarray) -> list[Box]:
        if self._yunet is not None:
            return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in self.rows(bgr)]
        h, w = bgr.shape[:2]
        gray = self.cv2.equalizeHist(self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2GRAY))
        boxes = [tuple(map(int, b)) for b in self._front.detectMultiScale(gray, 1.1, 4, minSize=(24, 24))]
        boxes += [tuple(map(int, b)) for b in self._profile.detectMultiScale(gray, 1.1, 4, minSize=(24, 24))]
        return boxes

    def blur(self, bgr: np.ndarray, boxes: list[Box], pad: float = 0.35) -> np.ndarray:
        """Pixelate and blur each box (made 35 % bigger, so hair line and chin are covered). Changes the picture in place."""
        cv2 = self.cv2
        h, w = bgr.shape[:2]
        for x, y, bw, bh in boxes:
            px, py = int(bw * pad), int(bh * pad)
            x0, y0, x1, y1 = max(0, x - px), max(0, y - py), min(w, x + bw + px), min(h, y + bh + py)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            roi = bgr[y0:y1, x0:x1]
            small = cv2.resize(roi, (max(2, (x1 - x0) // 14), max(2, (y1 - y0) // 14)), interpolation=cv2.INTER_LINEAR)
            roi[:] = cv2.GaussianBlur(cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST), (0, 0), max(3, (x1 - x0) // 12))
        return bgr

    def process(self, bgr: np.ndarray) -> int:
        boxes = self.detect(bgr)
        self.blur(bgr, boxes)
        return len(boxes)


SAME_PERSON = 0.363                          # cosine similarity above which SFace says "the same person" (OpenCV's own threshold)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).flatten()
    return v / (np.linalg.norm(v) or 1.0)


def pick_presenter(seen: list[tuple[np.ndarray, float]], min_seen: int = 3) -> Optional[np.ndarray]:
    """Who is the presenter? The person who is seen most, and biggest: faces are grouped by who they look like and the
    group with the largest total face size wins. `seen` = [(face feature, face area as a share of the picture)].
    Returns that person's average feature, or None when nobody was seen often enough to be sure."""
    groups: list[dict] = []
    for feat, area in seen:
        f = _unit(feat)
        best = max(groups, key=lambda g: float(f @ _unit(g["sum"])), default=None)
        if best is not None and float(f @ _unit(best["sum"])) >= SAME_PERSON:
            best["sum"] = best["sum"] + f
            best["n"] += 1
            best["weight"] += area
        else:
            groups.append({"sum": f.copy(), "n": 1, "weight": area})
    groups = [g for g in groups if g["n"] >= min_seen]
    return _unit(max(groups, key=lambda g: g["weight"])["sum"]) if groups else None


def is_presenter(feat: np.ndarray, presenter: Optional[np.ndarray]) -> bool:
    return presenter is not None and float(_unit(feat) @ presenter) >= SAME_PERSON


class FaceIdentity:
    """Tells whether two faces are the same person (OpenCV SFace). Without its model file `ok` is False and every face is blurred."""

    def __init__(self, blurrer: "FaceBlurrer", model: Optional[Path] = None):
        model = Path(model or config.FACE_RECOGNIZER)
        self.ok = blurrer.kind == "yunet" and model.exists()
        self._rec = _cv2().FaceRecognizerSF.create(str(model), "") if self.ok else None

    def feature(self, bgr: np.ndarray, row: np.ndarray) -> np.ndarray:
        return _unit(self._rec.feature(self._rec.alignCrop(bgr, row)))


def find_presenter(src: Path, blurrer: "FaceBlurrer", identity: FaceIdentity, info: dict, samples: int = 300) -> Optional[np.ndarray]:
    """A first quick look at the video: one small picture every few seconds, to learn the presenter's face."""
    step = max(3.0, info["duration"] / samples)
    w = min(640, info["width"]) // 2 * 2
    h = round(info["height"] * w / info["width"] / 2) * 2
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(src), "-an", "-vf", f"fps=1/{step},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    seen: list[tuple[np.ndarray, float]] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, preexec_fn=_nice)
    try:
        while True:
            raw = proc.stdout.read(w * h * 3)
            if len(raw) < w * h * 3:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            for row in blurrer.rows(frame):
                seen.append((identity.feature(frame, row), float(row[2] * row[3]) / (w * h)))
    finally:
        proc.stdout.close()
        proc.wait()
    return pick_presenter(seen)


def blur_image_file(src: Path, dst: Path, blurrer: Optional[FaceBlurrer] = None, quality: int = 88) -> int:
    """Read a picture, blur the faces, write it. Returns how many faces were found."""
    cv2 = _cv2()
    blurrer = blurrer or FaceBlurrer()
    img = cv2.imread(str(src))
    if img is None:
        raise MissingInputError(f"cannot read the picture {src}")
    n = blurrer.process(img)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return n


def _probe(media: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-show_entries",
           "format=duration", "-of", "default=nw=1", str(media)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60).stdout
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        raise ExternalToolError(f"ffprobe could not read {Path(media).name}: {e}") from None
    info = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    try:
        return {"width": int(info["width"]), "height": int(info["height"]), "duration": float(info.get("duration", 0) or 0)}
    except (KeyError, ValueError):
        raise ExternalToolError(f"{Path(media).name} has no picture to blur") from None


def _nice() -> None:
    try:
        os.nice(10)                         # background work yields the CPU to the notes of other people
    except (AttributeError, OSError):
        pass


def self_follow(box: Box, last: Optional[tuple[Box, int]], frame_no: int, hold: int) -> bool:
    """The presenter was just here and this face is in nearly the same place: still the presenter (no need to look again)."""
    if last is None or frame_no - last[1] > hold:
        return False
    (x, y, w, h), (lx, ly, lw, lh) = box, last[0]
    return abs((x + w / 2) - (lx + lw / 2)) < lw * 0.5 and abs((y + h / 2) - (ly + lh / 2)) < lh * 0.5 and 0.6 < w / max(lw, 1) < 1.6


def blur_video(src: Path, dst: Path, progress: Optional[Callable[[int], None]] = None, max_height: int = 720, fps: int = 25,
               every: int = 4, hold: int = 12, blurrer: Optional[FaceBlurrer] = None, keep_presenter: bool = False) -> dict:
    """Re-encode a video to a web-friendly MP4 (H.264 + AAC, at most `max_height` high) with every face blurred.

    Faces are looked for on every `every`-th frame and the boxes are held for `hold` frames, so a face that
    the detector misses for a moment stays covered. The sound is copied over. progress(percent) is called now and then.

    keep_presenter: the person seen most and biggest in the video (the lecturer) is left unblurred; everybody else is blurred.
    If that person cannot be found, or the recognition model is missing, every face is blurred."""
    src, dst = Path(src), Path(dst)
    info = _probe(src)
    h = min(max_height, info["height"]) // 2 * 2
    w = round(info["width"] * h / info["height"] / 2) * 2
    blurrer = blurrer or FaceBlurrer()
    identity = FaceIdentity(blurrer) if keep_presenter else None
    presenter = find_presenter(src, blurrer, identity, info) if identity is not None and identity.ok else None
    if keep_presenter and presenter is None:
        logger.info("replay %s: no presenter found (or no recognition model): every face is blurred", dst.name)
    frame_bytes = w * h * 3
    total = max(1, int(info["duration"] * fps))
    tmp = Path(tempfile.mkstemp(suffix=".mp4", dir=dst.parent)[1])
    err = tempfile.TemporaryFile()
    dec = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(src), "-an", "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    enc = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-i", str(src), "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart", str(tmp)]
    t0 = time.time()
    faces_seen = frames = 0
    held: list[tuple[Box, int]] = []         # (box, frame number when it was last seen)
    last_presenter: Optional[tuple[Box, int]] = None
    try:
        pd = subprocess.Popen(dec, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, preexec_fn=_nice)
        pe = subprocess.Popen(enc, stdin=subprocess.PIPE, stderr=err, preexec_fn=_nice)
        try:
            while True:
                raw = pd.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
                if frames % every == 0:
                    if presenter is not None:
                        found = []
                        for row in blurrer.rows(frame):
                            box = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
                            if self_follow(box, last_presenter, frames, hold) or is_presenter(identity.feature(frame, row), presenter):
                                last_presenter = (box, frames)
                            else:
                                found.append(box)
                    else:
                        found = blurrer.detect(frame)
                    faces_seen += len(found)
                    held = [(b, n) for b, n in held if frames - n < hold] + [(b, frames) for b in found]
                boxes = [b for b, n in held if frames - n < hold]
                if boxes:
                    blurrer.blur(frame, boxes)
                pe.stdin.write(frame.tobytes())
                frames += 1
                if progress and frames % 250 == 0:
                    progress(min(99, int(100 * frames / total)))
        finally:
            pe.stdin.close()
            pd.stdout.close()
        pd.wait()
        pe.wait()
        if pe.returncode != 0 or frames == 0:
            err.seek(0)
            raise ExternalToolError(f"ffmpeg could not make the replay of {src.name}: {err.read().decode(errors='replace')[-300:]}")
        tmp.replace(dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        err.close()
    if progress:
        progress(100)
    logger.info("replay %s: %d frames, %d face detections (%s) in %.0f s", dst.name, frames, faces_seen, blurrer.kind, time.time() - t0)
    return {"frames": frames, "faces_found": faces_seen, "detector": blurrer.kind, "seconds": round(time.time() - t0, 1),
            "width": w, "height": h, "presenter_kept": presenter is not None}

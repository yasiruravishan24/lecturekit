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

    def detect(self, bgr: np.ndarray) -> list[Box]:
        h, w = bgr.shape[:2]
        if self._yunet is not None:
            if self._size != (w, h):
                self._yunet.setInputSize((w, h))
                self._size = (w, h)
            _, faces = self._yunet.detect(bgr)
            return [] if faces is None else [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces]
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


def blur_video(src: Path, dst: Path, progress: Optional[Callable[[int], None]] = None, max_height: int = 720, fps: int = 25,
               every: int = 4, hold: int = 12, blurrer: Optional[FaceBlurrer] = None) -> dict:
    """Re-encode a video to a web-friendly MP4 (H.264 + AAC, at most `max_height` high) with every face blurred.

    Faces are looked for on every `every`-th frame and the boxes are held for `hold` frames, so a face that
    the detector misses for a moment stays covered. The sound is copied over. progress(percent) is called now and then."""
    src, dst = Path(src), Path(dst)
    info = _probe(src)
    h = min(max_height, info["height"]) // 2 * 2
    w = round(info["width"] * h / info["height"] / 2) * 2
    blurrer = blurrer or FaceBlurrer()
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
            "width": w, "height": h}

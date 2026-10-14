"""Speech recognition with Whisper (small, English) via faster-whisper.

Audio is decoded with the ffmpeg command line (not PyAV), which avoids library
version clashes: ffmpeg -> 16 kHz mono float32 -> Whisper.
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from . import config, log
from .errors import ExternalToolError, MissingInputError

logger = log.get("asr")

SR = 16000


def load_audio(media: Path, start: float = 0.0, end: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start)]
    if end:
        cmd += ["-t", str(end - start)]
    cmd += ["-i", str(media), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    if not Path(media).exists():
        raise MissingInputError(f"recording not found: {media}")
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError:
        raise ExternalToolError("ffmpeg is not installed (brew install ffmpeg)") from None
    except subprocess.CalledProcessError as e:
        tail = e.stderr.decode(errors="replace").strip()[-300:]
        raise ExternalToolError(f"ffmpeg could not read {Path(media).name}: {tail}") from None
    if not raw:
        raise ExternalToolError(f"ffmpeg produced no audio from {Path(media).name} (no audio track?)")
    return np.frombuffer(raw, dtype=np.float32)


_MODELS: dict = {}


def get_model(size: str | None = None):
    """Load the Whisper model once per process (loading takes seconds; chunks call transcribe often)."""
    from faster_whisper import WhisperModel
    size = size or config.WHISPER_SIZE
    if size not in _MODELS:
        logger.info("loading whisper-%s", size)
        _MODELS[size] = WhisperModel(size, device="cpu", compute_type="int8")
    return _MODELS[size]


def transcribe(media: Path, out: Path, model_size: str | None = None,
               start: float = 0.0, end: float | None = None, time_offset: float = 0.0) -> dict:
    """Transcribe media[start:end]; segment times are lecture time = file time + start + time_offset.

    `start` cuts the file (the clip begins there). `time_offset` is for a chunk file whose own
    time 0 is lecture second X: decode from 0 and add X to every segment time.
    """
    size = model_size or config.WHISPER_SIZE
    logger.info("decoding audio %s [%s-%s]", Path(media).name, start, end)
    audio = load_audio(media, start, end)
    logger.info("transcribing %.0f s of audio with whisper-%s", len(audio) / SR, size)
    model = get_model(size)
    t0 = time.time()
    shift = start + time_offset
    segs, _ = model.transcribe(audio, language="en", vad_filter=True, beam_size=5)
    segments = [{"id": i, "start": round(s.start + shift, 2), "end": round(s.end + shift, 2),
                 "text": s.text.strip()} for i, s in enumerate(segs)]
    result = {"model": f"whisper-{size}", "seconds": round(time.time() - t0, 1),
              "audio_seconds": round(len(audio) / SR, 1), "segments": segments}
    out.write_text(json.dumps(result, indent=1))
    logger.info("%d segments in %s s", len(segments), result["seconds"])
    log.event("transcribe_done", media=Path(media).name, segments=len(segments),
              seconds=result["seconds"], audio_seconds=result["audio_seconds"])
    return result

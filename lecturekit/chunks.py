"""Chunked recording: transcribe a lecture in pieces while it is still being recorded.

Chunk contract (the client and the server must use the same values):
  index          0, 1, 2, ...; uploads may arrive out of order or twice (a repeat redoes that chunk)
  chunk_seconds  L, fixed for the whole session (default 300)
  overlap        extra seconds of audio at the start of every chunk after the first (default 5)
  chunk k        a complete audio file covering lecture time [max(0, k*L - overlap), (k+1)*L];
                 the last chunk ends at Stop
Merge rule: chunk k owns lecture time [k*L, (k+1)*L) (the last chunk owns everything after its start);
a segment is kept only from the chunk that owns its midpoint, so the overlap lets a sentence cut at a
boundary be heard whole in the next chunk and never duplicates text.

Session folder:  <session>/chunks/chunk_00000.<ext>  audio, kept
                 <session>/chunks/chunk_00000.json   Whisper result, times in lecture time
"""
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from . import asr, log
from .errors import ExternalToolError, MissingInputError, ValidationError

logger = log.get("chunks")

CHUNK_SECONDS = 300
OVERLAP = 5
MEDIA_TYPES = {".mp4", ".mov", ".m4a", ".mp3", ".wav", ".webm", ".mkv", ".aac", ".ogg"}


def chunks_dir(session: Path) -> Path:
    return Path(session) / "chunks"


def chunk_start(index: int, chunk_seconds: int = CHUNK_SECONDS, overlap: int = OVERLAP) -> float:
    """Lecture second at which chunk `index` begins (its own time 0)."""
    return 0.0 if index == 0 else max(0.0, index * chunk_seconds - overlap)


def _check_args(index: int, chunk_seconds: int, overlap: int) -> None:
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValidationError(f"chunk index must be 0, 1, 2, ...; got {index!r}")
    if chunk_seconds <= 0 or overlap < 0 or overlap >= chunk_seconds:
        raise ValidationError("need chunk_seconds > 0 and 0 <= overlap < chunk_seconds")


def _result_file(session: Path, index: int) -> Path:
    return chunks_dir(session) / f"chunk_{index:05d}.json"


def add_chunk(session: Path, index: int, audio: Path, chunk_seconds: int = CHUNK_SECONDS,
              overlap: int = OVERLAP, whisper_size: Optional[str] = None) -> dict:
    """Store one chunk file and run Whisper on it now. A repeated index redoes that chunk."""
    session, audio = Path(session), Path(audio)
    _check_args(index, chunk_seconds, overlap)
    try:
        if not audio.exists():
            raise MissingInputError(f"chunk {index}: file not found: {audio}")
        if audio.suffix.lower() not in MEDIA_TYPES:
            raise ValidationError(f"chunk {index}: unsupported file type {audio.suffix!r}")
        d = chunks_dir(session)
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob(f"chunk_{index:05d}.*"):          # idempotent: replace an earlier upload
            if old.resolve() != audio.resolve():
                old.unlink()
        stored = d / f"chunk_{index:05d}{audio.suffix.lower()}"
        if audio.resolve() != stored.resolve():
            shutil.copy2(audio, stored)
        offset = chunk_start(index, chunk_seconds, overlap)
        t0 = time.time()
        res = asr.transcribe(stored, _result_file(session, index), whisper_size, 0.0, None, time_offset=offset)
    except Exception as e:
        log.event("chunk_failed", session=str(session), index=index,
                  error=type(e).__name__, code=getattr(e, "code", "unexpected"), message=str(e)[:300])
        logger.exception("chunk %s failed for %s", index, session)
        raise
    log.event("chunk_transcribed", session=str(session), index=index, offset=offset,
              segments=len(res["segments"]), audio_seconds=res["audio_seconds"],
              seconds=round(time.time() - t0, 1))
    return {"index": index, "offset": offset, "segments": len(res["segments"]),
            "audio_seconds": res["audio_seconds"], "seconds": res["seconds"]}


def received(session: Path) -> list:
    """Chunk indexes that have a finished Whisper result, sorted."""
    d = chunks_dir(session)
    if not d.exists():
        return []
    return sorted(int(m.group(1)) for f in d.glob("chunk_*.json")
                  if (m := re.fullmatch(r"chunk_(\d+)\.json", f.name)))


def status(session: Path, total: Optional[int] = None) -> dict:
    got = received(session)
    out = {"received": got, "count": len(got)}
    if total is not None:
        out["total"] = total
        out["missing"] = [k for k in range(total) if k not in got]
    return out


def merge(session: Path, total_chunks: int, chunk_seconds: int = CHUNK_SECONDS,
          overlap: int = OVERLAP) -> dict:
    """Build <session>/transcript.json from the chunk results (same format as asr.transcribe)."""
    session = Path(session)
    _check_args(0, chunk_seconds, overlap)
    if not isinstance(total_chunks, int) or total_chunks < 1:
        raise ValidationError("total_chunks must be at least 1")
    missing = status(session, total_chunks)["missing"]
    if missing:
        raise MissingInputError(f"chunks missing: {missing}; upload them and finish again")
    segments, whisper_s, model, end_time = [], 0.0, "whisper", 0.0
    for k in range(total_chunks):
        r = json.loads(_result_file(session, k).read_text())
        model = r.get("model", model)
        whisper_s += r.get("seconds", 0)
        lo = k * chunk_seconds
        hi = (k + 1) * chunk_seconds if k < total_chunks - 1 else math.inf
        end_time = chunk_start(k, chunk_seconds, overlap) + r.get("audio_seconds", 0)
        for s in r["segments"]:
            if lo <= (s["start"] + s["end"]) / 2 < hi:
                segments.append({"id": len(segments), "start": s["start"], "end": s["end"], "text": s["text"]})
    result = {"model": model, "seconds": round(whisper_s, 1), "audio_seconds": round(end_time, 1),
              "segments": segments}
    (session / "transcript.json").write_text(json.dumps(result, indent=1))
    meta_file = session / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {"title": session.name, "clip_start": 0}
    meta["clip_start"], meta["clip_end"] = 0, result["audio_seconds"]
    meta_file.write_text(json.dumps(meta, indent=1))
    log.event("chunks_merged", session=str(session), chunks=total_chunks, segments=len(segments),
              audio_seconds=result["audio_seconds"], whisper_seconds=result["seconds"])
    logger.info("merged %d chunks: %d segments, %.0f s", total_chunks, len(segments), end_time)
    return result


def _duration(media: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(media)]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True, text=True).stdout.strip()
        return float(out)
    except FileNotFoundError:
        raise ExternalToolError("ffprobe is not installed (brew install ffmpeg)") from None
    except (subprocess.CalledProcessError, ValueError) as e:
        raise ExternalToolError(f"ffprobe could not read the length of {Path(media).name}") from e


def split_recording(media: Path, out_dir: Path, chunk_seconds: int = CHUNK_SECONDS,
                    overlap: int = OVERLAP, end: Optional[float] = None) -> list:
    """Cut a recording into chunk files exactly as a client would (for simulate-chunks and tests).

    Returns [(index, path), ...]. `end` limits the lecture to its first `end` seconds.
    """
    media, out_dir = Path(media), Path(out_dir)
    _check_args(0, chunk_seconds, overlap)
    if not media.exists():
        raise MissingInputError(f"recording not found: {media}")
    total_s = min(_duration(media), end) if end else _duration(media)
    n = max(1, math.ceil(total_s / chunk_seconds))
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for k in range(n):
        a, b = chunk_start(k, chunk_seconds, overlap), min((k + 1) * chunk_seconds, total_s)
        dst = out_dir / f"src_{k:05d}.wav"
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(a), "-t", str(b - a), "-i", str(media),
               "-vn", "-ac", "1", "-ar", "16000", str(dst)]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except FileNotFoundError:
            raise ExternalToolError("ffmpeg is not installed (brew install ffmpeg)") from None
        except subprocess.CalledProcessError as e:
            raise ExternalToolError(f"ffmpeg could not cut chunk {k}: "
                                    f"{e.stderr.decode(errors='replace').strip()[-300:]}") from None
        files.append((k, dst))
    return files

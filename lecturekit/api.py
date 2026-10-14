"""Backend-facing interface: the only module a web server needs to import.

    from lecturekit.api import process_session, check_environment

    result = process_session(recording="/path/lecture.mp4", out_dir="storage/sessions/12",
                             slides="/path/slides.pdf",          # optional; a path or a list of paths (pdf, pptx, docx, ...)
                             on_progress=lambda stage, pct, msg: ...)

process_session is synchronous and slow (minutes): run it in a worker thread or process.
It raises LectureKitError subclasses (see errors.py); a server maps `err.code` to a message.
Result schema (also written to <out_dir>/out/<config>.json):
  {"status": "done", "session_dir": str, "config": "A|B|C", "model": str,
   "notes": {"title", "full_notes": [{heading, summary, key_points}], "short_notes": [str]},
   "quiz": [{question, options[4], answer_index, bloom_level, explanation}],
   "stats": {asr_seconds, align, filter, notes, quiz, total_seconds, ratio, tokens_in, tokens_out, ...},
   "files": {"json": path, "markdown": path}}
"""
import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

from . import asr, chunks, config, log, pipeline, slidefiles, slides as slides_mod
from .errors import MissingInputError, ValidationError
from .llm import LLM

logger = log.get("api")

STAGES = ["prepare", "transcribe", "slides", "align", "filter", "notes", "quiz", "write", "done"]
MEDIA_TYPES = {".mp4", ".mov", ".m4a", ".mp3", ".wav", ".webm", ".mkv", ".aac", ".ogg"}

ProgressFn = Callable[[str, int, str], None]


def choose_config(has_slides: bool) -> str:
    """Default rule for the app: slides given -> C (full pipeline), otherwise A (speech only)."""
    return "C" if has_slides else "A"


def check_environment() -> dict:
    """Which required tools and settings are present. Never calls the Claude API, never returns the key."""
    import importlib.util
    import os
    import sys

    def has_mod(name):
        return importlib.util.find_spec(name) is not None

    checks = {
        "python>=3.10": sys.version_info >= (3, 10),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "tesseract": shutil.which("tesseract") is not None,
        "libreoffice": any(shutil.which(n) for n in slidefiles._SOFFICE_NAMES),
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
        "anthropic": has_mod("anthropic"),
        "faster_whisper": has_mod("faster_whisper"),
        "pdfplumber": has_mod("pdfplumber"),
        "pytesseract": has_mod("pytesseract"),
        "jiwer": has_mod("jiwer"),
        "sklearn": has_mod("sklearn"),
    }
    required = ["python>=3.10", "ffmpeg", "ANTHROPIC_API_KEY", "anthropic", "faster_whisper", "pdfplumber"]
    return {"checks": checks, "ok": all(checks[k] for k in required),
            "optional_missing": [k for k in ("tesseract", "libreoffice", "pytesseract", "jiwer", "sklearn") if not checks[k]],
            "models": {"dev": config.MODEL_DEV, "final": config.MODEL_FINAL},
            "whisper": config.WHISPER_SIZE, "data_dir": str(config.DATA_DIR),
            "log_dir": str(config.LOG_DIR)}


def _prepare(recording: Path, slides: list, out_dir: Path, title: Optional[str]) -> Path:
    if not recording.exists():
        raise MissingInputError(f"recording not found: {recording}")
    if recording.suffix.lower() not in MEDIA_TYPES:
        raise ValidationError(f"unsupported recording type {recording.suffix!r}; use one of {sorted(MEDIA_TYPES)}")
    slidefiles.validate(slides)
    out_dir.mkdir(parents=True, exist_ok=True)
    media = out_dir / f"media{recording.suffix.lower()}"
    if recording.resolve() != media.resolve():
        for old in out_dir.glob("media.*"):
            old.unlink()
        shutil.copy2(recording, media)
    if slides:
        slidefiles.build_pdf(slides, out_dir / "slides.pdf")
    meta = out_dir / "meta.json"
    if not meta.exists():
        meta.write_text(json.dumps({"title": title or recording.stem, "clip_start": 0}))
    return media


def _run_pipeline(out_dir: Path, cfg: str, llm: Optional[LLM], n_questions: int, final: bool,
                  reuse: bool, prog) -> dict:
    """Slide text (configs B, C) + pipeline.run; shared by process_session and finish_chunked_session."""
    if cfg != "A":
        prog("slides", "reading slides")
        sfile = out_dir / "slides.json"
        if not (reuse and sfile.exists()):
            slides_mod.extract(out_dir / "slides.pdf", sfile, force_ocr=False)
    llm = llm or LLM(model=config.MODEL_FINAL if final else config.MODEL_DEV)
    return pipeline.run(out_dir, cfg, llm, n_questions,
                        progress=lambda stage, message="": prog(stage, message))


def _result(out_dir: Path, cfg: str, run: dict) -> dict:
    return {"status": "done", "session_dir": str(out_dir), **{k: run[k] for k in
            ("config", "model", "notes", "quiz", "stats")},
            "files": {"json": str(out_dir / "out" / f"{cfg}.json"),
                      "markdown": str(out_dir / "out" / f"{cfg}.md")}}


def _progress(on_progress: Optional[ProgressFn]):
    def prog(stage: str, msg: str = "") -> None:
        pct = int(100 * STAGES.index(stage) / (len(STAGES) - 1))
        logger.info("[%3d%%] %s %s", pct, stage, msg)
        if on_progress:
            on_progress(stage, pct, msg)
    return prog


def _check_key(llm: Optional[LLM]) -> None:
    if llm is None and not os.getenv("ANTHROPIC_API_KEY"):
        raise MissingInputError("ANTHROPIC_API_KEY is not set (add it to .env)")


def process_session(recording, out_dir, slides=None, config_name: Optional[str] = None,
                    title: Optional[str] = None, n_questions: int = 10, final: bool = False,
                    whisper_size: Optional[str] = None, llm: Optional[LLM] = None,
                    reuse: bool = True, on_progress: Optional[ProgressFn] = None) -> dict:
    """Recording (+ optional slides: one or several PDF/PowerPoint/Word files) -> notes + quiz in out_dir. See the module docstring."""
    recording, out_dir = Path(recording), Path(out_dir)
    slides = slidefiles.as_list(slides)      # [] when none; PDF/PowerPoint/Word, one or several

    prog = _progress(on_progress)
    t0 = time.time()
    cfg = (config_name or choose_config(bool(slides))).upper()
    log.event("process_start", session=str(out_dir), config=cfg, has_slides=bool(slides))
    try:
        if cfg != "A" and not slides:
            raise ValidationError(f"configuration {cfg} needs slides; give a slides PDF or use configuration A")
        _check_key(llm)
        prog("prepare", f"config {cfg}")
        media = _prepare(recording, slides, out_dir, title)

        prog("transcribe", "speech to text")
        tfile = out_dir / "transcript.json"
        if reuse and tfile.exists():
            logger.info("reusing %s", tfile)
        else:
            res = asr.transcribe(media, tfile, whisper_size, 0.0, None)
            meta = json.loads((out_dir / "meta.json").read_text())
            meta["clip_end"] = res["audio_seconds"]
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

        result = _run_pipeline(out_dir, cfg, llm, n_questions, final, reuse, prog)
    except Exception as e:
        log.event("process_failed", session=str(out_dir), config=cfg,
                  error=type(e).__name__, code=getattr(e, "code", "unexpected"), message=str(e)[:300])
        logger.exception("process_session failed for %s", out_dir)
        raise
    prog("done", "finished")
    result = _result(out_dir, cfg, result)
    log.event("process_done", session=str(out_dir), config=cfg,
              seconds=round(time.time() - t0, 1), tokens_in=result["stats"]["tokens_in"],
              tokens_out=result["stats"]["tokens_out"])
    return result


# ---- chunked recording (iPhone app, browser extension): see CHUNKED_RECORDING_LATER.md --------------

def add_chunk(session_dir, index: int, chunk_file, chunk_seconds: int = chunks.CHUNK_SECONDS,
              overlap: int = chunks.OVERLAP, title: Optional[str] = None,
              whisper_size: Optional[str] = None) -> dict:
    """Run Whisper on one chunk while the lecture is still being recorded. Repeating an index redoes it.

    Synchronous (about 0.13 x chunk length). Raises LectureKitError subclasses like process_session.
    Returns {"index", "offset", "segments", "audio_seconds", "seconds", "received": [...]}.
    """
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    meta = session_dir / "meta.json"
    if not meta.exists():
        meta.write_text(json.dumps({"title": title or session_dir.name, "clip_start": 0}))
    res = chunks.add_chunk(session_dir, index, Path(chunk_file), chunk_seconds, overlap, whisper_size)
    return {**res, "received": chunks.received(session_dir)}


def chunk_status(session_dir, total: Optional[int] = None) -> dict:
    """{"received": [indexes], "count": n} and, if `total` is given, "total" and "missing"."""
    return chunks.status(Path(session_dir), total)


def finish_chunked_session(session_dir, total_chunks: int, slides=None, config_name: Optional[str] = None,
                           chunk_seconds: int = chunks.CHUNK_SECONDS, overlap: int = chunks.OVERLAP,
                           n_questions: int = 10, final: bool = False, llm: Optional[LLM] = None,
                           on_progress: Optional[ProgressFn] = None) -> dict:
    """After Stop: merge the chunk transcripts, then slides -> align -> filter -> notes -> quiz.

    Same result as process_session, with `stats["finish_seconds"]` = the wait after the last chunk was
    transcribed (stats["asr_seconds"] is the sum of the chunk times, not a wait). Missing chunks raise
    MissingInputError listing them; upload them and call this again. Slides may be given here, or
    already be in the session folder as slides.pdf.
    """
    session_dir = Path(session_dir)
    slides = slidefiles.as_list(slides)
    prog = _progress(on_progress)
    t0 = time.time()
    has_slides = bool(slides) or (session_dir / "slides.pdf").exists()
    cfg = (config_name or choose_config(has_slides)).upper()
    log.event("finish_start", session=str(session_dir), config=cfg, chunks=total_chunks, has_slides=has_slides)
    try:
        if cfg != "A" and not has_slides:
            raise ValidationError(f"configuration {cfg} needs slides; give a slides file or use configuration A")
        _check_key(llm)
        prog("prepare", f"config {cfg}, {total_chunks} chunks")
        slidefiles.validate(slides)
        if slides:
            slidefiles.build_pdf(slides, session_dir / "slides.pdf")
            (session_dir / "slides.json").unlink(missing_ok=True)     # new slides: read them again
        prog("transcribe", "merging chunk transcripts")
        chunks.merge(session_dir, total_chunks, chunk_seconds, overlap)
        run = _run_pipeline(session_dir, cfg, llm, n_questions, final, True, prog)
    except Exception as e:
        log.event("process_failed", session=str(session_dir), config=cfg,
                  error=type(e).__name__, code=getattr(e, "code", "unexpected"), message=str(e)[:300])
        logger.exception("finish_chunked_session failed for %s", session_dir)
        raise
    prog("done", "finished")
    result = _result(session_dir, cfg, run)
    result["stats"] = {**result["stats"], "finish_seconds": round(time.time() - t0, 1)}
    log.event("process_done", session=str(session_dir), config=cfg, seconds=round(time.time() - t0, 1),
              tokens_in=result["stats"]["tokens_in"], tokens_out=result["stats"]["tokens_out"])
    return result

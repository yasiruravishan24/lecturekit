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

from . import asr, chunks, config, enrich, faces, generate, log, pipeline, slidefiles, slides as slides_mod, visuals
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


def _review_new_quiz(out_dir: Path, cfg: str, run: dict, llm: Optional[LLM], final: bool) -> dict:
    """A second look at a new quiz (only when LK_QUIZ_REVIEW=1): unclear questions are dropped, a wrong answer mark is corrected."""
    if not config.QUIZ_REVIEW or not run.get("quiz"):
        return run
    llm = llm or LLM(model=config.MODEL_FINAL if final else config.MODEL_DEV)
    quiz, changes = enrich.review_quiz(llm, enrich._notes_text(run["notes"]), run["quiz"])
    if changes["dropped"] or changes["fixed"]:
        run = {**run, "quiz": quiz, "stats": {**run["stats"], "quiz_review": {"dropped": len(changes["dropped"]), "fixed": len(changes["fixed"])}}}
        result_file = out_dir / "out" / f"{cfg}.json"
        if result_file.exists():                                  # keep the files on disk in step with the quiz
            data = json.loads(result_file.read_text())
            data["quiz"], data["stats"] = run["quiz"], run["stats"]
            result_file.write_text(json.dumps(data, indent=1))
            (out_dir / "out" / f"{cfg}.md").write_text(pipeline.render_markdown(data))
        logger.info("quiz review: %d dropped, %d corrected", len(changes["dropped"]), len(changes["fixed"]))
    return run


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
            res = asr.transcribe(media, tfile, whisper_size, 0.0, None,
                                 on_segment=lambda done, total: prog("transcribe", f"{int(100 * done / max(total, 1))}% of the audio"))
            meta = json.loads((out_dir / "meta.json").read_text())
            meta["clip_end"] = res["audio_seconds"]
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

        result = _run_pipeline(out_dir, cfg, llm, n_questions, final, reuse, prog)
        result = _review_new_quiz(out_dir, cfg, result, llm, final)
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
        run = _review_new_quiz(session_dir, cfg, run, llm, final)
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


# ---- the picture side (optional; the notes and quiz above never wait for it) ------------------------------------
def extract_visuals(video, out_dir, final: bool = True, llm: Optional[LLM] = None, max_frames: Optional[int] = None,
                    on_progress: Optional[Callable[[str, int], None]] = None) -> dict:
    """Pictures of the board and screen from a video: faces blurred, read by Claude, cropped screenshots saved.

    Writes out_dir/shots/shot_NNN.jpg and out_dir/visuals.json. Returns
    {"model", "frames_sampled", "faces_blurred", "detector", "seconds",
     "shots": [{"t": seconds, "kind": "board|slide|screen", "text", "description", "file"}]}."""
    video, out_dir = Path(video), Path(out_dir)
    if not video.exists():
        raise MissingInputError(f"recording not found: {video}")
    _check_key(llm)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = llm or LLM(model=config.MODEL_FINAL if final else config.MODEL_DEV)
    return visuals.extract_visuals(video, out_dir, llm, max_frames=max_frames, progress=on_progress)


def refine_notes_with_visuals(out_dir, notes: dict, final: bool = True, llm: Optional[LLM] = None,
                              config_name: Optional[str] = None) -> dict:
    """The notes with the board and screen content merged in (needs visuals.json from extract_visuals).
    Returns the new notes, or the same notes when nothing useful was seen. The quiz is not changed."""
    out_dir = Path(out_dir)
    vfile = out_dir / "visuals.json"
    if not vfile.exists():
        raise MissingInputError(f"{vfile} not found: run extract_visuals first")
    shots = json.loads(vfile.read_text()).get("shots", [])
    if not shots:
        return notes
    _check_key(llm)
    llm = llm or LLM(model=config.MODEL_FINAL if final else config.MODEL_DEV)
    new = generate.refine_notes(llm, notes, visuals.visual_context(shots))
    cfg = (config_name or "").upper()
    result_file = out_dir / "out" / f"{cfg}.json"
    if cfg and result_file.exists():                           # keep the files on disk in step with the notes
        result = json.loads(result_file.read_text())
        result["notes"] = new
        result_file.write_text(json.dumps(result, indent=1))
        (out_dir / "out" / f"{cfg}.md").write_text(pipeline.render_markdown(result))
    return new


def add_visual_questions(out_dir, quiz: list, n: int = 3, final: bool = True, llm: Optional[LLM] = None,
                         config_name: Optional[str] = None) -> list:
    """Extra quiz questions about the board and screen content (needs visuals.json). Returns only the NEW questions, in the
    same shape as the quiz, each with "source": "board"; the existing questions are not changed. Returns [] when nothing
    useful was seen or n is 0."""
    out_dir = Path(out_dir)
    vfile = out_dir / "visuals.json"
    if not vfile.exists():
        raise MissingInputError(f"{vfile} not found: run extract_visuals first")
    shots = json.loads(vfile.read_text()).get("shots", [])
    if n <= 0 or not shots:
        return []
    _check_key(llm)
    llm = llm or LLM(model=config.MODEL_FINAL if final else config.MODEL_DEV)
    new = generate.make_visual_quiz(llm, visuals.visual_context(shots), [q["question"] for q in quiz], n)
    new = [{**q, "source": "board"} for q in new]
    cfg = (config_name or "").upper()
    result_file = out_dir / "out" / f"{cfg}.json"
    if new and cfg and result_file.exists():                    # keep the files on disk in step with the quiz
        result = json.loads(result_file.read_text())
        result["quiz"] = list(result.get("quiz", [])) + new
        result_file.write_text(json.dumps(result, indent=1))
        (out_dir / "out" / f"{cfg}.md").write_text(pipeline.render_markdown(result))
    return new


def go_deeper(out_dir, notes: dict, quiz: list, n_questions: int = 3, max_searches: int = 4, final: bool = True, llm: Optional[LLM] = None,
              config_name: Optional[str] = None, on_progress: Optional[Callable[[str, int], None]] = None) -> dict:
    """Beyond the lecture: per topic of the notes, missing background, an example, common mistakes, things to check and pages to read
    (found with web search), plus up to n_questions extra quiz questions (each with "source": "extra", already checked).

    Writes out_dir/deeper.json and returns {"model", "web", "seconds", "topics", "not_covered", "questions"}.
    The notes and the old questions are not changed; the new questions are appended to out/<config>.json."""
    out_dir = Path(out_dir)
    if not notes.get("full_notes"):
        raise ValidationError("there are no notes to go deeper on")
    _check_key(llm)
    p = on_progress or (lambda stage, pct: None)
    t0 = time.time()
    llm = llm or LLM(model=config.MODEL_FINAL if final else config.MODEL_DEV, max_tokens=12000)
    result = enrich.deeper(llm, notes, max_searches, progress=p)
    p("questions", 75)
    questions = enrich.make_extra_quiz(llm, result, [q["question"] for q in quiz], n_questions)
    p("save", 95)
    data = {"model": llm.model, "seconds": round(time.time() - t0, 1), **result, "questions": questions}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "deeper.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
    cfg = (config_name or "").upper()
    result_file = out_dir / "out" / f"{cfg}.json"
    if questions and cfg and result_file.exists():
        saved = json.loads(result_file.read_text())
        saved["quiz"] = list(saved.get("quiz", [])) + questions
        result_file.write_text(json.dumps(saved, indent=1))
        (out_dir / "out" / f"{cfg}.md").write_text(pipeline.render_markdown(saved))
    p("done", 100)
    return data


def _audio_replay(src: Path, dst: Path) -> None:
    import subprocess
    from .errors import ExternalToolError
    done = subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src), "-vn", "-c:a", "aac", "-b:a", "96k", str(dst)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise ExternalToolError(f"ffmpeg could not make the audio replay: {done.stderr[-300:]}")


def make_replay(src, dst_dir, name: str = "replay", on_progress: Optional[Callable[[int], None]] = None) -> dict:
    """A web-friendly copy for watching again, written to dst_dir/<name>.mp4 or .m4a.

    Video: H.264 + AAC, at most 720 p, every face blurred except the presenter's. Audio (or a video file without a picture): a small AAC file.
    Returns {"kind": "video" | "audio", "file": "<name>.mp4" | "<name>.m4a", "bytes": int}."""
    src, dst_dir = Path(src), Path(dst_dir)
    if not src.exists():
        raise MissingInputError(f"recording not found: {src}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    has_picture = False
    if src.suffix.lower() not in (".m4a", ".mp3", ".wav", ".aac", ".ogg"):
        try:
            has_picture = faces._probe(src)["height"] > 0
        except Exception:
            has_picture = False
    if not has_picture:
        dst = dst_dir / f"{name}.m4a"
        _audio_replay(src, dst)
        return {"kind": "audio", "file": dst.name, "bytes": dst.stat().st_size}
    dst = dst_dir / f"{name}.mp4"
    info = faces.blur_video(src, dst, progress=on_progress, keep_presenter=config.KEEP_PRESENTER)
    return {"kind": "video", "file": dst.name, "bytes": dst.stat().st_size, **{k: info[k] for k in ("faces_found", "detector", "seconds", "presenter_kept")}}

"""Minimal example of how a backend exposes lecturekit (NOT the final server).

    pip install -r requirements-server.txt
    uvicorn examples.backend_stub:app --reload --port 8000

Whole file (web app):
    curl -F recording=@talk.mp4 -F slides=@slides.pdf localhost:8000/process     -> {"job_id": "..."}
    curl localhost:8000/jobs/<job_id>                                            -> status, stage, percent
    curl localhost:8000/jobs/<job_id>/result                                     -> notes + quiz
Live, in chunks (iPhone app, browser extension; contract in lecturekit/chunks.py):
    curl -X POST localhost:8000/live                                             -> {"live_id": "..."}
    curl -F chunk=@c0.m4a "localhost:8000/live/<id>/chunks?index=0"              -> queued (repeat = redo)
    curl localhost:8000/live/<id>                                                -> received / failed chunks
    curl -X POST -F slides=@slides.pdf "localhost:8000/live/<id>/finish?total=N" -> {"job_id": "..."}  (then /jobs)
The real server (server/ in the project plan) does the same with a database around it.

All Whisper and Claude work goes through ONE worker thread fed by a FIFO queue, so "finish" always runs
after the chunks sent before it and the unlocked cache/usage files are never written by two jobs at once.
"""
import queue
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from lecturekit import log
from lecturekit.api import add_chunk, check_environment, chunk_status, finish_chunked_session, process_session
from lecturekit.chunks import CHUNK_SECONDS, OVERLAP
from lecturekit.errors import LectureKitError

log.setup()
app = FastAPI(title="lecturekit stub")
STORAGE = Path("storage/stub")
JOBS: dict[str, dict] = {}
LIVE: dict[str, dict] = {}
QUEUE: "queue.Queue" = queue.Queue()            # FIFO; one worker: Whisper + Claude calls never run in parallel


def _worker_loop() -> None:
    while True:
        task = QUEUE.get()
        try:
            task()
        finally:
            QUEUE.task_done()


threading.Thread(target=_worker_loop, daemon=True, name="lecturekit-worker").start()


def _error(e: Exception) -> dict:
    if isinstance(e, LectureKitError):           # expected failure: show the message to the user
        return {"code": e.code, "message": str(e)}
    return {"code": "unexpected", "message": type(e).__name__}      # bug: details are in logs/errors.log


def _run_job(job_id: str, fn) -> None:
    job = JOBS[job_id]
    job["status"] = "processing"
    try:
        job["result"] = fn(lambda stage, pct, msg: job.update(stage=stage, percent=pct, message=msg))
        job["status"] = "done"
    except Exception as e:
        job.update(status="failed", error=_error(e))


def _new_job() -> str:
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "queued", "stage": "queued", "percent": 0, "message": "", "error": None}
    return job_id


def _save(upload: UploadFile, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


@app.get("/health")
def health():
    return check_environment()


# ---- whole-file upload ------------------------------------------------------------------------------

@app.post("/process")
def start(recording: UploadFile = File(...), slides: UploadFile | None = File(None)):
    job_id = _new_job()
    inbox = STORAGE / job_id / "upload"
    rec_path = _save(recording, inbox / (recording.filename or "recording.mp4"))
    slides_path = _save(slides, inbox / (slides.filename or "slides.pdf")) if slides is not None and slides.filename else None
    session = STORAGE / job_id / "session"
    QUEUE.put(lambda: _run_job(job_id, lambda progress: process_session(
        rec_path, session, slides=slides_path, on_progress=progress)))
    return {"job_id": job_id}


# ---- live recording in chunks -----------------------------------------------------------------------

@app.post("/live")
def live_start(chunk_seconds: int = CHUNK_SECONDS, overlap: int = OVERLAP):
    live_id = uuid.uuid4().hex[:8]
    LIVE[live_id] = {"chunk_seconds": chunk_seconds, "overlap": overlap, "failed": {}, "queued": {},
                     "session": STORAGE / live_id / "session"}
    return {"live_id": live_id, "chunk_seconds": chunk_seconds, "overlap": overlap}


def _live(live_id: str) -> dict:
    if live_id not in LIVE:
        raise HTTPException(404, "unknown live session")
    return LIVE[live_id]


@app.post("/live/{live_id}/chunks")
def live_chunk(live_id: str, index: int, chunk: UploadFile = File(...)):
    """Idempotent: uploading an index again redoes that chunk."""
    live = _live(live_id)
    suffix = Path(chunk.filename or "chunk.m4a").suffix or ".m4a"
    path = _save(chunk, STORAGE / live_id / "uploads" / f"{index:05d}{suffix}")
    live["failed"].pop(index, None)
    live["queued"][index] = live["queued"].get(index, 0) + 1      # repeats of one index can be pending together

    def task():
        try:
            add_chunk(live["session"], index, path, live["chunk_seconds"], live["overlap"])
        except Exception as e:                    # keep going: the client re-uploads, finish lists what is missing
            live["failed"][index] = _error(e)
        finally:
            live["queued"][index] -= 1
            if live["queued"][index] == 0:
                del live["queued"][index]

    QUEUE.put(task)
    return {"queued": index}


@app.get("/live/{live_id}")
def live_status(live_id: str):
    live = _live(live_id)
    got = chunk_status(live["session"])
    return {"received": got["received"], "queued": sorted(live["queued"]), "failed": live["failed"]}


@app.post("/live/{live_id}/finish")
def live_finish(live_id: str, total: int, slides: UploadFile | None = File(None)):
    live = _live(live_id)
    slides_path = _save(slides, STORAGE / live_id / "uploads" / (slides.filename or "slides.pdf")) \
        if slides is not None and slides.filename else None
    job_id = _new_job()
    QUEUE.put(lambda: _run_job(job_id, lambda progress: finish_chunked_session(
        live["session"], total, slides=slides_path, chunk_seconds=live["chunk_seconds"],
        overlap=live["overlap"], on_progress=progress)))
    return {"job_id": job_id}


# ---- jobs -------------------------------------------------------------------------------------------

@app.get("/jobs/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return {k: job[k] for k in ("status", "stage", "percent", "message", "error")}


@app.get("/jobs/{job_id}/result")
def result(job_id: str):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "result not ready")
    return job["result"]

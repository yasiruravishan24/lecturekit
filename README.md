# lecturekit - lecture recording + slides -> notes + quiz

The pipeline package behind the whole project. The terminal, the evaluation scripts and the backend
(web app, iPhone app, browser extension) all use this same code. Plan: `LECTUREKIT_PLAN.md`.
Thesis testing: `TESTING.md` (separate).

## Set up on the MacBook (once)
```
brew install ffmpeg tesseract
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # put your ANTHROPIC_API_KEY in .env (never share or commit it)
python -m pytest -q         # no network, no API cost
python -m lecturekit.cli doctor
```

## Process one recording (what the backend does)
```
python -m lecturekit.cli process talk.mp4 --slides slides.pdf --out storage/demo
python -m lecturekit.cli process talk.mp4 --out storage/demo2        # no slides -> config A
python -m lecturekit.cli process talk.mp4 --slides slides.pdf --out storage/demo --json
```
Result: `storage/demo/out/C.md` (readable) and `C.json`. Exit code 0 ok, 2 missing/invalid input, 1 other failure.

## Use from Python (backend)
```python
from lecturekit.api import process_session, check_environment
result = process_session("talk.mp4", "storage/sessions/12", slides="slides.pdf",
                         on_progress=lambda stage, pct, msg: print(pct, stage))
```
See `examples/backend_stub.py` (FastAPI, background thread, job status).

## Logs (folder `logs/`)
| File | Content |
|---|---|
| `lecturekit.log` | everything (debug), rotating |
| `errors.log` | errors with tracebacks only |
| `events.jsonl` | one JSON line per event: `transcribe_done`, `run_done`, `process_start/done/failed` |
Token use per Claude call: `.cache/usage.jsonl`. Console level: `-v` debug, `-q` warnings only.

## Evaluation data folder: data/<ID>/ (see TESTING.md)
`media.*`, `slides.pdf`, `meta.json`, `ref_transcript.txt`, `ref_slides.txt`, `off_topic_spans.json`.

## Evaluation commands
```
python -m lecturekit.cli transcribe L02
python -m lecturekit.cli slides L02 --force-ocr
python -m lecturekit.cli refslides L02
python -m lecturekit.cli run L02 --config all            # dev model; add --final for the final model
python -m lecturekit.cli eval L02
python -m lecturekit.cli filter-diff L02                 # false positives / negatives of the filter
python -m lecturekit.cli pack L02 L05 L09                # blind rating packs
python -m lecturekit.cli report L02 L05 L09
```
Prompts are cached on disk (`.cache/llm`), so repeating a run costs nothing and shows 0.0 s.

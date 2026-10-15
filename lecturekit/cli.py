"""Terminal interface.  python -m lecturekit.cli --help"""
import argparse
import json
import sys
from pathlib import Path

from . import config, evaluate, log, pipeline, rating, report
from .errors import LectureKitError, MissingInputError
from .llm import LLM

logger = log.get("cli")


def ldir(name: str) -> Path:
    return config.DATA_DIR / name


def main(argv=None) -> int:
    """Run a command. Returns 0 on success, 2 for missing/invalid input, 1 for other failures."""
    ap = argparse.ArgumentParser(prog="lecturekit")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug output on the console")
    ap.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="check ffmpeg, tesseract, API key, packages (no API call)")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("process", help="recording (+ slides) -> notes + quiz in one folder, like the backend does")
    p.add_argument("recording"); p.add_argument("--slides", nargs="+", default=None, help="one or more slide files (pdf, pptx, docx, ...) in lecture order")
    p.add_argument("--out", required=True, help="session folder to create/use")
    p.add_argument("--config", default=None, help="A, B or C (default: C with slides, else A)")
    p.add_argument("--title", default=None); p.add_argument("--questions", type=int, default=10)
    p.add_argument("--final", action="store_true"); p.add_argument("--no-reuse", action="store_true")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")

    p = sub.add_parser("simulate-chunks", help="cut a recording into chunks, transcribe them one by one as a live "
                                               "client would, merge, and (unless --no-notes) make notes + quiz")
    p.add_argument("recording"); p.add_argument("--out", required=True, help="session folder to create/use")
    p.add_argument("--slides", nargs="+", default=None, help="slide files in lecture order (optional)")
    p.add_argument("--chunk-seconds", type=int, default=300); p.add_argument("--overlap", type=int, default=5)
    p.add_argument("--end", type=float, default=None, help="use only the first END seconds of the recording")
    p.add_argument("--compare", default=None, help="whole-file transcript.json; prints WER of the merged transcript")
    p.add_argument("--no-notes", action="store_true", help="stop after merging (no Claude calls)")
    p.add_argument("--final", action="store_true"); p.add_argument("--questions", type=int, default=10)

    p = sub.add_parser("filter-diff", help="list filter false positives / false negatives for one lecture")
    p.add_argument("lecture")

    p = sub.add_parser("transcribe", help="Whisper transcript of the clip")
    p.add_argument("lecture"); p.add_argument("--model", default=None)

    p = sub.add_parser("slides", help="extract slide text from slides.pdf")
    p.add_argument("lecture"); p.add_argument("--force-ocr", action="store_true")

    p = sub.add_parser("refslides", help="write ref_slides.txt from the slides.pdf text layer")
    p.add_argument("lecture")

    p = sub.add_parser("run", help="run configuration A, B, C or all")
    p.add_argument("lecture"); p.add_argument("--config", default="all")
    p.add_argument("--final", action="store_true", help="use the final (stronger) model")
    p.add_argument("--questions", type=int, default=10)

    p = sub.add_parser("eval", help="WER, slide accuracy, filter P/R/F1 for one lecture")
    p.add_argument("lecture")

    p = sub.add_parser("pack", help="make blind rating packs")
    p.add_argument("lectures", nargs="+")
    p.add_argument("--rated-questions", type=int, default=4, help="questions per set shown to raters (default 4)")

    sub.add_parser("env-info", help="write environment.md (hardware, OS, tools, packages) to the results folder")

    p = sub.add_parser("report", help="build the result tables")
    p.add_argument("lectures", nargs="+")

    a = ap.parse_args(argv)
    log.setup(verbose=a.verbose, quiet=a.quiet)
    logger.debug("command: %s", a)
    try:
        _dispatch(a)
    except LectureKitError as e:
        logger.error("%s: %s", e.code, e)
        return e.exit_code
    except Exception:
        logger.exception("unexpected error in '%s' (traceback: %s)", a.cmd, config.LOG_DIR / "errors.log")
        return 1
    return 0


def _dispatch(a) -> None:
    d = ldir(a.lecture) if hasattr(a, "lecture") else None

    if a.cmd == "doctor":
        from .api import check_environment
        env = check_environment()
        if a.json:
            print(json.dumps(env, indent=1))
        else:
            for k, v in env["checks"].items():
                print(f"{'ok     ' if v else 'MISSING'} {k}")
            print(f"models: {env['models']}  whisper: {env['whisper']}  logs: {env['log_dir']}")
            print("READY" if env["ok"] else "NOT READY (fix the MISSING items above)")
        if not env["ok"]:
            raise LectureKitError("environment check failed")
    elif a.cmd == "process":
        from .api import process_session
        res = process_session(a.recording, a.out, slides=a.slides, config_name=a.config, title=a.title,
                              n_questions=a.questions, final=a.final, reuse=not a.no_reuse)
        if a.json:
            print(json.dumps(res, indent=1))
        else:
            s = res["stats"]
            print(f"done: config {res['config']}, {len(res['quiz'])} questions, {s['total_seconds']} s, "
                  f"tokens {s['tokens_in']}/{s['tokens_out']}\n  {res['files']['markdown']}")
    elif a.cmd == "simulate-chunks":
        _simulate_chunks(a)
    elif a.cmd == "filter-diff":
        rows = evaluate.filter_diff(d)
        for r in rows:
            print(f"{r['type']}  {r['start']:7.1f}  [{r['id']}] {r['text']}")
        print(f"{sum(r['type'] == 'FP' for r in rows)} false positives, "
              f"{sum(r['type'] == 'FN' for r in rows)} false negatives")
    elif a.cmd == "transcribe":
        from . import asr
        m = pipeline.load_meta(d)
        media = next(d.glob("media.*"))
        r = asr.transcribe(media, d / "transcript.json", a.model, m.get("clip_start", 0), m.get("clip_end"))
        print(f"{len(r['segments'])} segments in {r['seconds']} s")
    elif a.cmd == "slides":
        from . import slides
        r = slides.extract(d / "slides.pdf", d / "slides.json", a.force_ocr)
        print(f"{len(r['slides'])} slides in {r['seconds']} s")
    elif a.cmd == "refslides":
        import pdfplumber
        with pdfplumber.open(d / "slides.pdf") as doc:
            text = "\n\n".join((pg.extract_text() or "").strip() for pg in doc.pages)
        (d / "ref_slides.txt").write_text(text)
        print(f"{len(text.split())} words written to {d / 'ref_slides.txt'} - check it by eye")
    elif a.cmd == "run":
        llm = LLM(model=config.MODEL_FINAL if a.final else config.MODEL_DEV)
        for c in (pipeline.CONFIGS if a.config == "all" else [a.config.upper()]):
            r = pipeline.run(d, c, llm, a.questions)
            s = r["stats"]
            print(f"{a.lecture} {c}: {len(r['quiz'])} questions, {s['total_seconds']} s, "
                  f"ratio {s['ratio']}, tokens {s['tokens_in']}/{s['tokens_out']}")
    elif a.cmd == "eval":
        if not d.is_dir():
            raise MissingInputError(f"lecture folder not found: {d}")
        out, missing = {}, []
        if (d / "ref_transcript.txt").exists():
            out["wer"] = evaluate.wer_lecture(d)
            out["wer_top_errors"] = evaluate.wer_top_errors(d)
        else:
            missing.append("ref_transcript.txt (WER)")
        if (d / "off_topic_spans.json").exists() and (d / "out" / "C.json").exists():
            out["filter"] = evaluate.filter_prf(d)
            excl = evaluate.filter_prf_without_licence(d)
            if excl:
                out["filter_without_licence"] = excl
        else:
            missing.append("off_topic_spans.json and out/C.json (filter scores)")
        corr = evaluate.slide_corrections(d)
        if corr:
            out["slide_corrections"] = {k: [list(c) for c in v] for k, v in corr.items()}
        print(json.dumps(out, indent=1))
        if (d / "ref_slides.txt").exists():
            print("slide accuracy (order-sensitive):", round(evaluate.slide_accuracy(d), 3))
            print("slide words (order-independent):",
                  {k: round(v, 3) for k, v in evaluate.slide_word_scores(d).items()})
        else:
            missing.append("ref_slides.txt (slide accuracy)")
        if missing:
            print("NOT COMPUTED, missing:", "; ".join(missing))
    elif a.cmd == "env-info":
        _env_info()
    elif a.cmd == "pack":
        rating.make_packs(config.DATA_DIR, a.lectures, config.RESULTS_DIR / "rating", rated_questions=a.rated_questions)
        print(f"Written to {config.RESULTS_DIR / 'rating'} (give each rater rating_pack.md and their own three CSVs; keep the KEY file)")
    elif a.cmd == "report":
        text = report.build(config.DATA_DIR, a.lectures, config.RESULTS_DIR / "rating")
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (config.RESULTS_DIR / "summary.md").write_text(text)
        svg = report.figure_note_ratings_svg(config.RESULTS_DIR / "rating")
        if svg:
            (config.RESULTS_DIR / "figure_note_ratings.svg").write_text(svg)
        print(text)


def _simulate_chunks(a) -> None:
    import shutil
    from . import api, chunks
    session = Path(a.out)
    src = session / "simulated_upload"
    parts = chunks.split_recording(a.recording, src, a.chunk_seconds, a.overlap, a.end)
    print(f"{len(parts)} chunks of {a.chunk_seconds} s (overlap {a.overlap} s)")
    times = []
    for k, f in parts:
        r = api.add_chunk(session, k, f, a.chunk_seconds, a.overlap)
        times.append(r["seconds"])
        print(f"  chunk {k}: {r['audio_seconds']:.0f} s audio, whisper {r['seconds']} s, {r['segments']} segments")
    shutil.rmtree(src, ignore_errors=True)
    keeps_up = all(t < a.chunk_seconds for t in times)
    print(f"whisper total {sum(times):.0f} s; keeps up with real time: {'yes' if keeps_up else 'NO'}")
    if a.no_notes:
        merged = chunks.merge(session, len(parts), a.chunk_seconds, a.overlap)
        wait = times[-1]
    else:
        res = api.finish_chunked_session(session, len(parts), slides=a.slides, chunk_seconds=a.chunk_seconds,
                                         overlap=a.overlap, n_questions=a.questions, final=a.final)
        merged = json.loads((session / "transcript.json").read_text())
        wait = times[-1] + res["stats"]["finish_seconds"]
        print(f"config {res['config']}, {len(res['quiz'])} questions, tokens "
              f"{res['stats']['tokens_in']}/{res['stats']['tokens_out']}\n  {res['files']['markdown']}")
    print(f"estimated wait after Stop: {wait:.0f} s (last chunk {times[-1]} s + the rest)")
    if a.compare:
        ref = json.loads(Path(a.compare).read_text())["segments"]
        limit = merged["audio_seconds"]
        ref_text = " ".join(g["text"] for g in ref if (g["start"] + g["end"]) / 2 <= limit)
        hyp_text = " ".join(g["text"] for g in merged["segments"])
        print(f"WER of chunked vs whole-file transcript: {evaluate.wer(ref_text, hyp_text):.3f}")


def _env_info() -> None:
    """results/environment.md: hardware, OS, tools, settings and key package versions (thesis 5.2.3)."""
    import datetime, platform, subprocess
    from .api import check_environment

    def sh(*cmd) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return "unavailable"

    pk = [l for l in sh(sys.executable, "-m", "pip", "list").splitlines()
          if l.split()[:1] and l.split()[0].lower() in
          ("anthropic", "faster-whisper", "ctranslate2", "pdfplumber", "pytesseract", "numpy",
           "scikit-learn", "jiwer", "pytest")]
    env = check_environment()
    mem = sh("sysctl", "-n", "hw.memsize")
    text = "\n".join([
        "# Environment record", "", f"Recorded: {datetime.datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}", "",
        "## Hardware and OS", "```", sh("sw_vers") or platform.platform(),
        f"cpu:    {sh('sysctl', '-n', 'machdep.cpu.brand_string') or platform.processor()}",
        f"memory: {int(mem) // 1073741824} GB" if mem.isdigit() else "memory: unavailable",
        f"arch:   {platform.machine()}", "```", "",
        "## Tools", "```", f"python:    {platform.python_version()}",
        f"ffmpeg:    {(sh('ffmpeg', '-version').splitlines() or ['unavailable'])[0]}",
        f"tesseract: {(sh('tesseract', '--version').splitlines() or ['unavailable'])[0]}", "```", "",
        "## Pipeline settings", "```json", json.dumps({k: env[k] for k in ("models", "whisper")}, indent=1), "```", "",
        "## Key packages", "```", *pk, "```", ""])
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "environment.md").write_text(text)
    print(text)


if __name__ == "__main__":
    sys.exit(main())

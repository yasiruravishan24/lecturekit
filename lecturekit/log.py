"""Logging for lecturekit.

Files (in config.LOG_DIR, default ./logs):
  lecturekit.log   everything (DEBUG and above), rotating
  errors.log       ERROR and above only, with tracebacks
  events.jsonl     one JSON line per pipeline event (success, failure, timing): easy to parse
Console: INFO (or DEBUG with -v, WARNING with -q).
"""
import json
import logging
import logging.handlers
import sys
import time

from . import config

ROOT = "lecturekit"


class _ConsoleFormatter(logging.Formatter):
    """Console shows the message only; tracebacks go to the log files."""
    def formatException(self, ei):
        return ""


def setup(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    root = logging.getLogger(ROOT)
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    root.propagate = False
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO)
    console.setFormatter(_ConsoleFormatter("%(levelname)-7s %(message)s"))
    root.addHandler(console)
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        allf = logging.handlers.RotatingFileHandler(
            config.LOG_DIR / "lecturekit.log", maxBytes=2_000_000, backupCount=3)
        allf.setLevel(logging.DEBUG); allf.setFormatter(fmt)
        errf = logging.FileHandler(config.LOG_DIR / "errors.log")
        errf.setLevel(logging.ERROR); errf.setFormatter(fmt)
        root.addHandler(allf); root.addHandler(errf)
    except OSError as e:                      # read-only folder etc.: keep working, console only
        root.warning("cannot write log files in %s: %s", config.LOG_DIR, e)
    return root


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT}.{name}")


def event(name: str, **fields) -> None:
    """Append one structured event to events.jsonl (best effort, never raises)."""
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": name, **fields}
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        with (config.LOG_DIR / "events.jsonl").open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        pass

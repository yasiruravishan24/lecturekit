"""Settings read from environment / .env (never commit the .env file)."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv optional in tests
    pass

DATA_DIR = Path(os.getenv("LK_DATA", "data"))
CACHE_DIR = Path(os.getenv("LK_CACHE", ".cache"))
LOG_DIR = Path(os.getenv("LK_LOG", "logs"))
RESULTS_DIR = Path(os.getenv("LK_RESULTS", "results"))      # evaluation outputs: rating packs, summary.md
MODEL_DEV = os.getenv("LK_MODEL_DEV", "claude-haiku-5-5")
MODEL_FINAL = os.getenv("LK_MODEL_FINAL", "claude-sonnet-5-5")
WHISPER_SIZE = os.getenv("LK_WHISPER_SIZE", "small")
TEMPERATURE = 0.0          # requested; the SDK/model may refuse it (recorded as temperature_fixed)

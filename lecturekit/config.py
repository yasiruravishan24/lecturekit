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

# optional visual stage: pictures of the board and screen, faces blurred, replay video
FACE_MODEL = Path(os.getenv("LK_FACE_MODEL", "/opt/models/face_detection_yunet_2023mar.onnx"))   # OpenCV YuNet; missing: Haar cascade
FACE_RECOGNIZER = Path(os.getenv("LK_FACE_RECOGNIZER", "/opt/models/face_recognition_sface_2021dec.onnx"))   # OpenCV SFace: finds the presenter
KEEP_PRESENTER = os.getenv("LK_KEEP_PRESENTER", "1") != "0"      # the presenter's face stays clear in the replay, all others are blurred
QUIZ_CODE_BLOCKS = os.getenv("LK_QUIZ_CODE_BLOCKS", "0") == "1"       # ask for code in fenced blocks (the web page shows them as code); 0 = the prompt exactly as evaluated
QUIZ_REVIEW = os.getenv("LK_QUIZ_REVIEW", "0") == "1"            # a second look at every new quiz (drops unclear questions); 0 = the quiz exactly as evaluated
VISUAL_MAX_FRAMES = int(os.getenv("LK_VISUAL_MAX_FRAMES", "80"))                                   # most pictures sent to the AI per lecture

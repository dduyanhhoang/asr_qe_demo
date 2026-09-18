"""All knobs live here; every one can be overridden with an env var."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SAMPLE_RATE = 16_000

# --- VAD / segmentation --------------------------------------------------
VAD_MODEL_PATH = ROOT / "models" / "silero_vad.onnx"
VAD_START_THRESHOLD = float(os.getenv("VAD_START_THRESHOLD", "0.5"))
VAD_END_THRESHOLD = float(os.getenv("VAD_END_THRESHOLD", "0.35"))
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "700"))   # silence that closes an utterance
VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "300"))     # shorter blips are dropped
VAD_PRE_ROLL_MS = int(os.getenv("VAD_PRE_ROLL_MS", "300"))         # audio kept before speech onset
VAD_MAX_UTTERANCE_S = float(os.getenv("VAD_MAX_UTTERANCE_S", "30"))

# --- ASR -----------------------------------------------------------------
ASR_BACKEND = os.getenv("ASR_BACKEND", "whisper")          # whisper | mock
ASR_MODEL = os.getenv("ASR_MODEL", "small")                # faster-whisper size or CT2 path
ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8")
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE") or None           # e.g. "vi"; None = auto-detect

# --- QE ------------------------------------------------------------------
QE_LOW_CONF = float(os.getenv("QE_LOW_CONF", "0.5"))
QE_MID_CONF = float(os.getenv("QE_MID_CONF", "0.75"))
QE_GAP_S = float(os.getenv("QE_GAP_S", "0.4"))             # untranscribed speech gap -> "missing words?"

# --- LLM -----------------------------------------------------------------
LLM_BACKEND = os.getenv("LLM_BACKEND", "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "mock")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-5")
LLM_EFFORT = os.getenv("LLM_EFFORT", "low")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
ASSISTANT_PROMPT = os.getenv(
    "ASSISTANT_PROMPT",
    "You are a helpful voice assistant. The user's message is a speech-to-text transcript. "
    "Reply briefly (1-3 sentences) in the same language the user speaks.",
)

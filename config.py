"""
config.py

Central configuration for the Survey Voice Agent platform.

Priority order for secrets / config values:
  1. Environment variable  (always checked first — fastest, works everywhere)
  2. GCP Secret Manager    (only attempted if env var is absent AND GCP creds exist)
  3. Hard-coded default    (safe fallback so the app at least starts locally)

This means the app runs locally with zero GCP setup — just set the env vars
you care about in a .env file or your shell.
"""
import os
from typing import List


# ---------------------------------------------------------------------------
# Secret helper
# ---------------------------------------------------------------------------

def _get_secret(secret_id: str, default: str = "") -> str:
    """
    Return a config value from (in priority order):
      1. Environment variable named <secret_id>
      2. GCP Secret Manager secret named <secret_id>
      3. Provided default string

    Never raises — always returns a string so startup never crashes.
    """
    # 1. Env var takes priority (covers local dev + CI + Cloud Run env injection)
    val = os.getenv(secret_id, "").strip()
    if val:
        return val

    # 2. Try GCP Secret Manager (only if running on GCP or ADC is configured)
    try:
        from secret import access_secret_version
        val = access_secret_version(secret_id).strip()
        if val:
            return val
    except Exception:
        pass   # No GCP creds locally — that's fine

    return default


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
PROJECT_NAME: str = "ABC Survey Voice Agent"
VERSION: str = "2.0.0"
DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1")

# ---------------------------------------------------------------------------
# Google Cloud / Vertex AI
# ---------------------------------------------------------------------------
VERTEX_PROJECT_ID: str  = _get_secret("VERTEX_PROJECT_ID")
VERTEX_LOCATION: str    = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL_ID: str    = _get_secret("VERTEX_MODEL_ID", "gemini-2.0-flash-live-001")

VERTEX_WS_URL: str = (
    f"wss://{VERTEX_LOCATION}-aiplatform.googleapis.com/ws/"
    "google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"
)

VERTEX_MODEL_RESOURCE: str = (
    f"projects/{VERTEX_PROJECT_ID}/locations/{VERTEX_LOCATION}/"
    f"publishers/google/models/{VERTEX_MODEL_ID}"
    if VERTEX_PROJECT_ID else ""
)

# ---------------------------------------------------------------------------
# Google Sheets
# ---------------------------------------------------------------------------
GOOGLE_SHEET_ID: str = _get_secret("SURVEY_SHEET_ID", os.getenv("GOOGLE_SHEET_ID", ""))

# ---------------------------------------------------------------------------
# FastAPI / CORS
# ---------------------------------------------------------------------------
CORS_ORIGINS: List[str] = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*",
]

SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))

# ---------------------------------------------------------------------------
# Voice agent tuning
# ---------------------------------------------------------------------------
INACTIVITY_TIMEOUT_SECONDS: int    = int(os.getenv("INACTIVITY_TIMEOUT_SECONDS", "90"))
USER_SILENCE_DETECTION_SECONDS: float = float(os.getenv("USER_SILENCE_DETECTION_SECONDS", "1.2"))
GRACEFUL_SHUTDOWN_DELAY_SECONDS: int  = 2
MIN_AUDIO_CHUNK_MS: int               = 200

# ---------------------------------------------------------------------------
# Goodbye trigger words  (all languages supported by the agent)
# ---------------------------------------------------------------------------
GOODBYE_TRIGGERS: List[str] = [
    # English
    "goodbye", "good bye",
    # Hindi
    "अलविदा", "alvida",
    # Marathi
    "निरोप", "nirop",
    # Tamil
    "போயிட்டு வர்றேன்", "poituvaren", "vanakkam",
    # Telugu
    "వెళ్లొస్తాను", "veltanu",
    # Kannada
    "ಹೋಗಿ ಬರ್ತೀನಿ", "hogibistini",
    # Malayalam
    "പോകുന്നു", "pokunnu",
    # Bengali
    "আসছি", "asche", "aschi",
    # Gujarati
    "આવજો", "aavjo",
    # Urdu
    "خدا حافظ", "khuda hafiz",
]

# ---------------------------------------------------------------------------
# Legacy system prompt  (used by old application.py only)
# The new backend/main.py builds prompts dynamically via prompt_builder.py
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = """You are Sneha, a professional phone interviewer calling on behalf of ABC.
Conduct the survey as defined. Speak warmly and professionally."""

# ---------------------------------------------------------------------------
# Legacy extraction prompt  (used by survey_storage.py / application.py only)
# ---------------------------------------------------------------------------
EXTRACT_SURVEY_PROMPT: str = """You are given a transcript of a phone survey call.
Extract the survey answers and return ONLY a valid JSON object."""
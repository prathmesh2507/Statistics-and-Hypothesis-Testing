"""
config/settings.py
──────────────────
Central configuration hub for EVA.
All modules import from here — never read os.environ directly elsewhere.

Loads .env via python-dotenv and exposes a single `Settings` dataclass
so every setting has a type, a default, and one source of truth.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (two levels up from config/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass
class Settings:
    # ── Project paths ──────────────────────────────────────────
    PROJECT_ROOT: Path = field(default_factory=lambda: _PROJECT_ROOT)
    LOG_DIR: Path = field(default_factory=lambda: _PROJECT_ROOT / "logs")
    DATA_DIR: Path = field(default_factory=lambda: _PROJECT_ROOT / "data")
    MODELS_DIR: Path = field(default_factory=lambda: _PROJECT_ROOT / "models")

    # ── Ollama / LLM ───────────────────────────────────────────
    OLLAMA_BASE_URL: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    OLLAMA_MODEL: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    )
    OLLAMA_TIMEOUT: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT", "60"))
    )

    # ── Whisper ────────────────────────────────────────────────
    WHISPER_MODEL_SIZE: str = field(
        default_factory=lambda: os.getenv("WHISPER_MODEL_SIZE", "small")
    )
    WHISPER_DEVICE: str = field(
        default_factory=lambda: os.getenv("WHISPER_DEVICE", "cuda")
    )
    WHISPER_COMPUTE_TYPE: str = field(
        default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    )
    # None → auto-detect language (best for Hinglish)
    WHISPER_LANGUAGE: str | None = field(
        default_factory=lambda: _parse_optional_str(os.getenv("WHISPER_LANGUAGE", "None"))
    )
    WHISPER_BEAM_SIZE: int = field(
        default_factory=lambda: int(os.getenv("WHISPER_BEAM_SIZE", "5"))
    )

    # ── Audio ──────────────────────────────────────────────────
    SAMPLE_RATE: int = field(
        default_factory=lambda: int(os.getenv("SAMPLE_RATE", "16000"))
    )
    CHANNELS: int = field(
        default_factory=lambda: int(os.getenv("CHANNELS", "1"))
    )
    # Chunk size fed to Silero VAD — 512 samples ≈ 32ms @16kHz
    VAD_CHUNK_SIZE: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512"))
    )

    # ── VAD ────────────────────────────────────────────────────
    VAD_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv("VAD_THRESHOLD", "0.45"))
    )
    SILENCE_DURATION: float = field(
        default_factory=lambda: float(os.getenv("SILENCE_DURATION", "1.2"))
    )
    PRE_SPEECH_BUFFER_MS: int = field(
        default_factory=lambda: int(os.getenv("PRE_SPEECH_BUFFER_MS", "300"))
    )
    MAX_SPEECH_DURATION: float = field(
        default_factory=lambda: float(os.getenv("MAX_SPEECH_DURATION", "30.0"))
    )

    # ── MongoDB (Phase 2) ──────────────────────────────────────
    MONGO_URI: str = field(
        default_factory=lambda: os.getenv("MONGO_URI", "mongodb://localhost:27017")
    )
    MONGO_DB_NAME: str = field(
        default_factory=lambda: os.getenv("MONGO_DB_NAME", "eva_memory")
    )

    # ── ChromaDB (Phase 2) ─────────────────────────────────────
    CHROMA_PERSIST_DIR: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    )

    # ── Logging ────────────────────────────────────────────────
    LOG_LEVEL: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    LOG_FILE: str = field(
        default_factory=lambda: os.getenv("LOG_FILE", "./logs/eva.log")
    )
    
    # ── Piper TTS ──────────────────────────────────────────────
    PIPER_PATH: str = field(
        default_factory=lambda: os.getenv(
            "PIPER_PATH",
            "./models/piper/piper.exe"
        )
    )

    VOICE_MODEL: str = field(
        default_factory=lambda: os.getenv(
            "VOICE_MODEL",
            "./models/piper/voices/en_US-lessac-medium.onnx"
        )
    )

    TTS_ENABLED: bool = field(
        default_factory=lambda: os.getenv(
            "TTS_ENABLED",
            "True"
        ).lower() == "true"
    )

    # ── Personality ────────────────────────────────────────────
    EVA_NAME: str = field(
        default_factory=lambda: os.getenv("EVA_NAME", "EVA")
    )
    USER_NAME: str = field(
        default_factory=lambda: os.getenv("USER_NAME", "Friend")
    )

    def __post_init__(self):
        """Ensure required directories exist."""
        for d in (self.LOG_DIR, self.DATA_DIR, self.MODELS_DIR):
            d.mkdir(parents=True, exist_ok=True)


def _parse_optional_str(value: str) -> str | None:
    """Convert the string 'None' to Python None."""
    return None if value.strip().lower() == "none" else value


# ── Singleton ──────────────────────────────────────────────────
# Import `settings` from anywhere: `from config.settings import settings`
settings = Settings()

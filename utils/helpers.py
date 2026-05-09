"""
utils/helpers.py
────────────────
Small, stateless utility functions used across EVA.
No business logic here — just pure helpers.
"""

import re
import time
import unicodedata
from functools import wraps
from typing import Callable, Any

from utils.logger import get_logger

logger = get_logger(__name__)


# ─── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_transcription(text: str) -> str:
    """
    Post-process Whisper output to remove common hallucination artifacts.

    Whisper sometimes hallucinates:
      - Repeated filler phrases ("Thank you.", "Thanks for watching.")
      - Music/sound notations ("[Music]", "♪")
      - Pure punctuation runs
      - Excessive whitespace
    """
    if not text:
        return ""

    # Remove Whisper sound/music tags
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)

    # Remove music symbols
    text = re.sub(r"[♪♫♬♩]", "", text)

    # Strip known hallucination phrases Whisper generates on silence
    _HALLUCINATION_PATTERNS = [
    r"^(thank you[.!,]?)+$",
    r"^(thanks for watching[.!,]?)+$",   # ← add [.!,]? to catch ! and ,
    r"^(thanks for watching\.?)+$",
    r"^(\.+)$",
    r"^(\s*\.\s*)+$",
    r"^subscribe.*$",
    r"^(bye[.!]?)+$",
    r"^(you're welcome[.!,]?)+$",        # ← add this, EVA's TTS triggers it
    r"^(no problem[.!,]?)+$",
]
    stripped = text.strip().lower()
    for pattern in _HALLUCINATION_PATTERNS:
        if re.fullmatch(pattern, stripped, re.IGNORECASE):
            logger.debug(f"Hallucination filtered: '{text.strip()}'")
            return ""

    # Normalize Unicode (handles Devanagari / Hindi characters correctly)
    text = unicodedata.normalize("NFC", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_empty_transcription(text: str) -> bool:
    """Return True if transcription is effectively empty or noise."""
    if not text or not text.strip():
        return True
    # Less than 2 non-space characters → not real speech
    meaningful_chars = [c for c in text if c.strip() and c.isalnum()]
    return len(meaningful_chars) < 2


# ─── Timing ────────────────────────────────────────────────────────────────────

def timed(label: str = ""):
    """Decorator to log execution time of a function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            name = label or func.__name__
            logger.debug(f"⏱  {name} took {elapsed:.1f}ms")
            return result
        return wrapper
    return decorator


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, label: str = "block"):
        self.label = label
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        logger.debug(f"⏱  {self.label}: {self.elapsed_ms:.1f}ms")


# ─── Text Formatting ───────────────────────────────────────────────────────────

def truncate(text: str, max_chars: int = 200) -> str:
    """Truncate text for display/logging purposes."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


def sanitize_for_tts(text: str) -> str:
    """
    Prepare LLM response for TTS.
    Remove markdown, code blocks, URLs, excessive punctuation.
    """
    # Remove markdown code blocks
    text = re.sub(r"```[\s\S]*?```", "...code block...", text)
    text = re.sub(r"`[^`]+`", "", text)

    # Remove markdown bold/italic
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "link", text)

    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove bullet points
    text = re.sub(r"^[\-\*\+]\s+", "", text, flags=re.MULTILINE)

    # Collapse whitespace and newlines
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

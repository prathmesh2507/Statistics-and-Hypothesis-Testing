"""
speech/tts_cache.py
────────────────────
Synthesis cache for repeated phrases.

Why cache matters for conversational TTS:
  - Common filler phrases ("yeah", "hmm", "okay") appear constantly
  - XTTS-v2 takes 1-3 seconds even for short phrases
  - Cached responses play in <10ms — instant
  - Cache also warms up on startup with predictable phrases

Architecture:
  - In-memory dict: text_hash → numpy array
  - Disk backup: saves/loads from data/tts_cache/ as .npy files
  - LRU eviction when memory limit reached
  - Thread-safe for concurrent access
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Common phrases to pre-synthesize at startup ────────────────
# Short, frequent, conversational. Hearing these instantly feels very natural.
WARMUP_PHRASES = [
    ("hmm", "en"),
    ("yeah", "en"),
    ("okay", "en"),
    ("got it", "en"),
    ("right", "en"),
    ("sure", "en"),
    ("let me think", "en"),
    ("interesting", "en"),
    ("haan", "en"),          # Hinglish "yes"
    ("achha", "en"),         # Hinglish "I see / okay"
    ("okay yaar", "en"),
    ("haan bilkul", "en"),   # "yes absolutely"
]


class TTSCache:
    """
    In-memory + disk synthesis cache.

    Usage:
        cache = TTSCache(max_entries=200, persist_dir=Path("./data/tts_cache"))
        audio = cache.get("hello there", "en")
        if audio is None:
            audio = synthesize(...)
            cache.set("hello there", "en", audio)
    """

    def __init__(
        self,
        max_entries: int = 200,
        persist_dir: Optional[Path] = None,
        sample_rate: int = 24000,
    ):
        self.max_entries = max_entries
        self.persist_dir = persist_dir
        self.sample_rate = sample_rate

        # OrderedDict as LRU: most recently used at end
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

        if persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

        logger.debug(f"TTSCache initialized | max={max_entries} | disk={persist_dir}")

    # ── Public API ─────────────────────────────────────────────

    def get(self, text: str, language: str = "en") -> Optional[np.ndarray]:
        """Return cached audio or None if not found."""
        key = self._key(text, language)
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                logger.debug(f"Cache HIT: '{text[:40]}'")
                return self._cache[key].copy()
        return None

    def set(self, text: str, language: str, audio: np.ndarray) -> None:
        """Store synthesized audio. Evicts LRU entry if at capacity."""
        if audio is None or len(audio) == 0:
            return

        key = self._key(text, language)

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_entries:
                    # Evict least recently used (first item)
                    evicted_key, _ = self._cache.popitem(last=False)
                    logger.debug(f"Cache evict: {evicted_key[:20]}")

            self._cache[key] = audio.copy()

        # Async disk save (don't block the synthesis thread)
        if self.persist_dir:
            t = threading.Thread(
                target=self._save_entry,
                args=(key, audio),
                daemon=True,
            )
            t.start()

    def warm_up(self, synthesize_fn) -> None:
        """
        Pre-synthesize common phrases at startup.

        Args:
            synthesize_fn: callable(text, language) → np.ndarray
        """
        logger.info(f"Warming up TTS cache ({len(WARMUP_PHRASES)} phrases)...")
        warmed = 0

        for text, lang in WARMUP_PHRASES:
            if self.get(text, lang) is not None:
                continue  # Already cached (from disk)
            try:
                audio = synthesize_fn(text, lang)
                if audio is not None and len(audio) > 0:
                    self.set(text, lang, audio)
                    warmed += 1
            except Exception as exc:
                logger.warning(f"Warmup failed for '{text}': {exc}")

        logger.info(f"Cache warmed: {warmed} new phrases synthesized")

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate_estimate(self) -> str:
        return f"{self.size}/{self.max_entries} entries cached"

    # ── Disk persistence ───────────────────────────────────────

    def _key(self, text: str, language: str) -> str:
        """Deterministic cache key from text + language."""
        normalized = text.strip().lower()
        combined = f"{language}:{normalized}"
        return hashlib.md5(combined.encode("utf-8")).hexdigest()

    def _save_entry(self, key: str, audio: np.ndarray) -> None:
        if not self.persist_dir:
            return
        try:
            np.save(self.persist_dir / f"{key}.npy", audio)
        except Exception as exc:
            logger.warning(f"Cache disk save failed: {exc}")

    def _load_from_disk(self) -> None:
        if not self.persist_dir or not self.persist_dir.exists():
            return
        loaded = 0
        for npy_file in self.persist_dir.glob("*.npy"):
            if loaded >= self.max_entries:
                break
            try:
                audio = np.load(npy_file)
                key = npy_file.stem
                with self._lock:
                    self._cache[key] = audio
                loaded += 1
            except Exception:
                npy_file.unlink(missing_ok=True)  # Corrupt file — delete

        if loaded > 0:
            logger.info(f"Loaded {loaded} cached phrases from disk")

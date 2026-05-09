"""
speech/tts.py
─────────────
Text-to-Speech engine for EVA.

Phase 1:  pyttsx3 (offline, cross-platform, no extra downloads)
Phase 2:  Piper TTS (replace speak() internals — interface stays the same)

Piper TTS migration path:
  - Install: pip install piper-tts
  - Replace _speak_pyttsx3() with _speak_piper()
  - Point to a downloaded .onnx model file
  - Everything else stays identical

Architecture goal: the rest of EVA only calls tts.speak(text).
No other module knows which engine is running.
"""

from __future__ import annotations

import io
import subprocess
import threading
from pathlib import Path
from typing import Literal

import numpy as np

from utils.logger import get_logger
from utils.helpers import sanitize_for_tts
from config.settings import Settings

logger = get_logger(__name__)

TTSBackend = Literal["pyttsx3", "piper"]


class TTSEngine:
    """
    Speaks text aloud using the configured backend.

    Usage:
        tts = TTSEngine(settings)
        tts.speak("Haan yaar, kya scene hai?")
    """

    def __init__(self, settings: Settings, backend: TTSBackend = "pyttsx3"):
        self.settings = settings
        self.backend = backend
        self._engine = None
        self._lock = threading.Lock()   # pyttsx3 is not thread-safe
        self._setup()

    # ── Setup ──────────────────────────────────────────────────

    def _setup(self):
        if self.backend == "pyttsx3":
            self._setup_pyttsx3()
        elif self.backend == "piper":
            self._setup_piper()
        else:
            raise ValueError(f"Unknown TTS backend: {self.backend}")

    def _setup_pyttsx3(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            # Tune voice properties
            self._engine.setProperty("rate", 165)    # words per minute (default ~200, too fast)
            self._engine.setProperty("volume", 0.9)

            # Try to select a pleasant voice
            voices = self._engine.getProperty("voices")
            if voices:
                # Prefer female voice if available (index 1 on most systems)
                preferred = next(
                    (v for v in voices if "female" in v.name.lower() or "zira" in v.name.lower()),
                    voices[0]
                )
                self._engine.setProperty("voice", preferred.id)
                logger.debug(f"TTS voice: {preferred.name}")

            logger.info("[green]pyttsx3 TTS ready ✓[/green]")

        except ImportError:
            logger.error("pyttsx3 not installed. Run: pip install pyttsx3")
            raise
        except Exception as exc:
            logger.error(f"pyttsx3 init failed: {exc}")
            raise

    def _setup_piper(self):
        """
        Phase 2 — Piper TTS setup.
        Expects PIPER_MODEL_PATH in settings pointing to a .onnx file.

        Download models from: https://rhasspy.github.io/piper-samples/
        Recommended for Hinglish: use English model (en_US-lessac-medium)
        """
        # TODO Phase 2: implement Piper
        # from piper import PiperVoice
        # model_path = getattr(self.settings, "PIPER_MODEL_PATH", None)
        # if not model_path:
        #     raise ValueError("PIPER_MODEL_PATH not set in .env")
        # self._piper_voice = PiperVoice.load(model_path)
        logger.warning("Piper TTS not yet implemented — falling back to pyttsx3")
        self.backend = "pyttsx3"
        self._setup_pyttsx3()

    # ── Public API ─────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """
        Speak `text` aloud.

        Automatically sanitizes text (strips markdown, code blocks, URLs)
        before passing to the TTS engine.

        Blocks until speech completes.
        """
        if not text or not text.strip():
            return

        clean_text = sanitize_for_tts(text)
        if not clean_text:
            return

        logger.debug(f"TTS speaking: '{clean_text[:80]}...'")

        if self.backend == "pyttsx3":
            self._speak_pyttsx3(clean_text)
        elif self.backend == "piper":
            self._speak_piper(clean_text)

    def speak_async(self, text: str) -> threading.Thread:
        """
        Non-blocking speak. Returns the thread so caller can join() if needed.
        Useful for future UI overlays.
        """
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
        return t

    # ── Backend Implementations ────────────────────────────────

    def _speak_pyttsx3(self, text: str) -> None:
        with self._lock:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:
                logger.error(f"pyttsx3 speak failed: {exc}")

    def _speak_piper(self, text: str) -> None:
        """
        Phase 2 placeholder.
        Piper synthesis to wav → play via sounddevice.
        """
        # TODO Phase 2:
        # import sounddevice as sd
        # audio_bytes = io.BytesIO()
        # self._piper_voice.synthesize(text, audio_bytes)
        # audio_np = np.frombuffer(audio_bytes.getvalue(), dtype=np.int16)
        # sd.play(audio_np / 32768.0, samplerate=22050)
        # sd.wait()
        pass

    # ── Utility ────────────────────────────────────────────────

    def stop(self) -> None:
        """Stop any currently playing speech."""
        if self.backend == "pyttsx3" and self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass

    def __repr__(self) -> str:
        return f"TTSEngine(backend={self.backend})"

"""
speech/tts.py
─────────────
EVA's Text-to-Speech engine — XTTS-v2 powered.
Drop-in replacement for old pyttsx3-based tts.py.

External interface is IDENTICAL to before:
    tts = TTSEngine(settings)
    tts.speak("Hey yaar, kya chal raha hai?")   # blocking
    tts.speak_async("...")                        # non-blocking
    tts.stop()                                    # interrupt

Pipeline for each tts.speak(text) call:
  1. ExpressiveSpeechProcessor splits text into SpeechSegments (per language)
  2. For each segment:
     a. Check TTSCache - play instantly if hit
     b. VoiceManager.synthesize_stream() yields audio chunks
     c. AudioPlayer.enqueue() plays chunk while next is synthesizing
     d. Cache the full audio for future reuse
  3. AudioPlayer.wait_done() blocks until all audio is done
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from speech.voice_manager import VoiceManager
from speech.audio_player import AudioPlayer, StreamingPlaybackSession
from speech.tts_cache import TTSCache
from speech.expressive_speech import ExpressiveSpeechProcessor, SpeechSegment
from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_REFERENCE = Path("data/voices/eva_reference.wav")


class TTSEngine:
    """
    Production TTS engine for EVA using XTTS-v2.
    Same interface as old pyttsx3 engine — just better in every way.
    """

    def __init__(
        self,
        settings: Settings,
        reference_wav: Optional[str] = None,
        backend: str = "xtts",
    ):
        self.settings = settings

        self._reference_wav = str(
            Path(reference_wav) if reference_wav else
            settings.PROJECT_ROOT / _DEFAULT_REFERENCE
        )

        self._voice_manager = VoiceManager()
        self._player = AudioPlayer(
            sample_rate=VoiceManager.SAMPLE_RATE,
            volume=1.0,
        )
        self._cache = TTSCache(
            max_entries=300,
            persist_dir=settings.DATA_DIR / "tts_cache",
            sample_rate=VoiceManager.SAMPLE_RATE,
        )
        self._processor = ExpressiveSpeechProcessor(
            max_chunk_chars=180,
            normalize_hinglish=True,
        )

        self._is_speaking = threading.Event()
        self._initialized = False
        self._load()

    # ── Init ───────────────────────────────────────────────────

    def _load(self) -> None:
        ref_path = Path(self._reference_wav)

        if not ref_path.exists():
            logger.warning(
                f"[yellow]Reference WAV not found: {ref_path}[/yellow]\n"
                f"Run:  python speech/setup_voice.py\n"
                f"This records your 10-second voice sample for XTTS-v2."
            )
            self._try_create_fallback_reference()

        try:
            self._voice_manager.load(str(ref_path))
            self._player.start()

            t = threading.Thread(target=self._warm_cache, daemon=True, name="CacheWarmup")
            t.start()

            self._initialized = True
            logger.info("[green]TTSEngine (XTTS-v2) ready[/green]")

        except Exception as exc:
            logger.error(f"XTTS-v2 load failed: {exc}")
            logger.error(
                "Fixes:\n"
                "  pip install TTS\n"
                "  python speech/setup_voice.py\n"
                "  Set WHISPER_DEVICE=cpu in .env to free VRAM"
            )
            raise

    def _warm_cache(self) -> None:
        time.sleep(3.0)
        self._cache.warm_up(
            synthesize_fn=lambda text, lang: self._voice_manager.synthesize(
                text, lang, temperature=0.65
            )
        )

    def _try_create_fallback_reference(self) -> None:
        try:
            from speech.setup_voice import record_reference
            ref_path = Path(self._reference_wav)
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Auto-recording 10s voice sample from microphone...")
            record_reference(output_path=str(ref_path), duration=10)
        except Exception as exc:
            logger.warning(f"Auto-record failed ({exc}). Run setup_voice.py manually.")

    # ── Public API ─────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Synthesize and speak text. Blocks until complete."""
        if not text or not text.strip():
            return
        if not self._initialized:
            logger.warning("TTSEngine not initialized.")
            return

        self._is_speaking.set()
        try:
            self._speak_internal(text)
        except Exception as exc:
            logger.error(f"TTS error: {exc}", exc_info=True)
        finally:
            self._is_speaking.clear()

    def speak_async(self, text: str) -> threading.Thread:
        """Non-blocking speak. Returns thread."""
        t = threading.Thread(target=self.speak, args=(text,), daemon=True, name="EVA-TTS")
        t.start()
        return t

    def stop(self) -> None:
        """Immediately interrupt playback."""
        self._player.stop()
        self._is_speaking.clear()

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking.is_set()

    # ── Pipeline ───────────────────────────────────────────────

    def _speak_internal(self, text: str) -> None:
        """Full preprocessing -> synthesis -> streaming playback pipeline."""
        processed = self._processor.process(text)
        if processed.is_empty:
            return

        logger.debug(f"Speaking {len(processed.segments)} segment(s)")

        with StreamingPlaybackSession(self._player) as session:
            for segment in processed.segments:
                if not segment.text.strip():
                    continue

                # Cache hit — instant playback
                cached = self._cache.get(segment.text, segment.language)
                if cached is not None:
                    session.add(cached, pause_ms=segment.pause_after_ms)
                    continue

                # Stream synthesis + real-time playback
                full_chunks: list[np.ndarray] = []
                try:
                    for audio_chunk in self._voice_manager.synthesize_stream(
                        text=segment.text,
                        language=segment.language,
                        temperature=self._temperature_for(segment),
                        speed=0.95,
                        stream_chunk_size=20,
                    ):
                        session.add(audio_chunk)
                        full_chunks.append(audio_chunk)

                except Exception as exc:
                    logger.error(f"Synthesis failed '{segment.text[:40]}': {exc}")
                    continue

                if segment.pause_after_ms > 0:
                    session.add_pause(segment.pause_after_ms)

                if full_chunks:
                    self._cache.set(
                        segment.text,
                        segment.language,
                        np.concatenate(full_chunks),
                    )

    def _temperature_for(self, segment: SpeechSegment) -> float:
        text = segment.text
        words = len(text.split())
        if text.endswith("?"):
            return 0.75
        if words <= 4:
            return 0.72
        if words >= 20:
            return 0.65
        return 0.70

    # ── Utility ────────────────────────────────────────────────

    def set_voice(self, reference_wav: str) -> None:
        """Hot-swap reference voice without reloading model."""
        self._voice_manager.set_reference_voice(reference_wav)
        self._cache.clear()

    def presynthesize(self, text: str, language: str = "en") -> None:
        if self._cache.get(text, language) is not None:
            return
        try:
            audio = self._voice_manager.synthesize(text, language)
            self._cache.set(text, language, audio)
        except Exception as exc:
            logger.warning(f"Presynthesize failed: {exc}")

    def __repr__(self) -> str:
        return (
            f"TTSEngine(XTTS-v2, "
            f"device={self._voice_manager.device}, "
            f"cache={self._cache.hit_rate_estimate})"
        )

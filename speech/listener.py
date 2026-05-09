"""
speech/listener.py
──────────────────
Microphone capture with Silero VAD-driven utterance segmentation.

This is the heart of the speech pipeline. It solves the core problems
with naive "record for N seconds" approaches:

  Problem 1 — Premature cutoff: fixed recording cuts mid-sentence
  Solution: Keep recording as long as VAD detects speech

  Problem 2 — Hallucination on silence: Whisper invents text from silence
  Solution: Only send audio that contains confirmed speech (VAD-gated)

  Problem 3 — Missing speech onset: VAD triggers slightly late
  Solution: Pre-speech ring buffer captures 300ms before VAD fires

  Problem 4 — Audio glitches from buffer overruns
  Solution: sounddevice callback → thread-safe queue → consumer loop

State machine:
    IDLE  →  (VAD detects speech)  →  SPEAKING
    SPEAKING  →  (N seconds silence)  →  DONE
    SPEAKING  →  (max duration hit)   →  DONE
    IDLE  →  (timeout)  →  returns None
"""

import queue
import threading
from enum import Enum, auto
from typing import Optional

import numpy as np
import sounddevice as sd

from speech.vad import SileroVAD
from speech.audio_utils import preprocess_audio, is_audio_quality_sufficient, get_audio_stats
from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class _ListenerState(Enum):
    IDLE = auto()
    SPEAKING = auto()
    DONE = auto()


class VoiceListener:
    """
    Captures microphone audio and returns complete utterances as numpy arrays.

    Usage:
        listener = VoiceListener(settings)
        audio = listener.listen()   # blocks until utterance complete
        # audio is float32 @ 16kHz, ready for Whisper
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.sr = settings.SAMPLE_RATE           # 16000
        self.chunk = settings.VAD_CHUNK_SIZE     # 512

        # How many silent chunks = end of utterance
        self._silence_chunks = int(
            settings.SILENCE_DURATION * self.sr / self.chunk
        )
        # Max utterance length in chunks
        self._max_chunks = int(
            settings.MAX_SPEECH_DURATION * self.sr / self.chunk
        )
        # Pre-speech ring buffer size in chunks
        self._pre_buffer_size = max(
            1,
            int(settings.PRE_SPEECH_BUFFER_MS / 1000 * self.sr / self.chunk)
        )

        # Silero VAD
        self.vad = SileroVAD(
            threshold=settings.VAD_THRESHOLD,
            sampling_rate=self.sr,
        )

        # Thread-safe audio queue (callback → consumer)
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)

        logger.info(
            f"VoiceListener ready | SR={self.sr} chunk={self.chunk} "
            f"silence={settings.SILENCE_DURATION}s vad_thresh={settings.VAD_THRESHOLD}"
        )

    # ── Public API ─────────────────────────────────────────────

    def listen(self, idle_timeout: float = 10.0) -> Optional[np.ndarray]:
        """
        Block until one complete utterance is captured.

        Args:
            idle_timeout: Seconds to wait for speech before giving up.

        Returns:
            float32 numpy array @ 16kHz, or None if no speech detected.
        """
        self.vad.reset()
        self._drain_queue()

        state = _ListenerState.IDLE
        speech_chunks: list[np.ndarray] = []
        pre_buffer: list[np.ndarray] = []   # ring buffer: pre-speech audio
        silent_count = 0

        with sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            blocksize=self.chunk,
            callback=self._sd_callback,
        ):
            logger.info("[bold cyan]🎤 Listening...[/bold cyan]")

            while True:
                # ── Fetch one chunk ──────────────────────────
                try:
                    wait = idle_timeout if state == _ListenerState.IDLE else 5.0
                    chunk = self._audio_q.get(timeout=wait)
                except queue.Empty:
                    if state == _ListenerState.IDLE:
                        logger.debug("No speech detected within idle timeout.")
                        return None
                    else:
                        # Mid-speech timeout — use what we have
                        logger.warning("Mid-speech timeout — using partial audio.")
                        break

                # ── VAD inference ────────────────────────────
                speech_prob = self.vad.is_speech(chunk)
                is_voice = speech_prob >= self.settings.VAD_THRESHOLD

                # ── State transitions ────────────────────────
                if state == _ListenerState.IDLE:
                    # Maintain pre-speech ring buffer
                    pre_buffer.append(chunk)
                    if len(pre_buffer) > self._pre_buffer_size:
                        pre_buffer.pop(0)

                    if is_voice:
                        # Speech onset — prepend buffer for clean onset
                        speech_chunks = list(pre_buffer)
                        speech_chunks.append(chunk)
                        pre_buffer.clear()
                        silent_count = 0
                        state = _ListenerState.SPEAKING
                        logger.debug(f"Speech onset (p={speech_prob:.2f})")

                elif state == _ListenerState.SPEAKING:
                    speech_chunks.append(chunk)

                    if is_voice:
                        silent_count = 0
                    else:
                        silent_count += 1

                    # ── End conditions ────────────────────────
                    if silent_count >= self._silence_chunks:
                        logger.debug(
                            f"End of speech (silence × {silent_count} chunks)"
                        )
                        break

                    if len(speech_chunks) >= self._max_chunks:
                        logger.warning(
                            f"Max speech duration ({self.settings.MAX_SPEECH_DURATION}s) reached."
                        )
                        break

        if not speech_chunks:
            return None

        # ── Assemble and preprocess ──────────────────────────
        raw_audio = np.concatenate(speech_chunks, axis=0)
        processed = preprocess_audio(raw_audio, self.sr)

        stats = get_audio_stats(processed, self.sr)
        logger.info(
            f"Captured utterance | duration={stats['duration_s']}s "
            f"rms={stats['rms']} peak={stats['peak']}"
        )

        if not is_audio_quality_sufficient(processed):
            logger.warning("Captured audio failed quality check — discarding.")
            return None

        return processed

    # ── Internal ───────────────────────────────────────────────

    def _sd_callback(self, indata: np.ndarray, frames: int, time, status):
        """
        sounddevice callback — runs in audio thread.
        NEVER block here. Just put data in queue.
        """
        if status:
            logger.warning(f"sounddevice status: {status}")

        # Drop frames if consumer is lagging (prevents unbounded memory growth)
        try:
            self._audio_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # Silently drop — the VAD state machine handles gaps

    def _drain_queue(self):
        """Clear any leftover audio from previous calls."""
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break

    def __repr__(self) -> str:
        return (
            f"VoiceListener(sr={self.sr}, chunk={self.chunk}, "
            f"vad_thresh={self.settings.VAD_THRESHOLD})"
        )

import queue
from enum import Enum, auto
from typing import Optional

import numpy as np
import sounddevice as sd

from speech.vad import SileroVAD
from speech.audio_utils import (
    preprocess_audio,
    is_audio_quality_sufficient,
    get_audio_stats,
)
from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class _ListenerState(Enum):
    IDLE = auto()
    SPEAKING = auto()
    DONE = auto()


class VoiceListener:

    def __init__(self, settings: Settings):

        self.settings = settings
        self.sr = settings.SAMPLE_RATE
        self.chunk = settings.VAD_CHUNK_SIZE

        # Assistant speaking lock
        self.is_assistant_speaking = False

        self._silence_chunks = int(
            settings.SILENCE_DURATION * self.sr / self.chunk
        )

        self._max_chunks = int(
            settings.MAX_SPEECH_DURATION * self.sr / self.chunk
        )

        self._pre_buffer_size = max(
            1,
            int(settings.PRE_SPEECH_BUFFER_MS / 1000 * self.sr / self.chunk)
        )

        self.vad = SileroVAD(
            threshold=settings.VAD_THRESHOLD,
            sampling_rate=self.sr,
        )

        self._audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)

        logger.info(
            f"VoiceListener ready | "
            f"SR={self.sr} "
            f"chunk={self.chunk} "
            f"silence={settings.SILENCE_DURATION}s "
            f"vad_thresh={settings.VAD_THRESHOLD}"
        )

    # ──────────────────────────────────────────────────────────

    def listen(self, idle_timeout: float = 10.0) -> Optional[np.ndarray]:

        # Prevent EVA from hearing itself
        if self.is_assistant_speaking:
            return None

        self.vad.reset()
        self._drain_queue()

        state = _ListenerState.IDLE

        speech_chunks: list[np.ndarray] = []
        pre_buffer: list[np.ndarray] = []

        silent_count = 0

        with sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            blocksize=self.chunk,
            callback=self._sd_callback,
        ):

            logger.info("🎤 Listening...")

            while True:

                # Prevent self-listening during runtime
                if self.is_assistant_speaking:
                    self._drain_queue()
                    continue

                try:

                    wait = (
                        idle_timeout
                        if state == _ListenerState.IDLE
                        else 5.0
                    )

                    chunk = self._audio_q.get(timeout=wait)

                except queue.Empty:

                    if state == _ListenerState.IDLE:
                        logger.debug("No speech detected.")
                        return None

                    logger.warning("Mid-speech timeout.")
                    break

                # ── VAD ─────────────────────────────

                speech_prob = self.vad.is_speech(chunk)

                is_voice = (
                    speech_prob >= self.settings.VAD_THRESHOLD
                )

                # ── IDLE → SPEAKING ─────────────────

                if state == _ListenerState.IDLE:

                    pre_buffer.append(chunk)

                    if len(pre_buffer) > self._pre_buffer_size:
                        pre_buffer.pop(0)

                    if is_voice:

                        speech_chunks = list(pre_buffer)

                        speech_chunks.append(chunk)

                        pre_buffer.clear()

                        silent_count = 0

                        state = _ListenerState.SPEAKING

                        logger.debug(
                            f"Speech onset "
                            f"(p={speech_prob:.2f})"
                        )

                # ── SPEAKING ────────────────────────

                elif state == _ListenerState.SPEAKING:

                    speech_chunks.append(chunk)

                    if is_voice:
                        silent_count = 0
                    else:
                        silent_count += 1

                    # End of speech

                    if silent_count >= self._silence_chunks:

                        logger.debug(
                            f"End of speech "
                            f"(silence chunks={silent_count})"
                        )

                        break

                    # Max duration safety

                    if len(speech_chunks) >= self._max_chunks:

                        logger.warning(
                            f"Max speech duration "
                            f"({self.settings.MAX_SPEECH_DURATION}s)"
                        )

                        break

        if not speech_chunks:
            return None

        # ── Audio Assembly ────────────────────────

        raw_audio = np.concatenate(speech_chunks, axis=0)

        processed = preprocess_audio(raw_audio, self.sr)

        stats = get_audio_stats(processed, self.sr)

        logger.info(
            f"Captured utterance | "
            f"duration={stats['duration_s']}s "
            f"rms={stats['rms']} "
            f"peak={stats['peak']}"
        )

        if not is_audio_quality_sufficient(processed):

            logger.warning(
                "Captured audio failed quality check."
            )

            return None

        return processed

    # ──────────────────────────────────────────────────────────

    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time,
        status,
    ):

        if status:
            logger.warning(f"sounddevice status: {status}")

        try:
            self._audio_q.put_nowait(
                indata[:, 0].copy()
            )

        except queue.Full:
            pass

    # ──────────────────────────────────────────────────────────

    def _drain_queue(self):

        while not self._audio_q.empty():

            try:
                self._audio_q.get_nowait()

            except queue.Empty:
                break

    # ──────────────────────────────────────────────────────────

    def __repr__(self):

        return (
            f"VoiceListener("
            f"sr={self.sr}, "
            f"chunk={self.chunk}, "
            f"vad_thresh={self.settings.VAD_THRESHOLD}"
            f")"
        )


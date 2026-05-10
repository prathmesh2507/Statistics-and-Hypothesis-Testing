"""
speech/audio_player.py
───────────────────────
Non-blocking, interruptible audio playback engine.

Why a dedicated player module?
  - sounddevice.play() blocks OR uses callbacks that are tricky
  - We need to START playing chunk-1 while chunk-2 is still synthesizing
  - We need INSTANT interruption (user starts speaking mid-sentence)
  - We need precise gap control between sentence chunks

Architecture:
  - Producer (TTS engine) puts audio chunks into a queue
  - Consumer thread reads queue → plays via sounddevice stream
  - Stop event instantly kills the consumer thread and stops playback
  - "done" event signals when the queue is fully drained

Usage:
    player = AudioPlayer(sample_rate=24000)
    player.start()                        # spawn worker thread

    player.enqueue(chunk1)               # add audio immediately
    player.enqueue(chunk2)               # add more (plays after chunk1)
    player.wait_done()                   # block until queue is empty
    # OR
    player.stop()                        # interrupt mid-sentence
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from utils.logger import get_logger

logger = get_logger(__name__)

# Sentinel object — placed in queue to signal "no more chunks"
_QUEUE_END = object()

# Gap inserted between sentence segments (in samples)
_INTER_SEGMENT_SILENCE_MS = 60


class AudioPlayer:
    """
    Queue-based async audio player with instant interruption support.
    """

    def __init__(self, sample_rate: int = 24000, volume: float = 1.0):
        self.sample_rate = sample_rate
        self.volume = max(0.0, min(2.0, volume))  # clamp 0-2x

        self._queue: queue.Queue[object] = queue.Queue(maxsize=50)
        self._stop_event = threading.Event()
        self._done_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._is_playing = threading.Event()

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        """Start the background playback worker."""
        if self._worker and self._worker.is_alive():
            return

        self._stop_event.clear()
        self._done_event.clear()
        self._worker = threading.Thread(
            target=self._playback_worker,
            daemon=True,
            name="EVA-AudioPlayer",
        )
        self._worker.start()
        logger.debug("AudioPlayer worker started")

    def stop(self) -> None:
        """
        Immediately stop playback and drain the queue.
        Safe to call from any thread.
        """
        self._stop_event.set()
        sd.stop()  # Kills any currently playing audio instantly

        # Drain remaining items so the worker thread can exit
        self._drain_queue()

        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)

        self._stop_event.clear()
        self._done_event.set()
        self._is_playing.clear()

    def restart(self) -> None:
        """Stop, reset state, and restart the worker. Call before each utterance."""
        self.stop()
        self._done_event.clear()
        self._is_playing.clear()
        self.start()

    # ── Enqueue ────────────────────────────────────────────────

    def enqueue(self, audio: np.ndarray, pause_after_ms: int = 0) -> None:
        """
        Add an audio chunk to the playback queue.

        Args:
            audio:         float32 numpy array at self.sample_rate
            pause_after_ms: silence gap to insert AFTER this chunk finishes
        """
        if audio is None or len(audio) == 0:
            return

        # Apply volume
        chunk = (audio * self.volume).astype(np.float32)
        chunk = np.clip(chunk, -1.0, 1.0)

        # If pause requested, append silence directly to chunk
        if pause_after_ms > 0:
            silence_samples = int(pause_after_ms / 1000 * self.sample_rate)
            silence = np.zeros(silence_samples, dtype=np.float32)
            chunk = np.concatenate([chunk, silence])

        try:
            self._queue.put_nowait(chunk)
            self._done_event.clear()
        except queue.Full:
            logger.warning("Audio queue full — dropping chunk")

    def enqueue_silence(self, duration_ms: int) -> None:
        """Enqueue a silent gap (for natural pauses mid-utterance)."""
        samples = int(duration_ms / 1000 * self.sample_rate)
        silence = np.zeros(samples, dtype=np.float32)
        self.enqueue(silence)

    def signal_end(self) -> None:
        """
        Tell the worker no more chunks are coming.
        The worker will set done_event when the queue drains.
        """
        try:
            self._queue.put_nowait(_QUEUE_END)
        except queue.Full:
            pass

    # ── Waiting ────────────────────────────────────────────────

    def wait_done(self, timeout: float = 60.0) -> bool:
        """
        Block until all queued audio has finished playing.
        Returns True if done, False if timeout expired.
        """
        return self._done_event.wait(timeout=timeout)

    @property
    def is_playing(self) -> bool:
        return self._is_playing.is_set()

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    # ── Worker ─────────────────────────────────────────────────

    def _playback_worker(self) -> None:
        """
        Background thread: reads chunks from queue and plays them.
        Exits on stop event or queue sentinel.
        """
        logger.debug("Playback worker running")

        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Check for end sentinel
            if item is _QUEUE_END:
                self._done_event.set()
                self._is_playing.clear()
                break

            # It's an audio chunk
            chunk: np.ndarray = item
            self._is_playing.set()

            try:
                # Play blocking — sounddevice handles the audio thread
                sd.play(chunk, samplerate=self.sample_rate, blocking=True)

                # Check stop between chunks
                if self._stop_event.is_set():
                    sd.stop()
                    break

            except Exception as exc:
                logger.error(f"Playback error: {exc}")
                continue

        self._is_playing.clear()
        logger.debug("Playback worker exited")

    def _drain_queue(self) -> None:
        """Empty the queue without playing anything."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


class StreamingPlaybackSession:
    """
    Context manager for a single speaking utterance.
    Handles the producer side of the player.

    Usage:
        with StreamingPlaybackSession(player) as session:
            for chunk in tts.synthesize_stream(text):
                session.add(chunk)
        # session.__exit__ signals end and waits for drain
    """

    def __init__(self, player: AudioPlayer, pause_after_ms: int = 0):
        self.player = player
        self.pause_after_ms = pause_after_ms
        self._chunk_count = 0

    def __enter__(self) -> "StreamingPlaybackSession":
        self.player.restart()
        return self

    def add(self, audio: np.ndarray, pause_ms: int = 0) -> None:
        """Add an audio chunk from the synthesis stream."""
        self.player.enqueue(audio, pause_after_ms=pause_ms)
        self._chunk_count += 1

    def add_pause(self, ms: int) -> None:
        self.player.enqueue_silence(ms)

    def __exit__(self, *_) -> None:
        self.player.signal_end()
        self.player.wait_done(timeout=30.0)

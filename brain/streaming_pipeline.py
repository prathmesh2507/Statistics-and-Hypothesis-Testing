"""
brain/streaming_pipeline.py
────────────────────────────
Real-time sentence streaming pipeline.

Old flow (slow):
  LLM generates ALL → XTTS synthesizes ALL → play ALL
  Latency: LLM_time + XTTS_time = 5s + 4s = 9s before first word

New flow (fast):
  LLM streams tokens → detect sentence boundary →
  immediately synthesize that sentence → play while LLM continues

  Sentence 1: LLM(0.8s) + XTTS(2s) = first audio at ~2.8s
  Sentence 2: already synthesizing while sentence 1 plays
  Sentence 3: already in queue

Perceived latency drops from ~9s to ~2.5-3s.

Also handles:
  - Immediate filler audio ("hmm..") while pipeline starts — makes it feel instant
  - Short response fast path (< 6 words skip streaming overhead)
  - Sentence boundary detection for natural speech chunking
"""

from __future__ import annotations

import re
import threading
import queue
from typing import Generator, Optional

from utils.logger import get_logger
from utils.helpers import Timer

logger = get_logger(__name__)

# ── Sentence boundary detection ────────────────────────────────
# Fires when LLM token stream hits one of these patterns
_SENTENCE_END = re.compile(
    r"[.!?।]\s"          # sentence-ending punctuation + space
    r"|[.!?।]$"          # at end of buffer
    r"|,\s{1}(?=\w)"     # comma followed by word (clause break)
    r"|—\s"              # em dash pause
    r"|\n"               # newline always = new chunk
)

# Minimum chars before we consider splitting (avoid 1-word chunks)
_MIN_CHUNK_CHARS = 25

# Maximum chars before we force-split even without punctuation
_MAX_CHUNK_CHARS = 160

# Filler sounds to play IMMEDIATELY while pipeline warms up
# Must already be in TTS cache from warmup — plays in <10ms
FILLER_SOUNDS = [
    "hmm",
    "let me think",
    "okay",
    "haan",
]


def split_into_tts_chunks(text: str) -> list[str]:
    """
    Split a complete text into TTS-friendly sentence chunks.
    Used for non-streaming (short response) path.
    """
    if len(text) <= _MAX_CHUNK_CHARS:
        return [text]

    chunks = []
    current = ""

    for token in text.split(" "):
        current = (current + " " + token).strip() if current else token

        # Check for sentence end
        if len(current) >= _MIN_CHUNK_CHARS and _SENTENCE_END.search(current):
            chunks.append(current)
            current = ""
        elif len(current) >= _MAX_CHUNK_CHARS:
            # Force split at word boundary
            chunks.append(current)
            current = ""

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c.strip()]


class SentenceBuffer:
    """
    Accumulates LLM token stream and emits complete sentences.

    Usage:
        buf = SentenceBuffer()
        for token in llm.stream(...):
            sentence = buf.push(token)
            if sentence:
                tts.synthesize_async(sentence)
        remainder = buf.flush()
        if remainder:
            tts.synthesize_async(remainder)
    """

    def __init__(
        self,
        min_chars: int = _MIN_CHUNK_CHARS,
        max_chars: int = _MAX_CHUNK_CHARS,
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buffer = ""

    def push(self, token: str) -> Optional[str]:
        """
        Add a token. Returns a complete sentence chunk if boundary detected,
        otherwise returns None.
        """
        self._buffer += token

        # Don't split yet if too short
        if len(self._buffer) < self.min_chars:
            return None

        # Check for natural sentence boundary
        if _SENTENCE_END.search(self._buffer):
            chunk = self._buffer.strip()
            self._buffer = ""
            return chunk if chunk else None

        # Force split if too long
        if len(self._buffer) >= self.max_chars:
            # Find last space to avoid mid-word split
            last_space = self._buffer.rfind(" ")
            if last_space > self.min_chars:
                chunk = self._buffer[:last_space].strip()
                self._buffer = self._buffer[last_space:].strip()
                return chunk if chunk else None

        return None

    def flush(self) -> Optional[str]:
        """Return any remaining buffered text."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None

    def reset(self):
        self._buffer = ""


class StreamingTTSPipeline:
    """
    Connects LLM streaming output → XTTS synthesis → audio playback
    with sentence-level parallelism.

    Architecture:
      Thread 1 (main):   LLM token stream → SentenceBuffer → sentence_queue
      Thread 2 (synth):  sentence_queue → XTTS synthesis → audio_queue
      Thread 3 (player): audio_queue → sounddevice playback

    Threads 1, 2, 3 run concurrently — true pipeline parallelism.
    """

    def __init__(self, tts_engine, response_generator, context_manager):
        self.tts = tts_engine
        self.responder = response_generator
        self.context = context_manager

        self._sentence_queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=10)
        self._synthesis_thread: Optional[threading.Thread] = None

    def respond_and_speak(self, user_text: str) -> str:
        """
        Full pipeline:
          1. Play filler sound immediately (perceived instant response)
          2. Start LLM streaming
          3. Feed complete sentences to TTS as they arrive
          4. Return full assembled response text

        Returns:
            The complete response text (for context manager + logging)
        """
        # ── Step 1: Immediate filler (< 10ms from cache) ──────
        self._play_filler()

        # ── Step 2: Add user turn to context ──────────────────
        self.context.add_user(user_text)
        messages = self.context.get_messages()

        # ── Step 3: Start synthesis thread ────────────────────
        self._synthesis_thread = threading.Thread(
            target=self._synthesis_worker,
            daemon=True,
            name="EVA-SynthWorker",
        )
        self._synthesis_thread.start()

        # ── Step 4: Stream LLM → sentence buffer → queue ──────
        buf = SentenceBuffer()
        full_response = ""

        with Timer("LLM first sentence"):
            for token in self.responder.llm.stream(
                prompt=user_text,
                system=self.responder.personality.system_prompt,
                temperature=self.responder.personality.default_temperature,
                max_tokens=self.responder.personality.max_response_tokens,
            ):
                full_response += token
                sentence = buf.push(token)
                if sentence:
                    logger.debug(f"Sentence ready: '{sentence[:50]}'")
                    self._sentence_queue.put(sentence)

        # Flush remaining buffer
        remainder = buf.flush()
        if remainder:
            self._sentence_queue.put(remainder)

        # Signal synthesis thread: no more sentences
        self._sentence_queue.put(None)

        # ── Step 5: Wait for synthesis + playback ─────────────
        if self._synthesis_thread:
            self._synthesis_thread.join(timeout=60.0)

        # ── Step 6: Update context ─────────────────────────────
        if full_response:
            self.context.add_assistant(full_response)
        else:
            import random
            from config.prompts import LLM_FALLBACK_RESPONSES
            full_response = random.choice(LLM_FALLBACK_RESPONSES)
            self.context.add_assistant(full_response)

        return full_response

    def _synthesis_worker(self) -> None:
        """
        Background thread: pulls sentences from queue → synthesizes → plays.
        Runs concurrently with LLM generation.
        """
        while True:
            try:
                sentence = self._sentence_queue.get(timeout=15.0)
            except queue.Empty:
                logger.warning("Synthesis worker timed out waiting for sentence.")
                break

            if sentence is None:
                break

            if not sentence.strip():
                continue

            try:
                self.tts.speak(sentence)
            except Exception as exc:
                logger.error(f"Synthesis worker error: {exc}")
                continue

    def _play_filler(self) -> None:
        """
        Play a filler sound from cache immediately.
        This gives the user instant audio feedback while the pipeline starts.
        Falls back silently if cache miss.
        """
        import random
        for filler in random.sample(FILLER_SOUNDS, len(FILLER_SOUNDS)):
            cached = self.tts._cache.get(filler, "en")
            if cached is not None:
                try:
                    from speech.audio_player import AudioPlayer
                    # Quick fire-and-forget play
                    import sounddevice as sd
                    sd.play(cached, samplerate=self.tts._player.sample_rate)
                    # Don't wait — LLM starts streaming immediately
                    logger.debug(f"Filler played: '{filler}'")
                    return
                except Exception:
                    pass
        # Cache miss (warmup not complete yet) — skip filler silently

    def interrupt(self) -> None:
        """Stop current speech immediately (user interrupted)."""
        self.tts.stop()
        # Drain sentence queue
        while not self._sentence_queue.empty():
            try:
                self._sentence_queue.get_nowait()
            except queue.Empty:
                break

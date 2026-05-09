"""
speech/transcriber.py
─────────────────────
Faster-Whisper based speech-to-text engine.

Why Faster-Whisper over openai-whisper:
  - 4x faster inference on same hardware
  - Lower VRAM usage via CTranslate2 quantization
  - float16 on GPU, int8 on CPU — both accurate
  - Supports beam search + word-level timestamps

Anti-hallucination measures:
  1. no_speech_prob threshold — reject audio where model is uncertain
  2. compression_ratio check — hallucinated text has abnormally low compression
  3. log_prob threshold — reject low-confidence transcriptions
  4. Minimum audio length enforcement (done in listener + audio_utils)
  5. Post-processing via clean_transcription() in helpers.py

Hinglish support:
  - Setting language=None lets Whisper auto-detect per utterance
  - The 'small' model handles code-switching (EN+HI) reasonably well
  - 'medium' model would be better for Hinglish if VRAM allows
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.logger import get_logger
from utils.helpers import clean_transcription, is_empty_transcription, Timer
from config.settings import Settings

logger = get_logger(__name__)


@dataclass
class TranscriptionResult:
    """Structured result from Whisper transcription."""
    text: str                      # Cleaned transcription text
    raw_text: str                  # Unprocessed Whisper output
    language: str                  # Detected language code (e.g. "en", "hi")
    language_prob: float           # Confidence in detected language
    no_speech_prob: float          # Probability that audio contains no speech
    avg_log_prob: float            # Average token log probability
    duration_s: float              # Audio duration in seconds
    was_filtered: bool = False     # True if result was filtered as low quality


class Transcriber:
    """
    Wraps faster-whisper with quality filtering and anti-hallucination logic.

    Usage:
        transcriber = Transcriber(settings)
        result = transcriber.transcribe(audio_array)
        if result and result.text:
            print(result.text)
    """

    # Quality filter thresholds
    NO_SPEECH_THRESHOLD = 0.65      # Above this → likely silence/noise
    AVG_LOG_PROB_THRESHOLD = -1.2   # Below this → low-confidence output
    COMPRESSION_RATIO_THRESHOLD = (
        2.4  # Above this → repetitive (hallucination signal)
    )

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._load_model()

    # ── Setup ──────────────────────────────────────────────────

    def _load_model(self):
        """Load Faster-Whisper model. First call downloads to ~/.cache."""
        # Lazy import — only pull in CTranslate2 when needed
        from faster_whisper import WhisperModel

        device = self.settings.WHISPER_DEVICE
        compute_type = self.settings.WHISPER_COMPUTE_TYPE
        model_size = self.settings.WHISPER_MODEL_SIZE

        # Graceful fallback: CUDA unavailable → CPU with int8
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available — falling back to CPU with int8")
            device = "cpu"
            compute_type = "int8"

        logger.info(
            f"Loading Whisper '{model_size}' on {device} ({compute_type})..."
        )

        with Timer("Whisper model load"):
            self._model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=str(self.settings.MODELS_DIR / "whisper"),
                cpu_threads=4,
            )

        logger.info(f"[green]Whisper '{model_size}' loaded ✓[/green]")

    # ── Transcription ──────────────────────────────────────────

    def transcribe(self, audio: np.ndarray) -> Optional[TranscriptionResult]:
        """
        Transcribe a numpy audio array to text.

        Args:
            audio: float32 array @ 16kHz (from preprocess_audio)

        Returns:
            TranscriptionResult, or None if audio was empty/invalid.
        """
        if audio is None or len(audio) == 0:
            logger.warning("transcribe() called with empty audio.")
            return None

        duration_s = len(audio) / 16000
        logger.debug(f"Transcribing {duration_s:.2f}s of audio...")

        try:
            with Timer("Whisper inference"):
                segments_gen, info = self._model.transcribe(
                    audio,
                    language=self.settings.WHISPER_LANGUAGE,  # None = auto
                    beam_size=self.settings.WHISPER_BEAM_SIZE,
                    best_of=5,
                    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    # ── Anti-hallucination settings ────────────
                    no_speech_threshold=self.NO_SPEECH_THRESHOLD,
                    log_prob_threshold=self.AVG_LOG_PROB_THRESHOLD,
                    compression_ratio_threshold=self.COMPRESSION_RATIO_THRESHOLD,
                    condition_on_previous_text=False,  # Reduces hallucination chains
                    initial_prompt=None,
                    word_timestamps=False,
                    vad_filter=True,        # Built-in VAD as second pass
                    vad_parameters={
                        "min_silence_duration_ms": 500,
                        "speech_pad_ms": 200,
                    },
                )

                # Consume the generator (lazy evaluation)
                raw_text = " ".join(seg.text for seg in segments_gen).strip()

        except Exception as exc:
            logger.error(f"Whisper transcription failed: {exc}", exc_info=True)
            return None

        # ── Quality filtering ──────────────────────────────────
        result = TranscriptionResult(
            text="",
            raw_text=raw_text,
            language=info.language,
            language_prob=info.language_probability,
            no_speech_prob=getattr(info, "no_speech_prob", 0.0),
            avg_log_prob=getattr(info.transcription_options, "avg_log_prob", 0.0)
                if hasattr(info, "transcription_options") else 0.0,
            duration_s=duration_s,
        )

        filtered, reason = self._should_filter(result)
        if filtered:
            logger.info(f"Transcription filtered [{reason}]: '{raw_text[:60]}'")
            result.was_filtered = True
            result.text = ""
            return result

        # ── Post-processing ────────────────────────────────────
        clean = clean_transcription(raw_text)

        if is_empty_transcription(clean):
            result.was_filtered = True
            result.text = ""
        else:
            result.text = clean
            logger.info(
                f"[green]Transcribed ({info.language} "
                f"p={info.language_probability:.2f}): "
                f"'{clean}'[/green]"
            )

        return result

    # ── Filtering ──────────────────────────────────────────────

    def _should_filter(self, result: TranscriptionResult) -> tuple[bool, str]:
        """
        Check whether a transcription result should be discarded.

        Returns (should_filter, reason_string).
        """
        # Check no-speech probability from Whisper's internal VAD
        if result.no_speech_prob > self.NO_SPEECH_THRESHOLD:
            return True, f"no_speech_prob={result.no_speech_prob:.2f}"

        # Check average log probability (low = uncertain / hallucinating)
        if result.avg_log_prob < self.AVG_LOG_PROB_THRESHOLD:
            return True, f"avg_log_prob={result.avg_log_prob:.2f}"

        return False, ""

    # ── Utility ────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def __repr__(self) -> str:
        return (
            f"Transcriber(model={self.settings.WHISPER_MODEL_SIZE}, "
            f"device={self.settings.WHISPER_DEVICE}, "
            f"lang={self.settings.WHISPER_LANGUAGE or 'auto'})"
        )

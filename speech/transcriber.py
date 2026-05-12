"""
speech/transcriber.py
─────────────────────
Stable Faster-Whisper transcription engine for EVA.

Optimized for:
- English
- Hinglish
- conversational speech

Important:
We intentionally FORCE language="en"
because Hinglish performs far more reliably
through English phonetic decoding than
multilingual auto-detection.

This prevents:
- random Spanish
- Japanese hallucinations
- weird language switching
- unstable multilingual detection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.logger import get_logger
from utils.helpers import (
    clean_transcription,
    is_empty_transcription,
    Timer,
)
from config.settings import Settings

logger = get_logger(__name__)


@dataclass
class TranscriptionResult:

    text: str
    raw_text: str
    language: str
    language_prob: float
    no_speech_prob: float
    avg_log_prob: float
    duration_s: float
    was_filtered: bool = False


class Transcriber:

    # ──────────────────────────────────────────
    # Anti-hallucination thresholds
    # ──────────────────────────────────────────

    NO_SPEECH_THRESHOLD = 0.65

    AVG_LOG_PROB_THRESHOLD = -1.2

    COMPRESSION_RATIO_THRESHOLD = 2.4

    # ──────────────────────────────────────────

    def __init__(self, settings: Settings):

        self.settings = settings

        self._model = None

        self._load_model()

    # ──────────────────────────────────────────
    # Model Loading
    # ──────────────────────────────────────────

    def _load_model(self):

        from faster_whisper import WhisperModel

        device = self.settings.WHISPER_DEVICE

        compute_type = self.settings.WHISPER_COMPUTE_TYPE

        model_size = self.settings.WHISPER_MODEL_SIZE

        import torch

        if device == "cuda" and not torch.cuda.is_available():

            logger.warning(
                "CUDA unavailable → switching to CPU"
            )

            device = "cpu"

            compute_type = "int8"

        logger.info(
            f"Loading Whisper '{model_size}' "
            f"on {device} ({compute_type})..."
        )

        with Timer("Whisper model load"):

            self._model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=str(
                    self.settings.MODELS_DIR / "whisper"
                ),
                cpu_threads=4,
            )

        logger.info(
            f"[green]Whisper '{model_size}' loaded ✓[/green]"
        )

    # ──────────────────────────────────────────
    # Main Transcription
    # ──────────────────────────────────────────

    def transcribe(
        self,
        audio: np.ndarray
    ) -> Optional[TranscriptionResult]:

        if audio is None or len(audio) == 0:

            logger.warning(
                "Empty audio received."
            )

            return None

        duration_s = len(audio) / 16000

        logger.debug(
            f"Transcribing {duration_s:.2f}s audio..."
        )

        try:

            with Timer("Whisper inference"):

                segments_gen, info = (
                    self._model.transcribe(

                        audio,

                        # FORCE ENGLISH MODE
                        # Better for Hinglish stability
                        language="en",

                        beam_size=5,

                        best_of=5,

                        temperature=0.0,

                        # Anti-hallucination
                        no_speech_threshold=(
                            self.NO_SPEECH_THRESHOLD
                        ),

                        log_prob_threshold=(
                            self.AVG_LOG_PROB_THRESHOLD
                        ),

                        compression_ratio_threshold=(
                            self.COMPRESSION_RATIO_THRESHOLD
                        ),

                        condition_on_previous_text=False,

                        initial_prompt=(
                            "This is a casual conversation "
                            "in English and Hinglish."
                        ),

                        word_timestamps=False,

                        vad_filter=True,

                        vad_parameters={

                            "min_silence_duration_ms": 500,

                            "speech_pad_ms": 200,
                        },
                    )
                )

                raw_text = " ".join(
                    seg.text for seg in segments_gen
                ).strip()

        except Exception as exc:

            logger.error(
                f"Whisper failed: {exc}",
                exc_info=True
            )

            return None

        # ──────────────────────────────────────
        # Build Result
        # ──────────────────────────────────────

        result = TranscriptionResult(

            text="",

            raw_text=raw_text,

            language="en",

            language_prob=1.0,

            no_speech_prob=getattr(
                info,
                "no_speech_prob",
                0.0
            ),

            avg_log_prob=0.0,

            duration_s=duration_s,
        )

        # ──────────────────────────────────────
        # Quality Filtering
        # ──────────────────────────────────────

        filtered, reason = self._should_filter(
            result
        )

        if filtered:

            logger.info(
                f"Filtered [{reason}] → "
                f"'{raw_text[:60]}'"
            )

            result.was_filtered = True

            result.text = ""

            return result

        # ──────────────────────────────────────
        # Cleanup
        # ──────────────────────────────────────

        clean = clean_transcription(raw_text)

        if is_empty_transcription(clean):

            result.was_filtered = True

            result.text = ""

        else:

            result.text = clean

            logger.info(
                f"[green]Transcribed: "
                f"'{clean}'[/green]"
            )

        return result

    # ──────────────────────────────────────────
    # Filtering Logic
    # ──────────────────────────────────────────

    def _should_filter(
        self,
        result: TranscriptionResult
    ) -> tuple[bool, str]:

        if (
            result.no_speech_prob
            > self.NO_SPEECH_THRESHOLD
        ):

            return (
                True,
                f"no_speech="
                f"{result.no_speech_prob:.2f}"
            )

        if (
            result.avg_log_prob
            < self.AVG_LOG_PROB_THRESHOLD
        ):

            return (
                True,
                f"logprob="
                f"{result.avg_log_prob:.2f}"
            )

        return False, ""

    # ──────────────────────────────────────────

    @property
    def is_ready(self) -> bool:

        return self._model is not None

    # ──────────────────────────────────────────

    def __repr__(self) -> str:

        return (
            f"Transcriber("
            f"model={self.settings.WHISPER_MODEL_SIZE}, "
            f"device={self.settings.WHISPER_DEVICE})"
        )
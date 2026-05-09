"""
speech/vad.py
─────────────
Silero VAD wrapper for Voice Activity Detection.

Silero VAD is a lightweight, accurate neural network that outputs
a speech probability (0.0–1.0) for each audio chunk.

Why Silero over WebRTC VAD:
  - Much more accurate on noisy microphones
  - Handles accents and Hinglish well
  - Tiny model, runs in <1ms on CPU per chunk

Key behaviors:
  - Chunk size MUST be exactly 512 samples @ 16kHz (Silero constraint)
  - Call reset_states() between utterances to clear GRU memory
  - Speech probability > threshold → speech detected
"""

import numpy as np
import torch
from utils.logger import get_logger

logger = get_logger(__name__)

# Silero VAD requires exactly these chunk sizes at these sample rates
VALID_CHUNK_SIZES = {
    8000:  [256],
    16000: [512, 1024],
}


class SileroVAD:
    """
    Wraps the Silero VAD model with a clean, stateless-per-utterance API.

    Usage:
        vad = SileroVAD(threshold=0.45)
        prob = vad.is_speech(chunk)   # → float in [0, 1]
        vad.reset()                   # between utterances
    """

    def __init__(
        self,
        threshold: float = 0.45,
        sampling_rate: int = 16000,
        force_reload: bool = False,
    ):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self._model: torch.nn.Module | None = None
        self._force_reload = force_reload
        self._device = "cpu"   # Silero VAD always runs on CPU (tiny model)

        self._validate_sample_rate()
        self._load_model()

    # ── Setup ──────────────────────────────────────────────────

    def _validate_sample_rate(self):
        if self.sampling_rate not in VALID_CHUNK_SIZES:
            raise ValueError(
                f"Silero VAD only supports sample rates: {list(VALID_CHUNK_SIZES.keys())}. "
                f"Got {self.sampling_rate}."
            )

    def _load_model(self):
        """Download (first run) or load from torch hub cache."""
        logger.info("Loading Silero VAD model...")
        try:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=self._force_reload,
                onnx=False,           # PyTorch version — ONNX has no reset_states()
                verbose=False,
            )
            model.eval()
            self._model = model
            logger.info("[green]Silero VAD loaded ✓[/green]")

        except Exception as exc:
            logger.error(f"Failed to load Silero VAD: {exc}")
            logger.error("Ensure you have internet on first run for torch.hub.load.")
            raise RuntimeError("Silero VAD unavailable") from exc

    # ── Inference ──────────────────────────────────────────────

    def is_speech(self, chunk: np.ndarray) -> float:
        """
        Compute speech probability for one audio chunk.

        Args:
            chunk: float32 numpy array of exactly VAD_CHUNK_SIZE samples

        Returns:
            Float in [0.0, 1.0]. Values > self.threshold indicate speech.

        Important:
            Silero is stateful (GRU). Always call chunks sequentially.
            Call reset() between utterances.
        """
        if self._model is None:
            raise RuntimeError("VAD model not loaded.")

        if len(chunk) == 0:
            return 0.0

        # Silero expects a 1D float32 tensor
        tensor = torch.from_numpy(chunk.astype(np.float32))
        if tensor.ndim != 1:
            tensor = tensor.squeeze()

        with torch.no_grad():
            prob: float = self._model(tensor, self.sampling_rate).item()

        return prob

    def is_speech_bool(self, chunk: np.ndarray) -> bool:
        """Convenience wrapper: returns True/False instead of probability."""
        return self.is_speech(chunk) >= self.threshold

    # ── State Management ───────────────────────────────────────

    def reset(self):
        """
        Reset VAD internal GRU states.
        MUST call this between utterances to prevent state bleed.
        """
        if self._model is not None:
            self._model.reset_states()

    # ── Properties ─────────────────────────────────────────────

    @property
    def valid_chunk_sizes(self) -> list[int]:
        return VALID_CHUNK_SIZES[self.sampling_rate]

    def __repr__(self) -> str:
        return (
            f"SileroVAD(threshold={self.threshold}, "
            f"sr={self.sampling_rate}, "
            f"loaded={self._model is not None})"
        )

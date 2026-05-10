"""
speech/voice_manager.py
────────────────────────
XTTS-v2 model lifecycle manager optimized for GTX 1650 4GB.

Why XTTS-v2:
  - Only local TTS that genuinely handles Hindi + English + code-switching
  - Neural voice cloning from reference audio (6-30 seconds)
  - Streaming API — first audio chunk in ~1.5s instead of waiting full synthesis
  - Human-quality prosody — NOT robotic, NOT flat

VRAM strategy for 4GB GPU:
  - Load model in float16 → ~1.8GB instead of ~3.5GB in float32
  - Whisper small float16 → ~500MB
  - Total: ~2.3GB — fits comfortably in 4GB with headroom for activations
  - Call torch.cuda.empty_cache() between STT and TTS phases

Voice conditioning:
  - Computed ONCE from reference.wav at load time
  - Stored as (gpt_cond_latent, speaker_embedding) tensors
  - Reused for every synthesis call — fast
  - Can be updated without reloading the full model
"""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import torch

from utils.logger import get_logger

logger = get_logger(__name__)

XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_SAMPLE_RATE = 24000  # XTTS-v2 always outputs at 24kHz

# Minimum free VRAM (GB) required to run XTTS on GPU
MIN_VRAM_FOR_GPU = 2.5


class VoiceManager:
    """
    Manages the XTTS-v2 model for streaming speech synthesis.

    Key design decisions:
    - Uses XTTS lower-level API (not TTS.api.TTS) for streaming control
    - Conditioning latents pre-computed at load → fast per-call inference
    - Thread-safe: _lock prevents concurrent synthesis calls
    - Graceful CPU fallback if VRAM is too low

    Typical usage in EVA:
        vm = VoiceManager()
        vm.load(reference_wav="data/voices/eva_reference.wav")

        for chunk in vm.synthesize_stream("Hey yaar, kya chal raha hai?", "en"):
            player.enqueue(chunk)
    """

    SAMPLE_RATE = XTTS_SAMPLE_RATE

    def __init__(self):
        self._model = None
        self._config = None
        self._gpt_cond_latent: Optional[torch.Tensor] = None
        self._speaker_embedding: Optional[torch.Tensor] = None
        self._lock = threading.Lock()
        self._device = self._select_device()
        self._reference_wav: Optional[str] = None

    # ── Device Selection ───────────────────────────────────────

    def _select_device(self) -> str:
        if not torch.cuda.is_available():
            logger.warning("CUDA not available — XTTS-v2 will run on CPU (slow, ~15s/sentence)")
            return "cpu"

        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        used_gb  = torch.cuda.memory_allocated(0) / 1024**3
        free_gb  = total_gb - used_gb

        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)} | "
            f"VRAM {total_gb:.1f}GB total, {free_gb:.1f}GB free"
        )

        if free_gb >= MIN_VRAM_FOR_GPU:
            logger.info(f"Using CUDA for XTTS-v2 ✓")
            return "cuda"
        else:
            logger.warning(
                f"Only {free_gb:.1f}GB free VRAM — need {MIN_VRAM_FOR_GPU}GB. "
                f"Falling back to CPU. Tip: reduce WHISPER_DEVICE to cpu in .env "
                f"to free GPU for XTTS."
            )
            return "cpu"

    # ── Model Loading ──────────────────────────────────────────

    def load(self, reference_wav: str) -> None:
        """
        Load XTTS-v2 and compute voice conditioning from reference audio.

        Args:
            reference_wav: Path to a clean 6-30 second WAV file.
                           Quality of this file directly determines voice quality.
                           Recommended: 16-24 kHz, mono, no background noise.
                           Indian English accent → best Hinglish output.

        First call downloads XTTS-v2 weights (~1.7GB) to ~/.local/share/tts/
        Subsequent calls load from cache instantly.
        """
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.utils.manage import ModelManager

        logger.info(f"Loading XTTS-v2 on {self._device}...")
        self._free_vram()  # Clear any lingering activation cache

        # ── Download / locate model files ──────────────────────
        manager = ModelManager()
        try:
            model_path, config_path, _ = manager.download_model(XTTS_MODEL_NAME)
        except Exception as e:
            # Newer TTS versions changed the return signature
            result = manager.download_model(XTTS_MODEL_NAME)
            model_path = result[0] if isinstance(result, (list, tuple)) else result
            config_path = Path(model_path) / "config.json"

        model_path = Path(model_path)
        config_path = Path(config_path) if not isinstance(config_path, Path) else config_path

        logger.info(f"Model path: {model_path}")

        # ── Load config ────────────────────────────────────────
        self._config = XttsConfig()
        self._config.load_json(str(config_path))

        # ── Load model ─────────────────────────────────────────
        model = Xtts.init_from_config(self._config)
        model.load_checkpoint(
            self._config,
            checkpoint_dir=str(model_path),
            use_deepspeed=False,
            eval=True,
        )

        # float16 saves ~50% VRAM on GPU (1.8GB instead of 3.5GB)
        if self._device == "cuda":
            model = model.half().cuda()
        else:
            model.cpu()

        model.eval()
        self._model = model

        logger.info("XTTS-v2 weights loaded ✓")
        self._log_vram()

        # ── Compute voice conditioning ─────────────────────────
        self.set_reference_voice(reference_wav)

    # ── Voice Reference ────────────────────────────────────────

    def set_reference_voice(self, wav_path: str) -> None:
        """
        Update the reference voice without reloading the full model.
        Call this to switch between different voice personas at runtime.

        Args:
            wav_path: Path to reference WAV (6-30 seconds, clean speech)
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        wav_path = str(wav_path)
        if not Path(wav_path).exists():
            raise FileNotFoundError(f"Reference WAV not found: {wav_path}")

        logger.info(f"Computing voice conditioning from: {wav_path}")
        self._reference_wav = wav_path

        with torch.no_grad():
            gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
                audio_path=[wav_path],
                gpt_cond_len=30,        # Use up to 30s of reference
                gpt_cond_chunk_len=4,   # Process in 4s chunks
                max_ref_length=30,
            )

        # Keep on same device as model
        if self._device == "cuda":
            self._gpt_cond_latent = gpt_cond_latent.cuda().half()
            self._speaker_embedding = speaker_embedding.cuda().half()
        else:
            self._gpt_cond_latent = gpt_cond_latent.cpu()
            self._speaker_embedding = speaker_embedding.cpu()

        logger.info("Voice conditioning computed ✓")

    # ── Synthesis ──────────────────────────────────────────────

    def synthesize_stream(
        self,
        text: str,
        language: str = "en",
        temperature: float = 0.70,
        speed: float = 0.95,
        stream_chunk_size: int = 20,
    ) -> Generator[np.ndarray, None, None]:
        """
        Stream synthesis — yields audio chunks as they are generated.
        Start playing the first chunk while the rest are still synthesizing.
        This is the PRIMARY synthesis method — minimizes perceived latency.

        Args:
            text:              Clean text (already preprocessed)
            language:          "en" or "hi"
            temperature:       0.5 = stable/consistent, 0.9 = expressive/varied
                               0.7 is the sweet spot for conversational EVA
            speed:             0.85 = slower/deliberate, 1.0 = natural, 1.1 = fast
                               0.95 sounds natural for Indian English
            stream_chunk_size: tokens per yielded audio chunk
                               20 = ~0.5s of audio per chunk (low latency)
                               50 = ~1.0s per chunk (higher quality, more latency)

        Yields:
            float32 numpy arrays at 24000 Hz (XTTS sample rate)

        Note on float16 and synthesis:
            The model runs in float16 but inference_stream internally casts
            to float32 for numerical stability. The output is float32.
        """
        if not self.is_loaded:
            raise RuntimeError("VoiceManager not loaded. Call load() first.")

        if not text or not text.strip():
            return

        with self._lock:
            self._free_vram()  # Clear activation cache before inference

            try:
                with torch.no_grad():
                    chunks = self._model.inference_stream(
                        text,
                        language,
                        self._gpt_cond_latent,
                        self._speaker_embedding,
                        # ── Quality / style ──────────────────
                        temperature=temperature,
                        speed=speed,
                        top_p=0.85,
                        top_k=50,
                        repetition_penalty=10.0,
                        # ── Streaming ────────────────────────
                        stream_chunk_size=stream_chunk_size,
                        enable_text_splitting=False,  # We split text ourselves
                        # ── Anti-repetition ──────────────────
                        do_sample=True,
                    )

                    for chunk in chunks:
                        if chunk is None:
                            continue
                        audio = chunk.cpu().float().numpy()

                        # Normalize shape: (1, N) or (N,) → (N,)
                        if audio.ndim > 1:
                            audio = audio.squeeze()

                        if len(audio) > 0:
                            yield audio

            except torch.cuda.OutOfMemoryError:
                logger.error(
                    "CUDA OOM during XTTS synthesis!\n"
                    "Try: set WHISPER_DEVICE=cpu in .env to free VRAM for XTTS."
                )
                self._free_vram()
                raise

            except Exception as exc:
                logger.error(f"XTTS synthesis error: {exc}", exc_info=True)
                raise

    def synthesize(
        self,
        text: str,
        language: str = "en",
        temperature: float = 0.70,
        speed: float = 0.95,
    ) -> np.ndarray:
        """
        Blocking synthesis — collects all chunks and returns full array.
        Use for cache warmup and short phrases.
        """
        chunks = list(self.synthesize_stream(text, language, temperature, speed))
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)

    # ── Unload ─────────────────────────────────────────────────

    def unload(self) -> None:
        """Release GPU memory. Call when switching to STT phase if VRAM is tight."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._gpt_cond_latent is not None:
            del self._gpt_cond_latent
            self._gpt_cond_latent = None
        if self._speaker_embedding is not None:
            del self._speaker_embedding
            self._speaker_embedding = None
        self._free_vram()
        logger.info("XTTS-v2 unloaded — VRAM freed.")

    # ── Helpers ────────────────────────────────────────────────

    def _free_vram(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _log_vram(self) -> None:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / 1024**3
            res   = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"VRAM after XTTS load: {alloc:.2f}GB allocated, {res:.2f}GB reserved")

    @property
    def is_loaded(self) -> bool:
        return (
            self._model is not None
            and self._gpt_cond_latent is not None
            and self._speaker_embedding is not None
        )

    @property
    def device(self) -> str:
        return self._device

    def __repr__(self) -> str:
        return (
            f"VoiceManager(device={self._device}, "
            f"loaded={self.is_loaded}, "
            f"ref={Path(self._reference_wav).name if self._reference_wav else 'none'})"
        )

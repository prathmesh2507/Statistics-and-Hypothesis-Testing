"""
speech/audio_utils.py
─────────────────────
Low-level audio preprocessing utilities.

Responsibilities:
  - Normalize audio amplitude
  - Convert sample rates
  - Apply basic noise gate
  - Validate audio quality before transcription
  - Convert between numpy, bytes, and float32 formats

All functions are stateless and pure — they take arrays and return arrays.
"""

import numpy as np
from scipy import signal
from utils.logger import get_logger

logger = get_logger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
TARGET_SAMPLE_RATE = 16000          # Whisper expects 16kHz
TARGET_DTYPE = np.float32
MAX_AMPLITUDE = 1.0
NOISE_GATE_RMS_THRESHOLD = 0.003    # Below this = probably silence/noise


# ─── Core Preprocessing Pipeline ──────────────────────────────────────────────

def preprocess_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Full preprocessing pipeline.
    Input:  raw float32 mono audio at any sample rate
    Output: normalized, denoised float32 mono @ 16kHz

    Pipeline:
        1. Ensure float32 mono
        2. Resample to 16kHz (if needed)
        3. Apply noise gate
        4. Normalize amplitude
        5. Trim leading/trailing silence
    """
    if audio is None or len(audio) == 0:
        logger.warning("preprocess_audio received empty array")
        return np.array([], dtype=TARGET_DTYPE)

    # 1. Type / channel normalization
    audio = to_float32_mono(audio)

    # 2. Resample
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = resample(audio, sample_rate, TARGET_SAMPLE_RATE)

    # 3. Noise gate (zero out sub-threshold frames)
    audio = apply_noise_gate(audio)

    # 4. Normalize
    audio = normalize(audio)

    # 5. Trim edges
    audio = trim_silence(audio, threshold_db=-40.0)

    return audio


# ─── Individual Operations ────────────────────────────────────────────────────

def to_float32_mono(audio: np.ndarray) -> np.ndarray:
    """
    Ensure the array is float32 and single-channel.
    Handles: stereo (2D), int16, int32, float64 inputs.
    """
    # Collapse to 1D if stereo
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Convert integer PCM to float
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    return audio


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample audio using scipy's polyphase filter (high quality)."""
    if src_rate == dst_rate:
        return audio

    num_samples = int(len(audio) * dst_rate / src_rate)
    resampled = signal.resample_poly(
        audio,
        up=dst_rate,
        down=src_rate,
    )
    logger.debug(f"Resampled {src_rate}Hz → {dst_rate}Hz ({len(audio)} → {len(resampled)} samples)")
    return resampled.astype(TARGET_DTYPE)


def normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Peak normalization.
    Scales audio so that the loudest sample reaches `target_peak`.
    Does nothing if audio is already silent.
    """
    peak = np.abs(audio).max()
    if peak < 1e-6:
        return audio  # Avoid divide-by-zero on silent audio
    return (audio / peak * target_peak).astype(TARGET_DTYPE)


def apply_noise_gate(
    audio: np.ndarray,
    frame_length: int = 512,
    threshold: float = NOISE_GATE_RMS_THRESHOLD,
) -> np.ndarray:
    """
    Zero-out frames whose RMS energy is below the threshold.
    This removes mic hiss and background hum between speech.
    """
    output = audio.copy()
    for start in range(0, len(audio), frame_length):
        frame = audio[start : start + frame_length]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < threshold:
            output[start : start + frame_length] = 0.0
    return output


def trim_silence(audio: np.ndarray, threshold_db: float = -40.0) -> np.ndarray:
    """
    Trim leading and trailing silence below `threshold_db`.
    Uses a simple energy-based approach.
    """
    if len(audio) == 0:
        return audio

    threshold_linear = 10 ** (threshold_db / 20.0)
    abs_audio = np.abs(audio)

    # Find first and last sample above threshold
    nonsilent = np.where(abs_audio > threshold_linear)[0]
    if len(nonsilent) == 0:
        return audio  # All silence — return as-is, let caller decide

    start = max(0, nonsilent[0] - 800)       # 50ms pre-roll @ 16kHz
    end   = min(len(audio), nonsilent[-1] + 800)  # 50ms post-roll

    return audio[start:end]


# ─── Quality Checks ───────────────────────────────────────────────────────────

def is_audio_quality_sufficient(audio: np.ndarray, min_duration_s: float = 0.3) -> bool:
    """
    Quick sanity checks before sending audio to Whisper.
    Returns False if audio is too short or too quiet to transcribe reliably.
    """
    if audio is None or len(audio) == 0:
        return False

    duration = len(audio) / TARGET_SAMPLE_RATE
    if duration < min_duration_s:
        logger.debug(f"Audio too short: {duration:.2f}s < {min_duration_s}s")
        return False

    rms = np.sqrt(np.mean(audio ** 2))
    if rms < NOISE_GATE_RMS_THRESHOLD:
        logger.debug(f"Audio too quiet: RMS={rms:.5f}")
        return False

    return True


def get_audio_stats(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> dict:
    """Return diagnostic stats about an audio array (useful for debugging)."""
    if len(audio) == 0:
        return {"duration_s": 0, "rms": 0, "peak": 0}
    return {
        "duration_s": round(len(audio) / sample_rate, 3),
        "num_samples": len(audio),
        "rms": round(float(np.sqrt(np.mean(audio ** 2))), 6),
        "peak": round(float(np.abs(audio).max()), 6),
        "dtype": str(audio.dtype),
    }

"""
tests/test_audio_utils.py
─────────────────────────
Unit tests for audio preprocessing functions.
These run without hardware — all synthetic audio.

Run: pytest tests/ -v
"""

import numpy as np
import pytest

from speech.audio_utils import (
    to_float32_mono,
    normalize,
    apply_noise_gate,
    trim_silence,
    preprocess_audio,
    is_audio_quality_sufficient,
    get_audio_stats,
    TARGET_SAMPLE_RATE,
)
from utils.helpers import clean_transcription, is_empty_transcription


# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture
def silent_audio():
    """Pure silence at 16kHz, 1 second."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def speech_audio():
    """Synthetic sine wave representing speech."""
    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def noisy_audio():
    """Low-level noise below noise gate threshold."""
    return (np.random.randn(16000) * 0.001).astype(np.float32)


# ── to_float32_mono ────────────────────────────────────────────

class TestToFloat32Mono:
    def test_stereo_to_mono(self):
        stereo = np.random.randn(16000, 2).astype(np.float32)
        mono = to_float32_mono(stereo)
        assert mono.ndim == 1
        assert len(mono) == 16000

    def test_int16_to_float32(self):
        pcm = (np.random.randn(16000) * 16000).astype(np.int16)
        result = to_float32_mono(pcm)
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_float32_passthrough(self, speech_audio):
        result = to_float32_mono(speech_audio)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, speech_audio)


# ── normalize ──────────────────────────────────────────────────

class TestNormalize:
    def test_peak_reaches_target(self, speech_audio):
        result = normalize(speech_audio, target_peak=0.95)
        assert abs(result.max()) <= 0.96  # Allow small float error
        assert abs(result.max()) >= 0.90

    def test_silent_audio_unchanged(self, silent_audio):
        result = normalize(silent_audio)
        assert np.allclose(result, 0.0)

    def test_output_dtype(self, speech_audio):
        result = normalize(speech_audio)
        assert result.dtype == np.float32


# ── apply_noise_gate ───────────────────────────────────────────

class TestNoiseGate:
    def test_silence_zeroed(self, noisy_audio):
        result = apply_noise_gate(noisy_audio, threshold=0.003)
        # Very low noise should be gated to zero
        assert np.allclose(result, 0.0, atol=1e-6)

    def test_speech_preserved(self, speech_audio):
        result = apply_noise_gate(speech_audio, threshold=0.001)
        # Speech amplitude (0.5) >> threshold — should be mostly preserved
        assert result.max() > 0.3


# ── is_audio_quality_sufficient ───────────────────────────────

class TestQualityCheck:
    def test_good_audio_passes(self, speech_audio):
        assert is_audio_quality_sufficient(speech_audio) is True

    def test_silent_audio_fails(self, silent_audio):
        assert is_audio_quality_sufficient(silent_audio) is False

    def test_too_short_fails(self):
        short = np.random.randn(1000).astype(np.float32) * 0.5
        assert is_audio_quality_sufficient(short, min_duration_s=0.3) is False

    def test_none_fails(self):
        assert is_audio_quality_sufficient(None) is False


# ── clean_transcription ────────────────────────────────────────

class TestCleanTranscription:
    def test_removes_whisper_music_tags(self):
        assert clean_transcription("[Music] Hello") == "Hello"

    def test_filters_hallucination_phrases(self):
        assert clean_transcription("Thank you.") == ""
        assert clean_transcription("thanks for watching.") == ""

    def test_normal_text_unchanged(self):
        text = "Hey what's up yaar"
        assert clean_transcription(text) == text

    def test_empty_string(self):
        assert clean_transcription("") == ""

    def test_collapses_whitespace(self):
        result = clean_transcription("hello   world")
        assert result == "hello world"


# ── is_empty_transcription ─────────────────────────────────────

class TestIsEmpty:
    def test_empty_string(self):
        assert is_empty_transcription("") is True

    def test_whitespace_only(self):
        assert is_empty_transcription("   ") is True

    def test_punctuation_only(self):
        assert is_empty_transcription("...") is True

    def test_real_text(self):
        assert is_empty_transcription("hello") is False

    def test_single_char(self):
        assert is_empty_transcription("a") is True  # < 2 alnum chars

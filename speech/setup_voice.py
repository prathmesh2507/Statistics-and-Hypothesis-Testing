"""
speech/setup_voice.py
──────────────────────
One-time voice setup utility for EVA's XTTS-v2 TTS.

XTTS-v2 needs a 6-30 second reference WAV to clone the voice from.
This script offers three ways to provide one:

  Mode 1 (recommended): Record your own voice from the microphone
  Mode 2:               Download a public-domain Indian English voice sample
  Mode 3:               Use any existing WAV file on your system

Why your own voice is best:
  - Most natural match for Indian English / Hinglish accent
  - XTTS clones prosody, rhythm, and tone from the reference
  - 10-15 seconds of clean speech is enough

Recording tips for best results:
  - Speak naturally at normal conversational pace
  - No music, fan noise, or background chatter
  - USB microphone or headset is better than laptop mic
  - Read the sample script below aloud — it covers a range of phonemes

SAMPLE SCRIPT (read this when prompted):
  "Hi, I'm EVA. I'm here to chat, help you think, and keep you company.
   Whether you want to talk about life, tech, or just vent about your day —
   I'm all ears, yaar. Ask me anything, I'll do my best. Let's go."

Run:
    python speech/setup_voice.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

DEFAULT_OUTPUT = Path("data/voices/eva_reference.wav")
SAMPLE_RATE = 16000
MIN_DURATION = 6
MAX_DURATION = 30
RECOMMENDED_DURATION = 15


def record_reference(
    output_path: str = str(DEFAULT_OUTPUT),
    duration: int = RECOMMENDED_DURATION,
    sample_rate: int = SAMPLE_RATE,
) -> str:
    """
    Record a voice reference WAV from the default microphone.

    Args:
        output_path: Where to save the WAV file
        duration:    Recording length in seconds (6-30)
        sample_rate: Recording sample rate (16kHz is fine)

    Returns:
        Path to saved WAV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = max(MIN_DURATION, min(MAX_DURATION, duration))

    print(f"\n{'='*60}")
    print(f"  EVA Voice Reference Recorder")
    print(f"{'='*60}")
    print(f"\nRecording {duration} seconds of your voice.")
    print(f"Output: {output_path}")
    print(f"\nRead this script naturally when recording starts:")
    print(f"\n  \"Hi, I'm EVA. I'm here to chat, help you think, and")
    print(f"   keep you company. Whether you want to talk about life,")
    print(f"   tech, or just vent about your day — I'm all ears, yaar.")
    print(f"   Ask me anything, I'll do my best. Let's go.\"")
    print(f"\n{'─'*60}")

    for i in range(3, 0, -1):
        print(f"  Starting in {i}...", end="\r")
        time.sleep(1)

    print(f"\n🎤 Recording... speak now! ({duration}s)")
    print(f"   [{'─' * 40}]", end="")

    # Record
    audio = sd.rec(
        frames=int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32,
    )

    # Progress bar
    for i in range(duration):
        time.sleep(1)
        filled = int((i + 1) / duration * 40)
        print(f"\r   [{'█' * filled}{'─' * (40 - filled)}] {i+1}/{duration}s", end="")

    sd.wait()
    print(f"\n✅ Recording complete!")

    # Quality check
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.abs(audio).max())
    print(f"\n   Audio quality: RMS={rms:.4f}, Peak={peak:.4f}")

    if rms < 0.01:
        print("   ⚠️  WARNING: Very quiet recording. Check your microphone.")
    elif rms > 0.5:
        print("   ⚠️  WARNING: Recording may be clipping. Try moving mic further.")
    else:
        print("   ✅ Audio levels look good!")

    # Normalize
    if peak > 0:
        audio = audio / peak * 0.90

    # Save
    sf.write(str(output_path), audio, sample_rate)
    size_kb = output_path.stat().st_size / 1024
    print(f"\n✅ Saved: {output_path} ({size_kb:.0f}KB)")
    print(f"   You can now run: python main.py\n")

    return str(output_path)


def validate_reference(wav_path: str) -> dict:
    """
    Check if a WAV file is suitable as XTTS-v2 reference.
    Returns a report dict.
    """
    path = Path(wav_path)
    if not path.exists():
        return {"valid": False, "error": f"File not found: {wav_path}"}

    try:
        audio, sr = sf.read(str(path))
    except Exception as e:
        return {"valid": False, "error": f"Cannot read file: {e}"}

    duration = len(audio) / sr
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

    issues = []
    if duration < MIN_DURATION:
        issues.append(f"Too short ({duration:.1f}s < {MIN_DURATION}s minimum)")
    if duration > MAX_DURATION:
        issues.append(f"Too long ({duration:.1f}s > {MAX_DURATION}s maximum)")
    if rms < 0.005:
        issues.append(f"Too quiet (RMS={rms:.4f})")
    if sr < 16000:
        issues.append(f"Sample rate too low ({sr}Hz, recommend ≥16kHz)")

    return {
        "valid": len(issues) == 0,
        "path": str(path),
        "duration_s": round(duration, 2),
        "sample_rate": sr,
        "channels": audio.ndim,
        "rms": round(rms, 5),
        "issues": issues,
    }


def use_existing_file(src_path: str, output_path: str = str(DEFAULT_OUTPUT)) -> str:
    """Copy an existing WAV file as the reference voice."""
    import shutil
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate first
    report = validate_reference(src_path)
    if not report["valid"]:
        print(f"⚠️  Issues with reference: {report['issues']}")
        print("   Proceeding anyway — quality may be lower.")

    shutil.copy2(src_path, str(output_path))
    print(f"✅ Copied to: {output_path}")
    return str(output_path)


# ── Interactive CLI ────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  EVA Voice Setup")
    print(f"{'='*60}")
    print(f"\nChoose setup method:")
    print(f"  1. Record my voice now (recommended)")
    print(f"  2. Use an existing WAV file")
    print(f"  3. Validate an existing reference")
    print()

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        duration_str = input(f"Recording duration in seconds [{RECOMMENDED_DURATION}]: ").strip()
        duration = int(duration_str) if duration_str.isdigit() else RECOMMENDED_DURATION
        record_reference(duration=duration)

    elif choice == "2":
        src = input("Path to existing WAV file: ").strip().strip('"')
        use_existing_file(src)

    elif choice == "3":
        path = input("Path to WAV file: ").strip().strip('"')
        report = validate_reference(path)
        print(f"\n{'─'*40}")
        for k, v in report.items():
            print(f"  {k}: {v}")
        print(f"{'─'*40}")
        if report["valid"]:
            print("  ✅ File is suitable as XTTS-v2 reference!")
        else:
            print("  ❌ Issues found. Re-record or use a different file.")

    else:
        print("Invalid choice.")
        sys.exit(1)


if __name__ == "__main__":
    main()

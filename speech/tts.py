import subprocess
import tempfile
import threading
from pathlib import Path

import sounddevice as sd
import soundfile as sf

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class PiperTTS:

    def __init__(self, settings: Settings):

        self.settings = settings

        self.piper_path = Path(
            settings.PIPER_PATH
        )

        self.voice_model = Path(
            settings.VOICE_MODEL
        )

        # ─────────────────────────────
        # Speaking state
        # ─────────────────────────────

        self.is_speaking = False

        self._playback_lock = threading.Lock()

        logger.info(
            f"Piper initialized | "
            f"voice={self.voice_model.name}"
        )

    # ──────────────────────────────────
    # Main speak
    # ──────────────────────────────────

    def speak(self, text: str):

        if not text or not text.strip():
            return

        with self._playback_lock:

            self.is_speaking = True

            logger.info("🔊 EVA speaking...")

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as temp_audio:

                output_path = temp_audio.name

            command = [

                str(self.piper_path),

                "--model",

                str(self.voice_model),

                "--output_file",

                output_path
            ]

            try:

                process = subprocess.Popen(

                    command,

                    stdin=subprocess.PIPE,

                    stdout=subprocess.PIPE,

                    stderr=subprocess.PIPE,

                    text=True
                )

                process.communicate(input=text)

                data, samplerate = sf.read(
                    output_path
                )

                sd.play(data, samplerate)

                sd.wait()

            except Exception as e:

                logger.error(
                    f"Piper playback failed: {e}"
                )

            finally:

                self.is_speaking = False

    # ──────────────────────────────────
    # Stop playback
    # ──────────────────────────────────

    def stop(self):

        try:

            sd.stop()

        except Exception:
            pass

        self.is_speaking = False
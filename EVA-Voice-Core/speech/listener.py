import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile
import queue
import time

SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 2

audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())


def record_audio():
    print("🎤 Listening...")

    recording = []
    silence_start = None

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=audio_callback
    ):

        while True:
            data = audio_queue.get()

            recording.append(data)

            volume = np.linalg.norm(data)

            if volume < SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()

                elif time.time() - silence_start > SILENCE_DURATION:
                    break
            else:
                silence_start = None

    audio_data = np.concatenate(recording, axis=0)

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    write(temp_file.name, SAMPLE_RATE, audio_data)

    return temp_file.name
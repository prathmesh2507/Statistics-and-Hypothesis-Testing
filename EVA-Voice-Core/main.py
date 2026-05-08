from speech.listener import record_audio
from speech.transcriber import transcribe_audio

def main():

    while True:

        audio_path = record_audio()

        print("🧠 Transcribing...")

        text = transcribe_audio(audio_path)

        print(f"\n🗣 You Said: {text}\n")


if __name__ == "__main__":
    main()
"""
main.py
───────
EVA Voice Core — Main Entry Point (Low-Latency Edition)

Pipeline (old — slow):
  mic → Whisper → [wait full LLM] → [wait full XTTS] → play
  Latency: ~9-14 seconds

Pipeline (new — fast):
  mic → Whisper → filler plays instantly
               → LLM sentence 1 ready → XTTS sentence 1 → play
               → LLM sentence 2 ready → XTTS sentence 2 → play (overlap)
  Latency: ~2.5-4 seconds to first word

Key optimization: StreamingTTSPipeline overlaps LLM generation
and XTTS synthesis at sentence granularity.
"""

import sys
import signal

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from config.settings import settings
from brain.personality import eva_personality
from brain.llm_engine import LLMEngine
from brain.context_manager import ContextManager
from brain.response_generator import ResponseGenerator
from brain.streaming_pipeline import StreamingTTSPipeline
from speech.listener import VoiceListener
from speech.transcriber import Transcriber
from speech.tts import PiperTTS
from memory.conversation_store import ConversationStore
from utils.logger import get_root_logger

logger = get_root_logger()
console = Console()

_running = True

def _handle_exit(sig, frame):
    global _running
    console.print("\n[yellow]Shutting down EVA...[/yellow]")
    _running = False

signal.signal(signal.SIGINT, _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)


def print_banner():
    banner = Text()
    banner.append("  E V A  ", style="bold cyan")
    banner.append("Voice Core", style="dim")
    console.print(Panel(
        banner,
        subtitle=f"[dim]{eva_personality.tagline}[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print(
        f"[dim]Model:[/dim] {settings.OLLAMA_MODEL}  "
        f"[dim]Whisper:[/dim] {settings.WHISPER_MODEL_SIZE}/{settings.WHISPER_DEVICE}  "
        f"[dim]VAD:[/dim] Silero (thresh={settings.VAD_THRESHOLD})\n"
    )


def build_components():
    logger.info("Initializing EVA components...")

    try:
        llm = LLMEngine(settings)
    except RuntimeError as e:
        console.print(f"\n[bold red]❌ LLM Error:[/bold red] {e}\n")
        sys.exit(1)

    try:
        transcriber = Transcriber(settings)
    except Exception as e:
        console.print(f"\n[bold red]❌ Transcriber Error:[/bold red] {e}\n")
        sys.exit(1)

    try:
        listener = VoiceListener(settings)
    except Exception as e:
        console.print(f"\n[bold red]❌ Listener Error:[/bold red] {e}\n")
        sys.exit(1)

    tts = None
    try:
        tts = PiperTTS(settings, backend="xtts")
    except Exception as e:
        console.print(f"\n[bold yellow]⚠️  TTS Error:[/bold yellow] {e}")
        console.print("[yellow]Continuing without TTS — text-only mode.[/yellow]\n")

    context = ContextManager(
        system_prompt=eva_personality.system_prompt,
        max_turns=12,
        user_name=settings.USER_NAME,
        assistant_name=settings.EVA_NAME,
    )

    responder = ResponseGenerator(llm=llm, personality=eva_personality)
    store = ConversationStore(persist_path=settings.DATA_DIR / "sessions")

    # Build streaming pipeline if TTS is available
    pipeline = None
    if tts is not None:
        pipeline = StreamingTTSPipeline(
            tts_engine=tts,
            response_generator=responder,
            context_manager=context,
        )

    logger.info("[green]All components ready ✓[/green]")
    return listener, transcriber, tts, context, responder, store, pipeline


def run_conversation_loop(listener, transcriber, tts, context, responder, store, pipeline):
    global _running

    session_id = store.new_session()
    console.print("[bold green]EVA is ready. Start talking! (Ctrl+C to stop)[/bold green]\n")

    if tts:
        tts.speak(f"Hey! I'm {settings.EVA_NAME}. What's on your mind?")

    while _running:
        try:
            # ── Step 1: Listen ────────────────────────────────
            console.print("[dim]Listening...[/dim]", end="\r")
            audio = listener.listen(idle_timeout=15.0)

            if audio is None:
                continue

            # If TTS is speaking and user starts talking — interrupt
            if tts and tts.is_speaking:
                tts.stop()
                if pipeline:
                    pipeline.interrupt()

            # ── Step 2: Transcribe ────────────────────────────
            console.print("[dim]Transcribing...[/dim]", end="\r")
            result = transcriber.transcribe(audio)

            if result is None or result.was_filtered or not result.text:
                continue

            user_text = result.text
            console.print(f"[bold white]You:[/bold white] {user_text}")
            store.add_turn(session_id, "user", user_text)

            # Drain mic during response (prevent echo)
            listener._drain_queue()

            # ── Step 3+4: Generate + Speak (pipelined) ────────
            if pipeline:
                # FAST PATH: streaming LLM → per-sentence XTTS → play
                console.print("[dim]Thinking + speaking...[/dim]", end="\r")
                response = pipeline.respond_and_speak(user_text)
            else:
                # TEXT-ONLY fallback (no TTS)
                console.print("[dim]Thinking...[/dim]", end="\r")
                response = responder.respond(context, user_text)

            if not response:
                continue

            console.print(f"[bold cyan]{settings.EVA_NAME}:[/bold cyan] {response}\n")
            store.add_turn(session_id, "assistant", response)

            # Drain mic again after TTS finishes (prevent hearing TTS echo)
            listener._drain_queue()
            import time
            time.sleep(0.2)

        except KeyboardInterrupt:
            _running = False
            break
        except Exception as exc:
            logger.error(f"Loop error: {exc}", exc_info=True)
            import time
            time.sleep(0.5)
            continue

    # ── Shutdown ──────────────────────────────────────────────
    console.print("\n[dim]Saving session...[/dim]")
    store.save_session(session_id)
    console.print("[bold cyan]EVA: See ya! 👋[/bold cyan]")
    if tts:
        tts.speak("See you later!")


def main():
    print_banner()
    components = build_components()
    run_conversation_loop(*components)


if __name__ == "__main__":
    main()

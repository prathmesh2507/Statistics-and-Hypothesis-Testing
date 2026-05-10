"""
main.py
───────
EVA Voice Core — Main Entry Point

This is the top-level conversation loop:

    mic → VAD → Whisper STT → Ollama LLM → pyttsx3 TTS → speakers
         └── Silero VAD filters noise before any inference runs

Run:
    python main.py

Stop: Ctrl+C (graceful shutdown)

Phase 1 scope:
  ✅ Voice input (sounddevice + Silero VAD)
  ✅ Speech-to-text (Faster-Whisper small, GPU)
  ✅ LLM response (Ollama / Qwen2.5:3b)
  ✅ Voice output (pyttsx3)
  ✅ Rolling conversation context

Phase 2 (not yet):
  ⬜ MongoDB long-term memory
  ⬜ ChromaDB semantic recall
  ⬜ Piper TTS
  ⬜ System automation
"""

import sys
import signal
import traceback
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# ── EVA Modules ───────────────────────────────────────────────
from config.settings import settings
from brain.personality import eva_personality
from brain.llm_engine import LLMEngine
from brain.context_manager import ContextManager
from brain.response_generator import ResponseGenerator
from speech import listener
from speech.listener import VoiceListener
from speech.transcriber import Transcriber
from memory.conversation_store import ConversationStore
from utils.logger import get_root_logger
from speech.tts import TTSEngine

logger = get_root_logger()
console = Console()


# ── Graceful Shutdown ──────────────────────────────────────────
_running = True

def _handle_exit(sig, frame):
    global _running
    console.print("\n[yellow]Shutting down EVA...[/yellow]")
    _running = False

signal.signal(signal.SIGINT, _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)


# ── Startup Banner ─────────────────────────────────────────────

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
    console.print(f"[dim]Model:[/dim] {settings.OLLAMA_MODEL}  "
                  f"[dim]Whisper:[/dim] {settings.WHISPER_MODEL_SIZE}/"
                  f"{settings.WHISPER_DEVICE}  "
                  f"[dim]VAD:[/dim] Silero (thresh={settings.VAD_THRESHOLD})\n")


# ── Component Initialization ───────────────────────────────────

def build_components():
    """
    Initialize all EVA components.
    Fails loudly with a helpful message if anything is misconfigured.
    """
    logger.info("Initializing EVA components...")

    # LLM (checks Ollama is running)
    try:
        llm = LLMEngine(settings)
    except RuntimeError as e:
        console.print(f"\n[bold red]❌ LLM Error:[/bold red] {e}\n")
        sys.exit(1)

    # Whisper STT (downloads model on first run)
    try:
        transcriber = Transcriber(settings)
    except Exception as e:
        console.print(f"\n[bold red]❌ Transcriber Error:[/bold red] {e}\n")
        sys.exit(1)

    # Voice listener (Silero VAD + sounddevice)
    try:
        listener = VoiceListener(settings)
    except Exception as e:
        console.print(f"\n[bold red]❌ Listener Error:[/bold red] {e}\n")
        sys.exit(1)

    # TTS
    try:
        tts = TTSEngine(settings)
    except Exception as e:
        console.print(f"\n[bold yellow]⚠️  TTS Error:[/bold yellow] {e} (continuing without TTS)")
        tts = None

    # Conversation context (rolling window)
    context = ContextManager(
        system_prompt=eva_personality.system_prompt,
        max_turns=12,
        user_name=settings.USER_NAME,
        assistant_name=settings.EVA_NAME,
    )

    # Response generator
    responder = ResponseGenerator(llm=llm, personality=eva_personality)

    # Conversation persistence
    store = ConversationStore(
        persist_path=settings.DATA_DIR / "sessions"
    )

    logger.info("[green]All components ready ✓[/green]")
    return listener, transcriber, tts, context, responder, store


# ── Main Conversation Loop ─────────────────────────────────────

def run_conversation_loop(listener, transcriber, tts, context, responder, store):
    """
    The core listen → think → speak loop.

    Each iteration:
      1. Capture speech (VAD-gated microphone)
      2. Transcribe with Whisper
      3. Generate response with Qwen2.5 via Ollama
      4. Speak response via TTS
    """
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
                # Timeout — no speech detected
                logger.debug("No speech in timeout window.")
                continue

            # ── Step 2: Transcribe ────────────────────────────
            console.print("[dim]Transcribing...[/dim]", end="\r")
            result = transcriber.transcribe(audio)

            if result is None or result.was_filtered or not result.text:
                logger.debug("Transcription empty or filtered — listening again.")
                continue

            user_text = result.text
            console.print(f"[bold white]You:[/bold white] {user_text}")

            # Persist user turn
            store.add_turn(session_id, "user", user_text)

            # ── Step 3: Generate Response ─────────────────────
            console.print("[dim]Thinking...[/dim]", end="\r")
            response = responder.respond(context, user_text)

            if not response:
                continue

            console.print(f"[bold cyan]{settings.EVA_NAME}:[/bold cyan] {response}\n")
            tts.speak(response)

            # Persist assistant turn
            store.add_turn(session_id, "assistant", response)

            # ── Step 4: Speak ─────────────────────────────────
            if tts:
                listener.is_assistant_speaking = True
                tts.speak(response)
                listener.is_assistant_speaking = False
                listener._drain_queue()
                import time
                time.sleep(0.4)

        except KeyboardInterrupt:
            _running = False
            break

        except Exception as exc:
            logger.error(f"Loop error: {exc}", exc_info=True)
            console.print(f"[red]Error in conversation loop: {exc}[/red]")
            # Brief recovery pause before retrying
            import time
            time.sleep(0.5)
            continue

    # ── Shutdown ──────────────────────────────────────────────
    console.print("\n[dim]Saving session...[/dim]")
    store.save_session(session_id)
    console.print("[bold cyan]EVA: See ya! 👋[/bold cyan]")
    if tts:
        tts.speak("See you later!")


# ── Entry Point ────────────────────────────────────────────────

def main():
    print_banner()

    components = build_components()
    listener, transcriber, tts, context, responder, store = components

    run_conversation_loop(
        listener=listener,
        transcriber=transcriber,
        tts=tts,
        context=context,
        responder=responder,
        store=store,
    )


if __name__ == "__main__":
    main()

"""
main.py
────────
EVA Voice Core — Stable Piper Edition

Pipeline:
mic → Whisper → Qwen → Piper → audio

Optimized for:
- low latency
- conversational stability
- Hinglish support
- lightweight local execution
"""

import sys
import signal
import time

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from config.settings import settings

from brain.personality import eva_personality
from brain.llm_engine import LLMEngine
from brain.context_manager import ContextManager
from brain.response_generator import ResponseGenerator

from speech.listener import VoiceListener
from speech.transcriber import Transcriber
from speech.tts import PiperTTS

from memory.conversation_store import ConversationStore

from utils.logger import get_root_logger


logger = get_root_logger()

console = Console()

_running = True


# ──────────────────────────────────────────
# Graceful shutdown
# ──────────────────────────────────────────

def _handle_exit(sig, frame):

    global _running

    console.print(
        "\n[yellow]Shutting down EVA...[/yellow]"
    )

    _running = False


signal.signal(signal.SIGINT, _handle_exit)

signal.signal(signal.SIGTERM, _handle_exit)


# ──────────────────────────────────────────
# UI Banner
# ──────────────────────────────────────────

def print_banner():

    banner = Text()

    banner.append(
        "  E V A  ",
        style="bold cyan"
    )

    banner.append(
        "Voice Core",
        style="dim"
    )

    console.print(
        Panel(
            banner,

            subtitle=(
                f"[dim]"
                f"{eva_personality.tagline}"
                f"[/dim]"
            ),

            border_style="cyan",

            padding=(1, 4),
        )
    )

    console.print(
        f"[dim]Model:[/dim] "
        f"{settings.OLLAMA_MODEL}  "

        f"[dim]Whisper:[/dim] "
        f"{settings.WHISPER_MODEL_SIZE}/"
        f"{settings.WHISPER_DEVICE}  "

        f"[dim]TTS:[/dim] Piper\n"
    )


# ──────────────────────────────────────────
# Build Components
# ──────────────────────────────────────────

def build_components():

    logger.info(
        "Initializing EVA components..."
    )

    # ── LLM ───────────────────────────────

    try:

        llm = LLMEngine(settings)

    except RuntimeError as e:

        console.print(
            f"\n[bold red]"
            f"❌ LLM Error:"
            f"[/bold red] {e}\n"
        )

        sys.exit(1)

    # ── Transcriber ──────────────────────

    try:

        transcriber = Transcriber(settings)

    except Exception as e:

        console.print(
            f"\n[bold red]"
            f"❌ Transcriber Error:"
            f"[/bold red] {e}\n"
        )

        sys.exit(1)

    # ── Listener ─────────────────────────

    try:

        listener = VoiceListener(settings)

    except Exception as e:

        console.print(
            f"\n[bold red]"
            f"❌ Listener Error:"
            f"[/bold red] {e}\n"
        )

        sys.exit(1)

    # ── TTS ──────────────────────────────

    tts = None

    try:

        tts = PiperTTS(settings)

    except Exception as e:

        console.print(
            f"\n[bold yellow]"
            f"⚠️ TTS Error:"
            f"[/bold yellow] {e}"
        )

        console.print(
            "[yellow]"
            "Continuing without TTS."
            "[/yellow]\n"
        )

    # ── Context ──────────────────────────

    context = ContextManager(

        system_prompt=eva_personality.system_prompt,

        max_turns=12,

        user_name=settings.USER_NAME,

        assistant_name=settings.EVA_NAME,
    )

    # ── Response Generator ───────────────

    responder = ResponseGenerator(

        llm=llm,

        personality=eva_personality
    )

    # ── Conversation Store ───────────────

    store = ConversationStore(

        persist_path=settings.DATA_DIR / "sessions"
    )

    logger.info(
        "[green]All components ready ✓[/green]"
    )

    return (
        listener,
        transcriber,
        tts,
        context,
        responder,
        store
    )


# ──────────────────────────────────────────
# Main Loop
# ──────────────────────────────────────────

def run_conversation_loop(

    listener,
    transcriber,
    tts,
    context,
    responder,
    store
):

    global _running

    session_id = store.new_session()

    console.print(
        "[bold green]"
        "EVA is ready. Start talking!"
        " (Ctrl+C to stop)"
        "[/bold green]\n"
    )

    # ── Welcome ──────────────────────────

    if tts:

        tts.speak(
            f"Hey! I'm {settings.EVA_NAME}. "
            f"How's your day going?"
        )

    # ─────────────────────────────────────

    while _running:

        try:

            # ── Listen ───────────────────

            console.print(
                "[dim]Listening...[/dim]",
                end="\r"
            )

            audio = listener.listen(
                idle_timeout=15.0
            )

            if audio is None:
                continue

            # ── Interrupt TTS ────────────

            if tts and tts.is_speaking:

                tts.stop()

            # ── Transcribe ───────────────

            console.print(
                "[dim]Transcribing...[/dim]",
                end="\r"
            )

            result = transcriber.transcribe(
                audio
            )

            if (
                result is None
                or result.was_filtered
                or not result.text
            ):

                continue

            user_text = result.text

            console.print(
                f"[bold white]You:[/bold white] "
                f"{user_text}"
            )

            store.add_turn(
                session_id,
                "user",
                user_text
            )

            # Prevent echo
            listener._drain_queue()

            # ── Generate Response ────────

            console.print(
                "[dim]Thinking...[/dim]",
                end="\r"
            )

            response = responder.respond(
                context,
                user_text
            )

            if not response:
                continue

            # ── Print Response ───────────

            console.print(
                f"[bold cyan]"
                f"{settings.EVA_NAME}:"
                f"[/bold cyan] "
                f"{response}\n"
            )

            # ── Speak Response ───────────

            if tts:

                tts.speak(response)

            # ── Store Memory ─────────────

            store.add_turn(
                session_id,
                "assistant",
                response
            )

            # Prevent self-hearing
            listener._drain_queue()

            time.sleep(0.2)

        except KeyboardInterrupt:

            _running = False

            break

        except Exception as exc:

            logger.error(
                f"Loop error: {exc}",
                exc_info=True
            )

            time.sleep(0.5)

            continue

    # ─────────────────────────────────────
    # Shutdown
    # ─────────────────────────────────────

    console.print(
        "\n[dim]Saving session...[/dim]"
    )

    store.save_session(session_id)

    console.print(
        "[bold cyan]"
        "EVA: See ya! 👋"
        "[/bold cyan]"
    )

    if tts:

        tts.speak(
            "See you later!"
        )


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

def main():

    print_banner()

    components = build_components()

    run_conversation_loop(*components)


if __name__ == "__main__":

    main()
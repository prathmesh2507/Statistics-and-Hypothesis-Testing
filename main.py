"""
main.py
───────
EVA Voice Core — Phase 5: Memory & Personalization

Pipeline:
  User speaks
    → Whisper STT
    → [MEMORY] process_utterance() — extract + store memories (async)
    → [MEMORY] retrieve_for_prompt() — get relevant memories
    → Inject memory context into system prompt
    → LLM response (memory-aware, personalized)
    → XTTS TTS
    → User hears personalized response
"""

import sys
import signal
import time

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from pathlib import Path

from config.settings import settings
from brain.personality import eva_personality
from brain.llm_engine import LLMEngine
from brain.context_manager import ContextManager
from brain.response_generator import ResponseGenerator
from brain.streaming_pipeline import StreamingTTSPipeline
from speech.listener import VoiceListener
from speech.transcriber import Transcriber
from speech.tts import TTSEngine
from memory import MemorySystem
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
    banner.append("Voice Core  ", style="dim")
    banner.append("Phase 5 — Memory", style="bold magenta")
    console.print(Panel(
        banner,
        subtitle=f"[dim]{eva_personality.tagline}[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print(
        f"[dim]Model:[/dim] {settings.OLLAMA_MODEL}  "
        f"[dim]Whisper:[/dim] {settings.WHISPER_MODEL_SIZE}/{settings.WHISPER_DEVICE}  "
        f"[dim]Memory:[/dim] ChromaDB + sentence-transformers\n"
    )


def build_components():
    logger.info("Initializing EVA components...")

    # LLM
    try:
        llm = LLMEngine(settings)
    except RuntimeError as e:
        console.print(f"\n[bold red]❌ LLM Error:[/bold red] {e}\n")
        sys.exit(1)

    # Whisper
    try:
        transcriber = Transcriber(settings)
    except Exception as e:
        console.print(f"\n[bold red]❌ Transcriber Error:[/bold red] {e}\n")
        sys.exit(1)

    # Listener
    try:
        listener = VoiceListener(settings)
    except Exception as e:
        console.print(f"\n[bold red]❌ Listener Error:[/bold red] {e}\n")
        sys.exit(1)

    # TTS
    tts = None
    try:
        tts = TTSEngine(settings, backend="xtts")
    except Exception as e:
        console.print(f"\n[bold yellow]⚠️  TTS:[/bold yellow] {e} — text-only mode\n")

    # Memory System (Phase 5)
    memory = MemorySystem(persist_dir=settings.DATA_DIR / "chroma")
    try:
        memory.initialize(llm_engine=llm)
        console.print(
            f"[bold magenta]🧠 Memory:[/bold magenta] "
            f"{memory.memory_count} memories loaded\n"
        )
    except Exception as e:
        console.print(f"[yellow]⚠️  Memory system unavailable: {e}[/yellow]")
        memory = None

    # Context + Response
    context = ContextManager(
        system_prompt=eva_personality.system_prompt,
        max_turns=12,
        user_name=settings.USER_NAME,
        assistant_name=settings.EVA_NAME,
    )
    responder = ResponseGenerator(llm=llm, personality=eva_personality)
    store = ConversationStore(persist_path=settings.DATA_DIR / "sessions")

    # Streaming pipeline
    pipeline = None
    if tts is not None:
        pipeline = StreamingTTSPipeline(
            tts_engine=tts,
            response_generator=responder,
            context_manager=context,
        )

    logger.info("[green]All components ready ✓[/green]")
    return listener, transcriber, tts, context, responder, store, pipeline, memory


def run_conversation_loop(
    listener, transcriber, tts, context, responder, store, pipeline, memory
):
    global _running
    session_id = store.new_session()

    console.print("[bold green]EVA is ready. Start talking! (Ctrl+C to stop)[/bold green]\n")
    if tts:
        tts.speak(f"Hey! What's on your mind?")

    while _running:
        try:
            # ── Listen ────────────────────────────────────────
            console.print("[dim]Listening...[/dim]", end="\r")
            audio = listener.listen(idle_timeout=15.0)
            if audio is None:
                continue

            if tts and tts.is_speaking:
                tts.stop()
                if pipeline:
                    pipeline.interrupt()

            # ── Transcribe ────────────────────────────────────
            console.print("[dim]Transcribing...[/dim]", end="\r")
            result = transcriber.transcribe(audio)
            if result is None or result.was_filtered or not result.text:
                continue

            user_text = result.text
            console.print(f"[bold white]You:[/bold white] {user_text}")
            store.add_turn(session_id, "user", user_text)

            # ── Phase 5: Extract + Store Memories (async) ─────
            if memory and memory.is_ready:
                recent_msgs = context.get_recent_user_messages(n=5)
                memory.process_utterance(
                    user_text,
                    context_messages=recent_msgs,
                    async_store=True,    # background — zero added latency
                )

            # ── Phase 5: Retrieve + Inject Memories ───────────
            if memory and memory.is_ready:
                retrieval = memory.retrieve_for_prompt(user_text)
                if retrieval.has_memories:
                    memory_block = retrieval.format_for_prompt()
                    context.set_memory_block(memory_block)
                    logger.debug(
                        f"Injecting {len(retrieval.memories)} memories: "
                        + str([m.memory_type.value for m in retrieval.memories])
                    )
                else:
                    context.clear_memory_block()

            # Drain mic before response
            listener._drain_queue()

            # ── Generate + Speak ───────────────────────────────
            if pipeline:
                console.print("[dim]Thinking + speaking...[/dim]", end="\r")
                response = pipeline.respond_and_speak(user_text)
            else:
                console.print("[dim]Thinking...[/dim]", end="\r")
                response = responder.respond(context, user_text)

            if not response:
                continue

            console.print(f"[bold cyan]{settings.EVA_NAME}:[/bold cyan] {response}\n")
            store.add_turn(session_id, "assistant", response)

            # Drain mic after TTS
            listener._drain_queue()
            time.sleep(0.2)

        except KeyboardInterrupt:
            _running = False
            break
        except Exception as exc:
            logger.error(f"Loop error: {exc}", exc_info=True)
            time.sleep(0.5)
            continue

    # ── Shutdown ──────────────────────────────────────────────
    console.print("\n[dim]Saving session...[/dim]")
    store.save_session(session_id)
    if memory:
        stats = memory.stats()
        console.print(f"[dim]Memory bank: {stats.get('total', 0)} total memories[/dim]")
    console.print("[bold cyan]EVA: See ya![/bold cyan]")
    if tts:
        tts.speak("See you later!")


def main():
    print_banner()
    components = build_components()
    run_conversation_loop(*components)


if __name__ == "__main__":
    main()

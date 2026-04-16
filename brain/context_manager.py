"""
brain/context_manager.py
────────────────────────
Rolling conversation context + memory injection.

Phase 5 addition: memory_block injection into system prompt.

When MemorySystem is wired in, each get_messages() call prepends
a "Relevant context about this user:" block to the system prompt,
giving the LLM personalized background without bloating the context.

The injection is done at the SYSTEM PROMPT level (not as a user
message) so it doesn't confuse the model's turn-taking.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

Role = Literal["system", "user", "assistant"]


@dataclass
class Turn:
    role: Role
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ContextManager:
    """
    Rolling conversation window with optional memory injection.

    Usage:
        ctx = ContextManager(system_prompt="You are EVA...", max_turns=10)
        ctx.add_user("Hey what's up")
        ctx.add_assistant("Not much, yaar! You?")

        # With memory injection:
        ctx.set_memory_block("- User is building SkillSync AI\n- User stressed about exams")
        messages = ctx.get_messages()   # system prompt now includes memory block
    """

    def __init__(
        self,
        system_prompt: str,
        max_turns: int = 12,
        user_name: str = "User",
        assistant_name: str = "EVA",
    ):
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.user_name = user_name
        self.assistant_name = assistant_name

        self._turns: deque[Turn] = deque(maxlen=max_turns * 2)
        self._memory_block: str = ""   # ← Phase 5: injected memory context

        logger.debug(f"ContextManager ready | max_turns={max_turns}")

    # ── Memory Injection ───────────────────────────────────────

    def set_memory_block(self, memory_block: str) -> None:
        """
        Set the memory context to inject into the system prompt.
        Called before each LLM request with fresh retrieval results.

        Args:
            memory_block: Formatted string from RetrievalResult.format_for_prompt()
                         Empty string = no injection
        """
        self._memory_block = memory_block

    def clear_memory_block(self) -> None:
        self._memory_block = ""

    # ── Conversation Management ────────────────────────────────

    def add_user(self, text: str) -> None:
        self._turns.append(Turn(role="user", content=text.strip()))

    def add_assistant(self, text: str) -> None:
        self._turns.append(Turn(role="assistant", content=text.strip()))

    def clear(self) -> None:
        self._turns.clear()
        logger.info("Conversation history cleared.")

    # ── Message Building ───────────────────────────────────────

    def get_messages(self) -> list[dict]:
        """
        Build Ollama /api/chat compatible message list.

        Structure:
            [
              {"role": "system",    "content": "<base_prompt>\n\n<memory_block>"},
              {"role": "user",      "content": "..."},
              {"role": "assistant", "content": "..."},
              ...
            ]

        The memory block is appended to the system prompt when available.
        """
        # Build enriched system prompt
        system_content = self.system_prompt
        if self._memory_block:
            system_content = (
                f"{self.system_prompt}\n\n"
                f"{self._memory_block}"
            )

        messages = [{"role": "system", "content": system_content}]
        messages.extend(t.to_dict() for t in self._turns)
        return messages

    def get_plain_history(self) -> str:
        lines = []
        for turn in self._turns:
            name = self.user_name if turn.role == "user" else self.assistant_name
            lines.append(f"{name}: {turn.content}")
        return "\n".join(lines)

    def get_recent_user_messages(self, n: int = 5) -> list[str]:
        """Return last n user message strings (for memory extraction context)."""
        user_turns = [t.content for t in self._turns if t.role == "user"]
        return user_turns[-n:]

    # ── Properties ─────────────────────────────────────────────

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def is_empty(self) -> bool:
        return len(self._turns) == 0

    @property
    def last_user_message(self) -> Optional[str]:
        for t in reversed(self._turns):
            if t.role == "user":
                return t.content
        return None

    def __repr__(self) -> str:
        return (
            f"ContextManager(turns={self.turn_count}, "
            f"memory={'yes' if self._memory_block else 'no'})"
        )

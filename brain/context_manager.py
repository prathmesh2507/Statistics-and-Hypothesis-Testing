"""
brain/context_manager.py
────────────────────────
Manages the rolling conversation context window.

Keeps the last N turns of conversation in memory and formats them
as an Ollama-compatible message list for the LLM.

Phase 2 will persist this to MongoDB. For now it's in-memory only.
The interface is identical so the upgrade is seamless.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from utils.logger import get_logger

logger = get_logger(__name__)

Role = Literal["system", "user", "assistant"]


@dataclass
class Turn:
    """A single conversation turn."""
    role: Role
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ContextManager:
    """
    Maintains a rolling window of conversation turns.

    Usage:
        ctx = ContextManager(system_prompt="You are EVA...", max_turns=10)
        ctx.add_user("Hey what's up")
        ctx.add_assistant("Not much, yaar! You?")
        messages = ctx.get_messages()  # → list of {role, content} dicts
    """

    def __init__(
        self,
        system_prompt: str,
        max_turns: int = 12,        # Keep last 12 exchanges = 24 messages
        user_name: str = "User",
        assistant_name: str = "EVA",
    ):
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.user_name = user_name
        self.assistant_name = assistant_name

        # Deque of (user_turn, assistant_turn) pairs
        # Using deque with maxlen auto-removes oldest turns
        self._turns: deque[Turn] = deque(maxlen=max_turns * 2)

        logger.debug(f"ContextManager ready | max_turns={max_turns}")

    # ── Mutation ───────────────────────────────────────────────

    def add_user(self, text: str) -> None:
        """Add a user utterance."""
        self._turns.append(Turn(role="user", content=text.strip()))

    def add_assistant(self, text: str) -> None:
        """Add an assistant response."""
        self._turns.append(Turn(role="assistant", content=text.strip()))

    def clear(self) -> None:
        """Reset conversation history (but keep system prompt)."""
        self._turns.clear()
        logger.info("Conversation history cleared.")

    # ── Retrieval ──────────────────────────────────────────────

    def get_messages(self) -> list[dict]:
        """
        Return conversation as Ollama /api/chat compatible message list.

        Format:
            [
                {"role": "system",    "content": "..."},
                {"role": "user",      "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(t.to_dict() for t in self._turns)
        return messages

    def get_plain_history(self) -> str:
        """
        Return conversation as a plain text string.
        Used for building prompts in non-chat API formats.
        """
        lines = []
        for turn in self._turns:
            name = self.user_name if turn.role == "user" else self.assistant_name
            lines.append(f"{name}: {turn.content}")
        return "\n".join(lines)

    # ── Properties ─────────────────────────────────────────────

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def is_empty(self) -> bool:
        return len(self._turns) == 0

    @property
    def last_user_message(self) -> str | None:
        for t in reversed(self._turns):
            if t.role == "user":
                return t.content
        return None

    def __repr__(self) -> str:
        return f"ContextManager(turns={self.turn_count}, max={self.max_turns * 2})"

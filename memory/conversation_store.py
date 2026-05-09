"""
memory/conversation_store.py
─────────────────────────────
In-memory conversation persistence with Phase 2 MongoDB upgrade path.

Phase 1: Stores sessions in memory (lost on restart — that's fine for now).
Phase 2: Swap the in-memory dict for MongoDB calls. Interface is identical.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class ConversationStore:
    """
    Stores and retrieves conversation sessions.

    A 'session' is a list of {role, content, timestamp} dicts.

    Usage:
        store = ConversationStore()
        sid = store.new_session()
        store.add_turn(sid, "user", "Hey EVA")
        store.add_turn(sid, "assistant", "Hey! What's up?")
        history = store.get_session(sid)
    """

    def __init__(self, persist_path: Optional[Path] = None):
        """
        Args:
            persist_path: If provided, sessions are JSON-saved here on exit.
                         Pass None for pure in-memory operation.
        """
        self._sessions: dict[str, list[dict]] = {}
        self._persist_path = persist_path

        if persist_path:
            persist_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"ConversationStore persist path: {persist_path}")

    # ── Session Management ─────────────────────────────────────

    def new_session(self) -> str:
        """Create a new session, return its ID."""
        sid = str(uuid.uuid4())
        self._sessions[sid] = []
        logger.debug(f"New conversation session: {sid[:8]}...")
        return sid

    def get_session(self, session_id: str) -> list[dict]:
        """Return all turns for a session."""
        return self._sessions.get(session_id, [])

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Append a turn to a session."""
        if session_id not in self._sessions:
            logger.warning(f"Session {session_id[:8]} not found — creating.")
            self._sessions[session_id] = []

        self._sessions[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })

    def save_session(self, session_id: str) -> Optional[Path]:
        """Save session to JSON (only if persist_path is set)."""
        if not self._persist_path:
            return None

        session = self._sessions.get(session_id)
        if not session:
            return None

        filename = self._persist_path / f"session_{session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "turns": session,
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Session saved: {filename}")
        return filename

    @property
    def session_count(self) -> int:
        return len(self._sessions)

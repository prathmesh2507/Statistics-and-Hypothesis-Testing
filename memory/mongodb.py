"""
memory/mongodb.py
─────────────────
MongoDB persistence layer — Phase 2.

This stub defines the interface so callers can import it now
without errors. Implement the body when wiring Phase 2.

To enable:
    1. pip install pymongo
    2. Set MONGO_URI in .env
    3. Uncomment and implement the methods below
"""

from utils.logger import get_logger

logger = get_logger(__name__)


class MongoMemory:
    """Phase 2: Long-term conversation persistence in MongoDB."""

    def __init__(self, uri: str, db_name: str):
        self.uri = uri
        self.db_name = db_name
        self._client = None
        self._db = None
        logger.info("MongoMemory: Phase 2 — not yet connected.")

    def connect(self):
        """TODO Phase 2: Initialize MongoDB connection."""
        # from pymongo import MongoClient
        # self._client = MongoClient(self.uri)
        # self._db = self._client[self.db_name]
        # logger.info(f"MongoDB connected: {self.db_name}")
        pass

    def save_conversation(self, session_id: str, turns: list[dict]) -> None:
        """TODO Phase 2: Persist a conversation session."""
        pass

    def load_conversation(self, session_id: str) -> list[dict]:
        """TODO Phase 2: Load a conversation session."""
        return []

    def search_recent(self, user_id: str, limit: int = 5) -> list[dict]:
        """TODO Phase 2: Fetch recent sessions for a user."""
        return []

    def close(self):
        """TODO Phase 2: Close MongoDB connection."""
        pass

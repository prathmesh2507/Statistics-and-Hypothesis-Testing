"""
memory/semantic_memory.py
──────────────────────────
ChromaDB-backed semantic memory — Phase 2.

When implemented, EVA will be able to:
  - Remember facts mentioned in past conversations
  - Recall relevant context by semantic similarity
  - Build a persistent "knowledge about the user" layer

Interface is stubbed so imports work in Phase 1.
"""

from utils.logger import get_logger

logger = get_logger(__name__)


class SemanticMemory:
    """Phase 2: Vector memory using ChromaDB + sentence-transformers."""

    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self._collection = None
        logger.info("SemanticMemory: Phase 2 — not yet active.")

    def initialize(self):
        """TODO Phase 2: Set up ChromaDB collection."""
        # import chromadb
        # from sentence_transformers import SentenceTransformer
        # client = chromadb.PersistentClient(path=self.persist_dir)
        # self._collection = client.get_or_create_collection("eva_memory")
        # self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        pass

    def store(self, text: str, metadata: dict | None = None) -> None:
        """TODO Phase 2: Embed and store a memory."""
        pass

    def recall(self, query: str, top_k: int = 3) -> list[str]:
        """TODO Phase 2: Retrieve semantically similar memories."""
        return []

    def forget(self, memory_id: str) -> None:
        """TODO Phase 2: Delete a specific memory."""
        pass

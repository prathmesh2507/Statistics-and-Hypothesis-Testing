"""
memory/semantic_memory.py
──────────────────────────
ChromaDB-backed vector memory store for EVA.

Stores MemoryEntry objects as vector embeddings using
sentence-transformers/all-MiniLM-L6-v2.

Each stored memory has:
  - A vector embedding (for semantic similarity search)
  - The raw text content
  - Metadata: type, importance, timestamp, tags, access_count

ChromaDB architecture:
  - Persistent local storage in data/chroma/
  - One collection: "eva_memory"
  - L2 distance metric (smaller = more similar)
  - No server required — embedded DB

Thread safety:
  ChromaDB's embedded client is NOT thread-safe for writes.
  The _write_lock ensures only one write at a time.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from memory.memory_types import (
    MemoryEntry, MemoryType, RetrievalResult,
    SIMILARITY_DISTANCE_THRESHOLD, MAX_INJECTED_MEMORIES,
)
from utils.logger import get_logger

logger = get_logger(__name__)

_COLLECTION_NAME = "eva_memory"
_EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticMemory:
    """
    Vector memory store backed by ChromaDB.

    Usage:
        mem = SemanticMemory(persist_dir=Path("./data/chroma"))
        mem.initialize()

        mem.add(entry)                      # store a memory
        results = mem.search("exams")       # semantic search
        mem.delete(memory_id)               # remove a memory
        all_mem = mem.get_all()             # dump all memories
    """

    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None
        self._embedder = None
        self._write_lock = threading.Lock()
        self._ready = False

    # ── Initialization ─────────────────────────────────────────

    def initialize(self) -> None:
        """
        Set up ChromaDB and load the embedding model.
        Safe to call multiple times (idempotent).
        """
        if self._ready:
            return

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing ChromaDB...")
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )

            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "l2"},  # L2 distance
            )

            count = self._collection.count()
            logger.info(f"ChromaDB ready — {count} memories loaded ✓")

        except ImportError:
            raise ImportError(
                "ChromaDB not installed. Run: pip install chromadb"
            )
        except Exception as exc:
            raise RuntimeError(f"ChromaDB init failed: {exc}") from exc

        logger.info(f"Loading embedding model: {_EMBEDDING_MODEL}")
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                _EMBEDDING_MODEL,
                device="cpu",   # CPU is fine — embeddings are fast and small
            )
            logger.info("Embedding model loaded ✓")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Run: pip install sentence-transformers"
            )

        self._ready = True

    # ── Write Operations ───────────────────────────────────────

    def add(self, entry: MemoryEntry) -> bool:
        """
        Add a memory entry to the vector store.

        Checks for near-duplicates before inserting to prevent
        the same memory being added multiple times.

        Returns True if added, False if duplicate detected.
        """
        if not self._ready:
            raise RuntimeError("SemanticMemory not initialized. Call initialize() first.")

        # Deduplication check: if very similar memory already exists, skip
        if self._is_duplicate(entry.content):
            logger.debug(f"Duplicate memory skipped: '{entry.content[:50]}'")
            return False

        embedding = self._embed(entry.content)
        metadata = entry.to_chroma_metadata()

        with self._write_lock:
            self._collection.add(
                ids=[entry.memory_id],
                embeddings=[embedding],
                documents=[entry.content],
                metadatas=[metadata],
            )

        logger.info(
            f"Memory stored [{entry.memory_type.value}] "
            f"importance={entry.importance}: '{entry.content[:60]}'"
        )
        return True

    def add_batch(self, entries: list[MemoryEntry]) -> int:
        """Add multiple memories. Returns count of successfully added."""
        added = 0
        for entry in entries:
            if self.add(entry):
                added += 1
        return added

    def update_access(self, memory_id: str) -> None:
        """Increment access count and update last_accessed timestamp."""
        with self._write_lock:
            try:
                result = self._collection.get(ids=[memory_id])
                if not result["ids"]:
                    return

                meta = result["metadatas"][0]
                meta["access_count"] = int(meta.get("access_count", 0)) + 1
                meta["last_accessed"] = datetime.now().isoformat()

                self._collection.update(
                    ids=[memory_id],
                    metadatas=[meta],
                )
            except Exception as exc:
                logger.debug(f"update_access failed: {exc}")

    # ── Read Operations ────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = MAX_INJECTED_MEMORIES,
        memory_types: Optional[list[MemoryType]] = None,
        min_importance: int = 1,
        distance_threshold: float = SIMILARITY_DISTANCE_THRESHOLD,
    ) -> list[MemoryEntry]:
        """
        Semantic similarity search.

        Args:
            query:              Search query text
            top_k:              Max results to return
            memory_types:       Filter by type (None = all types)
            min_importance:     Minimum importance score filter
            distance_threshold: Max L2 distance to include (lower = stricter)

        Returns:
            List of MemoryEntry sorted by relevance (most similar first)
        """
        if not self._ready:
            raise RuntimeError("Not initialized.")

        if self._collection.count() == 0:
            return []

        embedding = self._embed(query)

        # Build ChromaDB where filter
        where_filter = self._build_where_filter(memory_types, min_importance)

        try:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k * 2, max(1, self._collection.count())),
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error(f"ChromaDB query failed: {exc}")
            return []

        if not results["ids"] or not results["ids"][0]:
            return []

        entries = []
        ids        = results["ids"][0]
        docs       = results["documents"][0]
        metas      = results["metadatas"][0]
        distances  = results["distances"][0]

        for mid, doc, meta, dist in zip(ids, docs, metas, distances):
            if dist > distance_threshold:
                continue  # Too dissimilar — skip

            entry = MemoryEntry.from_chroma_result(doc, meta, mid)
            entries.append((entry, dist))

        # Sort by distance (ascending = most similar first)
        entries.sort(key=lambda x: x[1])

        # Update access counts asynchronously
        top_entries = [e for e, _ in entries[:top_k]]
        for entry in top_entries:
            threading.Thread(
                target=self.update_access,
                args=(entry.memory_id,),
                daemon=True,
            ).start()

        return top_entries

    def get_by_type(
        self,
        memory_type: MemoryType,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """Retrieve all memories of a specific type."""
        if not self._ready or self._collection.count() == 0:
            return []

        try:
            results = self._collection.get(
                where={"memory_type": memory_type.value},
                limit=limit,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.error(f"get_by_type failed: {exc}")
            return []

        entries = []
        for mid, doc, meta in zip(
            results["ids"], results["documents"], results["metadatas"]
        ):
            entries.append(MemoryEntry.from_chroma_result(doc, meta, mid))

        # Sort by importance (highest first)
        entries.sort(key=lambda e: e.importance, reverse=True)
        return entries

    def get_all(self, limit: int = 200) -> list[MemoryEntry]:
        """Return all stored memories (for debugging / export)."""
        if not self._ready or self._collection.count() == 0:
            return []

        results = self._collection.get(
            limit=limit,
            include=["documents", "metadatas"],
        )
        return [
            MemoryEntry.from_chroma_result(doc, meta, mid)
            for mid, doc, meta in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
        ]

    # ── Delete Operations ──────────────────────────────────────

    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        with self._write_lock:
            try:
                self._collection.delete(ids=[memory_id])
                logger.info(f"Memory deleted: {memory_id[:8]}")
                return True
            except Exception as exc:
                logger.error(f"Delete failed: {exc}")
                return False

    def delete_by_type(self, memory_type: MemoryType) -> int:
        """Delete all memories of a given type. Returns count deleted."""
        entries = self.get_by_type(memory_type, limit=500)
        deleted = 0
        for entry in entries:
            if self.delete(entry.memory_id):
                deleted += 1
        logger.info(f"Deleted {deleted} memories of type '{memory_type.value}'")
        return deleted

    def clear_all(self) -> None:
        """Wipe ALL memories. Use with caution."""
        with self._write_lock:
            self._client.delete_collection(_COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "l2"},
            )
        logger.warning("All memories cleared.")

    # ── Stats ──────────────────────────────────────────────────

    @property
    def count(self) -> int:
        if not self._ready:
            return 0
        return self._collection.count()

    @property
    def is_ready(self) -> bool:
        return self._ready

    def stats(self) -> dict:
        """Return memory statistics grouped by type."""
        if not self._ready:
            return {}

        all_mem = self.get_all(limit=1000)
        type_counts: dict[str, int] = {}
        for m in all_mem:
            key = m.memory_type.value
            type_counts[key] = type_counts.get(key, 0) + 1

        return {
            "total": len(all_mem),
            "by_type": type_counts,
            "avg_importance": (
                sum(m.importance for m in all_mem) / len(all_mem)
                if all_mem else 0
            ),
        }

    # ── Internal Helpers ───────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Generate embedding vector for text."""
        return self._embedder.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def _is_duplicate(
        self,
        content: str,
        threshold: float = 0.15,  # L2 distance — very tight for dedup
    ) -> bool:
        """Check if a nearly identical memory already exists."""
        if self._collection.count() == 0:
            return False

        try:
            embedding = self._embed(content)
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=1,
                include=["distances"],
            )
            if results["distances"] and results["distances"][0]:
                return results["distances"][0][0] < threshold
        except Exception:
            pass
        return False

    def _build_where_filter(
        self,
        memory_types: Optional[list[MemoryType]],
        min_importance: int,
    ) -> Optional[dict]:
        """Build ChromaDB where clause."""
        conditions = []

        if memory_types:
            conditions.append({
                "memory_type": {"$in": [t.value for t in memory_types]}
            })

        if min_importance > 1:
            conditions.append({
                "importance": {"$gte": min_importance}
            })

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

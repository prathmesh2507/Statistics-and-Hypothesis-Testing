"""
memory/memory_system.py
────────────────────────
MemorySystem — unified facade over all memory components.

This is the ONLY class the rest of EVA imports from the memory package.
It wires together:
  SemanticMemory  ←→  MemoryExtractor  ←→  MemoryRetriever
                              ↓
                       MemoryRanker (used by extractor)

Usage in the conversation loop:

    # Initialize once at startup
    memory = MemorySystem(persist_dir=Path("./data/chroma"))
    memory.initialize(llm_engine=llm)

    # After each user utterance — extract and store
    memory.process_utterance(user_text, context_messages=recent_msgs)

    # Before each LLM call — retrieve and inject
    result = memory.retrieve_for_prompt(user_text)
    memory_block = result.format_for_prompt()
    # Prepend memory_block to system prompt

Async-safe:
    process_utterance() can be called from a background thread.
    retrieve_for_prompt() is read-only and thread-safe.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from memory.semantic_memory import SemanticMemory
from memory.memory_ranker import MemoryRanker
from memory.memory_extractor import MemoryExtractor
from memory.memory_retriever import MemoryRetriever
from memory.memory_types import (
    MemoryEntry, MemoryType, RetrievalResult,
    ExtractionResult, STORE_THRESHOLD,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class MemorySystem:
    """
    Unified memory facade for EVA.

    Designed to be a drop-in addition to the existing conversation loop —
    no existing files need to be heavily restructured.
    """

    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir

        # Components (initialized in initialize())
        self._store    = SemanticMemory(persist_dir=persist_dir)
        self._ranker   = MemoryRanker()
        self._extractor: Optional[MemoryExtractor] = None
        self._retriever: Optional[MemoryRetriever] = None

        self._ready = False
        self._process_lock = threading.Lock()

    # ── Initialization ─────────────────────────────────────────

    def initialize(self, llm_engine=None) -> None:
        """
        Initialize the full memory system.

        Args:
            llm_engine: Optional LLM engine (enables LLM-assisted extraction
                        for complex utterances. Pass None to use rules only.)
        """
        logger.info("Initializing Memory System...")

        self._store.initialize()

        self._extractor = MemoryExtractor(
            ranker=self._ranker,
            llm_engine=llm_engine,
            use_llm_fallback=(llm_engine is not None),
            llm_min_words=15,
        )

        self._retriever = MemoryRetriever(
            semantic_memory=self._store,
            max_results=5,
        )

        self._ready = True

        stats = self._store.stats()
        logger.info(
            f"[green]Memory System ready ✓[/green] | "
            f"{stats.get('total', 0)} memories loaded | "
            f"types: {stats.get('by_type', {})}"
        )

    # ── Processing Pipeline ────────────────────────────────────

    def process_utterance(
        self,
        text: str,
        context_messages: Optional[list[str]] = None,
        async_store: bool = True,
    ) -> ExtractionResult:
        """
        Extract memories from a user utterance and store worthy ones.

        Call this AFTER every user message, before generating a response.
        Uses async_store=True by default so it doesn't add latency.

        Args:
            text:             User's utterance
            context_messages: Recent conversation history (list of strings)
            async_store:      If True, store in background thread

        Returns:
            ExtractionResult (even if async — reflects what was found)
        """
        if not self._ready or not text:
            return ExtractionResult(source_text=text)

        result = self._extractor.extract(text, context_messages)

        storable = result.storable
        if not storable:
            logger.debug(f"No storable memories in: '{text[:50]}'")
            return result

        if async_store:
            t = threading.Thread(
                target=self._store_batch,
                args=(storable,),
                daemon=True,
                name="MemStore",
            )
            t.start()
        else:
            self._store_batch(storable)

        return result

    def _store_batch(self, entries: list[MemoryEntry]) -> None:
        """Store a batch of memories (thread-safe)."""
        with self._process_lock:
            added = self._store.add_batch(entries)
            if added > 0:
                logger.debug(f"Stored {added} new memories")

    # ── Retrieval Pipeline ─────────────────────────────────────

    def retrieve_for_prompt(
        self,
        query: str,
        extra_context: Optional[str] = None,
        memory_types: Optional[list[MemoryType]] = None,
    ) -> RetrievalResult:
        """
        Retrieve memories relevant to the current query for prompt injection.

        Call this BEFORE generating each LLM response.

        Args:
            query:         Current user message
            extra_context: Optional conversation summary for better retrieval
            memory_types:  Optionally filter to specific memory types

        Returns:
            RetrievalResult — call .format_for_prompt() to get injection string
        """
        if not self._ready:
            return RetrievalResult(query=query)

        return self._retriever.retrieve(
            query=query,
            memory_types=memory_types,
            extra_context=extra_context,
        )

    # ── Direct Memory Operations ───────────────────────────────

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.GENERAL,
        importance: int = 8,
        tags: Optional[list[str]] = None,
    ) -> MemoryEntry:
        """
        Explicitly store a memory (bypasses extraction + scoring).
        Use when EVA wants to remember something specific.

        Example:
            memory.remember(
                "User explicitly asked EVA to remember their exam on Monday",
                memory_type=MemoryType.SCHEDULE,
                importance=9,
            )
        """
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or [memory_type.value],
        )
        self._store.add(entry)
        return entry

    def forget(self, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        return self._store.delete(memory_id)

    def forget_type(self, memory_type: MemoryType) -> int:
        """Delete all memories of a specific type."""
        return self._store.delete_by_type(memory_type)

    # ── Introspection ──────────────────────────────────────────

    def recall_all(self, memory_type: Optional[MemoryType] = None) -> list[MemoryEntry]:
        """
        Return all stored memories, optionally filtered by type.
        Useful for: "EVA, what do you remember about me?"
        """
        if memory_type:
            return self._store.get_by_type(memory_type, limit=50)
        return self._store.get_all(limit=200)

    def stats(self) -> dict:
        """Return memory system statistics."""
        return self._store.stats()

    def debug_retrieval(self, query: str) -> str:
        """Human-readable retrieval debug output. For testing."""
        if not self._ready:
            return "Memory system not initialized."
        return self._retriever.debug_retrieve(query)

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def memory_count(self) -> int:
        return self._store.count

    def __repr__(self) -> str:
        return (
            f"MemorySystem(ready={self._ready}, "
            f"memories={self.memory_count}, "
            f"path={self.persist_dir})"
        )

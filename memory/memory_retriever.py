"""
memory/memory_retriever.py
───────────────────────────
Advanced conversational memory retriever for EVA.
FINAL FIXED VERSION
"""

from __future__ import annotations

import math
from typing import Optional

from memory.semantic_memory import SemanticMemory
from memory.memory_types import (
    MemoryEntry,
    MemoryType,
    RetrievalResult,
)

from utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────
# Retrieval Tuning (FIXED)
# ──────────────────────────────────────────

# LOWERED from 0.80
SIMILARITY_THRESHOLD = 0.15

# Increased candidate retrieval
TOP_K = 10

_RECENCY_HALFLIFE_DAYS = 45.0

# ──────────────────────────────────────────
# Type Priorities
# ──────────────────────────────────────────

_TYPE_PRIORITY: dict[MemoryType, float] = {
    MemoryType.PROJECT: 1.5,
    MemoryType.GOAL: 1.4,
    MemoryType.CODING: 1.3,
    MemoryType.IDENTITY: 1.2,
    MemoryType.WORK: 1.2,
    MemoryType.PREFERENCE: 1.1,
    MemoryType.HABIT: 1.0,
    MemoryType.SCHEDULE: 1.0,
    MemoryType.EMOTION: 1.0,
    MemoryType.RELATIONSHIP: 1.0,
    MemoryType.GENERAL: 0.8,
}

# ──────────────────────────────────────────
# Query Expansions
# ──────────────────────────────────────────

_QUERY_EXPANSIONS = {
    "project": "project building working creating startup app",
    "working on": "building creating developing making",
    "career": "goal ambition dream future placement job",
    "goals": "goal ambition future target mission",
    "tech stack": "python fastapi flask react coding programming",
    "coding": "python programming development software engineer",
    "feeling": "emotion stress mood mental state",
    "tired": "stress burnout exhausted tired",
    "exam": "study semester college exam stress",
    "preference": "likes prefers favorite settings style",
}

# ──────────────────────────────────────────


class MemoryRetriever:

    def __init__(
        self,
        semantic_memory: SemanticMemory,
        max_results: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):

        self.memory = semantic_memory

        self.max_results = max_results

        self.similarity_threshold = similarity_threshold

    # ──────────────────────────────────────
    # Main Retrieval
    # ──────────────────────────────────────

    def retrieve(
        self,
        query: str,
        memory_types: Optional[list[MemoryType]] = None,
        extra_context: Optional[str] = None,
    ) -> RetrievalResult:

        if not self.memory.is_ready:

            return RetrievalResult(query=query)

        if not query or not query.strip():

            return RetrievalResult(query=query)

        # ──────────────────────────────────
        # Normalize Query
        # ──────────────────────────────────

        search_query = self._normalize_query(query)

        if extra_context:

            search_query += f" {extra_context}"

        search_query = search_query.lower().strip()

        logger.debug(f"Memory query: '{search_query}'")

        # ──────────────────────────────────
        # Semantic Search
        # ──────────────────────────────────

        try:

            raw_results = self.memory.search(
                query=search_query,
                top_k=20,
                memory_types=memory_types,
                min_importance=1,
                distance_threshold=self.similarity_threshold,
            )

        except Exception as e:

            logger.error(f"Semantic search failed: {e}")

            raw_results = []

        print("\nDEBUG RAW RESULTS:")
        print(raw_results)

        # ──────────────────────────────────
        # Keyword Fallback
        # ──────────────────────────────────

        if not raw_results:

            logger.debug("No semantic results → trying keyword fallback.")

            raw_results = self._keyword_fallback(search_query)

        if not raw_results:

            logger.debug(f"No memories found for: '{query[:50]}'")

            return RetrievalResult(query=query)

        # ──────────────────────────────────
        # Re-rank
        # ──────────────────────────────────

        scored = []

        for entry in raw_results:

            try:

                score = self._combined_score(entry)

                if score < 1:
                    continue

                scored.append((entry, score))

            except Exception:
                continue

        scored.sort(key=lambda x: x[1], reverse=True)

        filtered = [
            entry for entry, score in scored
            if entry.importance >= 1
        ][:self.max_results]

        if not filtered:

            return RetrievalResult(query=query)

        logger.info(
            f"Retrieved {len(filtered)} memories "
            f"for '{query}'"
        )

        return RetrievalResult(
            memories=filtered,
            query=query,
            distances=[0.0] * len(filtered),
        )

    # ──────────────────────────────────────
    # Query Normalization
    # ──────────────────────────────────────

    def _normalize_query(self, query: str) -> str:

        original = query.lower().strip()

        expanded = [original]

        mappings = {
            "what project": "project building startup app",
            "working on": "project coding building developing",
            "career": "goal ambition future placement job",
            "goals": "goal ambition target dream",
            "tech stack": "python fastapi react coding programming",
            "coding": "python programming software development",
            "feeling": "emotion stress mood anxious tired",
            "feeling lately": "emotion stress mental state",
            "tired": "stress burnout exhausted",
            "exam": "study semester pressure stress",
            "preferences": "likes prefers favorite settings style",
            "what do i like": "preferences favorite likes",
            "who am i": "identity age location background",
        }

        for trigger, expansion in mappings.items():

            if trigger in original:
                expanded.append(expansion)

        return " ".join(expanded)

    # ──────────────────────────────────────
    # Keyword Fallback
    # ──────────────────────────────────────

    def _keyword_fallback(self, query: str) -> list[MemoryEntry]:

        keywords = [
            word.lower()
            for word in query.split()
            if len(word) > 2
        ]

        all_memories = self.memory.get_all()

        matches = []

        for memory in all_memories:

            content = memory.content.lower()

            score = 0

            for keyword in keywords:

                if keyword in content:
                    score += 1

            if score > 0:
                matches.append((memory, score))

        matches.sort(key=lambda x: x[1], reverse=True)

        return [m[0] for m in matches[:self.max_results]]

    # ──────────────────────────────────────
    # Combined Relevance Score
    # ──────────────────────────────────────

    def _combined_score(self, entry: MemoryEntry) -> float:

        importance_weight = entry.importance / 10.0

        age_days = entry.age_days()

        recency_weight = math.exp(
            -age_days / _RECENCY_HALFLIFE_DAYS
        )

        type_mult = _TYPE_PRIORITY.get(
            entry.memory_type,
            1.0,
        )

        access_bonus = min(
            entry.access_count * 0.05,
            0.3,
        )

        final = (
            importance_weight * 0.5
            + recency_weight * 0.3
        ) * type_mult + access_bonus

        return final * 10

    # ──────────────────────────────────────
    # Debug Retrieval
    # ──────────────────────────────────────

    def debug_retrieve(self, query: str) -> str:

        result = self.retrieve(query)

        if not result.has_memories:

            return f"No memories found for: '{query}'"

        lines = [
            f"Query: '{query}'",
            f"Found {len(result.memories)} memories:",
            "",
        ]

        for i, mem in enumerate(result.memories, 1):

            lines.append(
                f"{i}. "
                f"[{mem.memory_type.value}] "
                f"importance={mem.importance} "
                f"age={mem.age_days():.1f}d"
            )

            lines.append(f"   {mem.content}")

            lines.append("")

        lines.append("─" * 50)

        lines.append("Prompt injection:")

        lines.append(result.format_for_prompt())

        return "\n".join(lines)
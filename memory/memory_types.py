"""
memory/memory_types.py
───────────────────────
Central type definitions for EVA's memory system.

Everything the memory pipeline works with is defined here:
  - MemoryType enum       — what kind of memory it is
  - ImportanceLevel       — how important (with human-readable names)
  - MemoryEntry           — the core data object stored in ChromaDB
  - ExtractionResult      — what memory_extractor.py returns
  - RetrievalResult       — what memory_retriever.py returns

Keeping types in one file means:
  - No circular imports between memory modules
  - One place to add new memory categories
  - Clean isinstance() checks everywhere
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ── Memory Categories ──────────────────────────────────────────

class MemoryType(str, Enum):
    """
    What kind of information a memory represents.
    Using str Enum so values serialize to JSON cleanly.
    """
    PROJECT     = "project"      # "I'm building SkillSync AI"
    GOAL        = "goal"         # "I want to get into IIT"
    PREFERENCE  = "preference"   # "I prefer dark mode"
    EMOTION     = "emotion"      # "I'm stressed about exams"
    HABIT       = "habit"        # "I code every night after 10pm"
    IDENTITY    = "identity"     # "I'm 20 years old, from Nagpur"
    RELATIONSHIP = "relationship" # "my friend Rohan is helping me"
    WORK        = "work"         # "I have a deadline on Friday"
    CODING      = "coding"       # "I'm learning FastAPI"
    SCHEDULE    = "schedule"     # "I have an exam next Monday"
    GENERAL     = "general"      # catch-all for misc important info


# ── Importance Levels ──────────────────────────────────────────

class ImportanceLevel(int, Enum):
    """
    Human-readable importance bands.
    Maps to scores 1-10 used by memory_ranker.
    """
    NOISE       = 1   # greetings, filler — should NOT be stored
    LOW         = 2   # minor contextual detail
    MEDIUM_LOW  = 3   # could be useful later
    MEDIUM      = 4   # relevant context
    MEDIUM_HIGH = 5   # worth remembering
    SIGNIFICANT = 6   # clearly important
    HIGH        = 7   # strong signal (goal/project mention)
    VERY_HIGH   = 8   # core identity / recurring theme
    CRITICAL    = 9   # major life event / explicit statement
    CORE        = 10  # "remember this" — explicitly asked to save

    @classmethod
    def from_score(cls, score: int) -> "ImportanceLevel":
        """Map a raw 1-10 score to the enum."""
        clamped = max(1, min(10, int(score)))
        return cls(clamped)

    @property
    def should_store(self) -> bool:
        """Only store memories above the noise threshold."""
        return self.value >= ImportanceLevel.MEDIUM_LOW.value


# Minimum importance score required to persist a memory
STORE_THRESHOLD = ImportanceLevel.MEDIUM_LOW.value   # = 3

# Minimum importance to inject into prompt (high-signal only)
INJECT_THRESHOLD = ImportanceLevel.MEDIUM.value      # = 4

# Maximum memories injected into a single LLM prompt
MAX_INJECTED_MEMORIES = 5

# Similarity threshold for semantic search (0.0–1.0, higher = more similar)
# ChromaDB uses cosine distance: 0.0 = identical, 2.0 = opposite
# We filter results where distance > this value
SIMILARITY_DISTANCE_THRESHOLD = 1.2


# ── Core Memory Entry ──────────────────────────────────────────

@dataclass
class MemoryEntry:
    """
    The canonical memory object throughout the system.

    Stored in ChromaDB as:
      - document: self.content
      - id:       self.memory_id
      - metadata: everything else (flattened to str/int/float for Chroma)
    """
    content: str                              # The actual memory text
    memory_type: MemoryType                   # Category
    importance: int                           # Score 1-10
    tags: list[str] = field(default_factory=list)  # Searchable labels

    # Auto-populated
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    source_text: str = ""                     # Original utterance that triggered this
    access_count: int = 0                     # How many times retrieved (relevance signal)
    last_accessed: Optional[datetime] = None

    def to_chroma_metadata(self) -> dict:
        """
        Flatten to ChromaDB-compatible metadata dict.
        Chroma only supports str, int, float metadata values.
        """
        return {
            "memory_type":   self.memory_type.value,
            "importance":    self.importance,
            "timestamp":     self.timestamp.isoformat(),
            "source_text":   self.source_text[:500],    # cap length
            "tags":          ",".join(self.tags),        # flatten list to CSV
            "access_count":  self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else "",
        }

    @classmethod
    def from_chroma_result(cls, doc: str, meta: dict, memory_id: str) -> "MemoryEntry":
        """Reconstruct a MemoryEntry from ChromaDB query result."""
        tags_raw = meta.get("tags", "")
        return cls(
            content=doc,
            memory_type=MemoryType(meta.get("memory_type", "general")),
            importance=int(meta.get("importance", 3)),
            tags=[t for t in tags_raw.split(",") if t] if tags_raw else [],
            memory_id=memory_id,
            timestamp=_parse_dt(meta.get("timestamp")),
            source_text=meta.get("source_text", ""),
            access_count=int(meta.get("access_count", 0)),
            last_accessed=_parse_dt(meta.get("last_accessed")),
        )

    def age_days(self) -> float:
        """How old this memory is in days."""
        return (datetime.now() - self.timestamp).total_seconds() / 86400

    def __repr__(self) -> str:
        return (
            f"MemoryEntry(type={self.memory_type.value}, "
            f"importance={self.importance}, "
            f"content='{self.content[:60]}...')"
        )


# ── Extraction Result ──────────────────────────────────────────

@dataclass
class ExtractionResult:
    """
    Output of memory_extractor.py — one or more memories found in an utterance.
    """
    entries: list[MemoryEntry] = field(default_factory=list)
    source_text: str = ""

    @property
    def has_memories(self) -> bool:
        return len(self.entries) > 0

    @property
    def storable(self) -> list[MemoryEntry]:
        """Only the entries that meet the store threshold."""
        return [e for e in self.entries if e.importance >= STORE_THRESHOLD]


# ── Retrieval Result ───────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    Output of memory_retriever.py — memories relevant to the current query.
    """
    memories: list[MemoryEntry] = field(default_factory=list)
    query: str = ""
    distances: list[float] = field(default_factory=list)  # 0.0 = identical

    @property
    def has_memories(self) -> bool:
        return len(self.memories) > 0

    def format_for_prompt(self) -> str:
        """
        Format retrieved memories as a prompt injection block.

        Example output:
            Relevant context about this user:
            - [project] Working on SkillSync AI (a student productivity tool)
            - [goal] Wants to get into IIT Bombay for CS
            - [emotion] Stressed about upcoming exams
        """
        if not self.memories:
            return ""

        lines = ["Relevant context about this user:"]
        for mem in self.memories:
            lines.append(f"- [{mem.memory_type.value}] {mem.content}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"RetrievalResult(count={len(self.memories)}, query='{self.query[:40]}')"


# ── Helpers ────────────────────────────────────────────────────

def _parse_dt(value: Optional[str]) -> datetime:
    """Safely parse an ISO format datetime string."""
    if not value:
        return datetime.now()
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return datetime.now()

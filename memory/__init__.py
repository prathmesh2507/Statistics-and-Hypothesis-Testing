"""
memory/__init__.py
Public API for EVA's memory system.

    from memory import MemorySystem
    mem = MemorySystem(persist_dir=Path("./data/chroma"))
    mem.initialize()
    mem.process_utterance("I'm building SkillSync AI", context_messages=[])
    result = mem.retrieve_for_prompt("what project was I working on?")
"""

from memory.memory_types import (
    MemoryType, ImportanceLevel, MemoryEntry,
    ExtractionResult, RetrievalResult,
    STORE_THRESHOLD, INJECT_THRESHOLD,
)
from memory.semantic_memory import SemanticMemory
from memory.memory_ranker import MemoryRanker
from memory.memory_extractor import MemoryExtractor
from memory.memory_retriever import MemoryRetriever
from memory.memory_system import MemorySystem

__all__ = [
    "MemorySystem", "MemoryType", "ImportanceLevel",
    "MemoryEntry", "ExtractionResult", "RetrievalResult",
    "SemanticMemory", "MemoryRanker", "MemoryExtractor",
    "MemoryRetriever", "STORE_THRESHOLD", "INJECT_THRESHOLD",
]

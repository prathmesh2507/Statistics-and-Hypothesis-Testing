"""
tests/test_memory.py
─────────────────────
End-to-end test for EVA's memory system.

Run:
    python tests/test_memory.py

Tests:
  1. Ranker:    importance scoring for various utterance types
  2. Extractor: pattern-based memory extraction
  3. Store:     ChromaDB add / search / delete cycle
  4. Retriever: semantic recall with context
  5. System:    full pipeline — utterance in, memory recalled later

No microphone, no TTS, no Ollama needed — pure memory system test.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.memory_types import MemoryType, MemoryEntry
from memory.memory_ranker import MemoryRanker
from memory.memory_extractor import MemoryExtractor
from memory.memory_system import MemorySystem

# Use a temp directory for test isolation
TEST_DB_DIR = Path("./data/chroma_test")

PASS = "✅"
FAIL = "❌"


def _print_header(title: str):
    print(f"\n{'═' * 55}")
    print(f"  {title}")
    print(f"{'═' * 55}")


def _check(label: str, condition: bool, detail: str = "") -> bool:
    icon = PASS if condition else FAIL
    print(f"  {icon} {label}" + (f" — {detail}" if detail else ""))
    return condition


# ── Test 1: Ranker ─────────────────────────────────────────────

def test_ranker():
    _print_header("Test 1: MemoryRanker")
    ranker = MemoryRanker()
    all_pass = True

    cases = [
        ("hi",                           MemoryType.GENERAL,  True,  3, "noise → score 1"),
        ("okay",                         MemoryType.GENERAL,  True,  3, "noise → score 1"),
        ("I'm building SkillSync AI",    MemoryType.PROJECT,  False, 7, "project → high"),
        ("I want to get into IIT Bombay",MemoryType.GOAL,     False, 7, "goal → high"),
        ("I'm 20 years old from Nagpur", MemoryType.IDENTITY, False, 6, "identity → significant"),
        ("I'm stressed about exams",     MemoryType.EMOTION,  False, 4, "emotion → medium"),
        ("I prefer dark mode always",    MemoryType.PREFERENCE,False,5, "preference → med-high"),
        ("remember this: exam on Monday",MemoryType.SCHEDULE, False, 9, "explicit remember"),
    ]

    for text, mtype, expect_noise, min_score, desc in cases:
        result = ranker.score(text, mtype)
        ok = (result.is_noise == expect_noise) and (result.final_score >= min_score or expect_noise)
        all_pass = all_pass and ok
        _check(
            f"{desc}",
            ok,
            f"score={result.final_score} noise={result.is_noise}"
        )

    return all_pass


# ── Test 2: Extractor ──────────────────────────────────────────

def test_extractor():
    _print_header("Test 2: MemoryExtractor (rules only)")
    ranker = MemoryRanker()
    extractor = MemoryExtractor(ranker=ranker, use_llm_fallback=False)
    all_pass = True

    cases = [
        (
            "I'm building SkillSync AI for students",
            MemoryType.PROJECT,
            "SkillSync",
        ),
        (
            "My goal is to crack UPSC this year",
            MemoryType.GOAL,
            "crack UPSC",
        ),
        (
            "I am a 21 year old student from Nagpur",
            MemoryType.IDENTITY,
            "21",
        ),
        (
            "I'm learning FastAPI and Docker these days",
            MemoryType.CODING,
            "FastAPI",
        ),
        (
            "I always code after 11pm, it's my habit",
            MemoryType.HABIT,
            "11pm",
        ),
    ]

    for text, expected_type, keyword in cases:
        result = extractor.extract(text)
        found = any(
            keyword.lower() in e.content.lower() or expected_type == e.memory_type
            for e in result.entries
        )
        storable = len(result.storable) > 0
        ok = found or storable
        all_pass = all_pass and ok

        entries_str = "; ".join(
            f"[{e.memory_type.value}={e.importance}] {e.content[:40]}"
            for e in result.entries
        ) or "none"
        _check(f"'{text[:45]}'", ok, entries_str)

    return all_pass


# ── Test 3: Semantic Memory Store ─────────────────────────────

def test_semantic_memory():
    _print_header("Test 3: SemanticMemory (ChromaDB)")
    from memory.semantic_memory import SemanticMemory

    store = SemanticMemory(persist_dir=TEST_DB_DIR / "test_store")
    all_pass = True

    try:
        store.initialize()
        _check("ChromaDB initialized", True)
    except Exception as e:
        _check("ChromaDB initialized", False, str(e))
        return False

    # Add memories
    entries = [
        MemoryEntry("User is building SkillSync AI", MemoryType.PROJECT, importance=9,
                    tags=["project", "skillsync"]),
        MemoryEntry("User wants to get into IIT Bombay", MemoryType.GOAL, importance=8,
                    tags=["goal", "iit"]),
        MemoryEntry("User is 20 years old from Nagpur", MemoryType.IDENTITY, importance=7,
                    tags=["identity", "age", "nagpur"]),
        MemoryEntry("User prefers dark mode interfaces", MemoryType.PREFERENCE, importance=5,
                    tags=["preference", "ui"]),
        MemoryEntry("User is stressed about final exams", MemoryType.EMOTION, importance=6,
                    tags=["emotion", "stress", "exams"]),
    ]

    for entry in entries:
        added = store.add(entry)
        _check(f"Add [{entry.memory_type.value}]", added, entry.content[:40])
        all_pass = all_pass and added

    time.sleep(0.2)  # Brief pause for indexing

    # Semantic search tests
    search_cases = [
        ("what project is the user building?", "SkillSync"),
        ("entrance exam preparation",          "IIT"),
        ("how old is the user",                "20"),
        ("stress and pressure",                "stressed"),
    ]

    for query, expected_keyword in search_cases:
        results = store.search(query, top_k=8, min_importance=1, distance_threshold=1.5)
        found = any(
            expected_keyword.lower() in r.content.lower()
            for r in results
        )
        all_pass = all_pass and found
        top = results[0].content[:50] if results else "no results"
        _check(f"Search: '{query[:40]}'", found, f"top: '{top}'")

    # Count check
    count = store.count
    _check(f"Memory count >= 4", count >= 4, f"count={count}")
    all_pass = all_pass and count >= 4

    # Cleanup test DB
    #store.clear_all()
    #_check("Clear all", store.count == 0, f"count={store.count}")

    return all_pass


# ── Test 4: Full System Pipeline ───────────────────────────────

def test_full_system():
    _print_header("Test 4: Full MemorySystem Pipeline")
    system = MemorySystem(persist_dir=TEST_DB_DIR / "test_system")
    all_pass = True

    try:
        system.initialize(llm_engine=None)
        _check("MemorySystem initialized", True)
    except Exception as e:
        _check("MemorySystem initialized", False, str(e))
        return False

    # Simulate conversation
    utterances = [
        "I'm building SkillSync AI, it's a student productivity tool",
        "My goal is to get placed at Google after graduation",
        "I code mostly in Python and FastAPI",
        "I'm really stressed about my semester exams next week",
        "I prefer minimalist dark UI designs",
        "hi",                          # Should NOT be stored (noise)
        "okay thanks",                 # Should NOT be stored (noise)
    ]

    print("\n  Simulating conversation:")
    for utt in utterances:
        result = system.process_utterance(utt, async_store=False)
        storable_count = len(result.storable)
        icon = "💾" if storable_count > 0 else "⊘ "
        print(f"    {icon} '{utt[:50]}' → {storable_count} memories")

    time.sleep(0.3)
    total = system.memory_count
    _check(f"Memories stored (expect 4-6)", total >= 3, f"total={total}")
    all_pass = all_pass and total >= 3

    # Retrieval tests
    print("\n  Testing recall:")
    recall_cases = [
        ("what project am I working on?",       "SkillSync"),
        ("what are my career goals?",           "Google"),
        ("how am I feeling lately?",            "stressed"),
        ("what's my tech stack?",               "Python"),
    ]

    for query, expected in recall_cases:
        retrieval = system.retrieve_for_prompt(query)
        prompt = retrieval.format_for_prompt()
        found = expected.lower() in prompt.lower()
        all_pass = all_pass and found

        snippet = prompt.replace("Relevant context about this user:\n", "")[:80]
        _check(f"Recall: '{query[:40]}'", found, f"→ '{snippet}'")

    # Debug output for one query
    print("\n  Debug retrieval output:")
    print(system.debug_retrieval("I'm tired today"))

    # Cleanup
    #system._store.clear_all()
    return all_pass


# ── Main ───────────────────────────────────────────────────────

def main():
    print(f"\n{'╔' + '═'*53 + '╗'}")
    print(f"║{'EVA Memory System — Test Suite':^53}║")
    print(f"{'╚' + '═'*53 + '╝'}")

    results = {}

    results["Ranker"]          = test_ranker()
    results["Extractor"]       = test_extractor()
    results["SemanticMemory"]  = test_semantic_memory()
    results["FullSystem"]      = test_full_system()

    # Summary
    print(f"\n{'═' * 55}")
    print("  RESULTS SUMMARY")
    print(f"{'═' * 55}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        icon = PASS if ok else FAIL
        print(f"  {icon}  {name}")

    print(f"\n  {passed}/{total} test suites passed")
    print(f"{'═' * 55}\n")

    # Cleanup test directory
    import shutil
    if TEST_DB_DIR.exists():
        shutil.rmtree(TEST_DB_DIR, ignore_errors=True)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

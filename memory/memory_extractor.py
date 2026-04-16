"""
memory/memory_extractor.py
───────────────────────────
Extracts structured memory candidates from user utterances.

Two-pass extraction strategy:
  Pass 1 — Rule-based patterns (fast, zero latency)
            Catches explicit signals: "I'm building X", "my goal is Y"
            Returns results immediately without LLM call

  Pass 2 — LLM extraction (used only for ambiguous / rich text)
            Calls Ollama to extract memories from complex utterances
            Only fires when Pass 1 yields nothing AND text is > 20 words

This hybrid approach:
  - Handles clear cases instantly (no API call, ~0ms)
  - Uses LLM intelligence for nuanced extraction
  - Avoids unnecessary LLM calls on noise/greetings
  - Keeps latency low in the happy path

Output: ExtractionResult containing list[MemoryEntry]
"""

from __future__ import annotations

import json
import re
from typing import Optional

from memory.memory_types import (
    MemoryType,
    MemoryEntry,
    ExtractionResult,
    STORE_THRESHOLD,
)
from memory.memory_ranker import MemoryRanker
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Rule-Based Extraction Patterns ────────────────────────────
#
# Each entry: (compiled_regex, MemoryType, content_template)
# {0} = full match, {1} = first capture group, etc.

_EXTRACTION_RULES: list[tuple[re.Pattern, MemoryType, str]] = [
    # ── PROJECT patterns ──────────────────────────────────────
    (
        re.compile(
            r"(?:i(?:'m| am) (?:building|developing|creating|making|working on))\s+(.+?)(?:\.|,|$)",
            re.I,
        ),
        MemoryType.PROJECT,
        "User is building: {1}",
    ),
    (
        re.compile(
            r"(?:my (?:project|app|system|startup|tool) (?:is|called|named?))\s+(.+?)(?:\.|,|$)",
            re.I,
        ),
        MemoryType.PROJECT,
        "User's project is called: {1}",
    ),
    (
        re.compile(r"(?:launched|shipped|released|deployed)\s+(.+?)(?:\.|,|$)", re.I),
        MemoryType.PROJECT,
        "User launched: {1}",
    ),
    # ── GOAL patterns ─────────────────────────────────────────
    (
        re.compile(
            r"(?:my goal is|i want to|i'm trying to|i plan to|i aim to)\s+(.+?)(?:\.|,|$)",
            re.I,
        ),
        MemoryType.GOAL,
        "User's goal: {1}",
    ),
    (
        re.compile(
            r"(?:i want to get into|i'm preparing for|studying for)\s+(.+?)(?:\.|,|$)",
            re.I,
        ),
        MemoryType.GOAL,
        "User is preparing for: {1}",
    ),
    (
        re.compile(r"(?:my dream is(?: to)?)\s+(.+?)(?:\.|,|$)", re.I),
        MemoryType.GOAL,
        "User's dream: {1}",
    ),
    # ── IDENTITY patterns ─────────────────────────────────────
    # Age
    (
        re.compile(r"(?:i(?:'m| am)) (\d{1,2}) years old", re.I),
        MemoryType.IDENTITY,
        "User is {1} years old",
    ),
    (
        re.compile(r"\bi am (\d{1,2}) years old\b", re.I),
        MemoryType.IDENTITY,
        "User is {1} years old",
    ),
    # Profession / role
    (
        re.compile(
            r"(?:i(?:'m| am) (?:a|an))\s+(student|developer|engineer|designer|doctor|teacher|freelancer|entrepreneur)",
            re.I,
        ),
        MemoryType.IDENTITY,
        "User is a {1}",
    ),
    (
        re.compile(r"\bi am (?:a )?student\b", re.I),
        MemoryType.IDENTITY,
        "User is a student",
    ),
    # Location
    (
        re.compile(
            r"(?:i(?:'m| am) from|i live in|i(?:'m| am) based in)\s+(.+?)(?:\.|,|$)",
            re.I,
        ),
        MemoryType.IDENTITY,
        "User is from {1}",
    ),
    (
        re.compile(r"\bfrom ([A-Za-z ]+)\b", re.I),
        MemoryType.IDENTITY,
        "User is from {1}",
    ),
    # Name
    (
        re.compile(r"my name is\s+(\w+)", re.I),
        MemoryType.IDENTITY,
        "User's name is {1}",
    ),
    # ── PREFERENCE patterns ────────────────────────────────────
    (
        re.compile(r"i (?:prefer|love|like|always use|use)\s+(.+?)(?:\.|,|$)", re.I),
        MemoryType.PREFERENCE,
        "User prefers: {1}",
    ),
    (
        re.compile(
            r"i (?:hate|dislike|never use|can't stand|don't like)\s+(.+?)(?:\.|,|$)",
            re.I,
        ),
        MemoryType.PREFERENCE,
        "User dislikes: {1}",
    ),
    (
        re.compile(
            r"my (?:favourite|favorite|preferred)\s+\w+\s+is\s+(.+?)(?:\.|,|$)", re.I
        ),
        MemoryType.PREFERENCE,
        "User's preference: {1}",
    ),
    # ── CODING patterns ───────────────────────────────────────
    (
        re.compile(
            r"(?:i(?:'m| am) learning|i(?:'m| am) studying)\s+(.+?)(?:\.|,|$)", re.I
        ),
        MemoryType.CODING,
        "User is learning: {1}",
    ),
    (
        re.compile(
            r"(?:i(?:'m| am) using|i use|we(?:'re| are) using)\s+(.+?)(?:\.|,|$)", re.I
        ),
        MemoryType.CODING,
        "User uses: {1}",
    ),
    (
        re.compile(
            r"i(?:'m| am) (?:stuck on|debugging|fixing|working on)\s+(.+?)(?:\.|,|$)",
            re.I,
        ),
        MemoryType.CODING,
        "User working on technical problem: {1}",
    ),
    # ── CODING patterns ───────────────────────────────────────
    (
        re.compile(
            r"(?:i(?:'m| am) learning|i(?:'m| am) studying)\s+(.+?)(?:\.|,|$)",
            re.I,
        ),
        MemoryType.CODING,
        "User is learning: {1}",
    ),
    (
        re.compile(
            r"(?:i(?:'m| am) using|i use|we(?:'re| are) using)\s+(.+?)(?:\.|,|$)",
            re.I,
        ),
        MemoryType.CODING,
        "User uses: {1}",
    ),
    (
        re.compile(
            r"(?:i code mostly in|i code in|i primarily use|i mainly use)\s+(.+?)(?:\.|,|$)",
            re.I,
        ),
        MemoryType.CODING,
        "User codes in: {1}",
    ),
    (
        re.compile(
            r"i(?:'m| am) (?:stuck on|debugging|fixing|working on)\s+(.+?)(?:\.|,|$)",
            re.I,
        ),
        MemoryType.CODING,
        "User working on technical problem: {1}",
    ),
    # ── EMOTION patterns ──────────────────────────────────────
(
    re.compile(
        r"i(?:'m| am|'ve been| feel)\s+"
        r"(?:really |very |super |extremely )?"
        r"(stressed|anxious|depressed|overwhelmed|burned out|exhausted|sad|happy|excited|frustrated|proud|scared|worried)"
        r"(?:\s+(?:about|because of|because|since)\s+(.+?))?"
        r"(?:\.|,|$)",
        re.I,
    ),
    MemoryType.EMOTION,
    "User feels {1} about {2}",
),
    # ── WORK / DEADLINE patterns ──────────────────────────────
    (
        re.compile(
            r"(?:i have a|my) (?:deadline|submission|presentation|meeting|interview)\s+(?:on|by|at|this|next)\s+(.+?)(?:\.|,|$)",
            re.I,
        ),
        MemoryType.WORK,
        "User has a deadline: {1}",
    ),
    (
        re.compile(
            r"(?:i need to finish|i have to complete|i must submit)\s+(.+?)(?:\.|,|$)",
            re.I,
        ),
        MemoryType.WORK,
        "User needs to complete: {1}",
    ),
    # ── SCHEDULE patterns ─────────────────────────────────────
    (
        re.compile(
            r"(?:i have an? exam|exam is|test is)\s+(?:on|next|this)\s+(.+?)(?:\.|,|$)",
            re.I,
        ),
        MemoryType.SCHEDULE,
        "User has exam: {1}",
    ),
    (
        re.compile(
            r"(?:i(?:'m| am) (?:going|travelling|flying) to)\s+(.+?)(?:\.|,|$)", re.I
        ),
        MemoryType.SCHEDULE,
        "User going to: {1}",
    ),
    # ── HABIT patterns ────────────────────────────────────────
    (
        re.compile(
            r"i (?:always|usually|normally|typically|every (?:day|night|morning|evening))\s+(.+?)(?:\.|,|$)",
            re.I,
        ),
        MemoryType.HABIT,
        "User's habit: {1}",
    ),
    (
        re.compile(
            r"i (?:wake up|sleep|eat|exercise|code|work) (?:at|around)\s+(.+?)(?:\.|,|$)",
            re.I,
        ),
        MemoryType.HABIT,
        "User's routine: wakes/sleeps/etc at {1}",
    ),
    # ── RELATIONSHIP patterns ─────────────────────────────────
    (
        re.compile(
            r"my (?:friend|best friend|colleague|teammate|partner|boyfriend|girlfriend|brother|sister|mom|dad|bhai|yaar)\s+(\w+)\s+(.+?)(?:\.|,|$)",
            re.I,
        ),
        MemoryType.RELATIONSHIP,
        "Person in user's life: {1} — {2}",
    ),
]


# ── LLM Extraction Prompt ─────────────────────────────────────

_LLM_EXTRACTION_PROMPT = """Extract any important personal information from this message.

Message: "{text}"

Return a JSON array of memory objects. Each object must have:
  "content":     string — what to remember (clear, complete sentence)
  "memory_type": one of: project, goal, preference, emotion, habit, identity, relationship, work, coding, schedule, general
  "tags":        array of string keywords

Only extract GENUINELY important information worth remembering long-term.
Return [] if nothing important.
Return ONLY the JSON array, no other text.

Example:
[{{"content": "User is building a music recommendation app called BeatSync", "memory_type": "project", "tags": ["project", "music", "app", "beatsync"]}}]"""


class MemoryExtractor:
    """
    Extracts structured memory candidates from user utterances.

    Usage:
        extractor = MemoryExtractor(ranker, llm_engine)
        result = extractor.extract("I'm building SkillSync AI for students")
        for entry in result.storable:
            semantic_memory.add(entry)
    """

    def __init__(
        self,
        ranker: MemoryRanker,
        llm_engine=None,  # Optional — used for LLM-assisted extraction
        use_llm_fallback: bool = True,
        llm_min_words: int = 15,  # Only use LLM if text has ≥ this many words
    ):
        self.ranker = ranker
        self.llm = llm_engine
        self.use_llm_fallback = use_llm_fallback and (llm_engine is not None)
        self.llm_min_words = llm_min_words

    # ── Main Entry Point ───────────────────────────────────────

    def extract(
        self,
        text: str,
        context_messages: Optional[list[str]] = None,
    ) -> ExtractionResult:
        """
        Extract memories from a single user utterance.

        Pass 1: rule-based patterns (always runs, ~0ms)
        Pass 2: LLM extraction (only if pass 1 found nothing AND text is rich)

        Args:
            text:             The user's message
            context_messages: Recent conversation (for repetition scoring)

        Returns:
            ExtractionResult with all found memory entries
        """
        if not text or len(text.strip()) < 5:
            return ExtractionResult(source_text=text)

        entries: list[MemoryEntry] = []

        # ── Pass 1: Rule-based extraction ─────────────────────
        rule_entries = self._extract_rules(text)
        entries.extend(rule_entries)

        # ── Pass 2: LLM extraction (fallback for rich text) ────
        word_count = len(text.split())
        if not entries and self.use_llm_fallback and word_count >= self.llm_min_words:
            llm_entries = self._extract_llm(text)
            entries.extend(llm_entries)

        # ── Score and filter ───────────────────────────────────
        scored_entries = []
        for entry in entries:
            ranking = self.ranker.score(
                entry.content,
                entry.memory_type,
                context_messages,
            )
            entry.importance = ranking.final_score
            entry.source_text = text[:500]

            if not ranking.is_noise:
                scored_entries.append(entry)

        logger.debug(f"Extracted {len(scored_entries)} memories from: '{text[:60]}'")

        return ExtractionResult(entries=scored_entries, source_text=text)

    # ── Pass 1: Rule Extraction ────────────────────────────────

    def _extract_rules(self, text: str) -> list[MemoryEntry]:
        """Apply regex patterns to extract explicit memory signals."""
        entries = []
        seen_contents = set()

        for pattern, memory_type, template in _EXTRACTION_RULES:
            match = pattern.search(text)
            if not match:
                continue

            # Build content from template
            try:
                groups = match.groups()
                content = template
                for i, g in enumerate(groups, 1):
                    if g:
                        content = content.replace(f"{{{i}}}", g.strip())
                    else:
                        content = content.replace(f" about {{{i}}}", "")
                        content = content.replace(f"{{{i}}}", "")

                content = content.strip(" :")
                if not content or content in seen_contents:
                    continue

                seen_contents.add(content)

                # Extract tags from matched text
                tags = self._extract_tags(text, memory_type)

                entry = MemoryEntry(
                    content=content,
                    memory_type=memory_type,
                    importance=5,  # placeholder — scored in extract()
                    tags=tags,
                    source_text=text[:500],
                )
                entries.append(entry)

            except Exception as exc:
                logger.debug(f"Rule extraction error: {exc}")
                continue

        return entries

    # ── Pass 2: LLM Extraction ─────────────────────────────────

    def _extract_llm(self, text: str) -> list[MemoryEntry]:
        """Use LLM to extract memories from rich, ambiguous text."""
        if not self.llm:
            return []

        try:
            prompt = _LLM_EXTRACTION_PROMPT.format(text=text)
            raw = self.llm.generate(
                prompt=prompt,
                temperature=0.1,  # Low temperature = structured, consistent output
                max_tokens=400,
            )

            if not raw:
                return []

            # Extract JSON array from response
            json_match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if not json_match:
                return []

            data = json.loads(json_match.group())
            entries = []

            for item in data:
                if not isinstance(item, dict):
                    continue
                content = item.get("content", "").strip()
                if not content:
                    continue

                mtype_str = item.get("memory_type", "general")
                try:
                    mtype = MemoryType(mtype_str)
                except ValueError:
                    mtype = MemoryType.GENERAL

                tags = item.get("tags", [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",")]

                entries.append(
                    MemoryEntry(
                        content=content,
                        memory_type=mtype,
                        importance=5,
                        tags=tags,
                        source_text=text[:500],
                    )
                )

            logger.debug(f"LLM extracted {len(entries)} memories")
            return entries

        except (json.JSONDecodeError, Exception) as exc:
            logger.debug(f"LLM extraction failed: {exc}")
            return []

    # ── Tag Extraction ─────────────────────────────────────────

    @staticmethod
    def _extract_tags(text: str, memory_type: MemoryType) -> list[str]:
        """Extract keyword tags from text for easier filtering later."""
        tags = [memory_type.value]

        # Capitalize proper nouns (potential project/tool names)
        proper_nouns = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
        tags.extend([n.lower() for n in proper_nouns[:3]])

        # Common tech keywords
        tech_keywords = re.findall(
            r"\b(python|javascript|react|fastapi|llm|ai|ml|docker|api|"
            r"mongodb|chromadb|whisper|ollama|flutter|android|ios)\b",
            text,
            re.I,
        )
        tags.extend([k.lower() for k in tech_keywords])

        return list(dict.fromkeys(tags))[:8]  # Deduplicate, cap at 8

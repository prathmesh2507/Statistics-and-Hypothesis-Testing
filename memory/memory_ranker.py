"""
memory/memory_ranker.py
────────────────────────
Importance scoring for memory candidates.

Scores every candidate memory on a 1-10 scale before deciding
whether to persist it to ChromaDB.

Scoring model (additive):
  1. Base score by memory type   (projects score higher than emotions)
  2. Keyword signal weight       (power words push score up)
  3. Emotional intensity bonus   (strong emotion = worth remembering)
  4. Repetition signal           (same topic mentioned again = more important)
  5. Explicit save signal        ("remember this" = score 10)
  6. Negative filters            (greetings, noise = score 1)

Design philosophy:
  This is NOT an ML model. It's a carefully tuned rule system.
  Rules are fast, explainable, and easily adjusted.
  An ML ranker would need training data we don't have yet.
  Switch to a classifier in Phase 6 when conversation logs accumulate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from memory.memory_types import MemoryType, ImportanceLevel, STORE_THRESHOLD
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Type Base Scores ───────────────────────────────────────────
# Starting score before applying modifiers

_TYPE_BASE_SCORES: dict[MemoryType, int] = {
    MemoryType.PROJECT:      8,   # Explicit project mention → very high
    MemoryType.GOAL:         8,   # Goals are core context
    MemoryType.IDENTITY:     7,   # Who they are → high value
    MemoryType.CODING:       7,   # Technical interests → high value
    MemoryType.WORK:         6,   # Work deadlines → significant
    MemoryType.SCHEDULE:     6,   # Time-bound info → significant
    MemoryType.PREFERENCE:   5,   # Preferences → medium-high
    MemoryType.HABIT:        5,   # Patterns → medium-high
    MemoryType.EMOTION:      4,   # Emotional state → medium
    MemoryType.RELATIONSHIP: 5,   # People they mention → medium-high
    MemoryType.GENERAL:      3,   # Catch-all → conservative
}

# ── High-Signal Keywords ───────────────────────────────────────
# Presence of these words boosts score

_HIGH_SIGNAL_KEYWORDS: dict[str, int] = {
    # Project / work signals
    "building":     +2, "developing":   +2, "project":    +2,
    "startup":      +2, "app":          +1, "system":     +1,
    "ai":           +1, "bot":          +1, "api":        +1,
    "working on":   +2, "launched":     +2, "deployed":   +2,
    "deadline":     +2, "submission":   +2, "release":    +2,

    # Goal / aspiration signals
    "want to":      +1, "goal":         +2, "dream":      +2,
    "planning to":  +1, "trying to":    +1, "learn":      +1,
    "become":       +2, "achieve":      +2, "target":     +1,
    "iit":          +3, "exam":         +2, "placement":  +2,
    "college":      +1, "university":   +1, "degree":     +1,

    # Identity signals
    "i am":         +1, "i'm":          +1, "my name":    +3,
    "years old":    +2, "from":         +1, "live in":    +2,
    "work as":      +2, "study":        +1, "student":    +1,

    # Preference signals
    "prefer":       +2, "love":         +1, "hate":       +1,
    "always use":   +2, "favourite":    +2, "favorite":   +2,
    "never use":    +2, "can't stand":  +2,

    # Explicit memory request
    "remember":     +4, "don't forget": +4, "keep in mind": +3,
    "important":    +2, "note this":    +3,

    # Emotional intensity
    "stressed":     +1, "anxious":      +1, "excited":    +1,
    "exhausted":    +1, "happy":        +1, "depressed":  +2,
    "frustrated":   +1, "overwhelmed":  +2, "proud":      +1,
}

# ── Noise Patterns — Score Overrides → 1 ──────────────────────
# These patterns immediately set score to 1 (do not store)

_NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(hi|hello|hey|hii|helo|hellow)[\s!.,]*$", re.I),
    re.compile(r"^(okay|ok|k|hmm|uh|um|yeah|yep|yup|nope|no)[\s!.,]*$", re.I),
    re.compile(r"^(thanks|thank you|thx|ty)[\s!.,]*$", re.I),
    re.compile(r"^(bye|goodbye|see ya|cya|later)[\s!.,]*$", re.I),
    re.compile(r"^(haan|nahi|achha|acha|theek hai)[\s!.,]*$", re.I),
    re.compile(r"^\W+$"),                              # Only punctuation
    re.compile(r"^.{1,8}$"),                           # Too short (< 9 chars)
]

# ── Emotional Intensity ────────────────────────────────────────

_EMOTION_WORDS: dict[str, int] = {
    "very":         +1,
    "extremely":    +2,
    "so":           +1,
    "really":       +1,
    "totally":      +1,
    "completely":   +1,
    "absolutely":   +1,
    "devastated":   +3,
    "thrilled":     +2,
    "terrified":    +3,
    "desperate":    +3,
    "furious":      +2,
    "ecstatic":     +2,
}


@dataclass
class RankingResult:
    """Full scoring breakdown for a memory candidate."""
    final_score: int
    base_score: int
    keyword_bonus: int
    emotion_bonus: int
    length_bonus: int
    is_noise: bool
    reasoning: str

    @property
    def should_store(self) -> bool:
        return (not self.is_noise) and (self.final_score >= STORE_THRESHOLD)


class MemoryRanker:
    """
    Scores memory candidates on a 1-10 importance scale.

    Usage:
        ranker = MemoryRanker()
        result = ranker.score("I'm building SkillSync AI for students", MemoryType.PROJECT)
        if result.should_store:
            store_memory(...)
    """

    def score(
        self,
        text: str,
        memory_type: MemoryType = MemoryType.GENERAL,
        context_messages: Optional[list[str]] = None,
    ) -> RankingResult:
        """
        Compute importance score for a memory candidate.

        Args:
            text:             The memory content to score
            memory_type:      Category (affects base score)
            context_messages: Recent conversation turns (used for repetition bonus)

        Returns:
            RankingResult with final_score and breakdown
        """
        if not text or not text.strip():
            return self._noise_result("empty text")

        clean = text.strip().lower()

        # ── Noise check (immediate override) ──────────────────
        for pattern in _NOISE_PATTERNS:
            if pattern.match(clean):
                return self._noise_result(f"matched noise pattern: {pattern.pattern[:30]}")

        # ── Base score from memory type ────────────────────────
        base = _TYPE_BASE_SCORES.get(memory_type, 3)

        # ── Keyword bonus ──────────────────────────────────────
        kw_bonus = 0
        matched_keywords = []
        for keyword, weight in _HIGH_SIGNAL_KEYWORDS.items():
            if keyword in clean:
                kw_bonus += weight
                matched_keywords.append(keyword)
        kw_bonus = min(kw_bonus, 4)  # Cap at +4

        # ── Emotional intensity bonus ──────────────────────────
        em_bonus = 0
        for word, weight in _EMOTION_WORDS.items():
            if word in clean:
                em_bonus += weight
        em_bonus = min(em_bonus, 2)  # Cap at +2

        # ── Length bonus (longer = more context = more valuable) ─
        word_count = len(text.split())
        if word_count >= 15:
            len_bonus = 1
        elif word_count >= 8:
            len_bonus = 0
        else:
            len_bonus = -1  # Very short = less context

        # ── Repetition bonus ─────────────────────────────────
        # If same topic mentioned multiple times in context → more important
        rep_bonus = 0
        if context_messages:
            mention_count = sum(
                1 for msg in context_messages
                if any(kw in msg.lower() for kw in matched_keywords[:3])
            )
            if mention_count >= 3:
                rep_bonus = 2
            elif mention_count >= 2:
                rep_bonus = 1

        # ── Final score ────────────────────────────────────────
        raw = base + kw_bonus + em_bonus + len_bonus + rep_bonus
        final = max(1, min(10, raw))

        reasoning = (
            f"base={base} "
            f"keywords={kw_bonus}({', '.join(matched_keywords[:3])}) "
            f"emotion={em_bonus} "
            f"length={len_bonus} "
            f"repetition={rep_bonus}"
        )
        logger.debug(f"Ranked [{memory_type.value}] score={final} | {reasoning}")

        return RankingResult(
            final_score=final,
            base_score=base,
            keyword_bonus=kw_bonus,
            emotion_bonus=em_bonus,
            length_bonus=len_bonus,
            is_noise=False,
            reasoning=reasoning,
        )

    def score_batch(
        self,
        items: list[tuple[str, MemoryType]],
        context_messages: Optional[list[str]] = None,
    ) -> list[RankingResult]:
        """Score multiple candidates efficiently."""
        return [
            self.score(text, mtype, context_messages)
            for text, mtype in items
        ]

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _noise_result(reason: str) -> RankingResult:
        return RankingResult(
            final_score=1,
            base_score=1,
            keyword_bonus=0,
            emotion_bonus=0,
            length_bonus=0,
            is_noise=True,
            reasoning=f"noise: {reason}",
        )

    def quick_score(self, text: str, memory_type: MemoryType = MemoryType.GENERAL) -> int:
        """Return just the integer score (1-10). Convenience method."""
        return self.score(text, memory_type).final_score

"""
brain/personality.py
────────────────────
EVA's personality configuration and traits.

Keeping personality data separate from logic means you can tune
EVA's character without touching any inference code.
"""

from dataclasses import dataclass, field
from config.prompts import EVA_SYSTEM_PROMPT


@dataclass
class Personality:
    """
    Defines EVA's character, communication style, and behavioral constraints.
    """
    name: str = "EVA"
    tagline: str = "Your sharp, warm, no-BS AI companion"

    # Core system prompt injected at the start of every conversation
    system_prompt: str = field(default_factory=lambda: EVA_SYSTEM_PROMPT)

    # Response style configuration
    default_temperature: float = 0.75     # Creative but focused
    max_response_tokens: int = 300        # Keep responses concise

    # Languages EVA handles
    supported_languages: list = field(default_factory=lambda: ["en", "hi", "hinglish"])

    # Conversation style adjectives (for future dynamic prompt construction)
    traits: list = field(default_factory=lambda: [
        "casual",
        "witty",
        "direct",
        "empathetic",
        "curious",
        "bilingual",
    ])

    # Topics EVA is enthusiastic about
    interests: list = field(default_factory=lambda: [
        "technology",
        "science",
        "philosophy",
        "pop culture",
        "everyday life",
        "personal growth",
    ])


# ── Singleton ──────────────────────────────────────────────────
eva_personality = Personality()

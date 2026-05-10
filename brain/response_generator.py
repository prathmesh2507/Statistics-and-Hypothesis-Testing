from __future__ import annotations

import random
import re

from brain.llm_engine import LLMEngine
from brain.context_manager import ContextManager
from brain.personality import Personality, eva_personality
from config.prompts import LLM_FALLBACK_RESPONSES
from utils.logger import get_logger
from utils.helpers import Timer

logger = get_logger(__name__)


FORBIDDEN_PHRASES = [
    "as an ai",
    "as an ai language model",
    "i do not have feelings",
    "i don't have feelings",
    "i am just an assistant",
    "how can i assist you today",
    "certainly",
    "of course",
    "i understand your request",
]


class ResponseGenerator:

    def __init__(
        self,
        llm: LLMEngine,
        personality: Personality = eva_personality,
    ):
        self.llm = llm
        self.personality = personality

    def _clean_response(self, text: str) -> str:

        cleaned = text.strip()

        for phrase in FORBIDDEN_PHRASES:

            pattern = re.compile(re.escape(phrase), re.IGNORECASE)

            cleaned = pattern.sub("", cleaned)

        cleaned = re.sub(r"\s+", " ", cleaned)

        cleaned = cleaned.strip(" ,.-")

        return cleaned

    def respond(
        self,
        context: ContextManager,
        user_message: str,
    ) -> str:

        if not user_message or not user_message.strip():
            return random.choice(LLM_FALLBACK_RESPONSES)

        context.add_user(user_message)

        messages = context.get_messages()

        logger.debug(
            f"Sending {len(messages)} messages to LLM "
            f"(context turns: {context.turn_count})"
        )

        with Timer("LLM chat response"):

            response = self.llm.chat(
                messages=messages,
                temperature=self.personality.default_temperature,
                max_tokens=self.personality.max_response_tokens,
            )

        if not response:

            logger.warning("LLM returned empty response — using fallback.")

            fallback = random.choice(LLM_FALLBACK_RESPONSES)

            context.add_assistant(fallback)

            return fallback

        response = self._clean_response(response)

        context.add_assistant(response)

        logger.info(f"[magenta]EVA:[/magenta] {response[:120]}...")

        return response
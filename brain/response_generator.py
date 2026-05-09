"""
brain/response_generator.py
────────────────────────────
Orchestrates the full LLM response pipeline.

Takes a user message, pulls context, generates a response, and
returns it — handling errors and fallbacks gracefully.

This is the layer between the conversation loop and the raw LLM.
"""

from __future__ import annotations

import random
from typing import Optional

from brain.llm_engine import LLMEngine
from brain.context_manager import ContextManager
from brain.personality import Personality, eva_personality
from config.prompts import LLM_FALLBACK_RESPONSES
from utils.logger import get_logger
from utils.helpers import Timer

logger = get_logger(__name__)


class ResponseGenerator:
    """
    Generates EVA's responses given a user message and conversation context.

    Usage:
        gen = ResponseGenerator(llm_engine, personality)
        response = gen.respond(context_manager, user_message)
    """

    def __init__(
        self,
        llm: LLMEngine,
        personality: Personality = eva_personality,
    ):
        self.llm = llm
        self.personality = personality

    # ── Core ───────────────────────────────────────────────────

    def respond(
        self,
        context: ContextManager,
        user_message: str,
    ) -> str:
        """
        Generate a response to `user_message` using conversation context.

        1. Add user turn to context
        2. Build message list
        3. Call LLM
        4. Add assistant turn to context
        5. Return response text

        Returns a fallback phrase if LLM fails.
        """
        if not user_message or not user_message.strip():
            return random.choice(LLM_FALLBACK_RESPONSES)

        # Add user message to rolling context
        context.add_user(user_message)

        # Build message list with system prompt + history
        messages = context.get_messages()

        logger.debug(
            f"Sending {len(messages)} messages to LLM "
            f"(context turns: {context.turn_count})"
        )

        # Call LLM via chat API
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

        # Store assistant response in context
        context.add_assistant(response)

        logger.info(f"[magenta]EVA:[/magenta] {response[:120]}...")
        return response

    def respond_stream(
        self,
        context: ContextManager,
        user_message: str,
    ):
        """
        Streaming version — yields response fragments.
        Caller can print/speak each chunk as it arrives.

        Note: Context is updated only after full response is assembled.
        """
        context.add_user(user_message)
        messages = context.get_messages()

        full_response = ""
        for chunk in self.llm.stream(
            prompt=user_message,
            system=self.personality.system_prompt,
            temperature=self.personality.default_temperature,
            max_tokens=self.personality.max_response_tokens,
        ):
            full_response += chunk
            yield chunk

        if full_response:
            context.add_assistant(full_response)
        else:
            fallback = random.choice(LLM_FALLBACK_RESPONSES)
            context.add_assistant(fallback)
            yield fallback

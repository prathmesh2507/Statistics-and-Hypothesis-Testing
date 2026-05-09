"""
brain/llm_engine.py
───────────────────
Ollama API client for EVA's language understanding.

Talks to a locally running Ollama server via HTTP.
Supports both streaming (low latency first-token) and blocking modes.

Multi-model routing (future):
  The `model` parameter on generate() overrides the default, allowing
  easy routing to DeepSeek, Llama, Mistral etc. without changing callers.

Connection check:
  On init, pings Ollama to verify it's running. Gives a clear error
  message if not, rather than cryptic connection refused errors later.
"""

from __future__ import annotations

import json
import time
from typing import Generator, Optional

import requests

from utils.logger import get_logger
from utils.helpers import Timer
from config.settings import Settings

logger = get_logger(__name__)

_OLLAMA_API_GENERATE = "{base}/api/generate"
_OLLAMA_API_CHAT     = "{base}/api/chat"
_OLLAMA_API_TAGS     = "{base}/api/tags"

# How long to wait for connection check
_PING_TIMEOUT = 3.0


class LLMEngine:
    """
    Sends prompts to Ollama and returns completions.

    Usage:
        engine = LLMEngine(settings)
        response = engine.generate("What is the speed of light?")

    Streaming usage:
        for chunk in engine.stream("Tell me a joke"):
            print(chunk, end="", flush=True)
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.model = settings.OLLAMA_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT
        self._session = requests.Session()
        self._verify_connection()

    # ── Setup ──────────────────────────────────────────────────

    def _verify_connection(self):
        """Ping Ollama on init. Fail loudly if not running."""
        try:
            r = self._session.get(
                _OLLAMA_API_TAGS.format(base=self.base_url),
                timeout=_PING_TIMEOUT,
            )
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            logger.info(f"[green]Ollama connected ✓[/green] | available models: {models}")

            if self.model not in models:
                logger.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Run: ollama pull {self.model}"
                )

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start it with: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Ollama connection timed out at {self.base_url}")

    # ── Blocking Generation ────────────────────────────────────

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.75,
        max_tokens: int = 300,
    ) -> str:
        """
        Send a prompt and return the complete response as a string.

        Args:
            prompt:      The user message.
            model:       Override default model for routing.
            system:      System prompt (overrides settings default).
            temperature: Creativity (0 = focused, 1 = creative). 0.7-0.8 is ideal for chat.
            max_tokens:  Maximum response length.

        Returns:
            Response string, or empty string on failure.
        """
        payload = self._build_payload(
            prompt=prompt,
            model=model or self.model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        try:
            with Timer("LLM generate"):
                r = self._session.post(
                    _OLLAMA_API_GENERATE.format(base=self.base_url),
                    json=payload,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()
                return data.get("response", "").strip()

        except requests.exceptions.Timeout:
            logger.error(f"LLM timeout after {self.timeout}s")
            return ""
        except requests.exceptions.RequestException as exc:
            logger.error(f"LLM request failed: {exc}")
            return ""
        except (KeyError, json.JSONDecodeError) as exc:
            logger.error(f"LLM response parse error: {exc}")
            return ""

    # ── Streaming Generation ───────────────────────────────────

    def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.75,
        max_tokens: int = 300,
    ) -> Generator[str, None, None]:
        """
        Stream the response token by token.
        Yields string fragments as they arrive.

        Usage:
            full = ""
            for token in engine.stream("Tell me about stars"):
                print(token, end="", flush=True)
                full += token
        """
        payload = self._build_payload(
            prompt=prompt,
            model=model or self.model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        try:
            with self._session.post(
                _OLLAMA_API_GENERATE.format(base=self.base_url),
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.Timeout:
            logger.error("LLM stream timeout")
        except requests.exceptions.RequestException as exc:
            logger.error(f"LLM stream error: {exc}")

    # ── Chat-format Generation (multi-turn aware) ──────────────

    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.75,
        max_tokens: int = 300,
    ) -> str:
        """
        Send a list of {role, content} messages (OpenAI-compatible format).
        Better for maintaining conversation context than raw prompts.

        Args:
            messages: [{"role": "system", "content": "..."}, {"role": "user", ...}]
        """
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
            },
        }

        try:
            with Timer("LLM chat"):
                r = self._session.post(
                    _OLLAMA_API_CHAT.format(base=self.base_url),
                    json=payload,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()
                return data["message"]["content"].strip()

        except Exception as exc:
            logger.error(f"LLM chat failed: {exc}")
            return ""

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _build_payload(
        prompt: str,
        model: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict:
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "repeat_penalty": 1.1,   # Reduces repetitive output
                "stop": ["\nUser:", "\nHuman:", "\n###"],
            },
        }
        if system:
            payload["system"] = system
        return payload

    def __repr__(self) -> str:
        return f"LLMEngine(model={self.model}, url={self.base_url})"

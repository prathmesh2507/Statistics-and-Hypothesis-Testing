"""
speech/expressive_speech.py
────────────────────────────
Text preprocessing pipeline for human-like TTS output.

Responsibilities:
  1. Language detection  — Devanagari vs Roman vs Hinglish
  2. Sentence chunking   — split at natural breath boundaries
  3. Pronunciation fixes — Hinglish phoneme normalization
  4. Pause injection     — silence at commas, ellipses, dashes
  5. Text sanitization   — strip markdown, code, URLs before synthesis

Why this matters:
  Without preprocessing, XTTS-v2 receives a wall of mixed-script text
  and guesses language = wrong → horrible accent on Hindi words.
  This layer segments and routes each portion to the right language code.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Data Structures ────────────────────────────────────────────

@dataclass
class SpeechSegment:
    """
    A chunk of text ready for synthesis.
    One segment = one TTS call with consistent language.
    """
    text: str
    language: str           # "en" or "hi"
    pause_after_ms: int = 0 # silence gap AFTER this segment plays


@dataclass
class ProcessedSpeech:
    """Full preprocessed result — list of segments to synthesize in order."""
    segments: list[SpeechSegment] = field(default_factory=list)

    @property
    def total_text(self) -> str:
        return " ".join(s.text for s in self.segments)

    @property
    def is_empty(self) -> bool:
        return all(not s.text.strip() for s in self.segments)


# ── Devanagari Detection ───────────────────────────────────────

_DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F\u0980-\u09FF]+")

def _contains_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI_RANGE.search(text))

def _is_mostly_devanagari(text: str) -> bool:
    total = len(text.strip())
    if total == 0:
        return False
    deva_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    return deva_chars / total > 0.4


# ── Hinglish Pronunciation Dictionary ─────────────────────────
#
# XTTS-v2 with language="en" reads Roman characters phonetically.
# Most Romanized Hindi is already readable (yaar, bhai, kal).
# This table fixes the edge cases that sound wrong in English mode.
#
# Format: {"original": "phonetic_replacement"}
# These substitutions run BEFORE sending text to XTTS.

_HINGLISH_PRONUNCIATION = {
    # Common verbs / endings
    r"\bhai\b":   "hay",       # "है" romanized — English reads as "hye"
    r"\bhain\b":  "hain",      # plural "हैं" — already OK but clarify
    r"\bho\b":    "hoh",
    r"\bkya\b":   "kyaa",      # "क्या" — what
    r"\bnahi\b":  "nuhee",
    r"\bnahin\b": "nuheen",
    r"\bacha\b":  "uchha",     # "अच्छा" — good/okay
    r"\baur\b":   "owr",       # "और" — and
    r"\bbas\b":   "bus",       # "बस" — enough/just
    r"\bphir\b":  "phir",      # already OK
    r"\babi\b":   "ubhee",
    r"\babhi\b":  "ubhee",     # "अभी" — right now
    r"\bthik\b":  "theek",
    r"\btheek\b": "theek",     # "ठीक" — fine/okay
    r"\bkaro\b":  "kurro",
    r"\bkaro\b":  "kurro",
    r"\bsahi\b":  "suhee",     # "सही" — correct/right
    r"\bbaat\b":  "baat",      # already OK
    r"\bkuch\b":  "kuch",      # already OK
    r"\bwoh\b":   "woh",       # already OK
    r"\bkoi\b":   "koee",      # "कोई" — someone/any
    r"\bkab\b":   "kub",       # "कब" — when
    r"\bkahan\b": "kuhaan",    # "कहाँ" — where
    r"\bhoga\b":  "hogaa",
    r"\bhogi\b":  "hogee",
    r"\blaga\b":  "lugaa",
    r"\blagta\b": "lugta",     # "लगता" — feels like
    r"\bmujhe\b": "mujhey",    # "मुझे" — to me
    r"\btujhe\b": "tujhey",
    r"\bapna\b":  "upnaa",     # "अपना" — own
    r"\bmat\b":   "mut",       # "मत" — don't
    r"\bchalo\b": "chulo",     # "चलो" — let's go
    # Casual / slang
    r"\byaar\b":  "yaar",      # already natural
    r"\bbhai\b":  "bhay",      # "भाई" — bro
    r"\bbc\b":    "bay say",   # cleaned expletive abbreviation
    r"\bre\b":    "ray",       # vocative particle
    r"\bare\b":   "uray",      # "अरे" — hey!
}

def _normalize_hinglish(text: str) -> str:
    """Apply pronunciation normalization for Romanized Hindi words."""
    result = text
    for pattern, replacement in _HINGLISH_PRONUNCIATION.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


# ── Pause Markers ──────────────────────────────────────────────

# Maps punctuation patterns to pause duration in milliseconds
_PAUSE_RULES: list[tuple[str, int]] = [
    (r"\.{3}",       400),   # ellipsis — long thinking pause
    (r"\?",          300),   # question — let it land
    (r"—",           250),   # em dash — dramatic pause
    (r"–",           200),   # en dash
    (r",",           120),   # comma — breath pause
    (r";",           180),   # semicolon
    (r"\.\s",        250),   # sentence end
    (r"!",           200),   # exclamation
]

def _get_pause_after(text: str) -> int:
    """Return milliseconds of silence to insert after this segment."""
    stripped = text.rstrip()
    for pattern, ms in _PAUSE_RULES:
        if re.search(pattern + r"\s*$", stripped):
            return ms
    return 80  # default inter-segment gap


# ── Text Sanitization ──────────────────────────────────────────

def sanitize_for_tts(text: str) -> str:
    """
    Strip content that TTS should not read aloud.
    Handles markdown, code, URLs, emojis, excessive punctuation.
    """
    # Code blocks
    text = re.sub(r"```[\s\S]*?```", " code block. ", text)
    text = re.sub(r"`[^`]+`", "", text)

    # Markdown
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)

    # URLs
    text = re.sub(r"https?://\S+", "link", text)

    # Emojis and special symbols
    text = re.sub(r"[^\x00-\x7F\u0900-\u097F\u0980-\u09FF]", " ", text)

    # Multiple exclamations / question marks
    text = re.sub(r"[!?]{2,}", lambda m: m.group()[0], text)

    # Collapse whitespace and newlines
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


# ── Sentence Splitter ──────────────────────────────────────────

_SENTENCE_END = re.compile(
    r"(?<=[.!?…])\s+"          # After .!?… followed by space
    r"|(?<=।)\s+"              # After Hindi danda
    r"|(?<=[,;—–])\s+"         # After clause markers
)

def _split_sentences(text: str, max_chars: int = 180) -> list[str]:
    """
    Split text into natural speech-sized chunks.

    Rules:
    - Prefer splitting at sentence-ending punctuation
    - Never split mid-word
    - Keep chunks ≤ max_chars (XTTS quality degrades on very long inputs)
    - Keep chunks ≥ 3 words (too-short chunks sound choppy)
    """
    if len(text) <= max_chars:
        return [text]

    # First pass: split at sentence boundaries
    parts = _SENTENCE_END.split(text)

    # Second pass: merge very short fragments, split very long ones
    result = []
    buffer = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if len(buffer) + len(part) < max_chars:
            buffer = (buffer + " " + part).strip() if buffer else part
        else:
            if buffer:
                result.append(buffer)
            # If part itself is too long, split at word boundary
            if len(part) > max_chars:
                words = part.split()
                chunk = ""
                for w in words:
                    if len(chunk) + len(w) + 1 < max_chars:
                        chunk = (chunk + " " + w).strip() if chunk else w
                    else:
                        if chunk:
                            result.append(chunk)
                        chunk = w
                if chunk:
                    buffer = chunk
                else:
                    buffer = ""
            else:
                buffer = part

    if buffer:
        result.append(buffer)

    return [r for r in result if r.strip()]


# ── Script Segmentation ────────────────────────────────────────

def _segment_by_script(text: str) -> list[tuple[str, str]]:
    """
    Split text into (chunk, language_code) pairs based on character script.

    Devanagari characters → language="hi"
    Everything else        → language="en"

    Returns list of (text, lang) tuples in order.
    """
    segments: list[tuple[str, str]] = []
    current_text = ""
    current_lang = "en"

    for char in text:
        if "\u0900" <= char <= "\u097F":  # Devanagari
            char_lang = "hi"
        elif "\u0980" <= char <= "\u09FF":  # Bengali (in case)
            char_lang = "hi"
        else:
            char_lang = "en"

        if char_lang == current_lang:
            current_text += char
        else:
            if current_text.strip():
                segments.append((current_text.strip(), current_lang))
            current_text = char
            current_lang = char_lang

    if current_text.strip():
        segments.append((current_text.strip(), current_lang))

    # Merge very short segments into neighbors
    merged: list[tuple[str, str]] = []
    for text_chunk, lang in segments:
        word_count = len(text_chunk.split())
        if merged and word_count <= 2:
            # Attach short segment to previous
            prev_text, prev_lang = merged[-1]
            merged[-1] = (prev_text + " " + text_chunk, prev_lang)
        else:
            merged.append((text_chunk, lang))

    return merged


# ── Main Preprocessor ──────────────────────────────────────────

class ExpressiveSpeechProcessor:
    """
    Transforms raw LLM output into a list of synthesis-ready SpeechSegments.

    Usage:
        processor = ExpressiveSpeechProcessor()
        result = processor.process("yaar kal exam hai, I'm stressed.")
        for seg in result.segments:
            audio = tts.synthesize(seg.text, language=seg.language)
            time.sleep(seg.pause_after_ms / 1000)
    """

    def __init__(
        self,
        max_chunk_chars: int = 180,
        normalize_hinglish: bool = True,
        default_language: str = "en",
    ):
        self.max_chunk_chars = max_chunk_chars
        self.normalize_hinglish = normalize_hinglish
        self.default_language = default_language

    def process(self, text: str) -> ProcessedSpeech:
        """
        Full preprocessing pipeline.

        1. Sanitize (strip markdown etc.)
        2. Detect scripts and segment by language
        3. For each language segment: split into TTS-sized chunks
        4. Apply Hinglish pronunciation normalization
        5. Compute pause durations
        """
        if not text or not text.strip():
            return ProcessedSpeech()

        # Step 1: Sanitize
        clean = sanitize_for_tts(text)
        if not clean:
            return ProcessedSpeech()

        logger.debug(f"Processing: '{clean[:80]}...'")

        # Step 2: Segment by script (Devanagari vs Roman)
        script_segments = _segment_by_script(clean)

        # Step 3–5: Process each script segment
        result = ProcessedSpeech()

        for raw_chunk, lang in script_segments:
            # Normalize Hinglish pronunciation for Roman text
            if lang == "en" and self.normalize_hinglish:
                raw_chunk = _normalize_hinglish(raw_chunk)

            # Split into TTS-friendly sentence chunks
            sentences = _split_sentences(raw_chunk, self.max_chunk_chars)

            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue

                is_last = i == len(sentences) - 1
                pause = _get_pause_after(sentence) if not is_last else _get_pause_after(sentence)

                result.segments.append(SpeechSegment(
                    text=sentence,
                    language=lang,
                    pause_after_ms=pause,
                ))

        logger.debug(f"Produced {len(result.segments)} speech segments")
        return result

    def quick_clean(self, text: str) -> str:
        """Fast single-call version — returns clean string only, no segmentation."""
        return sanitize_for_tts(text)

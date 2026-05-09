# EVA Voice Core

> **Your sharp, warm, no-BS AI companion — powered entirely locally.**

EVA is a production-style conversational voice assistant that understands natural speech, handles Hinglish, and responds like a smart friend — not a corporate bot.

**100% local. No cloud APIs. No subscriptions. Your data stays on your machine.**

---

## Architecture Overview

```
User Speaks
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  VoiceListener (sounddevice + Silero VAD)           │
│  • Continuous mic capture in 32ms chunks            │
│  • VAD gates audio — only speech passes through     │
│  • Pre-speech buffer prevents onset clipping        │
│  • Silence detection ends utterance cleanly         │
└──────────────────────┬──────────────────────────────┘
                       │  float32 audio @ 16kHz
                       ▼
┌─────────────────────────────────────────────────────┐
│  Transcriber (Faster-Whisper small, CUDA/float16)   │
│  • Anti-hallucination quality filters               │
│  • Auto language detection (handles Hinglish)       │
│  • Post-processing removes noise artifacts          │
└──────────────────────┬──────────────────────────────┘
                       │  clean text
                       ▼
┌─────────────────────────────────────────────────────┐
│  ResponseGenerator (Qwen2.5:3b via Ollama)          │
│  • Rolling 12-turn conversation context             │
│  • EVA personality system prompt                    │
│  • Fallback handling                                │
└──────────────────────┬──────────────────────────────┘
                       │  response text
                       ▼
┌─────────────────────────────────────────────────────┐
│  TTSEngine (pyttsx3 → Piper in Phase 2)             │
│  • Sanitizes markdown/code before speaking          │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Voice Capture | sounddevice | Low-latency mic I/O |
| Voice Activity Detection | Silero VAD | Neural, accurate, fast |
| Speech-to-Text | Faster-Whisper (small) | CUDA float16, anti-hallucination |
| LLM | Qwen2.5:3b via Ollama | Fully local |
| TTS Phase 1 | pyttsx3 | Offline, cross-platform |
| TTS Phase 2 | Piper TTS | Natural voice (planned) |
| Memory Phase 2 | MongoDB + ChromaDB | Persistent + semantic (planned) |

---

## Prerequisites

### Hardware
- **GPU**: NVIDIA GPU with 4GB+ VRAM (for CUDA whisper)
- **CPU fallback**: Slower but works (auto-detected)
- **RAM**: 8GB+ recommended
- **Microphone**: Any USB or built-in mic

### Software
- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- CUDA 11.8+ (for GPU acceleration)

---

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/EVA-Voice-Core.git
cd EVA-Voice-Core
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# OR
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Ollama and pull the model
```bash
# Install Ollama: https://ollama.ai/download
ollama pull qwen2.5:3b
ollama serve                    # Start Ollama server
```

### 5. Configure environment
```bash
cp .env.example .env
# Edit .env to match your setup
```

Key settings in `.env`:
```env
WHISPER_DEVICE=cuda          # Change to cpu if no GPU
WHISPER_COMPUTE_TYPE=float16 # Change to int8 for CPU
VAD_THRESHOLD=0.45           # Raise if picking up too much noise
SILENCE_DURATION=1.2         # Seconds of silence to end utterance
```

### 6. Run EVA
```bash
python main.py
```

---

## Configuration Guide

### VAD Tuning
| Setting | Effect | When to adjust |
|---------|--------|----------------|
| `VAD_THRESHOLD` | 0.45 (default) | Raise (→0.6) if noisy room; lower (→0.3) for quiet environments |
| `SILENCE_DURATION` | 1.2s | Raise if EVA cuts off mid-sentence; lower for faster response |
| `PRE_SPEECH_BUFFER_MS` | 300ms | Raise if first syllable is missing |

### Whisper Tuning
| Setting | Effect |
|---------|--------|
| `WHISPER_MODEL_SIZE` | `small` (default), `medium` (better Hinglish), `large-v3` (best quality) |
| `WHISPER_DEVICE` | `cuda` or `cpu` |
| `WHISPER_LANGUAGE` | `None` (auto-detect), `en`, `hi` |

---

## Project Structure

```
EVA-Voice-Core/
├── main.py                     # Entry point — conversation loop
├── requirements.txt
├── .env.example                # Config template
│
├── config/
│   ├── settings.py             # Typed settings from .env
│   └── prompts.py              # EVA personality + all prompts
│
├── speech/
│   ├── listener.py             # Mic capture + VAD state machine ← KEY
│   ├── vad.py                  # Silero VAD wrapper
│   ├── transcriber.py          # Faster-Whisper with anti-hallucination
│   ├── tts.py                  # pyttsx3 (Piper-ready interface)
│   └── audio_utils.py          # Normalization, noise gate, resampling
│
├── brain/
│   ├── llm_engine.py           # Ollama HTTP client + streaming
│   ├── response_generator.py   # Context + LLM orchestration
│   ├── personality.py          # EVA character config
│   └── context_manager.py      # Rolling conversation window
│
├── memory/
│   ├── conversation_store.py   # JSON session persistence
│   ├── mongodb.py              # Phase 2 stub
│   └── semantic_memory.py      # Phase 2 stub (ChromaDB)
│
├── utils/
│   ├── logger.py               # Rich terminal + rotating file log
│   └── helpers.py              # Text cleaning, hallucination filter
│
├── models/                     # Whisper model cache (auto-downloaded)
├── logs/                       # Rotating log files
├── data/                       # Sessions, ChromaDB (Phase 2)
└── tests/
```

---

## Phase Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Current | Voice core: mic → VAD → Whisper → Ollama → TTS |
| **Phase 2** | 🔜 Next | Memory: MongoDB sessions + ChromaDB semantic recall |
| **Phase 3** | 📋 Planned | Piper TTS for natural voice |
| **Phase 4** | 📋 Planned | System automation hooks |
| **Phase 5** | 📋 Planned | Browser and app control |

---

## Common Issues

**"Cannot connect to Ollama"**
```bash
# Make sure Ollama is running:
ollama serve
```

**VAD cutting speech too early**
```env
# In .env — increase silence window:
SILENCE_DURATION=1.8
```

**Whisper hallucinating on silence**
```env
# Raise VAD threshold to filter more aggressively:
VAD_THRESHOLD=0.55
```

**CUDA out of memory**
```env
# Use CPU instead:
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

**Missing first syllable**
```env
PRE_SPEECH_BUFFER_MS=400
```

---

## License
MIT

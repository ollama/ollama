# Whisper.cpp Integration for Ollama

This directory contains the integration of [whisper.cpp](https://github.com/ggerganov/whisper.cpp) into Ollama, providing speech-to-text capabilities alongside the existing LLM functionality.

## Architecture Overview

The integration follows the same pattern as llama.cpp:

```
ollama/
├── whisper/              # Go bindings for whisper.cpp (this directory)
│   ├── whisper.go        # CGo bindings
│   └── whisper.cpp/      # whisper.cpp source (integrated, not external)
├── stt/                  # Speech-to-text server
│   ├── server.go         # WhisperServer interface + TranscribeRequest
│   └── client.go         # HTTP client for subprocess mode
├── api/
│   └── types.go          # API types (TranscribeRequest/Response)
├── server/
│   ├── routes.go         # HTTP handlers (TranscribeHandler, etc.)
│   └── sched.go          # Scheduler with Whisper model management
└── cmd/
    └── cmd.go            # CLI 'transcribe' command
```

## Features

- **Audio Transcription**: Convert speech to text
- **Translation**: Translate any language to English
- **Multiple Formats**: Output as text, SRT, VTT, or JSON
- **Streaming**: Real-time transcription support
- **GPU Acceleration**: CUDA, Metal, and Vulkan support
- **Language Detection**: Automatic source language detection
- **Scheduler Integration**: Model caching and keep-alive like LLM models
- **99 Languages**: Full multilingual support

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/transcribe` | Transcribe audio to text |
| `POST /api/translate` | Translate audio to English |
| `POST /api/detect-language` | Detect spoken language |
| `GET /api/languages` | List supported languages |
| `POST /v1/audio/transcriptions` | OpenAI-compatible transcription |
| `POST /v1/audio/translations` | OpenAI-compatible translation |

### POST /api/transcribe

Transcribe audio to text.

**Request (JSON):**
```json
{
  "model": "whisper:base",
  "audio": "<base64_encoded_audio>",
  "language": "en",
  "response_format": "json",
  "temperature": 0.0,
  "prompt": "Optional context",
  "options": {
    "sampling_strategy": "beam_search",
    "beam_size": 5,
    "num_threads": 4
  }
}
```

**Request (multipart/form-data):**
```bash
curl -X POST http://localhost:11434/api/transcribe \
  -F "file=@audio.wav" \
  -F "model=whisper:base" \
  -F "language=en"
```

**Response:**
```json
{
  "model": "whisper:base",
  "text": "Hello, world!",
  "language": "en",
  "task": "transcribe",
  "duration": 2500000000,
  "segments": [
    {
      "start": 0.5,
      "end": 2.5,
      "text": "Hello, world!"
    }
  ],
  "done": true
}
```

### Response Formats

| Format | Content-Type | Description |
|--------|--------------|-------------|
| `json` | application/json | Default, text only |
| `verbose_json` | application/json | With segments and timestamps |
| `text` | text/plain | Plain text only |
| `srt` | text/plain | SubRip subtitle format |
| `vtt` | text/vtt | WebVTT subtitle format |

### POST /api/translate

Same as `/api/transcribe` but translates to English.

### POST /api/detect-language

```json
{
  "model": "whisper:base",
  "audio": "<base64_encoded_audio>"
}
```

**Response:**
```json
{
  "language": "en",
  "probability": 0.978
}
```

### GET /api/languages

Returns list of 99 supported languages.

## Advanced Options

The `options` field supports these parameters:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `sampling_strategy` | string | `"greedy"` | `"greedy"` or `"beam_search"` |
| `beam_size` | int | 5 | Beam search width |
| `best_of` | int | 1 | Candidates for greedy sampling |
| `num_threads` | int | auto | CPU threads |
| `suppress_blank` | bool | true | Suppress blank tokens |
| `suppress_non_speech` | bool | true | Suppress non-speech tokens |
| `no_context` | bool | false | Don't use previous context |
| `single_segment` | bool | false | Force single segment |
| `max_segment_length` | int | 0 | Max chars per segment |
| `max_tokens_per_segment` | int | 0 | Max tokens per segment |
| `split_on_word` | bool | false | Split segments on word boundaries |
| `speaker_diarization` | bool | false | Enable speaker detection |
| `entropy_threshold` | float | 2.4 | Entropy threshold for fallback |
| `logprob_threshold` | float | -1.0 | Log probability threshold |
| `no_speech_threshold` | float | 0.6 | No-speech probability threshold |
| `temperature_increment` | float | 0.2 | Temperature increment on fallback |
| `word_timestamp_threshold` | float | 0.01 | Word timestamp threshold |

## Supported Models

| Model | Parameters | Size | Languages |
|-------|-----------|------|-----------|
| whisper:tiny | 39M | ~75MB | Multi |
| whisper:tiny.en | 39M | ~75MB | English |
| whisper:base | 74M | ~140MB | Multi |
| whisper:base.en | 74M | ~140MB | English |
| whisper:small | 244M | ~460MB | Multi |
| whisper:small.en | 244M | ~460MB | English |
| whisper:medium | 769M | ~1.5GB | Multi |
| whisper:medium.en | 769M | ~1.5GB | English |
| whisper:large-v3 | 1550M | ~3GB | Multi |
| whisper:large-v3-turbo | 809M | ~1.6GB | Multi |

## CLI Commands

### ollama transcribe

```bash
# Basic transcription
ollama transcribe audio.wav

# With specific model
ollama transcribe audio.mp3 --model whisper:large-v3

# Generate subtitles
ollama transcribe video.wav --format srt --output subtitles.srt

# Translate to English
ollama transcribe japanese_audio.wav --translate

# Verbose output with timestamps
ollama transcribe audio.wav --format verbose_json
```

## Building

### Prerequisites

1. Go 1.21+
2. C/C++ compiler (gcc/clang/MinGW)
3. CMake 3.14+
4. CUDA Toolkit (optional, for GPU support)

### Windows Build (MinGW)

```powershell
# Install prerequisites
winget install GoLang.Go
winget install MSYS2.MSYS2

# In MSYS2 UCRT64 terminal, install GCC
pacman -S mingw-w64-ucrt-x86_64-gcc

# Build Ollama (whisper.cpp is compiled automatically via CGo)
cd ollama
$env:Path = "C:\msys64\ucrt64\bin;" + $env:Path
$env:CGO_ENABLED = "1"
go build -o ollama.exe .
```

### Linux/macOS Build

```bash
# whisper.cpp is already included in ollama/whisper/whisper.cpp
# Just build Ollama directly
cd ollama
CGO_ENABLED=1 go build .
```

The build system automatically compiles `whisper.cpp` along with the rest of Ollama.

## Audio Format Support

Supported input formats:
- **WAV** (recommended, native support)
- MP3 (requires ffmpeg)
- FLAC (requires ffmpeg)
- OGG/Vorbis (requires ffmpeg)
- M4A/AAC (requires ffmpeg)
- WebM (requires ffmpeg)

Audio is automatically:
- Resampled to 16kHz
- Converted to mono
- Normalized to float32

## Memory Requirements

| Model | RAM (CPU) | VRAM (GPU) |
|-------|----------|------------|
| tiny | 200MB | 100MB |
| base | 300MB | 200MB |
| small | 600MB | 500MB |
| medium | 2GB | 1.5GB |
| large | 4GB | 3GB |

## Scheduler Integration

Whisper models are managed by Ollama's scheduler:

- **Caching**: Models stay loaded based on `keep_alive`
- **Reference counting**: Shared across concurrent requests
- **Auto-unload**: Idle models are automatically freed
- **Resource management**: VRAM/RAM tracked per model

```json
{
  "model": "whisper:base",
  "keep_alive": "10m"
}
```

## Performance Tips

1. **Use GPU**: Enable GPU acceleration for 5-10x faster transcription
2. **Choose Right Model**: Smaller models (tiny, base) for real-time, larger for accuracy
3. **English-Only Models**: Use `.en` variants for English-only content (faster)
4. **Beam Search**: Use for better accuracy (slower)
5. **Greedy Sampling**: Use for faster transcription (default)

## Language Codes

Whisper supports 99 languages. Common codes:
- `en` - English
- `zh` - Chinese
- `de` - German
- `es` - Spanish
- `fr` - French
- `ja` - Japanese
- `ko` - Korean
- `ru` - Russian

Use empty string `""` for auto-detection.

## Integration with LLM

Combine transcription with LLM for powerful workflows:

```bash
# Transcribe and summarize
ollama transcribe meeting.wav | ollama run llama3 "Summarize this meeting transcript:"

# Voice commands
ollama listen | ollama run llama3 --system "You are a voice assistant"
```

## Troubleshooting

### "ffmpeg not found"
Install ffmpeg for non-WAV audio support:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
winget install ffmpeg
```

### "CUDA out of memory"
Use a smaller model:
```bash
ollama transcribe audio.wav --model whisper:small
```

### "Model not found"
Pull the whisper model first or use direct path:
```bash
ollama pull whisper:base
# or
curl -X POST http://localhost:11434/api/transcribe \
  -d '{"model": "/path/to/ggml-base.bin", "audio": "..."}'
```

## License

whisper.cpp is licensed under the MIT License.
Ollama is licensed under the MIT License.

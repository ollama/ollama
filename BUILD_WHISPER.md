# Building Ollama with Whisper Support

## Architecture

```
ollama/
├── api/types.go           # TranscribeRequest/Response API types
├── server/routes.go       # HTTP handlers (TranscribeHandler, etc.)
├── server/sched.go        # Scheduler with Whisper model management
├── stt/server.go          # WhisperServer interface
├── stt/client.go          # HTTP client for subprocess mode
└── whisper/whisper.go     # CGo bindings to whisper.cpp
```

## Quick Start

> **Note**: `whisper.cpp` is integrated directly in `ollama/whisper/whisper.cpp`.
> No need to clone it separately - it's built automatically with Ollama.

### Windows (MinGW)

```powershell
# 1. Install prerequisites
winget install GoLang.Go
winget install MSYS2.MSYS2

# 2. Install GCC in MSYS2 UCRT64 terminal
pacman -S mingw-w64-ucrt-x86_64-gcc

# 3. Build Ollama (whisper.cpp is compiled automatically via CGo)
cd ollama
$env:Path = "C:\msys64\ucrt64\bin;" + $env:Path
$env:CGO_ENABLED = "1"
go build -o ollama.exe .
```

### Linux

```bash
# 1. Install prerequisites
sudo apt install golang gcc g++

# 2. Build Ollama (whisper.cpp is compiled automatically via CGo)
cd ollama
CGO_ENABLED=1 go build .
```

### macOS

```bash
# 1. Install prerequisites (Xcode command line tools include clang)
brew install go

# 2. Build Ollama (whisper.cpp is compiled automatically via CGo)
cd ollama
CGO_ENABLED=1 go build .
```

## Testing

```bash
# Start server
./ollama serve

# Test transcription (PowerShell)
$audio = [Convert]::ToBase64String([IO.File]::ReadAllBytes("audio.wav"))
$body = @{ model = "whisper:base"; audio = $audio } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/transcribe" -Method POST -Body $body -ContentType "application/json"

# Test with curl
curl -X POST http://localhost:11434/api/transcribe \
  -F "file=@audio.wav" \
  -F "model=whisper:base"
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/transcribe` | Transcribe audio |
| `POST /api/translate` | Translate to English |
| `POST /api/detect-language` | Detect language |
| `GET /api/languages` | List 99 supported languages |
| `POST /v1/audio/transcriptions` | OpenAI-compatible |
| `POST /v1/audio/translations` | OpenAI-compatible |

## Request Format

```json
{
  "model": "whisper:base",
  "audio": "<base64>",
  "language": "en",
  "response_format": "verbose_json",
  "temperature": 0.0,
  "prompt": "Context hint",
  "timestamp_granularity": "segment",
  "options": {
    "sampling_strategy": "beam_search",
    "beam_size": 5,
    "num_threads": 4,
    "suppress_blank": true,
    "suppress_non_speech": true
  }
}
```

## Response Formats

- `json` - Text only
- `verbose_json` - With segments
- `text` - Plain text
- `srt` - SubRip subtitles
- `vtt` - WebVTT subtitles

## Advanced Options

| Option | Type | Description |
|--------|------|-------------|
| `sampling_strategy` | string | `"greedy"` or `"beam_search"` |
| `beam_size` | int | Beam width (default: 5) |
| `best_of` | int | Greedy candidates |
| `num_threads` | int | CPU threads |
| `suppress_blank` | bool | Suppress blanks |
| `suppress_non_speech` | bool | Suppress non-speech |
| `no_context` | bool | Ignore context |
| `single_segment` | bool | Force single segment |
| `entropy_threshold` | float | Fallback threshold |
| `no_speech_threshold` | float | Silence threshold |

## Available Models

| Model | Size | Notes |
|-------|------|-------|
| `whisper:tiny` | 75 MB | Fastest |
| `whisper:base` | 150 MB | Good balance |
| `whisper:small` | 500 MB | Better accuracy |
| `whisper:medium` | 1.5 GB | High accuracy |
| `whisper:large-v3` | 3 GB | Best quality |
| `whisper:large-v3-turbo` | 2 GB | Fast + accurate |

## Troubleshooting

### "ffmpeg not found"
Install ffmpeg for non-WAV audio:
```bash
winget install ffmpeg  # Windows
brew install ffmpeg    # macOS
apt install ffmpeg     # Linux
```

### Model not loading
Use direct file path or pull the model:
```bash
ollama pull whisper:base
```

# Ollama Autotune

Automatic performance optimization for Ollama based on hardware detection.

## Overview

Autotune detects your hardware (CPU, GPU, RAM, VRAM) and automatically configures Ollama 
for optimal performance. It sets environment variables like `OLLAMA_FLASH_ATTENTION`, 
`OLLAMA_KV_CACHE_TYPE`, `OLLAMA_NUM_PARALLEL`, `OLLAMA_CONTEXT_LENGTH`, and others based 
on your system capabilities.

## Performance Profiles

| Profile      | Description |
|-------------|-------------|
| `speed`     | Maximize token generation speed (single-user, high VRAM usage) |
| `balanced`  | Good tradeoff between speed and resource usage (default) |
| `memory`    | Minimize memory usage for constrained hardware |
| `multiuser` | Optimize for concurrent users with shared cache |
| `max`       | Squeeze maximum performance at the cost of resources |

## Usage

### Environment Variable (before server start)

```bash
OLLAMA_PERFORMANCE_PROFILE=speed ollama serve
```

### Server Flag

```bash
ollama serve --profile speed
```

### CLI (runtime, requires running server)

```bash
# Show current autotune status
ollama autotune

# List available profiles
ollama autotune profiles

# Apply a profile at runtime
ollama autotune speed
```

### API Endpoints

```bash
# Get current status
GET /api/autotune

# Apply a profile
POST /api/autotune
{"profile": "speed"}

# List profiles
GET /api/autotune/profiles
```

## How It Works

1. **Hardware Detection** — Uses Ollama's native `discover` and `ml` packages to detect CPU model, GPU count/VRAM, total RAM, and Flash Attention capability.

2. **Profile Resolution** — Each profile defines base settings. The tuner then adjusts values dynamically based on actual hardware (e.g., VRAM-based context length, GPU count-based parallelism).

3. **Safe Application** — Settings are only applied if the user has NOT already set the corresponding environment variable, preserving manual overrides.

## Architecture

```
autotune/
├── profile.go    # Profile definitions and configurations
├── hardware.go   # Hardware detection wrapper
├── tuner.go      # Core auto-tuning engine
├── apply.go      # Apply settings to environment
├── api.go        # HTTP API handlers
└── autotune_test.go
```

## Integration Points

- `envconfig/config.go` — `OLLAMA_PERFORMANCE_PROFILE` variable
- `server/routes.go` — Startup auto-tune and API routes
- `cmd/cmd.go` — `ollama autotune` CLI command and `--profile` flag on `serve`

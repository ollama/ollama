# Image Generation in Ollama (Experimental)

Generate images from text prompts using local AI models.

## Quick Start

```bash
# Run with a prompt
ollama run z-image "a sunset over mountains"
Generating: step 30/30
Image saved to: /tmp/ollama-image-1704067200.png
```

On macOS, the generated image will automatically open in Preview.

## Supported Models

| Model | VRAM Required | Notes |
|-------|---------------|-------|
| z-image | ~12GB | Based on Flux architecture |

## CLI Usage

```bash
# Generate an image
ollama run z-image "a cat playing piano"

# Check if model is running
ollama ps

# Stop the model
ollama stop z-image
```

## API

### OpenAI-Compatible Endpoint

```bash
POST /v1/images/generations
```

**Request:**
```json
{
  "model": "z-image",
  "prompt": "a sunset over mountains",
  "size": "1024x1024",
  "response_format": "b64_json"
}
```

**Response:**
```json
{
  "created": 1704067200,
  "data": [
    {
      "b64_json": "iVBORw0KGgo..."
    }
  ]
}
```

### Example: cURL

```bash
curl http://localhost:11434/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "z-image",
    "prompt": "a white cat",
    "size": "1024x1024"
  }'
```

### Example: Save to File

```bash
curl -s http://localhost:11434/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "z-image",
    "prompt": "a white cat",
    "size": "1024x1024"
  }' | jq -r '.data[0].b64_json' | base64 -d > image.png
```

### Streaming Progress

Enable streaming to receive progress updates via SSE:

```bash
curl http://localhost:11434/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "z-image", "prompt": "a sunset", "stream": true}'
```

Events:
```
event: progress
data: {"step": 1, "total": 30}

event: progress
data: {"step": 2, "total": 30}
...

event: done
data: {"created": 1704067200, "data": [{"b64_json": "..."}]}
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | string | required | Model name |
| prompt | string | required | Text description of image |
| size | string | "1024x1024" | Image dimensions (WxH) |
| n | int | 1 | Number of images (currently only 1 supported) |
| response_format | string | "b64_json" | "b64_json" or "url" |
| stream | bool | false | Enable progress streaming |

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Sufficient VRAM (see model table above)
- Ollama built with MLX support

## Limitations

- macOS only (uses MLX backend)
- Single image per request
- Fixed step count (30 steps)

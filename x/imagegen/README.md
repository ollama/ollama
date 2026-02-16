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
- CUDA: tested on CUDA 12 Blackwell, more testing coming soon
- Sufficient VRAM (see model table above)
- Ollama built with MLX support

## Limitations

- macOS only (uses MLX backend)
- Single image per request
- Fixed step count (30 steps)
- Modelfiles not yet supported (use `ollama create` from model directory)

---

# Tensor Model Storage Format

Tensor models store each tensor as a separate blob with metadata in the manifest. This enables faster downloads (parallel fetching) and deduplication (shared tensors are stored once).

## Manifest Structure

The manifest follows the standard ollama format with tensor-specific layer metadata:

```json
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
  "config": { "digest": "sha256:...", "size": 1234 },
  "layers": [
    {
      "mediaType": "application/vnd.ollama.image.tensor",
      "digest": "sha256:25b36eed...",
      "size": 49807448,
      "name": "text_encoder/model.layers.0.mlp.down_proj.weight",
      "dtype": "BF16",
      "shape": [2560, 9728]
    },
    {
      "mediaType": "application/vnd.ollama.image.json",
      "digest": "sha256:abc123...",
      "size": 512,
      "name": "text_encoder/config.json"
    }
  ]
}
```

Each tensor layer includes:
- `name`: Path-style tensor name (e.g., `text_encoder/model.layers.0.mlp.down_proj.weight`)
- `dtype`: Data type (BF16, F32, etc.)
- `shape`: Tensor dimensions

Config layers use the same path-style naming (e.g., `tokenizer/tokenizer.json`).

## Blob Format

Each tensor blob is a minimal safetensors file:

```
[8 bytes: header size (uint64 LE)]
[~80 bytes: JSON header, padded to 8-byte alignment]
[N bytes: raw tensor data]
```

Header contains a single tensor named `"data"`:

```json
{"data":{"dtype":"BF16","shape":[2560,9728],"data_offsets":[0,49807360]}}
```

## Why Include the Header?

The ~88 byte safetensors header enables MLX's native `mlx_load_safetensors` function, which:

1. **Uses mmap** - Maps file directly into memory, no copies
2. **Zero-copy to GPU** - MLX reads directly from mapped pages
3. **No custom code** - Standard MLX API, battle-tested

Without the header, we'd need custom C++ code to create MLX arrays from raw mmap'd data. MLX's public API doesn't expose this - it always copies when creating arrays from external pointers.

The overhead is negligible: 88 bytes per tensor = ~100KB total for a 13GB model (0.0007%).

## Why Per-Tensor Blobs?

**Deduplication**: Blobs are content-addressed by SHA256. If two models share identical tensors (same weights, dtype, shape), they share the same blob file.

Example: Model A and Model B both use the same text encoder. The text encoder's 400 tensors are stored once, referenced by both manifests.

```
~/.ollama/models/
  blobs/
    sha256-25b36eed...  <- shared by both models
    sha256-abc123...
  manifests/
    library/model-a/latest  <- references sha256-25b36eed
    library/model-b/latest  <- references sha256-25b36eed
```

## Import Flow

```
cd ./weights/Z-Image-Turbo
ollama create z-image

1. Scan component directories (text_encoder/, transformer/, vae/)
2. For each .safetensors file:
   - Extract individual tensors
   - Wrap each in minimal safetensors format (88B header + data)
   - Write to blob store (SHA256 content-addressed)
   - Add layer entry to manifest with path-style name
3. Copy config files (*.json) as config layers
4. Write manifest
```

## FP8 Quantization

Z-Image supports FP8 quantization to reduce memory usage by ~50% while maintaining image quality.

### Usage

```bash
cd ./weights/Z-Image-Turbo
ollama create z-image-fp8 --quantize fp8
```

This quantizes weights during import. The resulting model will be ~15GB instead of ~31GB.


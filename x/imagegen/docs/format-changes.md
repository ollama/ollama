# Image Gen Storage Format

Image generation models are stored as per-tensor blobs with metadata in the manifest.

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
      "mediaType": "application/vnd.ollama.image.config",
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

## Loading Code

```go
// Load from manifest
weights, _ := LoadWeightsFromManifest(manifest, "text_encoder")
weights.Load(mlx.DtypeBFloat16)
arr, _ := weights.GetTensor("model.layers.0.mlp.down_proj.weight")

// Direct blob load
sf, _ := mlx.LoadSafetensorsNative(blobPath)
arr := sf.Get("data")
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

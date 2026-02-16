# Tensor Blob Format

Ollama stores model tensors as individual blobs in the safetensors format. Each blob contains a logical tensor (or a combined quantized tensor with its scale/bias components), or a group of logical tensors (e.g. shared experts for a given layer along with the scale/bias components for that tensor).

## Safetensors File Format

Every blob follows the [safetensors](https://github.com/huggingface/safetensors) layout:

```
[8 bytes: header_size (uint64 LE)] [header_size bytes: JSON header] [tensor data region]
```

The JSON header maps tensor names to their dtype, shape, and byte offsets within the data region. A special `__metadata__` key holds string-to-string metadata.

## Unquantized Blobs

An unquantized blob stores a single tensor keyed by its name:

```json
{
  "model.layers.0.self_attn.q_proj.weight": {
    "dtype": "BF16",
    "shape": [2560, 2560],
    "data_offsets": [0, 13107200]
  }
}
```

The tensor key is the full tensor name. Dtype is typically `BF16` or `F32`.

## Quantized Blobs (Combined Format)

A quantized blob stores the packed weight, scaling factors, and optional zero-point biases in a single file. Tensor keys use the tensor name, with `.scale` and `.bias` suffixes for the auxiliary tensors:

```json
{
  "__metadata__": {
    "quant_type": "int4",
    "group_size": "32"
  },
  "model.layers.0.mlp.up_proj.weight": {
    "dtype": "U32",
    "shape": [2560, 320],
    "data_offsets": [0, 3276800]
  },
  "model.layers.0.mlp.up_proj.weight.scale": {
    "dtype": "BF16",
    "shape": [2560, 80],
    "data_offsets": [3276800, 3686400]
  },
  "model.layers.0.mlp.up_proj.weight.bias": {
    "dtype": "BF16",
    "shape": [2560, 80],
    "data_offsets": [3686400, 4096000]
  }
}
```

### Metadata Fields

| Field | Description |
|---|---|
| `quant_type` | Quantization type: `int4`, `int8`, `nvfp4`, or `mxfp8` |
| `group_size` | Number of elements per quantization group (e.g., `32`, `64`) |

### Tensor Keys

| Key | Description |
|---|---|
| `{name}` | Packed quantized weights (dtype `U32`) |
| `{name}.scale` | Per-group scaling factors |
| `{name}.bias` | Per-group zero-point offsets (affine modes only) |

## Quantization Types

| Type | Bits | Group Size | Mode | Has Bias |
|---|---|---|---|---|
| `int4` | 4 | 32 | affine | yes |
| `int8` | 8 | 64 | affine | yes |
| `nvfp4` | 4 | 16 | nvfp4 | no |
| `mxfp8` | 8 | 32 | mxfp8 | no |

**Affine modes** (`int4`, `int8`) use `scale + bias` for dequantization. The bias tensor provides the zero-point offset.

**Non-affine modes** (`nvfp4`, `mxfp8`) use only `scale` with specialized E4M3 scale formats.

### Packed Weight Shape

Quantized weights are packed into `uint32` values:
- **4-bit** (int4, nvfp4): 8 values per uint32, so `packed_cols = original_cols / 8`
- **8-bit** (int8, mxfp8): 4 values per uint32, so `packed_cols = original_cols / 4`

Scale shape: `[rows, original_cols / group_size]`

## Manifest References

Blobs are referenced from the model manifest as layers:

```json
{
  "mediaType": "application/vnd.ollama.image.tensor",
  "digest": "sha256:abc123...",
  "size": 4096150,
  "name": "model.layers.0.mlp.up_proj.weight"
}
```

Each tensor (quantized or not) is one layer in the manifest. The layer name matches the tensor key in the blob header.

## Packed Blobs (Expert Groups)

For MoE (Mixture of Experts) models, expert tensors from the same layer are packed into a single blob to reduce blob count and improve loading efficiency. A packed blob is a standard safetensors file containing multiple tensor entries:

```json
{
  "model.layers.1.mlp.experts.0.down_proj.weight": {
    "dtype": "U32",
    "shape": [2560, 640],
    "data_offsets": [0, 6553600]
  },
  "model.layers.1.mlp.experts.0.down_proj.weight.scale": {
    "dtype": "BF16",
    "shape": [2560, 40],
    "data_offsets": [6553600, 6963200]
  },
  "model.layers.1.mlp.experts.0.gate_proj.weight": {
    "dtype": "U32",
    "shape": [10240, 320],
    "data_offsets": [6963200, 20070400]
  },
  "model.layers.1.mlp.experts.0.gate_proj.weight.scale": { "..." : "..." }
}
```

### Grouping Rules

- `model.layers.{L}.mlp.experts.*` tensors are packed into one blob per layer
- `model.layers.{L}.mlp.shared_experts.*` tensors are packed into one blob per layer
- All other tensors remain as individual blobs

### Manifest Representation

One manifest layer per packed group, using the group prefix as the layer name:

```json
{
  "mediaType": "application/vnd.ollama.image.tensor",
  "digest": "sha256:...",
  "size": 123456789,
  "name": "model.layers.1.mlp.experts"
}
```

## Loading

At load time, `mlx_load_safetensors` opens each blob via mmap for zero-copy access. For combined quantized blobs, the loader extracts `{name}`, `{name}.scale`, and `{name}.bias` tensors and caches them as `name`, `name + "_scale"`, and `name + "_qbias"` respectively, maintaining compatibility with the weight loading interface.

For packed blobs, if the manifest layer name (group prefix) is not found as a tensor key, the loader parses the blob header to discover all tensor names and loads each individually.

# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx>=0.30", "numpy"]
# ///
"""Golden-vector generator for the gemma4 MLX vision port.

Runs the *reference* vision forward (vendored verbatim from mlx-vlm main:
mlx_vlm/models/gemma4/vision.py and models/gemma4_unified/gemma4_unified.py,
plus the shared MultimodalEmbedder) over the ollama-quantized weights, on a
deterministic fixture whose budget-fill factor is exactly 1.0 — so the Go
side (x/models/gemma4 EncodeVision) sees pixel-identical patches and every
delta is model math.

Sizing is pinned to ADR 0008's budget-fill (llm.BudgetFillSize); mlx-vlm's
own sizing is deliberately not used (docs/maxusai/upstream-gemma4-sizing-issue.md).

Usage:
    uv run testdata/gen_vision_goldens.py gemma4:12b-nvfp4 testdata/vision_goldens_12b.json
    (repeat for 26b/31b; requires the model under $OLLAMA_MODELS or ~/.ollama/models-mlx)
"""

import json
import math
import os
import struct
import sys

import mlx.core as mx
import numpy as np

MODELS_ROOT = os.environ.get(
    "OLLAMA_MODELS", os.path.expanduser("~/.ollama/models-mlx")
)


# --- manifest / tensor loading -------------------------------------------------

def load_manifest(tag: str) -> dict:
    name, _, variant = tag.partition(":")
    path = os.path.join(
        MODELS_ROOT, "manifests/registry.ollama.ai/library", name, variant
    )
    with open(path) as f:
        return json.load(f)


def blob_path(digest: str) -> str:
    return os.path.join(MODELS_ROOT, "blobs", digest.replace(":", "-"))


def read_config(manifest: dict) -> dict:
    for layer in manifest["layers"]:
        if layer.get("name") == "config.json":
            with open(blob_path(layer["digest"])) as f:
                return json.load(f)
    raise SystemExit("no config.json layer in manifest")


def load_vision_tensors(manifest: dict) -> dict[str, mx.array]:
    tensors: dict[str, mx.array] = {}
    seen = set()
    for layer in manifest["layers"]:
        nm = layer.get("name", "")
        if "tensor" not in layer["mediaType"]:
            continue
        if not ("vision" in nm or "embed_vision" in nm):
            continue
        if layer["digest"] in seen:
            continue
        seen.add(layer["digest"])
        for k, v in mx.load(blob_path(layer["digest"]), format="safetensors").items():
            tensors[k] = v
    return tensors


def dequant(tensors: dict, path: str) -> mx.array:
    """Materialize a possibly nvfp4-quantized weight as bf16."""
    w = tensors[f"{path}.weight"]
    scales = tensors.get(f"{path}.weight.scale")
    if scales is None:
        return w.astype(mx.bfloat16)
    out = mx.dequantize(w, scales, None, group_size=16, bits=4, mode="nvfp4")
    gscale = tensors.get(f"{path}.weight.global_scale")
    if gscale is not None:
        out = out * gscale
    return out.astype(mx.bfloat16)


# --- reference building blocks (vendored from mlx-vlm main) --------------------

def gelu_approx(x):
    return 0.5 * x * (
        1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))
    )


def layer_norm(x, w, b, eps=1e-5):
    xf = x.astype(mx.float32)
    mu = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.var(xf, axis=-1, keepdims=True)
    out = (xf - mu) * mx.rsqrt(var + eps)
    return (out * w.astype(mx.float32) + b.astype(mx.float32)).astype(x.dtype)


def rms_norm_no_scale(x, eps):
    xf = x.astype(mx.float32)
    var = mx.mean(xf**2, axis=-1, keepdims=True)
    return (xf * mx.rsqrt(var + eps)).astype(x.dtype)


def vision_rms_norm(x, w, eps=1e-6):
    xf = x.astype(mx.float32)
    var = mx.mean(xf**2, axis=-1, keepdims=True)
    return ((xf * mx.rsqrt(var + eps)) * w.astype(mx.float32)).astype(x.dtype)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_multidimensional_rope(inputs, positions, base_frequency=100.0):
    head_dim = inputs.shape[-1]
    ndim = positions.shape[-1]
    channels_per_dim = 2 * (head_dim // (2 * ndim))
    half_per_dim = channels_per_dim // 2
    parts = []
    for d in range(ndim):
        x_part = inputs[..., d * channels_per_dim : (d + 1) * channels_per_dim]
        freq_exponents = (2.0 / channels_per_dim) * mx.arange(0, half_per_dim).astype(
            mx.float32
        )
        timescale = mx.power(base_frequency, freq_exponents)
        sinusoid = positions[..., d : d + 1].astype(mx.float32) / timescale
        cos_d = mx.concatenate([mx.cos(sinusoid)] * 2, axis=-1).astype(inputs.dtype)
        sin_d = mx.concatenate([mx.sin(sinusoid)] * 2, axis=-1).astype(inputs.dtype)
        cos_d = mx.expand_dims(cos_d, axis=2)
        sin_d = mx.expand_dims(sin_d, axis=2)
        parts.append(x_part * cos_d + _rotate_half(x_part) * sin_d)
    return mx.concatenate(parts, axis=-1)


def one_hot(indices, num_classes):
    return (mx.expand_dims(indices, -1) == mx.arange(num_classes)).astype(mx.float32)


def avg_pool_by_positions(x, patch_positions, length):
    input_seq_len = x.shape[1]
    k = int((input_seq_len // length) ** 0.5)
    clamped = mx.clip(patch_positions, 0, None)
    max_x = mx.max(clamped[..., 0], axis=-1, keepdims=True) + 1
    kernel_idxs = mx.floor(clamped.astype(mx.float32) / k).astype(mx.int32)
    kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
    weights = one_hot(kernel_idxs, length).astype(mx.float32) / (k * k)
    return mx.einsum("bLl,bLd->bld", weights, x.astype(mx.float32)).astype(x.dtype)


# --- fixture (matches the Go test's raster exactly) ----------------------------

def fixture_pixels(w=480, h=336):
    """Deterministic gradient raster: R=x%256, G=y%256, B=(x+y)%256, /255."""
    x = np.arange(w)[None, :]
    y = np.arange(h)[:, None]
    img = np.stack(
        [
            np.broadcast_to(x % 256, (h, w)),
            np.broadcast_to(y % 256, (h, w)),
            (x + y) % 256,
        ],
        axis=-1,
    ).astype(np.float32)
    return img / 255.0  # HWC, [0,1]


def patchify(img: np.ndarray, p: int):
    """[H,W,C] → ([pH*pW, p*p*C] channel-fastest, xs, ys) — the reference's
    _convert_image_to_model_patches layout."""
    h, w, c = img.shape
    ph, pw = h // p, w // p
    patches = img.reshape(ph, p, pw, p, c).transpose(0, 2, 1, 3, 4)
    patches = patches.reshape(ph * pw, p * p * c)
    xs, ys = np.meshgrid(np.arange(pw), np.arange(ph), indexing="xy")
    return patches, xs.flatten(), ys.flatten()


# --- forwards ------------------------------------------------------------------

def run_unified(cfg: dict, tensors: dict) -> mx.array:
    vc = cfg["vision_config"]
    p = vc.get("model_patch_size", 48)
    img = fixture_pixels()
    patches, xs, ys = patchify(img, p)

    x = mx.array(patches)[None].astype(mx.bfloat16)
    pfx = "model.vision_embedder."
    h = layer_norm(x, tensors[pfx + "patch_ln1.weight"], tensors[pfx + "patch_ln1.bias"])
    h = h @ dequant(tensors, pfx + "patch_dense").T + tensors[pfx + "patch_dense.bias"].astype(
        mx.bfloat16
    )
    h = layer_norm(h, tensors[pfx + "patch_ln2.weight"], tensors[pfx + "patch_ln2.bias"])

    pos = tensors[pfx + "pos_embedding"].astype(mx.bfloat16)  # [S, 2, D]
    h = h + (pos[mx.array(xs), 0] + pos[mx.array(ys), 1])[None]
    h = layer_norm(h, tensors[pfx + "pos_norm.weight"], tensors[pfx + "pos_norm.bias"])

    h = rms_norm_no_scale(h, vc.get("rms_norm_eps", 1e-6))
    return h @ dequant(tensors, "model.embed_vision.embedding_projection").T


def run_tower(cfg: dict, tensors: dict) -> mx.array:
    vc = cfg["vision_config"]
    p = vc["patch_size"]
    eps = vc.get("rms_norm_eps", 1e-6)
    theta = vc["rope_parameters"]["rope_theta"]
    img = fixture_pixels()
    patches, xs, ys = patchify(img, p)
    patches = 2 * (patches - 0.5)

    x = mx.array(patches)[None].astype(mx.bfloat16)
    pfx = "model.vision_tower."
    h = x @ dequant(tensors, pfx + "patch_embedder.input_proj").T

    table = tensors[pfx + "patch_embedder.position_embedding_table"].astype(mx.bfloat16)
    h = h + (table[0][mx.array(xs)] + table[1][mx.array(ys)])[None]

    positions = mx.array(np.stack([xs, ys], axis=-1))[None]  # [1, L, 2]
    H, D = vc["num_attention_heads"], vc["head_dim"]
    B, L = 1, h.shape[1]

    for i in range(vc["num_hidden_layers"]):
        lp = f"{pfx}encoder.layers.{i}."
        n = mx.fast.rms_norm(h, tensors[lp + "input_layernorm.weight"].astype(mx.bfloat16), eps)

        q = (n @ dequant(tensors, lp + "self_attn.q_proj.linear").T).reshape(B, L, H, D)
        k = (n @ dequant(tensors, lp + "self_attn.k_proj.linear").T).reshape(B, L, H, D)
        v = (n @ dequant(tensors, lp + "self_attn.v_proj.linear").T).reshape(B, L, H, D)
        q = vision_rms_norm(q, tensors[lp + "self_attn.q_norm.weight"], eps)
        k = vision_rms_norm(k, tensors[lp + "self_attn.k_norm.weight"], eps)
        v = rms_norm_no_scale(v, eps)
        q = apply_multidimensional_rope(q, positions, theta)
        k = apply_multidimensional_rope(k, positions, theta)
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0)
        attn = attn.transpose(0, 2, 1, 3).reshape(B, L, H * D)
        attn = attn @ dequant(tensors, lp + "self_attn.o_proj.linear").T
        attn = mx.fast.rms_norm(
            attn, tensors[lp + "post_attention_layernorm.weight"].astype(mx.bfloat16), eps
        )
        h = h + attn

        n = mx.fast.rms_norm(
            h, tensors[lp + "pre_feedforward_layernorm.weight"].astype(mx.bfloat16), eps
        )
        gate = n @ dequant(tensors, lp + "mlp.gate_proj.linear").T
        up = n @ dequant(tensors, lp + "mlp.up_proj.linear").T
        f = (gelu_approx(gate) * up) @ dequant(tensors, lp + "mlp.down_proj.linear").T
        f = mx.fast.rms_norm(
            f, tensors[lp + "post_feedforward_layernorm.weight"].astype(mx.bfloat16), eps
        )
        h = h + f

    length = L // vc["pooling_kernel_size"] ** 2
    h = avg_pool_by_positions(h, positions, length)
    h = h * (vc["hidden_size"] ** 0.5)

    if vc.get("standardize"):
        h = (h - tensors[pfx + "std_bias"].astype(mx.bfloat16)) * tensors[
            pfx + "std_scale"
        ].astype(mx.bfloat16)

    h = rms_norm_no_scale(h, eps)
    return h @ dequant(tensors, "model.embed_vision.embedding_projection").T


def main():
    tag, out_path = sys.argv[1], sys.argv[2]
    manifest = load_manifest(tag)
    cfg = read_config(manifest)
    tensors = load_vision_tensors(manifest)

    if cfg["vision_config"]["model_type"] == "gemma4_unified_vision":
        feats = run_unified(cfg, tensors)
    else:
        feats = run_tower(cfg, tensors)
    feats = feats.astype(mx.float32)
    mx.eval(feats)
    f = np.array(feats)[0]  # [n, hidden]

    norms = np.linalg.norm(f, axis=-1)
    golden = {
        "tag": tag,
        "fixture": "gradient-480x336@70",
        "shape": list(f.shape),
        "mean": float(f.mean()),
        "std": float(f.std()),
        "norms_head": [float(v) for v in norms[:8]],
        "norm_mean": float(norms.mean()),
        "row0_head": [float(v) for v in f[0, :32]],
        "rowlast_head": [float(v) for v in f[-1, :32]],
    }
    with open(out_path, "w") as out:
        json.dump(golden, out, indent=1)
    print(f"wrote {out_path}: shape={f.shape} norm_mean={golden['norm_mean']:.4f}")


if __name__ == "__main__":
    main()

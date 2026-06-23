#!/usr/bin/env python3
"""
calibrate-triattention.py  –  Generate .triattention calibration files.

The calibration file is consumed by src/llama-triattention.cpp at runtime to
score KV cache tokens using pre-RoPE query-frequency statistics.

Binary format (little-endian):
  magic       u32  = 0x54524941 ("TRIA")
  version     u32  = 1
  head_dim    u32
  num_layers  u32
  num_attn_heads u32
  num_kv_heads   u32
  rope_theta  f64
  rope_style  u32  (0 = GPT-NeoX / llama default)
  n_sampled   u32
  freq_count  u32  (= head_dim // 2)
  name_len    u32
  model_name  u8[name_len]   (no null terminator in file)
  --- repeated n_sampled times ---
  layer       u32
  head        u32
  q_mean_real f32[freq_count]   mean of real part of DFT per frequency
  q_mean_imag f32[freq_count]   mean of imag part of DFT per frequency
  q_abs_mean  f32[freq_count]   mean of |DFT| per frequency
  R_f         f32[freq_count]   validation field (written, skipped at load)

Usage:
  python3 scripts/calibrate-triattention.py \\
      --model <hf_model_id_or_path> \\
      --corpus <corpus.txt> \\
      --output <model_name>.triattention \\
      [--n-samples 512] \\
      [--sample-heads 32] \\
      [--max-seq-len 2048] \\
      [--device cuda]
"""

import argparse
import os
import struct
import sys

import numpy as np
import torch

TRIATTENTION_MAGIC   = 0x54524941
TRIATTENTION_VERSION = 1
ROPE_STYLE_NEOX      = 0


def parse_args():
    p = argparse.ArgumentParser(description="Generate .triattention calibration file")
    p.add_argument("--model",       required=True,  help="HuggingFace model id or local path")
    p.add_argument("--corpus",      required=True,  help="Plain-text corpus file (UTF-8)")
    p.add_argument("--output",      required=True,  help="Output .triattention file path")
    p.add_argument("--n-samples",   type=int, default=512,  help="Number of random windows to process")
    p.add_argument("--sample-heads",type=int, default=0,    help="Attention heads to sample (0 = all)")
    p.add_argument("--max-seq-len", type=int, default=2048, help="Token window length per sample")
    p.add_argument("--vram-gb",     type=int, default=2,    help="Max VRAM to use (GiB)")
    p.add_argument("--ram-gb",      type=int, default=20,   help="Max CPU RAM to use (GiB)")
    p.add_argument("--dtype",       default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def load_model_and_tokenizer(model_id, dtype_str, vram_gb=3, ram_gb=20):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[calibrate] Loading tokenizer from {model_id} …")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
    max_memory  = {0: f"{vram_gb}GiB", "cpu": f"{ram_gb}GiB"}
    print(f"[calibrate] Loading model ({dtype_str}, VRAM≤{vram_gb}GiB, RAM≤{ram_gb}GiB) …")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )
    model.eval()
    return tok, model


def get_model_config(model):
    cfg = model.config
    # Detect common field names across architectures
    n_layers     = getattr(cfg, "num_hidden_layers",  getattr(cfg, "n_layer", None))
    n_heads      = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))
    n_kv_heads   = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim_cfg = getattr(cfg, "head_dim", None)
    hidden_size  = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
    rope_theta   = float(getattr(cfg, "rope_theta", 10000.0))

    if n_layers is None or n_heads is None or hidden_size is None:
        raise ValueError("Cannot determine model dimensions from config.")

    head_dim = head_dim_cfg if head_dim_cfg else hidden_size // n_heads

    return dict(
        num_layers=n_layers,
        num_attn_heads=n_heads,
        num_kv_heads=n_kv_heads,
        head_dim=head_dim,
        rope_theta=rope_theta,
    )


def collect_q_activations(model, tok, corpus_text, cfg, args):
    """
    Run random corpus windows through the model and capture pre-RoPE Q tensors
    for every layer/head, returning a dict keyed by (layer, head) with
    shape [n_windows * seq_len, head_dim].
    """
    rng = np.random.default_rng(args.seed)
    n_layers    = cfg["num_layers"]
    n_heads     = cfg["num_attn_heads"]
    head_dim    = cfg["head_dim"]
    max_seq_len = args.max_seq_len

    # Tokenise corpus in chunks to avoid the "sequence too long" warning.
    # We split the raw text into ~4k-char pieces and tokenize each separately.
    chunk_size  = 16000  # characters per tokenization call
    token_lists = []
    for i in range(0, len(corpus_text), chunk_size):
        chunk = corpus_text[i:i + chunk_size]
        ids   = tok(chunk, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        token_lists.append(ids)
    tokens   = torch.cat(token_lists, dim=0)
    n_tokens = tokens.shape[0]
    print(f"[calibrate] Corpus: {n_tokens} tokens")
    if n_tokens < max_seq_len:
        print(f"[calibrate] WARNING: corpus only {n_tokens} tokens (< {max_seq_len}). "
              "Using the whole corpus as one sample.")
        max_seq_len = n_tokens

    # Select which (layer, head) pairs to sample
    all_pairs = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    if args.sample_heads > 0 and args.sample_heads < len(all_pairs):
        idxs = rng.choice(len(all_pairs), size=args.sample_heads, replace=False)
        sampled_pairs = [all_pairs[i] for i in sorted(idxs)]
    else:
        sampled_pairs = all_pairs

    print(f"[calibrate] Sampling {len(sampled_pairs)} (layer, head) pairs "
          f"over {args.n_samples} windows of {max_seq_len} tokens …")

    # Accumulators: list of arrays, one per sampled pair
    accum = {pair: [] for pair in sampled_pairs}
    sampled_set = set(sampled_pairs)

    hooks   = []
    storage = {}  # layer_idx -> list of tensors captured this forward pass

    def make_hook(layer_idx):
        def hook(module, args, kwargs):
            # Modern transformers passes hidden_states as kwarg; older as args[0].
            if args:
                hidden = args[0]
            elif "hidden_states" in kwargs:
                hidden = kwargs["hidden_states"]
            else:
                return

            q_proj = getattr(module, "q_proj", None)
            if q_proj is None:
                return

            with torch.no_grad():
                q = q_proj(hidden)               # [1, seq, n_heads * head_dim]

            batch, seq, _ = q.shape
            hd = cfg["head_dim"]
            nh = cfg["num_attn_heads"]
            q  = q.view(batch, seq, nh, hd).squeeze(0)  # [seq, n_heads, head_dim]
            storage[layer_idx] = q.cpu().float().numpy()

        return hook

    # Register hooks on attention submodules
    # Try common module paths: model.layers[i].self_attn, transformer.h[i].attn, etc.
    attn_modules = {}
    for name, module in model.named_modules():
        parts = name.split(".")
        # Match patterns like layers.N.self_attn or h.N.attn
        if len(parts) >= 2:
            try:
                layer_idx = int(parts[-2])
                if parts[-1] in ("self_attn", "attn", "attention") and layer_idx < n_layers:
                    attn_modules[layer_idx] = module
            except ValueError:
                pass

    if not attn_modules:
        raise RuntimeError(
            "Could not find attention submodules. "
            "Supported: LlamaAttention-style (layers.N.self_attn)."
        )

    for layer_idx, module in attn_modules.items():
        h = module.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
        hooks.append(h)

    input_device = next(model.parameters()).device

    try:
        for s in range(args.n_samples):
            start = int(rng.integers(0, n_tokens - max_seq_len + 1))
            window = tokens[start:start + max_seq_len].unsqueeze(0).to(input_device)

            storage.clear()
            with torch.no_grad():
                model(window)

            for (layer, head) in sampled_set:
                if layer in storage:
                    q_slice = storage[layer][:, head, :]   # [seq, head_dim]
                    accum[(layer, head)].append(q_slice)

            if (s + 1) % 50 == 0 or s == args.n_samples - 1:
                print(f"[calibrate]   {s+1}/{args.n_samples} windows done")
    finally:
        for h in hooks:
            h.remove()

    # Concatenate windows per head
    result = {}
    for pair in sampled_pairs:
        if accum[pair]:
            result[pair] = np.concatenate(accum[pair], axis=0)  # [total_tokens, head_dim]

    return sampled_pairs, result


def compute_head_stats(q_vecs, freq_count):
    """
    Given q_vecs of shape [N, head_dim], compute per-frequency statistics.

    RoPE pairs dimensions (0,1), (2,3), … as complex numbers.
    DFT here means: treat paired dims as (real, imag) complex value for each
    frequency f, then compute statistics across the N token samples.
    """
    N, head_dim = q_vecs.shape
    assert freq_count == head_dim // 2

    # Reshape to [N, freq_count, 2] → complex [N, freq_count]
    q_pairs  = q_vecs.reshape(N, freq_count, 2)
    q_cplx   = q_pairs[:, :, 0] + 1j * q_pairs[:, :, 1]   # [N, freq_count]

    q_mean_real = np.mean(q_cplx.real, axis=0).astype(np.float32)  # [freq_count]
    q_mean_imag = np.mean(q_cplx.imag, axis=0).astype(np.float32)
    q_abs_mean  = np.mean(np.abs(q_cplx), axis=0).astype(np.float32)

    # R_f: mean of squared magnitude (validation field)
    R_f = np.mean(np.abs(q_cplx) ** 2, axis=0).astype(np.float32)

    return q_mean_real, q_mean_imag, q_abs_mean, R_f


def write_calibration(path, model_name, cfg, sampled_pairs, head_data):
    n_layers      = cfg["num_layers"]
    n_attn_heads  = cfg["num_attn_heads"]
    n_kv_heads    = cfg["num_kv_heads"]
    head_dim      = cfg["head_dim"]
    rope_theta    = cfg["rope_theta"]
    freq_count    = head_dim // 2
    n_sampled     = len(sampled_pairs)
    name_bytes    = model_name.encode("utf-8")[:255]

    print(f"[calibrate] Writing {path} …")
    with open(path, "wb") as f:
        # Header
        f.write(struct.pack("<I", TRIATTENTION_MAGIC))
        f.write(struct.pack("<I", TRIATTENTION_VERSION))
        f.write(struct.pack("<I", head_dim))
        f.write(struct.pack("<I", n_layers))
        f.write(struct.pack("<I", n_attn_heads))
        f.write(struct.pack("<I", n_kv_heads))
        f.write(struct.pack("<d", rope_theta))
        f.write(struct.pack("<I", ROPE_STYLE_NEOX))
        f.write(struct.pack("<I", n_sampled))
        f.write(struct.pack("<I", freq_count))
        f.write(struct.pack("<I", len(name_bytes)))
        f.write(name_bytes)

        # Per-head data
        for (layer, head) in sampled_pairs:
            q_vecs = head_data.get((layer, head))
            if q_vecs is None or len(q_vecs) == 0:
                # Fallback: write zeros if no data collected for this pair
                zeros = np.zeros(freq_count, dtype=np.float32)
                f.write(struct.pack("<I", layer))
                f.write(struct.pack("<I", head))
                f.write(zeros.tobytes())
                f.write(zeros.tobytes())
                f.write(zeros.tobytes())
                f.write(zeros.tobytes())
                continue

            mean_real, mean_imag, abs_mean, R_f = compute_head_stats(q_vecs, freq_count)
            f.write(struct.pack("<I", layer))
            f.write(struct.pack("<I", head))
            f.write(mean_real.tobytes())
            f.write(mean_imag.tobytes())
            f.write(abs_mean.tobytes())
            f.write(R_f.tobytes())

    size_kb = os.path.getsize(path) / 1024
    print(f"[calibrate] Done. {n_sampled} heads written. File size: {size_kb:.1f} KB")


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    # Load corpus
    print(f"[calibrate] Reading corpus from {args.corpus} …")
    with open(args.corpus, "r", encoding="utf-8") as fh:
        corpus_text = fh.read()
    if not corpus_text.strip():
        print("ERROR: corpus file is empty.", file=sys.stderr)
        sys.exit(1)

    tok, model = load_model_and_tokenizer(args.model, args.dtype, args.vram_gb, args.ram_gb)
    cfg = get_model_config(model)

    print(f"[calibrate] Model: layers={cfg['num_layers']}, "
          f"attn_heads={cfg['num_attn_heads']}, kv_heads={cfg['num_kv_heads']}, "
          f"head_dim={cfg['head_dim']}, rope_theta={cfg['rope_theta']}")

    sampled_pairs, head_data = collect_q_activations(model, tok, corpus_text, cfg, args)

    model_name = os.path.basename(args.model.rstrip("/"))
    write_calibration(args.output, model_name, cfg, sampled_pairs, head_data)


if __name__ == "__main__":
    main()

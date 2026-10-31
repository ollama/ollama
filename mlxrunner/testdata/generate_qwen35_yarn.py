# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.6.0", "transformers==5.12.1"]
# ///

import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
)


def generate():
    torch.set_num_threads(1)
    defaults = {
        "rope_type": "default",
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [11, 11, 10],
    }
    yarn = {
        **defaults,
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 262144,
    }
    cases = [
        ("default", 262144, defaults, "float32"),
        ("no_metadata", 262144, None, "float32"),
        ("yarn4", 1048576, yarn, "float32"),
        ("unextended_limit", 262144, yarn, "float32"),
        (
            "implicit_original",
            262144,
            {k: v for k, v in yarn.items() if k != "original_max_position_embeddings"},
            "float32",
        ),
        ("beta", 1048576, {**yarn, "beta_fast": 16.0, "beta_slow": 2.0}, "float32"),
        ("fractional_range", 1048576, {**yarn, "truncate": False}, "float32"),
        ("mscale", 1048576, {**yarn, "mscale": 2.0, "mscale_all_dim": 1.0}, "float32"),
        ("attention_override", 1048576, {**yarn, "attention_factor": 1.2}, "float32"),
        ("zero_attention", 1048576, {**yarn, "attention_factor": 0.0}, "float32"),
        ("factor_one", 262144, {**yarn, "factor": 1.0}, "float32"),
        (
            "factor_one_attention",
            262144,
            {**yarn, "factor": 1.0, "attention_factor": 1.2},
            "float32",
        ),
        ("theta", 1048576, {**yarn, "rope_theta": 500000.0}, "float32"),
        (
            "half_rotary",
            1048576,
            {**yarn, "partial_rotary_factor": 0.5, "mrope_section": [22, 21, 21]},
            "float32",
        ),
        ("sections", 1048576, {**yarn, "mrope_section": [24, 4, 4]}, "float32"),
        ("default_bfloat16", 262144, defaults, "bfloat16"),
        ("yarn_bfloat16", 1048576, yarn, "bfloat16"),
    ]
    sample_tokens = [0, 1, 7, 15]
    sample_dims = [0, 7, 15, 16, 23, 31, 32, 39, 47, 48, 55, 63, 64, 95, 127, 128, 255]
    indices = torch.arange(16)
    text_positions = indices.reshape(1, -1)
    media_positions = torch.stack(
        (32768 + 2 * indices, 7 + indices % 4, 11 + indices // 4)
    ).unsqueeze(1)
    media_positions[:, 0, -1] = media_positions[:, :, :-1].max() + 1
    mask = torch.full((1, 1, 16, 16), float("-inf")).triu(1)
    results = []
    for name, limit, params, dtype_name in cases:
        raw_config = {
            "model_type": "qwen3_5_text",
            "hidden_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "intermediate_size": 256,
            "rms_norm_eps": 1e-6,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "layer_types": ["full_attention"],
            "max_position_embeddings": limit,
            "rope_theta": 10000000.0,
            "partial_rotary_factor": (params or defaults)["partial_rotary_factor"],
            "rope_parameters": params,
        }
        cfg = Qwen3_5TextConfig(
            **{
                k: v
                for k, v in raw_config.items()
                if k not in ("rope_theta", "rope_parameters")
            },
            rope_parameters=dict(params or defaults),
        )
        cfg._attn_implementation = "eager"
        dtype = getattr(torch, dtype_name)
        attention = Qwen3_5Attention(cfg, layer_idx=0).eval().to(dtype)
        rope = Qwen3_5TextRotaryEmbedding(cfg)
        assert rope.inv_freq.numel() * 2 == int(
            cfg.head_dim * (params or defaults)["partial_rotary_factor"]
        )
        with torch.no_grad():
            attention.q_proj.weight.zero_()
            attention.q_proj.weight[:256].copy_(torch.eye(256))
            for linear in (attention.k_proj, attention.v_proj, attention.o_proj):
                linear.weight.copy_(torch.eye(256))
            attention.q_norm.weight.fill_(-0.5)
            attention.k_norm.weight.fill_(-0.5)
            x = (
                (((torch.arange(16 * 256) % 37) - 18).float() / 16)
                .reshape(1, 16, 256)
                .to(dtype)
            )
            raw_keys = attention.k_norm(
                attention.k_proj(x).reshape(1, 16, 1, 256)
            ).transpose(1, 2)
            entry = {
                "name": name,
                "dtype": dtype_name,
                "config": raw_config,
                "inv_freq": rope.inv_freq.tolist(),
                "attention_factor": rope.attention_scaling,
            }
            modes = (
                ("text", text_positions),
                ("mrope", media_positions),
                ("long_text", text_positions + 32768),
            )
            if name == "sections":
                entry["modes"] = ["mrope"]
                modes = (("mrope", media_positions),)
            for mode, positions in modes:
                cos, sin = rope(x, positions)
                output, _ = attention(
                    x, position_embeddings=(cos, sin), attention_mask=mask
                )
                entry[mode + "_output"] = [
                    output[0, token, dim].float().item()
                    for token in sample_tokens
                    for dim in sample_dims
                ]
                if mode != "long_text":
                    _, keys = apply_rotary_pos_emb(raw_keys, raw_keys, cos, sin)
                    entry[mode + "_keys"] = [
                        keys[0, 0, token, dim].float().item()
                        for token in sample_tokens
                        for dim in sample_dims
                    ]
        results.append(entry)
    return {
        "source": f"transformers {transformers.__version__}; torch {torch.__version__}; CPU; Qwen3_5TextConfig/Qwen3_5TextRotaryEmbedding/Qwen3_5Attention",
        "sample_tokens": sample_tokens,
        "sample_dims": sample_dims,
        "mrope_positions": media_positions.flatten().tolist(),
        "cases": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fixture = generate()
    text = json.dumps(fixture, indent=2)
    for case in fixture["cases"]:
        for key in (
            "inv_freq",
            "text_output",
            "long_text_output",
            "text_keys",
            "mrope_output",
            "mrope_keys",
        ):
            if key not in case:
                continue
            multiline = json.dumps(case[key], indent=2)
            body = "\n".join("      " + line for line in multiline.splitlines()[1:-1])
            text = text.replace("[\n" + body + "\n      ]", json.dumps(case[key]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")
    print(
        f"Generated {len(fixture['cases'])} reference cases: {args.output} ({args.output.stat().st_size} bytes)"
    )

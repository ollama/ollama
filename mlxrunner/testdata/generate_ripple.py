# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.6.0", "transformers==4.57.6"]
# ///
"""CPU-only, test-only Ripple reference. Never loads checkpoint weights."""

import hashlib
import importlib
import sys
import types
from pathlib import Path

import torch

sys.dont_write_bytecode = True
DEFAULT_SOURCE = Path.home() / "git/Ripple-65B-Preview"
SOURCE_REVISION = "240d5966cbe00f0627c699e59ef58560b60303cf"
SOURCE_HASHES = {
    "configuration_ricursive.py": "002cb3915be9685eb962d617623bec14d45b5d6b5d46e689fac98a3b38cb6cf6",
    "modeling_ricursive.py": "f83865ef643e273a6c9aecf1575e85c95ad01fd995b1e43c3c7a892765006bb3",
    "config.json": "9a5654b256ead01855f0d74cf7a70fa65294267e6b7b754b5e5737457a45e14a",
}
SMALL_CONFIG = dict(
    vocab_size=19,
    hidden_size=8,
    intermediate_size=12,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=4,
    layer_types=["sliding_attention", "full_attention", "sliding_attention"],
    sliding_window=3,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    use_valres=True,
    use_gated_attention=True,
    gate_full_precision=True,
    ffn_clamp_threshold=20.0,
    embed_scale=71.55417527999327,
    max_position_embeddings=2097152,
    tie_word_embeddings=False,
)


def load_reference(source):
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"source hash mismatch: {name}")
    # Namespace import executes original files without copying or patching them.
    package_name = (
        "_frozen_ripple_" + hashlib.sha256(str(source).encode()).hexdigest()[:12]
    )
    package = types.ModuleType(package_name)
    package.__path__ = [str(source)]
    sys.modules[package_name] = package
    return importlib.import_module(package_name + ".modeling_ricursive")


def build_model(ref, dtype):
    torch.manual_seed(417)
    model = ref.RicursiveForCausalLM(ref.RicursiveConfig(**SMALL_CONFIG)).cpu().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("value_res_lambda"):
                p.fill_((-1.25, 0.75, 1.5)[int(name.split(".")[2])])
                continue
            phase = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 101
            v = ((torch.arange(p.numel()) * 37 + phase) % 101 - 50).float() / 50
            if "norm.weight" in name:
                v = 1 + v * 0.375
            elif "embed_tokens" in name:
                v = v * 0.045
            elif ".mlp.gate_proj" in name or ".mlp.up_proj" in name:
                v = v * 18
            else:
                v = v * 0.3
            p.copy_(v.reshape_as(p))
    return model.to(dtype)


def absolute_mask(query_positions, key_positions, window=None, padding=None):
    """Literal absolute-position contract, independent of HF mask helpers."""
    q, k = query_positions.tolist(), key_positions.tolist()
    batches = padding.tolist() if padding is not None else [[True] * len(k)]
    return torch.tensor(
        [
            [
                [
                    [
                        bool(valid[j])
                        and kp <= qp
                        and (window is None or kp >= qp - window + 1)
                        for j, kp in enumerate(k)
                    ]
                    for qp in q
                ]
            ]
            for valid in batches
        ],
        dtype=torch.bool,
    )


def observed_forward(
    model,
    ids,
    mode="hf_normal",
    use_cache=False,
    past=None,
    padding=None,
    position_ids=None,
):
    """Read-only hooks; the normal path calls the unchanged HF forward."""
    from transformers.cache_utils import DynamicCache

    intermediates, hooks = {}, []

    def observe(name):
        def hook(module, args, kwargs, output):
            x = args[0] if args else kwargs.get("hidden_states")
            if isinstance(x, torch.Tensor):
                intermediates[name + ".input"] = x.detach().clone()
            y = output[0] if isinstance(output, tuple) else output
            if isinstance(y, torch.Tensor):
                intermediates[name + ".output"] = y.detach().clone()
            if name.endswith("rotary_emb"):
                intermediates[name + ".cos"] = output[0].detach().clone()
                intermediates[name + ".sin"] = output[1].detach().clone()

        return hook

    for name, module in model.named_modules():
        if name and (not list(module.children()) or name.startswith("model.layers.")):
            hooks.append(module.register_forward_hook(observe(name), with_kwargs=True))
    try:
        with torch.no_grad():
            start = past.get_seq_length() if past is not None else 0
            positions = torch.arange(start, start + ids.shape[1])
            if position_ids is None:
                position_ids = positions.unsqueeze(0)
            if mode == "hf_normal":
                result = model(
                    input_ids=ids,
                    attention_mask=padding,
                    position_ids=position_ids,
                    cache_position=positions,
                    past_key_values=past,
                    use_cache=use_cache,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden, logits, cache = (
                    result.hidden_states[-1],
                    result.logits,
                    result.past_key_values,
                )
            elif mode == "absolute_mask_control":
                # Separate driver, NOT a patched model.forward or _update_causal_mask.
                cache = (
                    past
                    if past is not None
                    else (DynamicCache() if use_cache else None)
                )
                hidden = model.model.embed_tokens(ids)
                if model.config.embed_scale is not None:
                    hidden = hidden * model.config.embed_scale
                seed = hidden
                keys = torch.arange(start + ids.shape[1])
                for layer in model.model.layers:
                    window = (
                        model.config.sliding_window
                        if layer.self_attn.layer_type == "sliding_attention"
                        else None
                    )
                    mask = absolute_mask(positions, keys, window, padding)
                    hidden = layer(
                        hidden,
                        attention_mask=mask,
                        position_ids=position_ids,
                        cache_position=positions,
                        past_key_value=cache,
                        use_cache=use_cache,
                        value_residual=seed,
                    )[0]
                hidden = model.model.norm(hidden)
                logits = model.lm_head(hidden)
            else:
                raise ValueError(f"unknown reference mode: {mode}")
            return {
                "hidden": hidden.detach().clone(),
                "logits": logits.detach().clone(),
                "intermediates": intermediates,
                "positions": positions,
                "position_ids": position_ids,
            }, cache
    finally:
        for hook in hooks:
            hook.remove()


def mask_audit(model):
    import copy
    import torch.nn.functional as F

    cases = [
        ("prefill_none", 0, 6, None),
        ("prefill_allones", 0, 6, [1, 1, 1, 1, 1, 1]),
        ("cached_single_none", 4, 1, None),
        ("cached_multi_none", 4, 2, None),
        ("cached_multi_allones", 4, 2, [1, 1, 1, 1, 1, 1]),
        ("left_padding_prefill", 0, 6, [0, 0, 1, 1, 1, 1]),
        ("right_padding_prefill", 0, 6, [1, 1, 1, 1, 0, 0]),
        ("padded_cached_single", 4, 1, [0, 1, 1, 1, 1]),
        ("padded_cached_multi", 4, 2, [0, 1, 1, 1, 1, 1]),
    ]
    ids = torch.tensor([[2, 7, 3, 11, 5, 13]])
    dtype, results = model.lm_head.weight.dtype, []
    with torch.no_grad():
        for name, prefix, count, valid in cases:
            pad = torch.tensor([valid]) if valid is not None else None
            past = None
            if prefix:
                _, past = observed_forward(
                    model,
                    ids[:, :prefix],
                    use_cache=True,
                    padding=pad[:, :prefix] if pad is not None else None,
                )
            current = ids[:, prefix : prefix + count]
            pos = torch.arange(prefix, prefix + count)
            keys = torch.arange(prefix + count)
            raw = model.model._update_causal_mask(
                pad, model.model.embed_tokens(current), pos, past
            )
            masks = []
            for i, layer in enumerate(model.model.layers):
                attn = layer.self_attn
                source_mask = attn._apply_sliding_window_mask(
                    raw, count, len(keys), dtype, torch.device("cpu"), pos
                )
                expected = absolute_mask(
                    pos,
                    keys,
                    model.config.sliding_window
                    if attn.layer_type == "sliding_attention"
                    else None,
                    pad,
                )
                q = torch.zeros(1, 1, count, 4, dtype=dtype)
                k = torch.zeros(1, 1, len(keys), 4, dtype=dtype)
                v = torch.eye(len(keys), dtype=dtype)[None, None]
                probabilities = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=source_mask,
                    is_causal=source_mask is None and count > 1,
                )
                masks.append(
                    {
                        "layer": i,
                        "layer_type": attn.layer_type,
                        "source_mask": source_mask,
                        "absolute_mask": expected,
                        "source_zero_probabilities": probabilities,
                    }
                )
            normal, _ = observed_forward(
                model, current, use_cache=True, past=copy.deepcopy(past), padding=pad
            )
            control, _ = observed_forward(
                model,
                current,
                mode="absolute_mask_control",
                use_cache=True,
                past=copy.deepcopy(past),
                padding=pad,
            )
            # The full hook traces live in safe fixtures; audit keeps result boundaries.
            normal.pop("intermediates")
            control.pop("intermediates")
            results.append(
                {
                    "name": name,
                    "prefix_ids": ids[:, :prefix],
                    "input_ids": current,
                    "padding": pad,
                    "source_raw_mask": raw,
                    "masks": masks,
                    "prefix_mode": "hf_normal" if prefix else None,
                    "comparison_scope": "shared_prefix_continuation"
                    if prefix
                    else "prefill",
                    "normal": normal,
                    "control": control,
                }
            )
    return results


def validate_local_support(rows):
    """Check effective local attention against the independent mask."""
    for row in rows:
        if row["name"] == "padded_cached_single":
            continue  # Retained as a known source failure in the audit.
        valid = (
            row["padding"][:, -row["input_ids"].shape[1] :].bool()
            if row["padding"] is not None
            else torch.ones_like(row["input_ids"], dtype=torch.bool)
        )
        for layer in row["masks"]:
            if layer["layer_type"] != "sliding_attention":
                continue
            expected = layer["absolute_mask"]
            observed = layer["source_zero_probabilities"] > 0
            queries = valid[:, None, :, None] & expected.any(-1, keepdim=True)
            if torch.any((observed != expected) & queries):
                raise ValueError(
                    f"local mask support mismatch: {row['name']}, layer {layer['layer']}"
                )


def mutation_audit(ref, dtype):
    """Wrong behaviors affect disposable synthetic instances, never frozen files."""
    ids = torch.tensor([[2, 7, 3, 11, 5, 13]])
    normal, _ = observed_forward(build_model(ref, dtype), ids)
    norms = [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "q_norm",
        "k_norm",
    ]
    names = [
        "reverse_value_blend",
        *["omit_" + name for name in norms],
        "global_rope",
        "window_minus_one",
        "window_plus_one",
        "omit_attention_gate",
        "omit_embedding_scale",
        "omit_ffn_clipping",
        "omit_layer_zero_value_residual",
    ]
    results = []
    for name in names:
        model = build_model(ref, dtype)
        hooks = []
        if name == "reverse_value_blend":

            def capture_input(module, args, kwargs):
                module._probe_reversed = torch.lerp(
                    args[0],
                    kwargs["value_residual"],
                    module.value_res_lambda[0].sigmoid().to(dtype),
                )

            for layer in model.model.layers:
                attn = layer.self_attn
                hooks.append(
                    attn.register_forward_pre_hook(capture_input, with_kwargs=True)
                )
                hooks.append(
                    attn.v_proj.register_forward_pre_hook(
                        lambda module, args, a=attn: (a._probe_reversed,)
                    )
                )
        elif name.removeprefix("omit_") in norms:
            norm = name.removeprefix("omit_")
            for layer in model.model.layers:
                parent = layer.self_attn if norm in ("q_norm", "k_norm") else layer
                setattr(parent, norm, torch.nn.Identity())
        elif name == "global_rope":
            rope = ref.RicursiveRotaryEmbedding(model.config.head_dim)

            def rotate_output(module, args, output):
                x = output.transpose(1, 2)
                cos, sin = rope(x, torch.arange(ids.shape[1]).unsqueeze(0))
                rotated, _ = ref.apply_rotary_pos_emb(x, x, cos, sin)
                return rotated.transpose(1, 2)

            attn = model.model.layers[1].self_attn
            hooks.extend(
                [
                    attn.q_norm.register_forward_hook(rotate_output),
                    attn.k_norm.register_forward_hook(rotate_output),
                ]
            )
        elif name.startswith("window_"):
            model.config.sliding_window += -1 if name == "window_minus_one" else 1
        elif name == "omit_attention_gate":
            model.config.use_gated_attention = False
        elif name == "omit_embedding_scale":
            model.config.embed_scale = None
        elif name == "omit_ffn_clipping":
            for layer in model.model.layers:
                layer.mlp.clamp_threshold = None
        elif name == "omit_layer_zero_value_residual":

            def remove_seed(module, args, kwargs):
                return args, {**kwargs, "value_residual": None}

            hooks.append(
                model.model.layers[0].self_attn.register_forward_pre_hook(
                    remove_seed, with_kwargs=True
                )
            )
        try:
            mutant, _ = observed_forward(model, ids)
        finally:
            for hook in hooks:
                hook.remove()
        results.append(
            {
                "name": name,
                "normal_logits": normal["logits"],
                "mutant_logits": mutant["logits"],
                "mutant_hidden": mutant["hidden"],
            }
        )
    return results


@torch.no_grad()
def component_cases(ref, dtype):
    import torch.nn.functional as F

    x = (
        (((torch.arange(3 * 128) * 17) % 137 - 68).float() / 19)
        .reshape(1, 3, 128)
        .to(dtype)
    )
    norm = ref.RicursiveRMSNorm(128).to(dtype)
    norm.weight.copy_((0.6 + torch.arange(128).float() / 157).to(dtype))
    normalized = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + norm.eps
    )
    result = {
        "rmsnorm": {
            "origin": "RicursiveRMSNorm.forward",
            "input": x,
            "weight": norm.weight,
            "eps": norm.eps,
            "expected": norm(x),
            "alternatives": {
                "weight_before_cast": (normalized * norm.weight.float()).to(dtype)
            },
        }
    }
    model = build_model(ref, dtype)
    observed, _ = observed_forward(model, torch.tensor([[2, 7, 3, 11, 5, 13]]))
    traces = observed["intermediates"]
    embed = traces["model.embed_tokens.output"]
    seed = traces["model.layers.0.input"]
    result["embedding_scale"] = {
        "origin": "RicursiveModel.forward hook",
        "input": embed,
        "scale": model.config.embed_scale,
        "expected": seed,
        "alternatives": {
            "scalar_cast_first": embed
            * torch.tensor(model.config.embed_scale, dtype=dtype)
        },
    }
    u = traces["model.layers.0.input_layernorm.output"]
    lam = model.model.layers[0].self_attn.value_res_lambda[0]
    alpha = lam.sigmoid().to(dtype)
    result["value_blend"] = {
        "origin": "RicursiveAttention.forward v_proj input hook",
        "seed": seed,
        "normalized_hidden": u,
        "lambda": lam,
        "alpha": alpha,
        "expected": traces["model.layers.0.self_attn.v_proj.input"],
        "alternatives": {
            "reverse": torch.lerp(u, seed, alpha),
            "expanded_sum": (1 - alpha) * seed + alpha * u,
            "fp32_then_cast": torch.lerp(
                seed.float(), u.float(), lam.float().sigmoid()
            ).to(dtype),
        },
    }
    positions = torch.tensor([[32768, 1048576, 2097151]])
    rope = ref.RicursiveRotaryEmbedding(128)
    q = x.unsqueeze(1)
    cos, sin = rope(q, positions)
    output, _ = ref.apply_rotary_pos_emb(q, q, cos, sin)
    full_cos, full_sin = rope(q.float(), positions)
    unrounded, _ = ref.apply_rotary_pos_emb(q, q, full_cos, full_sin)
    result["rotary"] = {
        "origin": "RicursiveRotaryEmbedding/apply_rotary_pos_emb",
        "input": q,
        "positions": positions,
        "theta": 10000.0,
        "cos": cos,
        "sin": sin,
        "expected": output,
        "alternatives": {
            "unrounded_tables": unrounded,
            "activation_dtype_multiply": q * cos.unsqueeze(1)
            + ref.rotate_half(q) * sin.unsqueeze(1),
        },
    }
    gate = traces["model.layers.0.self_attn.g_proj.output"]
    result["gate_sigmoid"] = {
        "origin": "source expression probe, not an internal hook",
        "input": gate,
        "expected": gate.float().sigmoid().to(dtype),
        "alternatives": {"activation_dtype_sigmoid": gate.sigmoid()},
    }
    for threshold in (2, 20):
        config = ref.RicursiveConfig(
            hidden_size=4,
            intermediate_size=4,
            num_hidden_layers=1,
            layer_types=["full_attention"],
            ffn_clamp_threshold=threshold,
        )
        mlp = ref.RicursiveMLP(config).to(dtype)
        mlp.gate_proj.weight.copy_(torch.eye(4, dtype=dtype))
        mlp.up_proj.weight.copy_(torch.diag(torch.tensor([-2, 1, -3, 2], dtype=dtype)))
        mlp.down_proj.weight.copy_(
            torch.tensor(
                [
                    [1, 0.2, -0.3, 0.4],
                    [-0.5, 1, 0.25, 0.7],
                    [0.3, -0.1, 1, 0.2],
                    [0.4, 0.6, -0.8, 1],
                ],
                dtype=dtype,
            )
        )
        inp = torch.tensor(
            [[[-2 * threshold, 2 * threshold, -0.6 * threshold, 0.8 * threshold]]],
            dtype=dtype,
        )
        g, up = mlp.gate_proj(inp), mlp.up_proj(inp)
        result[f"ffn_clamp_{threshold}"] = {
            "origin": "RicursiveMLP.forward",
            "input": inp,
            "weights": dict(mlp.state_dict()),
            "threshold": threshold,
            "raw_gate": g,
            "raw_up": up,
            "clipped_gate": g.clamp(max=threshold),
            "clipped_up": up.clamp(-threshold, threshold),
            "expected": mlp(inp),
            "alternatives": {
                "symmetric_gate": mlp.down_proj(
                    F.silu(g.clamp(-threshold, threshold))
                    * up.clamp(-threshold, threshold)
                ),
                "no_clipping": mlp.down_proj(F.silu(g) * up),
            },
        }
    return result


def generate(source):
    import copy
    import importlib.metadata
    import json
    import platform
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ref = load_reference(source)
    fixture = {
        "schema_version": 1,
        "provenance": {
            "source": str(source),
            "revision": SOURCE_REVISION,
            "source_sha256": SOURCE_HASHES,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "packages": dict(
                sorted(
                    (d.metadata["Name"], d.version)
                    for d in importlib.metadata.distributions()
                )
            ),
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": "cpu",
            "threads": 1,
            "initialization_seed": 417,
            "weight_recipe": "name-hash/37-stride modular recipe in build_model; every parameter overwritten in FP32, then cast",
            "sdpa": "PyTorch default CPU, except explicitly labeled MATH calibration",
            "normal_forward": "unchanged source classes with observational module hooks",
            "explicit_control": "separate driver adds absolute-position masks to unchanged layers; local support is checked independently; cached audits share an HF-normal prefix",
        },
        "config": SMALL_CONFIG,
        "weights": {},
        "forward_cases": [],
        "components": {},
        "mask_audit": {},
        "mutations": {},
        "measurements": {},
        "incomplete_gates": [
            "Go/MLX port and tolerances",
            "SGLang disputed-mode confirmation (no authorized CUDA host)",
            "actual NVFP4 packed-weight dequantization/MLX comparison",
            "real-shape full-layer and full-model parity",
            "cache ring wrap/restore (HF DynamicCache retains all keys)",
            "high-position cache/mask continuation",
        ],
    }
    ids = torch.tensor([[2, 7, 3, 11, 5, 13]])
    baseline = {}
    for dtype_name in ("float32", "bfloat16"):
        dtype = getattr(torch, dtype_name)
        model = build_model(ref, dtype)
        fixture["weights"][dtype_name] = dict(model.state_dict())
        saved, past = {}, None
        for name in (
            "prefill",
            "cached_prefill",
            "decode_4",
            "decode_5",
            "high_rope_position_prefill",
            "explicit_prefill",
        ):
            if name == "cached_prefill":
                current, prefix, use_cache, current_past = (
                    ids[:, :4],
                    ids[:, :0],
                    True,
                    None,
                )
            elif name in ("decode_4", "decode_5"):
                i = int(name[-1])
                current, prefix, use_cache, current_past = (
                    ids[:, i : i + 1],
                    ids[:, :i],
                    True,
                    past,
                )
            else:
                current, prefix, use_cache, current_past = ids, ids[:, :0], False, None
            mode = (
                "absolute_mask_control" if name == "explicit_prefill" else "hf_normal"
            )
            position_ids = (
                32768 + torch.arange(6).unsqueeze(0)
                if name == "high_rope_position_prefill"
                else None
            )
            result, cache = observed_forward(
                model,
                current,
                mode=mode,
                use_cache=use_cache,
                past=current_past,
                position_ids=position_ids,
            )
            if use_cache:
                past = cache
            entry = {
                "name": name,
                "dtype": dtype_name,
                "mode": mode,
                "input_ids": current,
                "prefix_ids": prefix,
                **result,
                "cache": [
                    {"keys": layer.keys.clone(), "values": layer.values.clone()}
                    for layer in cache.layers
                ]
                if cache is not None
                else [],
            }
            fixture["forward_cases"].append(entry)
            saved[name] = result
        baseline[dtype_name] = saved["prefill"]
        fixture["components"][dtype_name] = component_cases(ref, dtype)
        for row in fixture["components"][dtype_name].values():
            row["differences"] = {
                name: difference(row["expected"], alt)
                for name, alt in row["alternatives"].items()
            }
        fixture["mask_audit"][dtype_name] = mask_audit(model)
        validate_local_support(fixture["mask_audit"][dtype_name])
        for row in fixture["mask_audit"][dtype_name]:
            valid = (
                row["padding"][:, -row["input_ids"].shape[1] :].bool()
                if row["padding"] is not None
                else torch.ones_like(row["input_ids"], dtype=torch.bool)
            )
            row["differences"] = {
                key: difference(row["control"][key][valid], row["normal"][key][valid])
                for key in ("hidden", "logits")
            }
        fixture["mutations"][dtype_name] = mutation_audit(ref, dtype)
        for row in fixture["mutations"][dtype_name]:
            row["difference"] = difference(row["normal_logits"], row["mutant_logits"])
        with sdpa_kernel(SDPBackend.MATH):
            math_result, _ = observed_forward(model, ids)
        joined = {
            key: torch.cat(
                [saved[n][key] for n in ("cached_prefill", "decode_4", "decode_5")],
                dim=1,
            )
            for key in ("hidden", "logits")
        }
        fixture["measurements"][dtype_name] = {
            "prefill_vs_serial_decode": {
                key: difference(saved["prefill"][key], joined[key]) for key in joined
            },
            "normal_vs_explicit_prefill": {
                key: difference(saved["prefill"][key], saved["explicit_prefill"][key])
                for key in joined
            },
            "default_vs_math_sdpa": {
                key: difference(saved["prefill"][key], math_result[key])
                for key in joined
            },
            "math_sdpa_hidden": math_result["hidden"],
            "math_sdpa_logits": math_result["logits"],
        }
        if dtype == torch.bfloat16:
            rounded_fp32, _ = observed_forward(copy.deepcopy(model).float(), ids)
            fixture["measurements"]["bf16_weights_fp32_math"] = {
                "hidden": rounded_fp32["hidden"],
                "logits": rounded_fp32["logits"],
                "vs_bf16": {
                    key: difference(rounded_fp32[key], saved["prefill"][key])
                    for key in joined
                },
            }
    fixture["measurements"]["fp32_vs_bf16"] = {
        key: difference(baseline["float32"][key], baseline["bfloat16"][key])
        for key in ("hidden", "logits")
    }
    load_reference(source)  # Fail if the pinned source changed during generation.
    return json.loads(json.dumps(fixture, default=tensor_record, allow_nan=False))


def difference(expected, actual):
    if expected.shape != actual.shape:
        raise ValueError(f"shape mismatch: {expected.shape} != {actual.shape}")
    if not (torch.isfinite(expected).all() and torch.isfinite(actual).all()):
        raise ValueError("nonfinite comparison")
    delta = (expected.double() - actual.double()).abs()
    return {
        "elements": expected.numel(),
        "different_elements": int(torch.count_nonzero(delta)),
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "rms": delta.square().mean().sqrt().item(),
    }


def tensor_record(x):
    """Export logical row-major order, not backing storage order."""
    import math

    data = x.detach().cpu().contiguous().reshape(-1).tolist()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype).removeprefix("torch."),
        "data": [
            v
            if math.isfinite(v)
            else ("nan" if math.isnan(v) else "+inf" if v > 0 else "-inf")
            for v in data
        ],
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = generate(args.source)
    fixture["provenance"]["invocation"] = sys.orig_argv
    fixture["provenance"]["working_directory"] = str(Path.cwd())
    text = (
        json.dumps(fixture, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    results = {
        "fixture_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "fixture_bytes": len(text.encode()),
        "forward_case_count": len(fixture["forward_cases"]),
        "measurements": fixture["measurements"],
        "mask_differences": {
            dtype: {row["name"]: row["differences"] for row in rows}
            for dtype, rows in fixture["mask_audit"].items()
        },
        "mutation_differences": {
            dtype: {row["name"]: row["difference"] for row in rows}
            for dtype, rows in fixture["mutations"].items()
        },
        "component_differences": {
            dtype: {name: row["differences"] for name, row in rows.items()}
            for dtype, rows in fixture["components"].items()
        },
        "incomplete_gates": fixture["incomplete_gates"],
    }
    args.output.with_suffix(".results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                key: results[key]
                for key in ("fixture_sha256", "fixture_bytes", "forward_case_count")
            }
        )
    )

#!/usr/bin/env python3
"""Inspect Hugging Face model directories before porting them to Ollama MLX.

The script reads config/tokenizer files and safetensors headers without loading
weights. It writes:

  - porting_manifest.json: structured data for later tooling
  - porting_manifest.md: human-readable summary for contributors/reviewers

Local directories require only the Python standard library. Hub model IDs are
supported when `huggingface_hub` is installed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import struct
import sys
from typing import Any


HEADER_LIMIT = 100 * 1024 * 1024


def load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else key
            out.update(flatten_config(value[key], child))
    else:
        out[prefix] = value
    return out


def read_safetensors_header(path: pathlib.Path) -> dict[str, Any]:
    with path.open("rb") as f:
        raw_size = f.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"{path}: missing safetensors header size")
        (header_size,) = struct.unpack("<Q", raw_size)
        if header_size > HEADER_LIMIT:
            raise ValueError(f"{path}: safetensors header too large: {header_size}")
        header = f.read(header_size)
        if len(header) != header_size:
            raise ValueError(f"{path}: truncated safetensors header")
    parsed = json.loads(header.decode("utf-8"))
    parsed.pop("__metadata__", None)
    return parsed


def resolve_model(spec: str, revision: str | None) -> tuple[pathlib.Path, str, str]:
    path = pathlib.Path(spec).expanduser()
    if path.exists():
        return path.resolve(), spec, "local"

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            f"{spec!r} is not a local path and huggingface_hub is not installed"
        ) from e

    patterns = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "processor_config.json",
        "image_processor_config.json",
        "*.safetensors",
        "*.safetensors.index.json",
    ]
    downloaded = snapshot_download(
        repo_id=spec,
        revision=revision,
        allow_patterns=patterns,
    )
    return pathlib.Path(downloaded), spec, "hub"


def safetensors_summary(model_dir: pathlib.Path) -> dict[str, Any]:
    dtype_counts: collections.Counter[str] = collections.Counter()
    dtype_bytes: collections.Counter[str] = collections.Counter()
    prefix_counts: collections.Counter[str] = collections.Counter()
    header_digests: list[dict[str, Any]] = []
    tensor_count = 0

    for path in sorted(model_dir.glob("*.safetensors")):
        header = read_safetensors_header(path)
        header_digests.append(
            {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "header_sha256": hashlib.sha256(
                    stable_json(header).encode("utf-8")
                ).hexdigest(),
                "tensor_count": len(header),
            }
        )
        for name, info in sorted(header.items()):
            if not isinstance(info, dict):
                continue
            tensor_count += 1
            dtype = str(info.get("dtype", "unknown"))
            dtype_counts[dtype] += 1
            offsets = info.get("data_offsets")
            if isinstance(offsets, list) and len(offsets) == 2:
                dtype_bytes[dtype] += int(offsets[1]) - int(offsets[0])
            parts = name.split(".")
            for depth in (1, 2, 3):
                if len(parts) >= depth:
                    prefix_counts[".".join(parts[:depth])] += 1

    return {
        "tensor_count": tensor_count,
        "dtype_histogram": dict(sorted(dtype_counts.items())),
        "dtype_bytes": dict(sorted(dtype_bytes.items())),
        "tensor_prefixes": [
            {"prefix": prefix, "count": count}
            for prefix, count in prefix_counts.most_common(12)
        ],
        "safetensors_headers": header_digests,
    }


def selected_config(raw: dict[str, Any]) -> dict[str, Any]:
    text = raw.get("text_config")
    if isinstance(text, dict):
        return text
    return raw


def tokenizer_summary(model_dir: pathlib.Path, raw_config: dict[str, Any]) -> dict[str, Any]:
    files = {
        "tokenizer_json": (model_dir / "tokenizer.json").exists(),
        "tokenizer_config_json": (model_dir / "tokenizer_config.json").exists(),
        "generation_config_json": (model_dir / "generation_config.json").exists(),
        "special_tokens_map_json": (model_dir / "special_tokens_map.json").exists(),
        "processor_config_json": (model_dir / "processor_config.json").exists(),
        "preprocessor_config_json": (model_dir / "preprocessor_config.json").exists(),
        "image_processor_config_json": (model_dir / "image_processor_config.json").exists(),
    }
    chat_template = ""
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        try:
            chat_template = str(load_json(tokenizer_config_path).get("chat_template") or "")
        except Exception:
            chat_template = ""
    if not chat_template:
        chat_template = str(raw_config.get("chat_template") or "")

    return {
        **files,
        "has_chat_template": bool(chat_template),
        "chat_template_mentions_thinking": "think" in chat_template.lower(),
    }


def compact_detail(value: Any, limit: int = 160) -> str:
    text = stable_json(value) if not isinstance(value, str) else value
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def risk_flags(
    raw_config: dict[str, Any],
    active_config: dict[str, Any],
    tok: dict[str, Any],
) -> list[dict[str, str]]:
    flat_raw = flatten_config(raw_config)
    flat_active = flatten_config(active_config)
    risks: list[dict[str, str]] = []

    rope = {k: v for k, v in flat_active.items() if "rope" in k.lower()}
    if rope:
        risks.append({"name": "rope", "detail": compact_detail(rope)})

    layer_types = active_config.get("layer_types")
    sliding = {
        k: v
        for k, v in flat_active.items()
        if "sliding" in k.lower() or "window" in k.lower()
    }
    if sliding or (
        isinstance(layer_types, list)
        and any("sliding" in str(x).lower() for x in layer_types)
    ):
        risks.append({"name": "sliding_window", "detail": compact_detail(sliding)})

    if any(
        key in active_config
        for key in (
            "num_experts",
            "n_routed_experts",
            "num_local_experts",
            "moe_intermediate_size",
        )
    ):
        detail = {
            k: active_config.get(k)
            for k in (
                "num_experts",
                "n_routed_experts",
                "num_local_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
            )
            if k in active_config
        }
        risks.append({"name": "moe", "detail": compact_detail(detail)})

    if isinstance(layer_types, list) and len({str(x) for x in layer_types}) > 1:
        risks.append(
            {
                "name": "hybrid_layers",
                "detail": compact_detail(collections.Counter(map(str, layer_types))),
            }
        )

    if active_config.get("tie_word_embeddings") is True:
        risks.append({"name": "tied_embeddings", "detail": "tie_word_embeddings=true"})

    if active_config.get("attention_bias") is True:
        risks.append({"name": "attention_bias", "detail": "attention_bias=true"})

    multimodal_keys = [
        k
        for k in flat_raw
        if any(part in k.lower() for part in ("vision", "audio", "image", "processor"))
    ]
    if multimodal_keys or tok.get("processor_config_json") or tok.get("image_processor_config_json"):
        risks.append(
            {
                "name": "multimodal",
                "detail": compact_detail(multimodal_keys[:10] or "processor files present"),
            }
        )

    if tok.get("chat_template_mentions_thinking"):
        risks.append({"name": "thinking_template", "detail": "chat template mentions think"})

    if "auto_map" in raw_config:
        risks.append({"name": "custom_transformers_code", "detail": compact_detail(raw_config["auto_map"])})

    return risks


def inspect_one(spec: str, revision: str | None) -> dict[str, Any]:
    model_dir, source, source_type = resolve_model(spec, revision)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"{model_dir}: missing config.json")

    raw = load_json(config_path)
    active = selected_config(raw)
    tok = tokenizer_summary(model_dir, raw)
    tensors = safetensors_summary(model_dir)

    arch = raw.get("architectures") or active.get("architectures") or []
    if isinstance(arch, str):
        arch = [arch]

    return {
        "label": model_dir.name,
        "source": source,
        "source_type": source_type,
        "path": str(model_dir),
        "config_sha256": file_sha256(config_path),
        "architectures": arch,
        "model_type": active.get("model_type") or raw.get("model_type"),
        "has_text_config": isinstance(raw.get("text_config"), dict),
        "tokenizer": tok,
        "safetensors": tensors,
        "risk_flags": risk_flags(raw, active, tok),
        "flat_config": flatten_config(active),
        "suggested_dump_activations": suggested_dump_command(model_dir, arch),
    }


def suggested_dump_command(model_dir: pathlib.Path, architectures: list[str]) -> str:
    parts = [
        ".venv/bin/python3",
        "x/models/scripts/dump_activations.py",
        "--model",
        str(model_dir),
    ]
    if architectures:
        parts += ["--model-class", architectures[0]]
    parts += ["--skip-logits"]
    return " ".join(parts)


def config_diffs(models: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if len(models) < 2:
        return {}
    keys = sorted({k for m in models for k in m["flat_config"]})
    out: dict[str, dict[str, Any]] = {}
    for key in keys:
        values = {m["label"]: m["flat_config"].get(key) for m in models}
        encoded = {stable_json(v) for v in values.values()}
        if len(encoded) > 1:
            out[key] = values
    return out


def write_markdown(manifest: dict[str, Any], path: pathlib.Path) -> None:
    lines: list[str] = ["# MLX Porting Manifest", ""]
    for model in manifest["models"]:
        lines += [
            f"## {model['label']}",
            "",
            f"- Source: `{model['source']}` ({model['source_type']})",
            f"- Path: `{model['path']}`",
            f"- Architecture(s): {', '.join(model['architectures']) or 'unknown'}",
            f"- Model type: `{model.get('model_type') or 'unknown'}`",
            f"- Nested text_config: {model['has_text_config']}",
            f"- Tensor count: {model['safetensors']['tensor_count']}",
            f"- Dtype histogram: `{stable_json(model['safetensors']['dtype_histogram'])}`",
            "",
            "### Risk Flags",
            "",
        ]
        if model["risk_flags"]:
            lines += [
                f"- `{risk['name']}`: {risk['detail']}" for risk in model["risk_flags"]
            ]
        else:
            lines.append("- none detected")

        lines += ["", "### Tensor Prefixes", ""]
        prefixes = model["safetensors"]["tensor_prefixes"]
        if prefixes:
            lines += [f"- `{p['prefix']}`: {p['count']}" for p in prefixes]
        else:
            lines.append("- none found")

        lines += [
            "",
            "### Suggested Reference Command",
            "",
            "```bash",
            model["suggested_dump_activations"],
            "```",
            "",
        ]

    diffs = manifest.get("config_diffs") or {}
    lines += ["## Config Differences Across Variants", ""]
    if diffs:
        for key in sorted(diffs):
            lines.append(f"- `{key}`: `{stable_json(diffs[key])}`")
    else:
        lines.append("- none; only one variant inspected or no differences found")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", action="append", required=True, help="Local HF model dir or Hub repo ID")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--revision", default=None, help="Hub revision used for repo IDs")
    args = parser.parse_args(argv)

    out_dir = pathlib.Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    full_models = [inspect_one(spec, args.revision) for spec in args.model]
    models = []
    for model in full_models:
        emitted = dict(model)
        emitted.pop("flat_config", None)
        models.append(emitted)
    manifest = {
        "schema_version": 1,
        "generated_by": "x/models/scripts/inspect_model.py",
        "models": models,
        "config_diffs": config_diffs(full_models),
    }

    json_path = out_dir / "porting_manifest.json"
    md_path = out_dir / "porting_manifest.md"
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(manifest, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(run())

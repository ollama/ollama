#!/usr/bin/env python3
"""Build a reviewer-ready MLX port validation report from recorded artifacts."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


def load_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_go_test_json(path: pathlib.Path) -> dict[str, Any]:
    counts = {"pass": 0, "fail": 0, "skip": 0}
    failures: list[str] = []
    skips: list[str] = []
    packages: set[str] = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        pkg = event.get("Package")
        if pkg:
            packages.add(pkg)
        test = event.get("Test")
        action = event.get("Action")
        if not test or action not in counts:
            continue
        counts[action] += 1
        name = f"{pkg}/{test}" if pkg else test
        if action == "fail":
            failures.append(name)
        elif action == "skip":
            skips.append(name)

    return {
        "counts": counts,
        "packages": sorted(packages),
        "failures": failures,
        "skips": skips,
    }


def summarize_models(manifest: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for model in manifest.get("models", []):
        risks = model.get("risk_flags") or []
        risk_names = ", ".join(r["name"] for r in risks) if risks else "none detected"
        lines += [
            f"- {model.get('label', 'unknown')}:",
            f"  - source: `{model.get('source', 'unknown')}`",
            f"  - architecture(s): {', '.join(model.get('architectures') or []) or 'unknown'}",
            f"  - model type: `{model.get('model_type') or 'unknown'}`",
            f"  - dtypes: `{json.dumps(model.get('safetensors', {}).get('dtype_histogram', {}), sort_keys=True)}`",
            f"  - risks: {risk_names}",
        ]
    return lines or ["- no models recorded"]


def summarize_activation_manifests(paths: list[pathlib.Path]) -> list[str]:
    if not paths:
        return ["- no activation manifests provided"]
    lines: list[str] = []
    for path in paths:
        data = load_json(path)
        output = data.get("output_path") or data.get("activation_path") or str(path)
        tensors = data.get("tensors") or {}
        tensor_count = len(tensors) if isinstance(tensors, dict) else data.get("tensor_count", "unknown")
        lines += [
            f"- `{output}`",
            f"  - model: `{data.get('model_path', 'unknown')}`",
            f"  - class: `{data.get('model_class', 'auto')}`",
            f"  - dtype: `{data.get('dtype', 'unknown')}`",
            f"  - attention: `{data.get('attn_implementation') or 'unknown'}`",
            f"  - use_cache: `{data.get('use_cache', 'unknown')}`",
            f"  - tokens: {data.get('num_tokens', 'unknown')}",
            f"  - tensors: {tensor_count}",
            f"  - prompt_sha256: `{data.get('prompt_sha256', 'unknown')}`",
        ]
        if data.get("prefill_num_tokens"):
            lines += [
                f"  - cached prefill tokens: {data.get('prefill_num_tokens')}",
                f"  - decode text: `{data.get('decode_text') or ''}`",
                f"  - decode token ids: `{data.get('decode_token_ids') or []}`",
            ]
    return lines


def summarize_activation_comparisons(paths: list[pathlib.Path]) -> list[str]:
    if not paths:
        return ["- no activation comparison JSON provided"]
    lines: list[str] = []
    for path in paths:
        data = load_json(path)
        counts = data.get("counts") or {}
        lines += [
            f"- `{path}`",
            f"  - got: `{data.get('got', 'unknown')}`",
            f"  - want: `{data.get('want', 'unknown')}`",
            f"  - filters: `{', '.join(data.get('filters') or []) or '*'}`",
            f"  - axis: `{data.get('axis', 'unknown')}`",
            f"  - counts: `{json.dumps(counts, sort_keys=True)}`",
        ]
        top = data.get("top_absolute") or []
        if top:
            worst = top[0]
            lines.append(
                "  - worst absolute: "
                f"`{worst.get('name', 'unknown')}` "
                f"max_diff={worst.get('max_diff', 'unknown')} "
                f"first_tol={worst.get('first_tol', 'unknown')}"
            )
    return lines


def summarize_ppl(path: pathlib.Path | None) -> list[str]:
    if path is None:
        return ["- no perplexity JSON provided"]
    data = load_json(path)
    lines = [
        f"- model: `{data.get('model', 'unknown')}`",
        f"- mode: `{data.get('mode', 'unknown')}`",
        f"- max length: {data.get('max_length', 'unknown')}",
        f"- tokens: {data.get('total_tokens', 'unknown')}",
        f"- token PPL: {data.get('token_perplexity', 'unknown')}",
    ]
    delta = data.get("baseline_delta")
    if isinstance(delta, dict):
        lines.append(
            "- baseline delta: "
            f"abs={delta.get('token_perplexity_abs')}, "
            f"rel={delta.get('token_perplexity_rel')}"
        )
    return lines


def summarize_go_tests(paths: list[pathlib.Path]) -> list[str]:
    if not paths:
        return ["- no go test JSON provided"]
    summaries = [parse_go_test_json(path) for path in paths]
    counts = {"pass": 0, "fail": 0, "skip": 0}
    packages: set[str] = set()
    failures: list[str] = []
    skips: list[str] = []
    for summary in summaries:
        for key in counts:
            counts[key] += summary["counts"][key]
        packages.update(summary["packages"])
        failures.extend(summary["failures"])
        skips.extend(summary["skips"])

    lines = [
        f"- files: {', '.join(str(path) for path in paths)}",
        f"- packages: {', '.join(sorted(packages)) or 'unknown'}",
        f"- passed: {counts['pass']}",
        f"- failed: {counts['fail']}",
        f"- skipped: {counts['skip']}",
    ]
    if failures:
        lines.append("- failures: " + ", ".join(failures))
    if skips:
        lines.append("- skips: " + ", ".join(skips))
    return lines


def read_transcript(path: pathlib.Path | None) -> str:
    if path is None:
        return "No generation transcript provided."
    text = path.read_text(encoding="utf-8").strip()
    if len(text) > 4000:
        return text[:3997] + "..."
    return text or "Generation transcript was empty."


def build_report(
    manifest: dict[str, Any],
    activation_manifest_paths: list[pathlib.Path],
    activation_comparison_paths: list[pathlib.Path],
    go_test_json: list[pathlib.Path],
    ppl_json: pathlib.Path | None,
    generation_transcript: pathlib.Path | None,
) -> str:
    lines: list[str] = [
        "# MLX Port Validation Report",
        "",
        "## Summary",
        "",
        *summarize_models(manifest),
        "",
        "## Variant And Config Coverage",
        "",
    ]
    diffs = manifest.get("config_diffs") or {}
    if diffs:
        lines += [f"- `{key}`: `{json.dumps(value, sort_keys=True)}`" for key, value in sorted(diffs.items())]
    else:
        lines.append("- no config differences recorded")

    lines += [
        "",
        "## Reference Activations",
        "",
        *summarize_activation_manifests(activation_manifest_paths),
        "",
        "## Activation Comparisons",
        "",
        *summarize_activation_comparisons(activation_comparison_paths),
        "",
        "## Go Test Results",
        "",
        *summarize_go_tests(go_test_json),
        "",
        "## Perplexity",
        "",
        *summarize_ppl(ppl_json),
        "",
        "## Generation Samples",
        "",
        read_transcript(generation_transcript),
        "",
        "## Known Skips Or Limitations",
        "",
        "- Review the Go test skips and missing artifacts above before merging.",
        "- Treat generation quality and PPL deltas as review evidence, not automatic approval.",
        "",
    ]
    return "\n".join(lines)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="porting_manifest.json from inspect_model.py")
    parser.add_argument("--activation-manifest", action="append", default=[], help="activation sidecar manifest; can repeat")
    parser.add_argument("--activation-comparison-json", action="append", default=[], help="compare_activations.py JSON output; can repeat")
    parser.add_argument("--go-test-json", action="append", default=[], help="go test -json output; can repeat")
    parser.add_argument("--ppl-json", default=None, help="x/cmd/ppl -format json output")
    parser.add_argument("--generation-transcript", default=None, help="optional generation transcript Markdown/text")
    parser.add_argument("--output", required=True, help="output Markdown report")
    args = parser.parse_args(argv)

    manifest_path = pathlib.Path(args.manifest)
    activation_paths = [pathlib.Path(p) for p in args.activation_manifest]
    activation_comparison_paths = [pathlib.Path(p) for p in args.activation_comparison_json]
    go_test_paths = [pathlib.Path(p) for p in args.go_test_json]
    ppl_path = pathlib.Path(args.ppl_json) if args.ppl_json else None
    generation_path = pathlib.Path(args.generation_transcript) if args.generation_transcript else None
    output_path = pathlib.Path(args.output)

    report = build_report(
        load_json(manifest_path),
        activation_paths,
        activation_comparison_paths,
        go_test_paths,
        ppl_path,
        generation_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(run())

#!/usr/bin/env python3
"""Compare two safetensors activation artifacts.

This is intentionally model-neutral: it compares tensors by key, reports
global drift, and optionally scans one axis (usually sequence position) to find
the first position that exceeds a tolerance. Use it to compare PyTorch
reference variants, or any two activation dumps produced during a port.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import pathlib
import statistics
import sys
from typing import Any

import torch
from safetensors import safe_open


def matches_any(name: str, patterns: list[str]) -> bool:
    return not patterns or any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def tensor_keys(path: pathlib.Path) -> list[str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return list(handle.keys())


def load_tensor(path: pathlib.Path, name: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def cosine_similarity(got: torch.Tensor, want: torch.Tensor) -> float:
    got_flat = got.reshape(-1).to(torch.float64)
    want_flat = want.reshape(-1).to(torch.float64)
    denom = torch.linalg.vector_norm(got_flat) * torch.linalg.vector_norm(want_flat)
    if float(denom) == 0.0:
        return 1.0 if torch.equal(got_flat, want_flat) else 0.0
    return float(torch.dot(got_flat, want_flat) / denom)


def position_stats(
    diff: torch.Tensor,
    got: torch.Tensor,
    want: torch.Tensor,
    axis: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if axis < 0 or axis >= diff.ndim:
        return {}

    moved_diff = torch.movedim(diff, axis, 0).reshape(diff.shape[axis], -1)
    moved_got = torch.movedim(got, axis, 0).reshape(diff.shape[axis], -1).to(torch.float64)
    moved_want = torch.movedim(want, axis, 0).reshape(diff.shape[axis], -1).to(torch.float64)
    moved_tol = atol + rtol * torch.abs(moved_want)

    pos_max = torch.amax(moved_diff, dim=1)
    pos_mean = torch.mean(moved_diff, dim=1)
    pos_passed = torch.all(moved_diff <= moved_tol, dim=1)

    first_tol = -1
    failing = torch.nonzero(~pos_passed, as_tuple=False)
    if failing.numel() > 0:
        first_tol = int(failing[0].item())

    worst_pos = int(torch.argmax(pos_max).item()) if pos_max.numel() else -1
    med = statistics.median(float(v) for v in pos_max.tolist()) if pos_max.numel() else 0.0
    rel_to_median = float(pos_max[worst_pos]) / max(med, 1e-6) if worst_pos >= 0 else 0.0

    pos_cos = None
    if worst_pos >= 0:
        g = moved_got[worst_pos]
        w = moved_want[worst_pos]
        denom = torch.linalg.vector_norm(g) * torch.linalg.vector_norm(w)
        pos_cos = 1.0 if float(denom) == 0.0 and torch.equal(g, w) else 0.0
        if float(denom) != 0.0:
            pos_cos = float(torch.dot(g, w) / denom)

    return {
        "axis": axis,
        "first_tol": first_tol,
        "worst_pos": worst_pos,
        "worst_pos_max_diff": float(pos_max[worst_pos]) if worst_pos >= 0 else 0.0,
        "worst_pos_mean_diff": float(pos_mean[worst_pos]) if worst_pos >= 0 else 0.0,
        "worst_pos_cos_sim": pos_cos,
        "worst_pos_rel_to_median": rel_to_median,
    }


def compare_tensor(
    name: str,
    got: torch.Tensor,
    want: torch.Tensor,
    axis: int | None,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "got_dtype": str(got.dtype),
        "want_dtype": str(want.dtype),
        "got_shape": list(got.shape),
        "want_shape": list(want.shape),
    }

    if tuple(got.shape) != tuple(want.shape):
        result["status"] = "shape_mismatch"
        return result

    got_f = got.to(torch.float32)
    want_f = want.to(torch.float32)
    diff = torch.abs(got_f - want_f)
    tol = atol + rtol * torch.abs(want_f)
    passed = bool(torch.all(diff <= tol).item())

    result.update({
        "status": "pass" if passed else "fail",
        "max_diff": float(torch.max(diff).item()) if diff.numel() else 0.0,
        "mean_diff": float(torch.mean(diff).item()) if diff.numel() else 0.0,
        "cos_sim": cosine_similarity(got_f, want_f),
    })
    if axis is not None:
        result.update(position_stats(diff, got_f, want_f, axis, atol, rtol))
    return result


def compare_files(
    got_path: pathlib.Path,
    want_path: pathlib.Path,
    patterns: list[str],
    axis: int | None,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    got_keys = set(tensor_keys(got_path))
    want_keys = set(tensor_keys(want_path))
    selected = sorted(k for k in got_keys & want_keys if matches_any(k, patterns))

    results: list[dict[str, Any]] = []
    for name in selected:
        results.append(compare_tensor(
            name,
            load_tensor(got_path, name),
            load_tensor(want_path, name),
            axis,
            atol,
            rtol,
        ))

    missing_from_got = sorted(k for k in want_keys - got_keys if matches_any(k, patterns))
    missing_from_want = sorted(k for k in got_keys - want_keys if matches_any(k, patterns))
    for name in missing_from_got:
        results.append({"name": name, "status": "missing_from_got"})
    for name in missing_from_want:
        results.append({"name": name, "status": "missing_from_want"})

    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1

    failed = [r for r in results if r["status"] != "pass"]
    ranked_abs = sorted(
        (r for r in results if "max_diff" in r),
        key=lambda r: r["max_diff"],
        reverse=True,
    )
    ranked_rel = sorted(
        (r for r in results if "worst_pos_rel_to_median" in r),
        key=lambda r: r["worst_pos_rel_to_median"],
        reverse=True,
    )

    return {
        "schema_version": 1,
        "got": str(got_path),
        "want": str(want_path),
        "filters": patterns,
        "axis": axis,
        "atol": atol,
        "rtol": rtol,
        "counts": counts,
        "all_passed": not failed,
        "results": results,
        "top_absolute": ranked_abs[:20],
        "top_relative": ranked_rel[:20],
    }


def fmt_float(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.6g}"
    return str(value)


def markdown_report(report: dict[str, Any], top: int) -> str:
    lines = [
        "# Activation Comparison",
        "",
        f"- got: `{report['got']}`",
        f"- want: `{report['want']}`",
        f"- filters: `{', '.join(report['filters']) if report['filters'] else '*'}`",
        f"- axis: `{report['axis']}`",
        f"- tolerance: `atol={report['atol']} rtol={report['rtol']}`",
        f"- counts: `{json.dumps(report['counts'], sort_keys=True)}`",
        "",
        "## Top Absolute Drift",
        "",
        "| tensor | status | max_diff | mean_diff | cos_sim | first_tol | worst_pos | rel_to_median |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report["top_absolute"][:top]:
        lines.append(
            "| {name} | {status} | {max_diff} | {mean_diff} | {cos_sim} | {first_tol} | {worst_pos} | {rel} |".format(
                name=result["name"],
                status=result["status"],
                max_diff=fmt_float(result.get("max_diff", "")),
                mean_diff=fmt_float(result.get("mean_diff", "")),
                cos_sim=fmt_float(result.get("cos_sim", "")),
                first_tol=result.get("first_tol", ""),
                worst_pos=result.get("worst_pos", ""),
                rel=fmt_float(result.get("worst_pos_rel_to_median", "")),
            )
        )

    lines += [
        "",
        "## Top Relative Position Drift",
        "",
        "| tensor | status | worst_pos | max_diff | mean_diff | cos_sim | rel_to_median |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report["top_relative"][:top]:
        lines.append(
            "| {name} | {status} | {worst_pos} | {max_diff} | {mean_diff} | {cos_sim} | {rel} |".format(
                name=result["name"],
                status=result["status"],
                worst_pos=result.get("worst_pos", ""),
                max_diff=fmt_float(result.get("worst_pos_max_diff", result.get("max_diff", ""))),
                mean_diff=fmt_float(result.get("worst_pos_mean_diff", result.get("mean_diff", ""))),
                cos_sim=fmt_float(result.get("worst_pos_cos_sim", result.get("cos_sim", ""))),
                rel=fmt_float(result.get("worst_pos_rel_to_median", "")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--got", required=True, help="candidate activation safetensors")
    parser.add_argument("--want", required=True, help="reference activation safetensors")
    parser.add_argument("--filter", action="append", default=[], help="fnmatch tensor pattern to compare; can repeat")
    parser.add_argument("--axis", type=int, default=1, help="axis to scan for first tolerance drift; use -1 to disable")
    parser.add_argument("--atol", type=float, default=0.5, help="absolute tolerance")
    parser.add_argument("--rtol", type=float, default=0.05, help="relative tolerance")
    parser.add_argument("--top", type=int, default=15, help="rows to include in Markdown tables")
    parser.add_argument("--json-output", help="write structured JSON report")
    parser.add_argument("--markdown-output", help="write Markdown report")
    parser.add_argument("--require-pass", action="store_true", help="exit nonzero if any compared tensor fails")
    args = parser.parse_args(argv)

    axis = None if args.axis < 0 else args.axis
    report = compare_files(
        pathlib.Path(args.got),
        pathlib.Path(args.want),
        args.filter,
        axis,
        args.atol,
        args.rtol,
    )

    if args.json_output:
        path = pathlib.Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {path}")

    markdown = markdown_report(report, args.top)
    if args.markdown_output:
        path = pathlib.Path(args.markdown_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        print(f"Wrote {path}")
    elif not args.json_output:
        print(markdown)

    if args.require_pass and not report["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())

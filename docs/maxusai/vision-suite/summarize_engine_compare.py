#!/usr/bin/env python3
"""Render the engine-comparison tables from scores_<tag>.json + ft_<tag>.json.

Usage:
    python3 summarize_engine_compare.py [--dir RUNDIR] <model> [model ...]

Models are the names given to run_engine_compare.sh, in row order; tags derive
the same way (':' and '.' become '_'). Output is the exact two-markdown-table
format of the 2026-08-08 MLX-vs-GGUF campaign
(../vision-campaign-2026-08-08-mlx.md) — keep it stable so runs diff cleanly.

Engine column: safetensors tags are the MLX engine on this fork; the store
names them by MLX-side quantization ("-nvfp4"). Anything else renders GGUF.
Override per model with ENGINE_MAP="model=Engine,model=Engine" if a store
breaks that naming convention.
"""
import json
import os
import sys


def tag_for(model):
    return model.replace(":", "_").replace(".", "_")


def engine_for(model, engine_map):
    if model in engine_map:
        return engine_map[model]
    return "MLX" if "nvfp4" in model else "GGUF"


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        return None


def fmt_bool(v):
    return "✅" if v else "❌"


def main():
    args = sys.argv[1:]
    rundir = os.path.dirname(os.path.abspath(__file__))
    if args and args[0] == "--dir":
        rundir = args[1]
        args = args[2:]
    if not args:
        sys.exit(__doc__)
    engine_map = {}
    for pair in os.environ.get("ENGINE_MAP", "").split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            engine_map[k.strip()] = v.strip()

    t1 = ["| Model | Engine | Scene bbox IoU | Boxes / labels / colors | Serial "
          "| Invoice (items · qty+price · total) | name_bbox hits |",
          "|---|---|---|---|---|---|---|"]
    t2 = ["| Model | Engine | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) "
          "| Gen tok/s | Prefill tok/s |",
          "|---|---|---|---|---|---|---|---|---|---|"]

    for model in args:
        tag = tag_for(model)
        eng = engine_for(model, engine_map)
        eng_cell = f"**{eng}**" if eng == "MLX" else eng
        scores = load(os.path.join(rundir, f"scores_{tag}.json")) or {}
        ft = load(os.path.join(rundir, f"ft_{tag}.json")) or {}
        sc = scores.get("scene_single", {})
        dc = scores.get("document_single", {})
        mu = scores.get("multi_3img", {})

        iou = sc.get("bbox_mean_iou")
        iou_cell = "—" if iou is None else (f"**{iou:.3f}**" if eng == "MLX" else f"{iou:.3f}")
        blc = (f"{sc.get('bbox_hits', '—')}/{sc.get('object_count', '—')} · "
               f"{sc.get('labels_found', '—')}/{sc.get('labels_total', '—')} · "
               f"{sc.get('colors_right', '—')}/{sc.get('labels_total', '—')}")
        inv = (f"{dc.get('items_found', '—')}/{dc.get('items_total', '—')} · "
               f"{dc.get('qty_price_right', '—')}/{dc.get('items_total', '—')} · "
               f"{fmt_bool(dc.get('total_right'))}")
        t1.append(f"| {model} | {eng_cell} | {iou_cell} | {blc} | "
                  f"{fmt_bool(sc.get('serial_found'))} | {inv} | {dc.get('name_bbox_hits', '—')} |")

        tiers = [str(ft.get(f"recall_{px}px", "—")) for px in (22, 16, 12, 9, 7)]
        if mu.get("q1_right") and mu.get("q2_right") and mu.get("q4_bbox_hit"):
            multi = "✅ all Qs + bbox"
        elif not mu:
            multi = "—"
        else:
            fails = [q for q in ("q1_right", "q2_right", "q4_bbox_hit") if not mu.get(q)]
            multi = "❌ " + ", ".join(fails)
        gen = sc.get("gen_tps")
        pre = sc.get("prefill_tps")
        t2.append(f"| {model} | {eng_cell} | " + " | ".join(tiers) +
                  f" | {multi} | {round(gen) if gen else '—'} | {round(pre) if pre else '—'} |")

    print("## Scene grounding (six objects, norm-1000 boxes) + document extraction\n")
    print("\n".join(t1))
    print("\n## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput\n")
    print("\n".join(t2))


if __name__ == "__main__":
    main()

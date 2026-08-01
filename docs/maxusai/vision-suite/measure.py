#!/usr/bin/env python3
"""Token-budget measurement against the ollama-rocm-nemotron test container.
Method: docs/maxusai/vision-token-budget-measurements.md — /api/generate,
num_predict:1, prompt_eval_count minus text-only baseline."""
import json, sys, base64, os, urllib.request

HOST = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11435"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "nemotron3:33b-q4_K_M"
IMGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testimgs")
SIZES = ["320x240", "640x480", "896x896", "1568x1568", "1920x1080",
         "2048x1664", "3000x2000", "3200x32"]

def gen(payload, timeout=900):
    req = urllib.request.Request(HOST + "/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

base = gen({"model": MODEL, "prompt": "Hi", "stream": False,
            "options": {"num_predict": 1}})["prompt_eval_count"]
print(f"text-only baseline: {base}")

results = {}
for name in SIZES:
    img = base64.b64encode(open(f"{IMGDIR}/{name}.png", "rb").read()).decode()
    try:
        r = gen({"model": MODEL, "prompt": "Describe briefly.", "images": [img],
                 "stream": False, "options": {"num_predict": 1}})
        delta = r["prompt_eval_count"] - base
        results[name] = delta
        print(f"{name:>10}: prompt_eval_count={r['prompt_eval_count']:>5}  visual+markers={delta}")
    except Exception as e:
        results[name] = f"ERROR: {e}"
        print(f"{name:>10}: ERROR {e}")

# knob check: image_max_tokens=1024 on a large image
img = base64.b64encode(open(f"{IMGDIR}/1920x1080.png", "rb").read()).decode()
r = gen({"model": MODEL, "prompt": "Describe briefly.", "images": [img], "stream": False,
         "options": {"num_predict": 1, "image_max_tokens": 1024}})
print(f"knob 1920x1080 @ image_max_tokens=1024: prompt_eval_count={r['prompt_eval_count']}  visual+markers={r['prompt_eval_count'] - base}")

# coherence smoke: one short caption
r = gen({"model": MODEL, "prompt": "What color is this image? Answer in one short sentence.",
         "images": [img], "stream": False, "options": {"num_predict": 60}})
print("coherence sample:", json.dumps(r["response"]))

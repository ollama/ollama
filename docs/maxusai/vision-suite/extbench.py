#!/usr/bin/env python3
"""External-benchmark slices against an ollama endpoint, scored locally.

Usage: extbench.py <host> <tag> [model] [benchmark]
e.g.   extbench.py http://127.0.0.1:11435 canon qwen3.6:35b-a3b-q4_k_m ocrbench

Benchmarks (rows pulled from the HF datasets-server REST API — no `datasets`
install, no HF token, public datasets only):

  ocrbench      echo840/OCRBench test        contains-match  (lmms-eval semantics)
  countbenchqa  vikhyatk/CountBenchQA test   integer match
  chartqa       lmms-lab-encoder/ChartQA test relaxed accuracy (+-5% numeric)
  refcoco       lmms-lab-encoder/RefCOCO val  dialect-aware bbox IoU

Env: LIMIT (default 50), OFFSET (0), THINK=on|false (false), ENDPOINT=generate|chat
(generate), NUM_PREDICT, NUM_CTX (16384), TIMEOUT (900), SLEEP (0 — seconds between
requests, to yield the GPU on a shared host).

Writes ext_<tag>_<bench>.json (per-item records + summary) beside the script and
caches images under extimgs/<bench>/.

The refcoco scorer reuses the dialect logic of vision_suite.py: it searches
pixel / norm-1000 / norm-0-1 spaces and xyxy / yxyx orders per item and keeps the
best, so qwen3.6 (bbox_2d, xyxy, norm-1000), gemma4 (box_2d, yxyx, norm-1000) and
nemotron3 (self-chosen key, pixel under reasoning) are all scored fairly. External
harnesses do not do this: lmms-eval's refcoco_bbox_rec demands normalized 0-1 xyxy
and scores everything else ~0, VLMEvalKit auto-detects the scale but not the order.
"""
import base64, json, os, re, sys, time, urllib.parse, urllib.request

DIR = os.path.dirname(os.path.abspath(__file__))
HOST = TAG = MODEL = BENCH = None

BENCHES = {
    "ocrbench":     dict(dataset="echo840/OCRBench",            config="default", split="test"),
    "countbenchqa": dict(dataset="vikhyatk/CountBenchQA",       config="default", split="test"),
    "chartqa":      dict(dataset="lmms-lab-encoder/ChartQA",    config="default", split="test"),
    "refcoco":      dict(dataset="lmms-lab-encoder/RefCOCO",    config="default", split="val"),
}

# Per-benchmark prompt suffixes. Kept close to the lmms-eval task YAMLs so scores are
# comparable to published harness numbers; refcoco asks for the model's own JSON
# dialect instead of forcing one convention (see module docstring).
SUFFIX = {
    "ocrbench": "\nAnswer the question using a single word or phrase.",
    "countbenchqa": "\nAnswer with a single number.",
    "chartqa": "\nAnswer the question using a single word or phrase.",
}

BBOX_KEYS = ("bbox", "bbox_2d", "box_2d", "box")


def http_json(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "maxusai-extbench/1"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def fetch_rows(bench, offset, limit):
    """HF datasets-server /rows, paged at its 100-row maximum."""
    spec = BENCHES[bench]
    out = []
    while len(out) < limit:
        n = min(100, limit - len(out))
        q = urllib.parse.urlencode(dict(dataset=spec["dataset"], config=spec["config"],
                                        split=spec["split"], offset=offset + len(out), length=n))
        page = http_json("https://datasets-server.huggingface.co/rows?" + q, timeout=120)
        rows = page.get("rows", [])
        if not rows:
            break
        out.extend(r["row"] for r in rows)
    return out


def cache_image(bench, idx, src):
    d = os.path.join(DIR, "extimgs", bench)
    os.makedirs(d, exist_ok=True)
    ext = ".png" if ".png" in src.lower().split("?")[0] else ".jpg"
    path = os.path.join(d, f"{idx:05d}{ext}")
    if not os.path.exists(path):
        req = urllib.request.Request(src, headers={"User-Agent": "maxusai-extbench/1"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    return path


def gen(prompt, image_path, fmt=None):
    num_ctx = int(os.environ.get("NUM_CTX", "16384"))
    num_predict = int(os.environ.get("NUM_PREDICT", "512"))
    img = base64.b64encode(open(image_path, "rb").read()).decode()
    payload = {
        "model": MODEL, "prompt": prompt, "images": [img], "stream": False,
        "options": {"num_predict": num_predict, "num_ctx": num_ctx, "temperature": 0},
    }
    if fmt:
        payload["format"] = fmt
    if os.environ.get("THINK", "false") != "on":
        payload["think"] = False
    timeout = int(os.environ.get("TIMEOUT", "900"))
    if os.environ.get("ENDPOINT", "generate") == "chat":
        payload["messages"] = [{"role": "user", "content": payload.pop("prompt"),
                                "images": payload.pop("images")}]
        req = urllib.request.Request(HOST + "/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
        msg = r.get("message") or {}
        r["response"], r["thinking"] = msg.get("content", ""), msg.get("thinking", "")
        return r
    req = urllib.request.Request(HOST + "/api/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


# ---------------------------------------------------------------- scorers

def norm_text(s):
    return re.sub(r"\s+", " ", str(s).lower().strip().replace("\n", " "))


def score_ocrbench(pred, row):
    """lmms-eval ocrbench_process_results: gold answer contained in the prediction.

    Handwritten-Mathematical-Expression-Recognition items are whitespace-stripped on
    both sides upstream; everything else is a plain normalized substring test.
    """
    golds = row.get("answer") or []
    if isinstance(golds, str):
        golds = [golds]
    p = norm_text(pred)
    if row.get("question_type") == "Handwritten Mathematical Expression Recognition":
        p_ns = re.sub(r"\s", "", p)
        return any(re.sub(r"\s", "", norm_text(g)) in p_ns for g in golds)
    return any(norm_text(g) in p for g in golds)


def score_count(pred, row):
    m = re.search(r"-?\d+", str(pred).replace(",", ""))
    if not m:
        return False
    try:
        return int(m.group()) == int(str(row["number"]).strip())
    except (ValueError, KeyError):
        return False


def score_chartqa(pred, row):
    """Relaxed accuracy: +-5% for numeric golds, normalized exact match otherwise."""
    gold = row.get("answer")
    if isinstance(gold, list):
        gold = gold[0] if gold else ""
    g = norm_text(gold)
    p = norm_text(pred)
    try:
        gv = float(re.sub(r"[%$,]", "", g))
    except ValueError:
        return p == g or g in p.split()
    m = re.findall(r"-?\d+\.?\d*", p.replace(",", "").replace("%", "").replace("$", ""))
    if not m:
        return False
    for cand in m:
        try:
            pv = float(cand)
        except ValueError:
            continue
        if gv == 0:
            if pv == 0:
                return True
        elif abs(pv - gv) / abs(gv) <= 0.05:
            return True
    return False


def parse_boxes(text):
    """Every plausible 4-number box in the response, with its JSON key when it had one."""
    out = []
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        obj = None

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in BBOX_KEYS and isinstance(v, list) and len(v) == 4:
                    try:
                        out.append((k, [float(x) for x in v]))
                    except (TypeError, ValueError):
                        pass
                else:
                    walk(v)
        elif isinstance(o, list):
            if len(o) == 4 and all(isinstance(x, (int, float)) for x in o):
                out.append((None, [float(x) for x in o]))
            for v in o:
                walk(v)

    if obj is not None:
        walk(obj)
    if not out:
        m = re.search(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,"
                      r"\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", str(text))
        if m:
            out.append((None, [float(g) for g in m.groups()]))
    return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def decode_box(box, space, order, W, H):
    if order == "yxyx":
        box = [box[1], box[0], box[3], box[2]]
    if space == "norm1000":
        fx, fy = W / 1000.0, H / 1000.0
    elif space == "norm01":
        fx, fy = float(W), float(H)
    else:
        fx = fy = 1.0
    x1, y1, x2, y2 = box[0] * fx, box[1] * fy, box[2] * fx, box[3] * fy
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def score_refcoco(pred, row):
    """Dialect-aware IoU: best over {pixel, norm-1000, norm-0-1} x {xyxy, yxyx}.

    Ground truth is COCO [x, y, w, h] in pixels. Returns the best IoU, the winning
    dialect, and the JSON key the model chose (so a run doubles as a dialect probe).
    """
    gt = row.get("bbox")
    if isinstance(gt, str):
        gt = json.loads(gt)
    gtb = [gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]
    W, H = row["_w"], row["_h"]
    best = (0.0, None, None)
    for key, box in parse_boxes(pred):
        for space in ("pixel", "norm1000", "norm01"):
            for order in ("xyxy", "yxyx"):
                v = iou(decode_box(box, space, order, W, H), gtb)
                if v > best[0]:
                    best = (v, f"{space}/{order}", key)
    return best


def image_size(path):
    """PNG/JPEG dimensions without Pillow (the suite's images come as either)."""
    with open(path, "rb") as f:
        head = f.read(2)
        if head == b"\x89P":
            f.seek(16)
            w, h = int.from_bytes(f.read(4), "big"), int.from_bytes(f.read(4), "big")
            return w, h
        f.seek(2)
        while True:
            b = f.read(1)
            while b and b != b"\xff":
                b = f.read(1)
            marker = f.read(1)
            while marker == b"\xff":
                marker = f.read(1)
            if marker in (b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6",
                          b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"):
                f.read(3)
                h = int.from_bytes(f.read(2), "big")
                w = int.from_bytes(f.read(2), "big")
                return w, h
            seg = int.from_bytes(f.read(2), "big")
            if seg < 2:
                raise ValueError(f"cannot read JPEG size: {path}")
            f.seek(seg - 2, 1)


def build_prompt(row):
    if BENCH == "refcoco":
        expr = row.get("answer")
        if isinstance(expr, str):
            try:
                expr = json.loads(expr.replace("'", '"'))
            except ValueError:
                expr = [expr]
        expr = expr[0] if isinstance(expr, list) and expr else str(expr)
        row["_expr"] = expr
        return ('Locate the region described as: "%s".\n'
                'Reply with JSON only: a single object with one key "bbox_2d" whose '
                'value is the bounding box of that region as four numbers. Use whatever '
                'coordinate convention you were trained on; do not add any other text.'
                % expr)
    return row["question"] + SUFFIX.get(BENCH, "")


def main():
    global HOST, TAG, MODEL, BENCH
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    HOST = sys.argv[1].rstrip("/")
    TAG = sys.argv[2]
    MODEL = sys.argv[3] if len(sys.argv) > 3 else "nemotron3:33b-q4_K_M"
    BENCH = sys.argv[4] if len(sys.argv) > 4 else "ocrbench"
    if BENCH not in BENCHES:
        sys.exit(f"unknown benchmark {BENCH!r}; pick one of {', '.join(BENCHES)}")

    limit = int(os.environ.get("LIMIT", "50"))
    offset = int(os.environ.get("OFFSET", "0"))
    sleep_s = float(os.environ.get("SLEEP", "0"))
    think = os.environ.get("THINK", "false")

    print(f"# {BENCH}: {BENCHES[BENCH]['dataset']} [{BENCHES[BENCH]['split']}] "
          f"rows {offset}..{offset + limit}")
    rows = fetch_rows(BENCH, offset, limit)
    print(f"# fetched {len(rows)} rows; model={MODEL} think={think} "
          f"endpoint={os.environ.get('ENDPOINT', 'generate')}")

    records, correct, ious, dialects, empty = [], 0, [], {}, 0
    for i, row in enumerate(rows):
        src = row["image"]["src"] if isinstance(row["image"], dict) else row["image"]
        path = cache_image(BENCH, offset + i, src)
        if BENCH == "refcoco":
            row["_w"], row["_h"] = image_size(path)
        prompt = build_prompt(row)
        t0 = time.time()
        try:
            r = gen(prompt, path, fmt="json" if BENCH == "refcoco" else None)
            pred = r.get("response", "")
        except Exception as e:                                    # noqa: BLE001
            records.append({"i": offset + i, "error": str(e)})
            print(f"{offset + i:>4}  ERROR {e}")
            continue
        dt = round(time.time() - t0, 1)
        if not pred.strip():
            empty += 1

        rec = {"i": offset + i, "prompt": prompt, "pred": pred, "secs": dt,
               "prompt_eval_count": r.get("prompt_eval_count"),
               "eval_count": r.get("eval_count")}
        if BENCH == "refcoco":
            v, dialect, key = score_refcoco(pred, row)
            ious.append(v)
            ok = v >= 0.5
            if dialect:
                dialects[f"{key or '-'} {dialect}"] = dialects.get(f"{key or '-'} {dialect}", 0) + 1
            rec.update(expr=row.get("_expr"), gt=row.get("bbox"), iou=round(v, 3),
                       dialect=dialect, key=key, ok=ok)
        else:
            ok = {"ocrbench": score_ocrbench, "countbenchqa": score_count,
                  "chartqa": score_chartqa}[BENCH](pred, row)
            rec.update(gold=row.get("answer") if BENCH != "countbenchqa" else row.get("number"),
                       ok=bool(ok))
        correct += bool(ok)
        records.append(rec)
        flag = "ok " if ok else "MISS"
        extra = f" iou={rec['iou']} {rec.get('dialect')}" if BENCH == "refcoco" else ""
        print(f"{offset + i:>4}  {flag}{extra}  {dt:>5}s  {norm_text(pred)[:70]!r}")
        if sleep_s:
            time.sleep(sleep_s)

    n = len([r for r in records if "error" not in r])
    summary = {
        "tag": TAG, "model": MODEL, "benchmark": BENCH,
        "dataset": BENCHES[BENCH]["dataset"], "split": BENCHES[BENCH]["split"],
        "offset": offset, "requested": limit, "scored": n,
        "errors": len(records) - n, "empty_responses": empty,
        "think": think, "endpoint": os.environ.get("ENDPOINT", "generate"),
        "correct": correct, "accuracy": round(correct / n, 4) if n else None,
    }
    if BENCH == "refcoco":
        summary["mean_iou"] = round(sum(ious) / len(ious), 4) if ious else None
        summary["acc@0.5"] = summary.pop("accuracy")
        summary["dialects"] = dialects
    out = os.path.join(DIR, f"ext_{TAG}_{BENCH}.json")
    json.dump({"summary": summary, "records": records}, open(out, "w"), indent=1)
    print("\n" + json.dumps(summary, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

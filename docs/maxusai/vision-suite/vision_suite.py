#!/usr/bin/env python3
"""Nemotron vision test suite: long prompts, single & multi image, JSON + bboxes.
Usage: vision_suite.py <host> <tag> [model]
e.g.   vision_suite.py http://127.0.0.1:11435 patched nemotron3:33b-q4_K_M
"""
import json, sys, base64, os, urllib.request

HOST = TAG = MODEL = None  # set in main()
DIR = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(DIR, "visimgs")
GT = json.load(open(f"{IMG}/ground_truth.json"))

def b64(name):
    return base64.b64encode(open(f"{IMG}/{name}", "rb").read()).decode()

def gen(prompt, images, num_predict=None, num_ctx=16384):
    if num_predict is None:
        num_predict = int(os.environ.get("NUM_PREDICT", "2200"))
    payload = {
        "model": MODEL, "prompt": prompt, "images": images,
        "stream": False, "format": "json",
        "options": {"num_predict": num_predict, "num_ctx": num_ctx, "temperature": 0},
    }
    if os.environ.get("THINK", "false") != "on":
        payload["think"] = False
    endpoint = os.environ.get("ENDPOINT", "generate")
    if endpoint == "chat":
        payload["messages"] = [{"role": "user", "content": payload.pop("prompt"),
                                "images": payload.pop("images")}]
        req = urllib.request.Request(HOST + "/api/chat",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        msg = r.get("message") or {}
        r["response"] = msg.get("content", "")
        r["thinking"] = msg.get("thinking", "")
        return r
    req = urllib.request.Request(HOST + "/api/generate",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))

SCENE_PROMPT = """You are a precision visual inspection system deployed in an industrial
quality-assurance pipeline. Your task on this frame is exhaustive object detection,
label transcription, and localization. Accuracy requirements are strict: downstream
robotic actuators consume your bounding boxes directly, so a box that misses its object
by more than a few percent of the frame causes a physical pick failure; a mis-transcribed
label causes the wrong part to be routed. Work methodically: first scan the entire frame
edge to edge, including corners and margins, then enumerate every distinct colored shape
you can find. For every shape, read the text label printed immediately above it — labels
are short uppercase code words, transcribe them EXACTLY, character by character, without
guessing or normalizing. If a label is genuinely illegible at the available resolution,
set "label" to null and "label_legible" to false rather than inventing a word; invented
labels are the single most damaging failure mode in this pipeline. Also transcribe any
other text present anywhere in the frame, however small, in the "other_text" array —
serial numbers, watermarks, footers, anything. Bounding boxes use ABSOLUTE PIXEL
coordinates in the original image coordinate system, formatted [x1, y1, x2, y2] where
(x1, y1) is the top-left corner and (x2, y2) the bottom-right corner of the shape itself
(not including its label text). The image is exactly {w} pixels wide and {h} pixels tall,
so all coordinates must lie in that range. For color, report the closest common English
color name (red, blue, green, orange, purple, teal, yellow, pink, brown, gray, black).
For shape kind use exactly "rectangle" or "ellipse". Respond with a SINGLE JSON object,
no prose before or after, following exactly this schema:
{{
  "image_width": <int>, "image_height": <int>,
  "object_count": <int>,
  "objects": [
    {{"label": <string or null>, "label_legible": <bool>, "kind": "rectangle"|"ellipse",
      "color": <string>, "bbox": [x1, y1, x2, y2], "confidence": <float 0..1>}}
  ],
  "other_text": [<string>, ...],
  "notes": <string, one short sentence on anything ambiguous>
}}
Do not omit any object. Do not merge adjacent objects. Count carefully before writing
object_count and make it equal to the length of the objects array."""

DOC_PROMPT = """You are an automated accounts-payable document parser. The attached image
is a scanned supplier invoice. Extract its contents COMPLETELY and EXACTLY into JSON for
direct ingestion into an ERP system; every field is compared against the purchase-order
database, so transcription must be verbatim — do not round numbers, do not paraphrase
item names, do not reformat identifiers. Read the entire page including headers, the
line-item table, totals, and any fine print at the bottom; fine-print reference codes
are mandatory fields for reconciliation. If any character is genuinely unreadable,
represent it as '?' rather than guessing. Amounts are in dollars; parse them as numbers
(strip the $ sign and thousands separators). For each line item give the bounding box of
the item-name text in ABSOLUTE PIXEL coordinates [x1, y1, x2, y2] (top-left and
bottom-right of the text run). The page is {w}x{h} pixels. Respond with a SINGLE JSON
object, no prose, exactly this schema:
{{
  "supplier": <string>, "invoice_number": <string>, "date": <string>,
  "customer": <string>,
  "line_items": [
    {{"name": <string>, "qty": <int>, "unit_price": <number>, "name_bbox": [x1,y1,x2,y2]}}
  ],
  "total": <number>,
  "fine_print": <string>,
  "all_reference_codes": [<string>, ...]
}}"""

MULTI_PROMPT = """You are a multi-document visual analyst. You receive THREE images in
order: image 1, image 2, image 3. Analyze each independently and then answer
cross-image questions. Be exhaustive but never invent content; if something is
unreadable, say so via null values rather than guessing. All bounding boxes are
ABSOLUTE PIXEL coordinates [x1, y1, x2, y2] in each image's own coordinate system.
For each image produce: a "type" classification (one of "shapes_scene",
"invoice_document", "bar_chart", "photo", "other"), a one-sentence "summary", a
"text_found" array with every distinct text string you can read in that image, and
"key_objects" — for a shapes scene: each shape with label+color+bbox; for a document:
the document identifier and the total amount; for a chart: every bar with its category
label and numeric value. Then answer the cross-image questions in the "answers" object:
q1: which image (1, 2 or 3) contains the reference code "INV-2026-0801"?
q2: in the bar chart, which category has the LARGEST value, and what is that value?
q3: does any single word that appears in image 1 also appear in image 2 or image 3?
    Answer with the word or null.
q4: give the bounding box, in image 1 pixel coordinates, of the shape whose label is
    "DYNAMO" (null if no such shape is legible).
Respond with a SINGLE JSON object, no prose:
{{
  "images": [
    {{"index": 1, "type": ..., "summary": ..., "text_found": [...], "key_objects": [...]}},
    {{"index": 2, ...}},
    {{"index": 3, ...}}
  ],
  "answers": {{"q1": <int>, "q2": {{"category": <string>, "value": <number>}},
               "q3": <string or null>, "q4": [x1,y1,x2,y2] or null}}
}}"""

def center_in(pred, gtb):
    try:
        cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
        return gtb[0] <= cx <= gtb[2] and gtb[1] <= cy <= gtb[3]
    except Exception:
        return False

def get_bbox(o):
    # Models speak different schema dialects: qwen-vl grounding uses "bbox_2d".
    for k in ("bbox", "bbox_2d", "box_2d"):
        if o.get(k):
            return o[k]
    return []

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0

def score_scene(resp_text):
    g = GT["scene_hd"]
    W, H = g["size"]
    s = {"json_valid": False, "labels_found": 0, "labels_total": len(g["objects"]),
         "bbox_hits": 0, "bbox_mean_iou": 0.0, "bbox_space": None,
         "colors_right": 0, "serial_found": g["serial"] in resp_text,
         "object_count": None}
    try:
        r = json.loads(resp_text); s["json_valid"] = True
    except Exception:
        return s
    objs = r.get("objects") or []
    s["object_count"] = len(objs)
    by_label = {o.get("label"): o for o in objs if o.get("label")}
    matched = []
    for gto in g["objects"]:
        o = by_label.get(gto["label"])
        if o:
            s["labels_found"] += 1
            if (o.get("color") or "").lower() == gto["color"]:
                s["colors_right"] += 1
            bb = get_bbox(o)
            if len(bb) == 4:
                matched.append((bb, gto["bbox"]))
    # Models emit boxes in different coordinate spaces regardless of prompt
    # instructions (qwen3.6: 0-1000 normalized; nemotron w/ reasoning: pixels).
    # Score both spaces and keep the better one — report which.
    best = (0, 0.0, None)
    for space, fx, fy in (("pixel", 1.0, 1.0), ("norm1000", W/1000.0, H/1000.0)):
        for order in ("xyxy", "yxyx"):  # gemma4/Gemini box_2d is [y1,x1,y2,x2]
            hits, ious = 0, []
            for bb, gtb in matched:
                x1, y1, x2, y2 = (bb[0], bb[1], bb[2], bb[3]) if order == "xyxy" else (bb[1], bb[0], bb[3], bb[2])
                px = [x1*fx, y1*fy, x2*fx, y2*fy]
                hits += center_in(px, gtb)
                ious.append(iou(px, gtb))
            mean_iou = round(sum(ious)/len(ious), 3) if ious else 0.0
            if (mean_iou, hits) > (best[1], best[0]):
                best = (hits, mean_iou, f"{space}/{order}")
    s["bbox_hits"], s["bbox_mean_iou"], s["bbox_space"] = best
    if not s["serial_found"]:
        s["serial_found"] = g["serial"] in json.dumps(r)
    return s

def score_doc(resp_text):
    g = GT["document"]
    s = {"json_valid": False, "invoice_no": False, "items_found": 0,
         "items_total": len(g["items"]), "qty_price_right": 0, "total_right": False,
         "name_bbox_hits": 0}
    try:
        r = json.loads(resp_text); s["json_valid"] = True
    except Exception:
        return s
    s["invoice_no"] = g["invoice_no"] in json.dumps(r)
    items = r.get("line_items") or []
    for gti in g["items"]:
        m = next((i for i in items if isinstance(i.get("name"), str)
                  and gti["name"].lower() in i["name"].lower()), None)
        if m:
            s["items_found"] += 1
            try:
                if int(m.get("qty")) == gti["qty"] and abs(float(m.get("unit_price")) - gti["unit_price"]) < 0.01:
                    s["qty_price_right"] += 1
            except Exception:
                pass
            bb = m.get("name_bbox") or m.get("name_bbox_2d") or []
            if len(bb) == 4 and bb[1] > 250 and bb[3] < 700 and bb[0] < 500:
                s["name_bbox_hits"] += 1
    try:
        s["total_right"] = abs(float(r.get("total")) - g["total"]) < 0.01
    except Exception:
        pass
    return s

def score_multi(resp_text):
    g = GT
    s = {"json_valid": False, "q1_right": False, "q2_right": False,
         "q4_bbox_hit": False, "chart_values_found": 0,
         "chart_total": len(g["chart"]["bars"])}
    try:
        r = json.loads(resp_text); s["json_valid"] = True
    except Exception:
        return s
    a = r.get("answers") or {}
    s["q1_right"] = a.get("q1") == 2
    q2 = a.get("q2") or {}
    try:
        s["q2_right"] = str(q2.get("category", "")).strip().rstrip("*").upper() == "Q4" \
            and abs(float(q2.get("value")) - 128) < 0.5
    except Exception:
        pass
    dyn = next(o for o in g["scene_hd"]["objects"] if o["label"] == "DYNAMO")
    s["q4_bbox_hit"] = center_in(a.get("q4") or [], dyn["bbox"])
    blob = json.dumps(r)
    for b in g["chart"]["bars"]:
        if str(b["value"]) in blob:
            s["chart_values_found"] += 1
    return s

tests = [
    ("scene_single", SCENE_PROMPT.format(w=1920, h=1080), ["scene_hd.png"], score_scene),
    ("document_single", DOC_PROMPT.format(w=1568, h=1568), ["document.png"], score_doc),
    ("multi_3img", MULTI_PROMPT, ["scene_hd.png", "document.png", "chart.png"], score_multi),
]

def main():
    global HOST, TAG, MODEL
    HOST = sys.argv[1]
    TAG = sys.argv[2]
    MODEL = sys.argv[3] if len(sys.argv) > 3 else "nemotron3:33b-q4_K_M"
    only = sys.argv[4] if len(sys.argv) > 4 else None
    results = {}
    for name, prompt, images, scorer in tests:
        if only and name != only:
            continue
        print(f"--- {name} [{TAG}] ---", flush=True)
        try:
            r = gen(prompt, [b64(i) for i in images])
        except Exception as e:
            print(f"ERROR: {e}")
            results[name] = {"error": str(e)}
            continue
        text = r.get("response", "")
        open(f"{DIR}/resp_{TAG}_{name}.json", "w").write(text)
        sc = scorer(text)
        sc["prompt_eval_count"] = r.get("prompt_eval_count")
        sc["eval_count"] = r.get("eval_count")
        results[name] = sc
        print(json.dumps(sc, indent=1), flush=True)
    
    open(f"{DIR}/scores_{TAG}.json", "w").write(json.dumps(results, indent=1))
    print("SUITE DONE", TAG)

if __name__ == "__main__":
    main()

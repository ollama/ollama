#!/usr/bin/env python3
"""Dense fine-text probe: does the gemma4 1120-token budget beat upstream's
default on genuinely small text? Generates a 1568x1568 compliance page with
reference codes at descending font sizes, asks for exact transcription,
scores per-size recall.

Usage: finetext_probe.py <host> <tag> <model>
Env: THINK=on|false, ENDPOINT=generate|chat, NUM_CTX, NUM_PREDICT, HTTP_TIMEOUT
"""
import json, os, sys, base64, random, urllib.request

DIR = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(DIR, "visimgs", "finetext.png")
GT = os.path.join(DIR, "visimgs", "finetext_gt.json")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

SIZES = [22, 16, 12, 9, 7]  # px per bucket, 4 codes each
CHARS = "ACDEFHJKMNPRTUVWXY"  # unambiguous set


def make_code(rng):
    return "%s%s%s-%d%d%d%d-%s%s%d%d" % tuple(
        [rng.choice(CHARS) for _ in range(3)] + [rng.randrange(10) for _ in range(4)]
        + [rng.choice(CHARS) for _ in range(2)] + [rng.randrange(10) for _ in range(2)])


def generate():
    from PIL import Image, ImageDraw, ImageFont
    rng = random.Random(42)
    img = Image.new("RGB", (1568, 1568), "white")
    d = ImageDraw.Draw(img)
    d.text((60, 40), "COMPLIANCE REGISTER — SECTION 7", fill="black",
           font=ImageFont.truetype(FONT, 34))
    d.text((60, 100), "Each entry below must be transcribed exactly for audit.",
           fill="black", font=ImageFont.truetype(FONT, 18))
    gt = {}
    y = 170
    for size in SIZES:
        f = ImageFont.truetype(FONT, size)
        d.text((60, y), f"[{size}px tier]", fill="gray", font=ImageFont.truetype(FONT, 14))
        y += 26
        codes = []
        for i in range(4):
            c = make_code(rng)
            codes.append(c)
            d.text((90 + (i % 2) * 700, y + (i // 2) * (size + 10)),
                   f"entry {c} status ACTIVE", fill="black", font=f)
        gt[str(size)] = codes
        y += 2 * (size + 10) + 28
    img.save(IMG)
    json.dump(gt, open(GT, "w"), indent=1)
    return gt


def run(host, tag, model):
    gt = json.load(open(GT)) if os.path.exists(GT) else generate()
    prompt = ("Transcribe EVERY reference code on this page exactly as printed. "
              "Codes look like ABC-1234-DE56 and appear at several text sizes, "
              "including very small ones; read carefully down to the smallest. "
              "Respond with a SINGLE JSON object, no prose: "
              '{"codes": [<string>, ...]} listing every code you can read.')
    img_b64 = base64.b64encode(open(IMG, "rb").read()).decode()
    num_ctx = int(os.environ.get("NUM_CTX", "32768"))
    num_predict = int(os.environ.get("NUM_PREDICT", "4000"))
    timeout = int(os.environ.get("HTTP_TIMEOUT", "1800"))
    ep = os.environ.get("ENDPOINT", "generate")
    think = os.environ.get("THINK", "false") == "on"
    opts = {"num_predict": num_predict, "num_ctx": num_ctx, "temperature": 0}
    if os.environ.get("KV_CACHE_TYPE"):
        opts["kv_cache_type"] = os.environ["KV_CACHE_TYPE"]
    if ep == "chat":
        payload = {"model": model, "stream": False, "format": "json", "options": opts,
                   "messages": [{"role": "user", "content": prompt, "images": [img_b64]}]}
        url = host + "/api/chat"
    else:
        payload = {"model": model, "prompt": prompt, "images": [img_b64],
                   "stream": False, "format": "json", "options": opts}
        url = host + "/api/generate"
    if not think:
        payload["think"] = False
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    body = r.get("response") or (r.get("message") or {}).get("content", "")
    s = {"tag": tag, "json_valid": False,
         "prompt_eval_count": r.get("prompt_eval_count"), "eval_count": r.get("eval_count")}
    found = []
    try:
        found = [str(x).strip().upper() for x in json.loads(body).get("codes", [])]
        s["json_valid"] = True
    except Exception:
        pass
    for size, codes in sorted(gt.items(), key=lambda kv: -int(kv[0])):
        s[f"recall_{size}px"] = sum(1 for c in codes if c in found)
    s["total_found"] = len(found)
    print(f"--- finetext [{tag}] ---")
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "gen":
        generate(); print("finetext.png + gt written")
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3])

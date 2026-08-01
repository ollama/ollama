import json, base64, sys, urllib.request
S = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
sys.path.insert(0, S)
from vision_suite import SCENE_PROMPT, score_scene, b64
HOST = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:11435"
def gen(extra, fmt, think):
    p = {"model": "nemotron3:33b-q4_K_M", "prompt": SCENE_PROMPT.format(w=1920, h=1080),
         "images": [b64("scene_hd.png")], "stream": False, "think": think,
         "options": {"num_predict": 3000, "num_ctx": 16384, "temperature": 0}}
    if fmt: p["format"] = "json"
    req = urllib.request.Request(HOST + "/api/generate", data=json.dumps(p).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))
mode = sys.argv[1]
fmt, think = {"nogrammar": (False, False), "thinkon": (True, True)}[mode]
r = gen(mode, fmt, think)
text = r.get("response", "")
open(f"{S}/resp_variant_{mode}.json", "w").write(text)
# strip code fences if present for scoring
t = text.strip()
if t.startswith("```"):
    t = t.split("```")[1]
    t = t[4:] if t.startswith("json") else t
sc = score_scene(t)
sc["eval_count"] = r.get("eval_count"); sc["thinking_len"] = len(r.get("thinking") or "")
print(mode, json.dumps(sc))

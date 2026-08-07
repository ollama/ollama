#!/usr/bin/env python3
"""Content-rich nemotron vision test images with deterministic ground truth."""
import json, os
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visimgs")
os.makedirs(OUT, exist_ok=True)
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
gt = {}

# ---------- scene_hd.png : 1920x1080, labeled shapes + tiny serial ----------
W, H = 1920, 1080
img = Image.new("RGB", (W, H), (245, 245, 240))
d = ImageDraw.Draw(img)
label_font = ImageFont.truetype(F, 20)
tiny_font = ImageFont.truetype(F, 14)
shapes = [
    # (kind, color name, rgb, bbox)
    ("rectangle", "red",    (200, 40, 40),   (140, 160, 420, 360)),
    ("ellipse",   "blue",   (40, 70, 200),   (620, 120, 900, 330)),
    ("rectangle", "green",  (40, 160, 70),   (1150, 180, 1500, 420)),
    ("ellipse",   "orange", (235, 140, 30),  (220, 600, 480, 860)),
    ("rectangle", "purple", (120, 50, 160),  (760, 640, 1040, 920)),
    ("ellipse",   "teal",   (0, 150, 150),   (1350, 620, 1720, 880)),
]
labels = ["ANCHOR", "BEACON", "CIPHER", "DYNAMO", "EMBER", "FALCON"]
gt_scene = []
for (kind, cname, rgb, bb), label in zip(shapes, labels):
    if kind == "rectangle":
        d.rectangle(bb, fill=rgb)
    else:
        d.ellipse(bb, fill=rgb)
    d.text((bb[0], bb[1] - 28), label, font=label_font, fill=(20, 20, 20))
    gt_scene.append({"label": label, "kind": kind, "color": cname, "bbox": list(bb)})
serial = "SN-4921-XK"
d.text((W - 150, H - 30), serial, font=tiny_font, fill=(90, 90, 90))
img.save(f"{OUT}/scene_hd.png")
gt["scene_hd"] = {"objects": gt_scene, "serial": serial, "size": [W, H]}

# ---------- document.png : 1568x1568, fake invoice ----------
W = H = 1568
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
title_f = ImageFont.truetype(FB, 44)
head_f = ImageFont.truetype(FB, 24)
body_f = ImageFont.truetype(F, 22)
small_f = ImageFont.truetype(F, 17)
d.text((90, 80), "MAXUS INDUSTRIAL SUPPLY", font=title_f, fill=(10, 10, 60))
d.text((90, 150), "INVOICE  INV-2026-0801", font=head_f, fill=(10, 10, 10))
d.text((90, 190), "Date: 2026-08-01    Customer: Glenn Neuber Pty Ltd", font=body_f, fill=(40, 40, 40))
rows = [
    ("Hydraulic actuator HA-220", 3, 412.50),
    ("Thermal camera module TC-9", 1, 1289.00),
    ("Cable loom, 12-way shielded", 8, 37.85),
    ("Torque wrench 5-60 Nm", 2, 148.20),
    ("Bearing kit BK-7731", 5, 64.4),
]
y = 300
d.line((90, y - 12, 1480, y - 12), fill=(0, 0, 0), width=2)
d.text((90, y), "Item", font=head_f, fill=(0, 0, 0))
d.text((950, y), "Qty", font=head_f, fill=(0, 0, 0))
d.text((1150, y), "Unit price", font=head_f, fill=(0, 0, 0))
y += 50
total = 0.0
name_bboxes = []
for name, qty, price in rows:
    d.text((90, y), name, font=body_f, fill=(30, 30, 30))
    # tight bbox of the item-name text run, for IoU scoring (score_doc's
    # name_bbox_mean_iou); analytic textbbox tracks measured pixel extents to
    # within a couple of px, which is negligible at IoU scale
    name_bboxes.append([int(v) for v in d.textbbox((90, y), name, font=body_f)])
    d.text((950, y), str(qty), font=body_f, fill=(30, 30, 30))
    d.text((1150, y), f"${price:,.2f}", font=body_f, fill=(30, 30, 30))
    total += qty * price
    y += 46
d.line((90, y + 6, 1480, y + 6), fill=(0, 0, 0), width=2)
d.text((950, y + 30), "TOTAL:", font=head_f, fill=(0, 0, 0))
d.text((1150, y + 30), f"${total:,.2f}", font=head_f, fill=(0, 0, 0))
d.text((90, 1450), "Payment due within 30 days. Quote reference INV-2026-0801 on all correspondence.",
       font=small_f, fill=(100, 100, 100))
img.save(f"{OUT}/document.png")
gt["document"] = {
    "invoice_no": "INV-2026-0801",
    "items": [{"name": n, "qty": q, "unit_price": p} for n, q, p in rows],
    "name_bboxes": name_bboxes,
    "total": round(total, 2), "size": [W, H],
}

# ---------- chart.png : 1280x960, bar chart ----------
W, H = 1280, 960
img = Image.new("RGB", (W, H), (252, 252, 252))
d = ImageDraw.Draw(img)
tf = ImageFont.truetype(FB, 30)
af = ImageFont.truetype(F, 20)
vf = ImageFont.truetype(F, 19)
d.text((360, 40), "Quarterly unit shipments (k)", font=tf, fill=(20, 20, 20))
bars = [("Q1", 62), ("Q2", 91), ("Q3", 47), ("Q4", 128), ("Q5*", 73)]
x0, y_base, bw, gap = 160, 840, 140, 70
maxv = max(v for _, v in bars)
colors = [(70, 120, 200), (200, 90, 60), (90, 170, 90), (150, 90, 180), (220, 170, 50)]
gt_bars = []
for i, ((lbl, v), c) in enumerate(zip(bars, colors)):
    x = x0 + i * (bw + gap)
    h = int(600 * v / maxv)
    d.rectangle((x, y_base - h, x + bw, y_base), fill=c)
    d.text((x + bw // 2 - 18, y_base - h - 30), str(v), font=vf, fill=(0, 0, 0))
    d.text((x + bw // 2 - 20, y_base + 14), lbl, font=af, fill=(0, 0, 0))
    gt_bars.append({"label": lbl, "value": v})
d.line((120, y_base, 1180, y_base), fill=(0, 0, 0), width=3)
img.save(f"{OUT}/chart.png")
gt["chart"] = {"bars": gt_bars, "tallest": "Q4", "size": [W, H]}

with open(f"{OUT}/ground_truth.json", "w") as f:
    json.dump(gt, f, indent=1)
print("wrote", sorted(os.listdir(OUT)))

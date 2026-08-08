#!/usr/bin/env python3
"""Probe layer for the pre-deploy regression harness.

Stdlib only, deliberately: this runs on build hosts that have an ollama image and
nothing else. Pillow is NOT required — ladder images are geometry, and token cost
is a function of geometry, not content.

Nothing in here asserts. Assertions live in checks.py, expected values live in
expectations.toml.
"""
import base64
import json
import os
import re
import shutil
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib

DIR = os.path.dirname(os.path.abspath(__file__))
IMGDIR = os.path.join(DIR, "ladderimgs")

# One prompt for both the text-only baseline and the image probes, so the text
# tokens cancel in the subtraction. See Ollama.text_baseline().
PROBE_PROMPT = "Describe briefly."


# --------------------------------------------------------------------------
# Deterministic PNG generation (no Pillow)
# --------------------------------------------------------------------------

def _chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def _render_png(w, h):
    """Gradient + 64px gridlines. Deterministic, compresses small, and has real
    edge structure so no degenerate all-one-colour decode path is exercised."""
    row0 = bytearray(w * 3)
    for x in range(w):
        row0[3 * x] = (x * 255) // max(1, w - 1)
        row0[3 * x + 2] = 200 if x % 64 == 0 else 60
    raw = bytearray()
    for y in range(h):
        row = bytearray(row0)
        row[1::3] = bytes([(y * 255) // max(1, h - 1)]) * w
        if y % 64 == 0:
            row[2::3] = bytes([220]) * w
        raw += b"\x00" + row
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + _chunk(b"IEND", b""))


def ladder_image_b64(size):
    """Base64 of the WxH ladder image, generating and caching it on first use."""
    path = os.path.join(IMGDIR, f"{size}.png")
    if not os.path.exists(path):
        w, h = (int(v) for v in size.lower().split("x"))
        os.makedirs(IMGDIR, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(_render_png(w, h))
        os.replace(tmp, path)
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


# --------------------------------------------------------------------------
# Ollama client
# --------------------------------------------------------------------------

class ProbeError(RuntimeError):
    pass


class Ollama:
    def __init__(self, host, timeout=1800):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.queue_waits = []          # every observed (label, seconds) queue delay

    def _post(self, path, payload, timeout=None):
        req = urllib.request.Request(
            self.host + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as fh:
                return json.load(fh)
        except urllib.error.HTTPError as exc:
            raise ProbeError(f"{path} HTTP {exc.code}: {exc.read()[:400]!r}") from exc
        except Exception as exc:
            raise ProbeError(f"{path}: {exc}") from exc

    def _get(self, path, timeout=30):
        try:
            with urllib.request.urlopen(self.host + path, timeout=timeout) as fh:
                return json.load(fh)
        except Exception as exc:
            raise ProbeError(f"{path}: {exc}") from exc

    def version(self):
        return self._get("/api/version").get("version", "")

    def tags(self):
        return [m["name"] for m in self._get("/api/tags").get("models", [])]

    def ps(self):
        return self._get("/api/ps").get("models", [])

    def unload(self, model):
        """Force the runner to drop the model so the next request emits a fresh
        load_hparams block. Without this, a payload proof can be read off a log
        line written by a PREVIOUS build."""
        try:
            self._post("/api/generate", {"model": model, "keep_alive": 0}, timeout=120)
        except ProbeError:
            pass
        time.sleep(2)

    def generate(self, model, prompt, images=None, num_predict=1, num_ctx=16384,
                 image_min_tokens=None, image_max_tokens=None, think=None,
                 fmt=None, label=""):
        """One /api/generate call. Returns the server response plus timing.

        `queue_wait` is wall-clock minus the server's own reported total_duration:
        the time this request spent waiting for a slot. It is the only reliable
        signal that another client is saturating the endpoint — a saturated server
        looks completely healthy and simply times requests out.
        """
        opts = {"num_predict": num_predict, "num_ctx": num_ctx, "temperature": 0}
        if image_min_tokens is not None:
            opts["image_min_tokens"] = image_min_tokens
        if image_max_tokens is not None:
            opts["image_max_tokens"] = image_max_tokens
        payload = {"model": model, "prompt": prompt, "stream": False, "options": opts}
        if images:
            payload["images"] = images
        if fmt:
            payload["format"] = fmt
        if think is not None:
            payload["think"] = think

        t0 = time.monotonic()
        resp = self._post("/api/generate", payload)
        wall = time.monotonic() - t0

        total = resp.get("total_duration", 0) / 1e9
        queue_wait = max(0.0, wall - total) if total else 0.0
        resp["_wall_s"] = round(wall, 2)
        resp["_server_total_s"] = round(total, 2)
        resp["_queue_wait_s"] = round(queue_wait, 2)
        self.queue_waits.append((label or "probe", round(queue_wait, 2)))
        return resp

    def visual_tokens(self, model, size, baseline, **kw):
        """prompt_eval_count for one image minus the text-only baseline."""
        resp = self.generate(model, PROBE_PROMPT,
                             images=[ladder_image_b64(size)], label=f"{size}", **kw)
        return resp["prompt_eval_count"] - baseline, resp

    def text_baseline(self, model):
        """Baseline for the SAME prompt the image probes use.

        This is load-bearing. measure.py takes its baseline with "Hi" but probes
        with "Describe briefly.", so its reported visual-token deltas silently
        carry the difference in prompt length — measured on nemotron3:33b-q8,
        "Hi" is 18 tokens and "Describe briefly." is 21, a constant +3 on every
        row. Using one prompt for both makes the subtraction cancel the text
        exactly and leaves only visual tokens plus markers.
        """
        resp = self.generate(model, PROBE_PROMPT, num_predict=1, label="baseline")
        return resp["prompt_eval_count"], resp


# --------------------------------------------------------------------------
# Container log access — for the payload patch proof
# --------------------------------------------------------------------------

PIXEL_RE = re.compile(
    r"load_hparams:\s+image_(min|max)_pixels:\s+(\d+)(\s+\(custom value\))?")


def find_container(port, explicit=None):
    """Resolve the container serving `port`. Explicit name always wins."""
    if explicit:
        return explicit
    if not shutil.which("docker"):
        return None
    try:
        out = subprocess.run(
            ["docker", "ps", "--filter", f"publish={port}", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=30).stdout.split()
        return out[0] if out else None
    except Exception:
        return None


def container_logs(container, since_epoch, log_cmd=None):
    """Logs written since `since_epoch`. `--since` is load-bearing: it is what
    guarantees the load_hparams line we read was emitted by THIS build during
    THIS run, not left over from a previous one."""
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(since_epoch))
    if log_cmd:
        cmd = log_cmd.format(container=container, since=since)
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    else:
        proc = subprocess.run(
            ["docker", "logs", "--since", since, container],
            capture_output=True, text=True, timeout=120)
    return (proc.stdout or "") + (proc.stderr or "")


def parse_pixel_lines(text):
    """[{'kind': 'max', 'value': 3407872, 'custom': True}, ...] in log order."""
    return [{"kind": m.group(1), "value": int(m.group(2)), "custom": bool(m.group(3))}
            for m in PIXEL_RE.finditer(text)]


def grep_binary_marker(container, path="/usr/bin/ollama", exec_cmd=None):
    """grep -c -- --image-max-tokens on the Go binary: 1 on a fork build, 0 on
    stock ollama/ollama. This is a GO-side marker only — it says nothing about
    whether the llama.cpp payload carries the compat patches. The payload proof
    is the model-load log, never a binary inspection."""
    cmd = ([exec_cmd.format(container=container)] if exec_cmd else
           ["docker", "exec", container, "sh", "-c",
            f"grep -c -- --image-max-tokens {path} || true"])
    proc = subprocess.run(cmd, shell=bool(exec_cmd), capture_output=True,
                          text=True, timeout=120)
    digits = re.findall(r"\d+", proc.stdout or "")
    if not digits:
        raise ProbeError(f"no count from grep: {(proc.stdout + proc.stderr)[:300]!r}")
    return int(digits[0])

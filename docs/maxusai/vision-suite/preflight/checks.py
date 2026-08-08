#!/usr/bin/env python3
"""The checks. Every expected value comes from expectations.toml; nothing is
hardcoded here except the shapes of the assertions themselves."""
import json
import os
import re
import subprocess
import time

from probes import (ProbeError, container_logs, grep_binary_marker,
                    ladder_image_b64, parse_pixel_lines)

PASS, FAIL, SKIP, NEEDS_BASELINE, ERROR, CONTENTION = (
    "PASS", "FAIL", "SKIP", "NEEDS_BASELINE", "ERROR", "CONTENTION")

SUITE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def result(name, status, summary, arch=None, expected=None, actual=None,
           diagnosis=None, **extra):
    r = {"check": name, "arch": arch, "status": status, "summary": summary}
    if expected is not None:
        r["expected"] = expected
    if actual is not None:
        r["actual"] = actual
    if diagnosis:
        r["diagnosis"] = diagnosis
    r.update(extra)
    return r


# --------------------------------------------------------------------------
# 1. Version string — the gate. Nothing downstream is trustworthy without it.
# --------------------------------------------------------------------------

def check_version(client, profile, profile_id):
    """11434/11435/11436 are all occupied on 10.8.0.6; a canary on 11436 once
    answered from the wrong server and only a mismatched version string caught
    it. This check gates the whole run: on FAIL, nothing else executes."""
    try:
        actual = client.version()
    except ProbeError as exc:
        return result("version", ERROR, f"cannot reach {client.host}: {exc}")
    pattern = profile["version_pattern"]
    if not re.match(pattern, actual):
        return result(
            "version", FAIL,
            f"server at {client.host} is not the build this profile describes",
            expected=f"match {pattern}  (profile {profile_id})", actual=actual,
            diagnosis="Wrong server on this port, or the image was built from a "
                      "different commit. Every measurement below would be "
                      "attributed to the wrong build — run aborted.")
    return result("version", PASS, f"{actual} matches profile {profile_id}",
                  expected=pattern, actual=actual)


def check_image_tag(client, profile, image_tag, container):
    """The operator names an image tag; confirm the running container is it."""
    expected = profile.get("reference_image")
    if not container:
        return result("image_tag", SKIP, "no container resolved (remote host?)",
                      expected=image_tag)
    try:
        actual = subprocess.run(
            ["docker", "inspect", container, "--format", "{{.Config.Image}}"],
            capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:
        return result("image_tag", SKIP, f"docker inspect unavailable: {exc}")
    if not actual:
        return result("image_tag", SKIP, f"could not inspect {container}")
    if image_tag and actual != image_tag:
        return result("image_tag", FAIL,
                      f"container {container} is not running the named image",
                      expected=image_tag, actual=actual,
                      diagnosis="The tag under test and the tag actually serving "
                                "this port differ.")
    return result("image_tag", PASS, f"{container} runs {actual}",
                  expected=image_tag or expected, actual=actual)


# --------------------------------------------------------------------------
# 2. Go-side patch marker
# --------------------------------------------------------------------------

def check_patch_marker(profile, container, exec_cmd=None):
    expected = profile.get("expect_patch_marker", 1)
    if not container:
        return result("go_patch_marker", SKIP,
                      "no container resolved; cannot grep the binary",
                      expected=expected)
    try:
        actual = grep_binary_marker(container, exec_cmd=exec_cmd)
    except Exception as exc:
        return result("go_patch_marker", ERROR, f"grep failed: {exc}",
                      expected=expected)
    if actual != expected:
        return result(
            "go_patch_marker", FAIL, "--image-max-tokens marker count is wrong",
            expected=expected, actual=actual,
            diagnosis="0 means a stock ollama/ollama binary — visionServerArgs "
                      "has no gemma4/nemotron_h_omni branch, so the budget flags "
                      "are never passed.")
    return result("go_patch_marker", PASS,
                  f"--image-max-tokens present in binary ({actual})",
                  expected=expected, actual=actual)


# --------------------------------------------------------------------------
# 3. Payload patch proof — from the MODEL-LOAD LOG, never the binary
# --------------------------------------------------------------------------

def check_payload_proof(expect, arch, container, since, log_cmd=None):
    """N == max_tokens * S^2 where S = patch_size * n_merge.

    Static inspection of libmtmd.so is explicitly NOT used: `strings` is absent
    from the ollama images so in-container greps return misleading zeros, and
    <img>/</img> literals appear in stock too (InternVL uses them). An RTTI
    occurrence-count delta was suggestive but never proof.
    """
    name = "payload_proof"
    if not container:
        return result(name, SKIP, "no container resolved; cannot read the load log",
                      arch=arch)

    expect_absent = expect.get("expect_no_pixel_log_line", False)
    try:
        logs = container_logs(container, since, log_cmd)
    except Exception as exc:
        return result(name, ERROR, f"could not read logs: {exc}", arch=arch)

    lines = parse_pixel_lines(logs)

    if expect_absent:
        # An unpatched projector never calls set_limit_image_tokens(), leaving the
        # values at the -1 sentinel, and the log lines are gated on value > 0.
        # Absence is the proof.
        if lines:
            return result(name, FAIL,
                          "pixel budget logged, but this payload should have none",
                          arch=arch, expected="no image_*_pixels line",
                          actual=[f"{d['kind']}={d['value']}" for d in lines],
                          diagnosis="This profile describes a pre-002 payload where "
                                    "the flags are inert. A budget line means the "
                                    "wrong payload is deployed.")
        return result(name, PASS, "no pixel budget logged, as expected for this payload",
                      arch=arch, expected="no image_*_pixels line", actual="absent")

    want = {"min": expect["image_min_pixels"], "max": expect["image_max_pixels"]}
    stride, bmin, bmax = (expect["patch_stride"], expect["budget_min_tokens"],
                          expect["budget_max_tokens"])
    derivation = (f"min {bmin}*{stride}^2={want['min']}, "
                  f"max {bmax}*{stride}^2={want['max']}")

    if not lines:
        return result(name, FAIL, "no load_hparams pixel budget in the fresh log",
                      arch=arch, expected=derivation, actual="no matching log line",
                      diagnosis="Either the model never loaded during this run (so "
                                "no fresh load_hparams block was emitted), or the "
                                "payload lacks the budget patch entirely.")

    # Pair the lines into (min, max) blocks in log order — one per model load —
    # and read the last DEFAULT-budget block. A pinned probe legitimately logs
    # min == max, which is not what this check is about. payload_proof already
    # runs before any pinned probe, so this is belt-and-braces against a future
    # reordering silently turning a pinned block into the "proof".
    blocks, cur = [], {}
    for d in lines:
        if d["kind"] in cur:
            blocks.append(cur)
            cur = {}
        cur[d["kind"]] = d
    if cur:
        blocks.append(cur)

    def is_pinned(b):
        return ("min" in b and "max" in b
                and b["min"]["value"] == b["max"]["value"])

    usable = blocks
    if want["min"] != want["max"]:
        usable = [b for b in blocks if not is_pinned(b)] or blocks
    got = usable[-1] if usable else {}

    bad = []
    for kind, value in want.items():
        if kind not in got:
            bad.append(f"{kind}: missing")
        elif got[kind]["value"] != value:
            bad.append(f"{kind}: expected {value}, got {got[kind]['value']}")
        elif not got[kind]["custom"]:
            bad.append(f"{kind}: value right but not marked '(custom value)' — "
                       f"the flags were not applied")
    actual = {k: f"{v['value']}{' (custom value)' if v['custom'] else ''}"
              for k, v in got.items()}
    if bad:
        return result(name, FAIL, "; ".join(bad), arch=arch,
                      expected=derivation, actual=actual,
                      diagnosis="The Go binary passes the flags but the llama.cpp "
                                "payload is not honouring them, or the budget "
                                "defaults changed without this file being updated.")
    return result(name, PASS, f"budget logged as custom ({derivation})",
                  arch=arch, expected=derivation, actual=actual)


# --------------------------------------------------------------------------
# 4. Token ladder — PER-ARCH verdict logic
# --------------------------------------------------------------------------

def check_ladder(client, expect, arch, sizes, baseline, tol_default=2):
    """Same image at five 16:9 geometries, delta against the text-only baseline.

    The verdict is per-arch and MUST NOT be shared. A flat ladder means an
    unpatched payload for nemotron (dynamic resolution never engaged) but is the
    CORRECT result for gemma4 under 004, which budget-fills every image to the
    ceiling. A shared heuristic gets this exactly backwards.
    """
    model = expect["model"]
    want = expect["ladder"]
    tol = expect.get("ladder_tolerance", tol_default)
    scaling = expect["scaling"]

    got, rows = [], []
    for size, exp in zip(sizes, want):
        try:
            delta, resp = client.visual_tokens(model, size, baseline)
        except ProbeError as exc:
            return result("token_ladder", ERROR, f"{size}: {exc}", arch=arch)
        got.append(delta)
        rows.append({"size": size, "expected": exp, "actual": delta,
                     "delta": delta - exp, "ok": abs(delta - exp) <= tol,
                     "queue_wait_s": resp["_queue_wait_s"]})

    is_flat = len(set(got)) == 1
    mismatches = [r for r in rows if not r["ok"]]

    diagnosis = None
    if mismatches:
        if scaling == "dynamic" and is_flat:
            diagnosis = (f"Ladder is FLAT at {got[0]} on an arch whose payload "
                         f"should scale with resolution — the dynamic-resolution "
                         f"patch is not in this payload (the Go flags are parsed "
                         f"but inert). This is the unpatched-payload signature.")
        elif scaling == "flat" and not is_flat:
            diagnosis = ("Ladder VARIES on an arch that should budget-fill to a "
                         "constant — the 004 budget-fill behaviour is missing, or "
                         "the ceiling changed.")
        else:
            diagnosis = ("Values moved but the shape is right: most likely a "
                         "budget or preprocessing change. Re-measure and update "
                         "expectations.toml if the new numbers are intended.")

    status = PASS if not mismatches else FAIL
    summary = (f"{len(rows) - len(mismatches)}/{len(rows)} geometries within +/-{tol}"
               + (f" (shape: {'flat' if is_flat else 'scaling'}, expected {scaling})"
                  if mismatches else ""))
    return result("token_ladder", status, summary, arch=arch,
                  expected=dict(zip(sizes, want)), actual=dict(zip(sizes, got)),
                  diagnosis=diagnosis, rows=rows, shape="flat" if is_flat else "scaling",
                  expected_shape=scaling)


# --------------------------------------------------------------------------
# 5. Pinned-budget probe — the 005 defect class
# --------------------------------------------------------------------------

def check_pinned_budget(client, expect, arch, baseline, marker_allowance=2):
    """image_min_tokens == image_max_tokens. Pre-005, nemotron pinned to 3328
    delivered 3390 — 60 grid tokens over its own ceiling.

    Two independent assertions:
      1. the exact measured regression value, and
      2. the class invariant `delivered - markers <= ceiling`, which catches a
         NEW overshoot at a number nobody has measured yet.
    """
    pin = expect.get("pinned")
    name = "pinned_budget"
    if not pin:
        return result(name, SKIP,
                      "no pinned-budget expectation recorded for this arch",
                      arch=arch,
                      diagnosis="Omitted deliberately: the 005 defect was found on "
                                "nemotron and this arch's probe has not been "
                                "measured. See README.md to add one.")

    model, size, pinv = expect["model"], pin["size"], pin["pin_tokens"]
    tol = pin.get("tolerance", 4)
    ceiling = expect["budget_max_tokens"]
    sub = []

    # --- pinned arm (forces a runner reload; grouped by the caller) ---
    try:
        delta, _ = client.visual_tokens(model, size, baseline,
                                        image_min_tokens=pinv, image_max_tokens=pinv)
    except ProbeError as exc:
        return result(name, ERROR, f"pinned probe failed: {exc}", arch=arch)

    want = pin["expect_tokens"]
    exact_ok = abs(delta - want) <= tol
    sub.append({"arm": "pinned", "pin": pinv, "expected": want, "actual": delta,
                "ok": exact_ok})

    grid = delta - marker_allowance
    ceiling_ok = True
    if pin.get("enforce_ceiling_invariant", True):
        ceiling_ok = grid <= ceiling
        sub.append({"arm": "ceiling_invariant",
                    "expected": f"grid <= {ceiling}", "actual": grid,
                    "ok": ceiling_ok})

    # --- unpinned control, same image, same run ---
    control_ok = True
    if "control_expect_tokens" in pin:
        ctol = pin.get("control_tolerance", 4)
        try:
            cdelta, _ = client.visual_tokens(model, size, baseline)
        except ProbeError as exc:
            return result(name, ERROR, f"control probe failed: {exc}", arch=arch)
        cwant = pin["control_expect_tokens"]
        control_ok = abs(cdelta - cwant) <= ctol
        sub.append({"arm": "unpinned_control", "expected": cwant, "actual": cdelta,
                    "ok": control_ok})

    diagnosis = None
    if not ceiling_ok:
        diagnosis = (f"OVERSHOOT: pinned to {pinv} but delivered {grid} grid tokens "
                     f"({delta} incl. markers) — {grid - ceiling} over the ceiling. "
                     f"This is the 005 defect class; the pinned dyn_size path is "
                     f"not clamping. Check llama/compat/005-*.patch is applied.")
    elif not exact_ok:
        diagnosis = (f"Pinned cost moved ({want} -> {delta}) but stays under the "
                     f"ceiling, so this is a behaviour change, not the 005 defect. "
                     f"Re-measure and update expectations.toml if intended.")
    elif not control_ok:
        diagnosis = ("The unpinned control moved while the pinned arm held — a "
                     "pinned probe leaked into the default budget, or the default "
                     "changed.")

    status = PASS if (exact_ok and ceiling_ok and control_ok) else FAIL
    return result(name, status,
                  f"pinned {pinv} -> {delta} (ceiling {ceiling}, +{marker_allowance} markers)",
                  arch=arch, expected=want, actual=delta, diagnosis=diagnosis, arms=sub)


# --------------------------------------------------------------------------
# 6. think + format non-empty
# --------------------------------------------------------------------------

def check_think_format(client, expect, arch, min_num_predict):
    """Stock returns an empty `response` for nemotron3/qwen3.6 when think:true is
    combined with format:"json". The fork emits valid JSON after thinking.

    num_predict is floored: at 120 three probes returned response:"" with
    eval_count exactly 120 — the whole allowance spent inside an unclosed
    thinking block. That reads as a vision failure and is not one, so `thinking`
    is judged alongside `response`.
    """
    cfg = expect.get("think_format")
    name = "think_format"
    if not cfg:
        return result(name, SKIP, "no think+format expectation recorded", arch=arch)

    np_ = cfg.get("num_predict", min_num_predict)
    if np_ < min_num_predict:
        return result(name, ERROR,
                      f"num_predict {np_} is below the enforced floor {min_num_predict}",
                      arch=arch,
                      diagnosis="Refusing to run: a low num_predict manufactures a "
                                "false vision failure. Raise it in expectations.toml.")
    try:
        resp = client.generate(
            expect["model"],
            "List three visual facts about this image as JSON: "
            '{"facts": ["...", "...", "..."]}',
            images=[ladder_image_b64("1024x576")], num_predict=np_,
            think=True, fmt="json", label="think_format")
    except ProbeError as exc:
        return result(name, ERROR, str(exc), arch=arch)

    body = (resp.get("response") or "").strip()
    thinking = (resp.get("thinking") or "").strip()
    eval_count = resp.get("eval_count", 0)

    failures, diagnosis = [], None
    if cfg.get("require_nonempty_response", True) and not body:
        failures.append("response is empty")
        if eval_count >= np_:
            diagnosis = (f"eval_count ({eval_count}) hit num_predict ({np_}) with an "
                         f"empty response and {len(thinking)} chars of thinking — "
                         f"this is the num_predict trap, NOT a vision failure. "
                         f"Raise num_predict for this arch.")
        else:
            diagnosis = (f"Generated {eval_count} tokens then emitted no JSON body, "
                         f"well under the {np_} budget. That is the stock "
                         f"think+format signature — the fork's fix is missing.")
    if cfg.get("require_nonempty_thinking", False) and not thinking:
        failures.append("thinking is empty")
    if cfg.get("require_valid_json", True) and body:
        try:
            json.loads(body)
        except Exception as exc:
            failures.append(f"response is not valid JSON: {exc}")

    status = PASS if not failures else FAIL
    return result(name, status,
                  "; ".join(failures) if failures
                  else f"valid JSON after thinking ({eval_count} tokens)",
                  arch=arch,
                  expected="non-empty JSON response with think:true + format:json",
                  actual={"response_chars": len(body), "thinking_chars": len(thinking),
                          "eval_count": eval_count},
                  diagnosis=diagnosis)


# --------------------------------------------------------------------------
# 7. Extraction quality — delegates scoring to the existing vision_suite.py
# --------------------------------------------------------------------------

def check_quality(host, quality, expect, arch, tag, timeout=5400):
    """Runs vision_suite.py and applies thresholds to the scores it already
    computes. The suite reports; this turns the report into a verdict."""
    name = "extraction_quality"
    if not quality or quality.get("status") != "measured":
        return result(name, SKIP, "no quality thresholds recorded for this arch",
                      arch=arch)

    tests = quality.get("tests", ["scene_single", "document_single"])
    model = expect["model"]

    # visimgs/ is gitignored, so on a fresh clone vision_suite.py fails at import
    # (it loads ground_truth.json at module level). Generate the scenes first.
    ground_truth = os.path.join(SUITE_DIR, "visimgs", "ground_truth.json")
    if not os.path.exists(ground_truth):
        gen = subprocess.run(["python3", "gen_scenes.py"], cwd=SUITE_DIR,
                             capture_output=True, text=True, timeout=600)
        if not os.path.exists(ground_truth):
            return result(name, ERROR, "could not generate scenes for the quality run",
                          arch=arch,
                          actual=(gen.stdout + gen.stderr)[-400:],
                          diagnosis="gen_scenes.py needs Pillow and the DejaVu fonts "
                                    "(/usr/share/fonts/truetype/dejavu/). Install them "
                                    "or run without --quality.")

    env = dict(os.environ, THINK="false", NUM_PREDICT="2200",
               ONLY_TESTS=",".join(tests), HTTP_TIMEOUT=str(timeout))
    scores_path = os.path.join(SUITE_DIR, f"scores_{tag}.json")
    try:
        proc = subprocess.run(
            ["python3", "vision_suite.py", host, tag, model],
            cwd=SUITE_DIR, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return result(name, ERROR, f"vision_suite.py exceeded {timeout}s", arch=arch,
                      diagnosis="A whole-suite timeout with a healthy server is the "
                                "queue-starvation signature — check for another "
                                "client on this endpoint.")
    if not os.path.exists(scores_path):
        return result(name, ERROR, "vision_suite.py wrote no scores file", arch=arch,
                      actual=(proc.stdout or proc.stderr)[-600:])

    with open(scores_path) as fh:
        scores = json.load(fh)

    metrics, failures = {}, []
    valid = [bool(s.get("json_valid")) for s in scores.values() if "error" not in s]
    errored = [t for t, s in scores.items() if "error" in s]
    if errored:
        failures.append(f"tests errored: {', '.join(errored)}")
    if valid:
        metrics["json_valid"] = sum(valid) / len(valid)
        if metrics["json_valid"] < quality.get("min_json_valid", 1.0):
            failures.append(f"json_valid {metrics['json_valid']:.2f} "
                            f"< {quality['min_json_valid']}")
    scene = scores.get("scene_single", {})
    if scene.get("labels_total"):
        metrics["label_recall"] = scene["labels_found"] / scene["labels_total"]
        floor = quality.get("min_label_recall")
        if floor is not None and metrics["label_recall"] < floor:
            failures.append(f"label_recall {metrics['label_recall']:.2f} < {floor}")
    doc = scores.get("document_single", {})
    if doc.get("items_total"):
        metrics["qty_price_exact"] = doc["qty_price_right"] / doc["items_total"]
        floor = quality.get("min_qty_price_exact")
        if floor is not None and metrics["qty_price_exact"] < floor:
            failures.append(f"qty_price_exact {metrics['qty_price_exact']:.2f} < {floor}")

    status = PASS if not failures else FAIL
    return result(name, status,
                  "; ".join(failures) if failures else
                  " ".join(f"{k}={v:.2f}" for k, v in metrics.items()),
                  arch=arch,
                  expected={k: v for k, v in quality.items()
                            if k.startswith("min_")},
                  actual={k: round(v, 3) for k, v in metrics.items()},
                  scores_file=os.path.relpath(scores_path, SUITE_DIR))


# --------------------------------------------------------------------------
# Endpoint exclusivity — queue starvation is invisible without this
# --------------------------------------------------------------------------

def check_exclusivity(client, threshold_s=10.0):
    """A vision_suite.py run once failed all three tests with "timed out" at
    exactly 3 x 1800s while the server was perfectly healthy — another client was
    saturating the single slot. A saturated endpoint reports a FALSE FAILURE, so
    contention is detected and named rather than measured through.

    Signal: wall-clock minus the server's own total_duration is time spent
    queued behind someone else's request. Every probe in the run already recorded
    one, so this is a verdict over the whole run rather than an extra request.
    """
    waits = client.queue_waits
    worst_label, worst = max(waits, key=lambda kv: kv[1]) if waits else ("", 0.0)
    loaded = []
    try:
        loaded = [m.get("name") for m in client.ps()]
    except ProbeError:
        pass

    if worst > threshold_s:
        return result(
            "endpoint_exclusive", CONTENTION,
            f"requests queued up to {worst:.1f}s behind another client "
            f"(worst: {worst_label})",
            expected=f"queue wait <= {threshold_s}s", actual=f"{worst:.1f}s",
            diagnosis="Another client is using this endpoint. Results from a "
                      "contended run are not trustworthy — a saturated single slot "
                      "produces timeouts that look like model failures. Stop the "
                      "other client (e.g. pilot_teacher_v3_exam.py) and re-run.",
            models_loaded=loaded)
    return result("endpoint_exclusive", PASS,
                  f"no contention detected (worst queue wait {worst:.1f}s)",
                  expected=f"queue wait <= {threshold_s}s", actual=f"{worst:.1f}s",
                  models_loaded=loaded)

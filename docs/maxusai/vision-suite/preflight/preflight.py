#!/usr/bin/env python3
"""Pre-deploy regression harness for freshly built ollama images.

Single entry point: takes an image tag and a platform, runs the checks that apply
to that (platform, payload) combination, prints a readable expected-vs-actual
diff, writes a machine-readable result file, and exits non-zero on failure.

    ./preflight.py --host http://127.0.0.1:11437 --platform cuda \\
                   --image-tag maxusai/ollama:4987dd49-dynres

Exit codes:
    0  all applicable checks passed
    1  one or more checks failed
    2  harness/config error — unknown (platform, version) combination, server
       unreachable, or the version gate rejected the server
    3  endpoint contention detected; results are not trustworthy
    4  an applicable expectation has never been measured (NEEDS_BASELINE)

Long runs must be detached — a backgrounded run has been SIGTERM'd (exit 143)
mid-suite. Use:

    setsid nohup ./preflight.py ... > preflight.log 2>&1 < /dev/null &

and poll the --out file. Do NOT use `pgrep -f preflight.py` to test whether the
run is alive: the pattern matches the checking shell's own command line, so the
loop never exits.
"""
import argparse
import json
import os
import re
import sys
import time
import tomllib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import checks  # noqa: E402
from checks import (CONTENTION, ERROR, FAIL, NEEDS_BASELINE, PASS,  # noqa: E402
                    SKIP)
from probes import Ollama, ProbeError, find_container  # noqa: E402

DIR = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(DIR, "runs")

STATUS_MARK = {PASS: "PASS", FAIL: "FAIL", SKIP: "skip",
               NEEDS_BASELINE: "BASE", ERROR: "ERR ", CONTENTION: "BUSY"}


def load_expectations(path):
    with open(path, "rb") as fh:
        return tomllib.load(fh)


class ConfigError(Exception):
    """Unknown or unresolvable (platform, version) combination. Exit code 2 —
    distinct from a check failure, because nothing was actually measured."""


def resolve_profile(exp, platform, version):
    """(platform, version) -> profile. Fails loudly rather than defaulting."""
    candidates = {pid: p for pid, p in exp["profiles"].items()
                  if p["platform"] == platform}
    if not candidates:
        known = sorted({p["platform"] for p in exp["profiles"].values()})
        raise ConfigError(
            f"no profile for platform {platform!r}.\n"
            f"       Known platforms: {', '.join(known)}\n"
            f"       Add one to expectations.toml — see README.md.")
    for pid, prof in candidates.items():
        if re.match(prof["version_pattern"], version):
            return pid, prof
    raise ConfigError(
        f"unknown (platform, version) combination — refusing to guess.\n"
        f"       platform: {platform}\n"
        f"       version:  {version}\n"
        f"       Profiles for this platform and the versions they describe:\n"
        + "".join(f"         {pid}: {p['version_pattern']}\n"
                  for pid, p in candidates.items())
        + "       If this build is intentionally new, add a profile and its\n"
          "       measured expectations (README.md, 'Adding an expectation').")


def emit(results, out_path, meta):
    """Written after every check so a killed run still leaves usable data."""
    payload = {"meta": meta, "results": results,
               "summary": summarize(results)}
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    os.replace(tmp, out_path)
    return payload


def summarize(results):
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return counts


def print_result(r):
    arch = f" [{r['arch']}]" if r.get("arch") else ""
    print(f"  {STATUS_MARK.get(r['status'], r['status'])}  {r['check']}{arch}: {r['summary']}")
    if r["status"] in (PASS, SKIP):
        if r["status"] == SKIP and r.get("diagnosis"):
            print(f"        why: {r['diagnosis']}")
        return
    if "rows" in r:
        print(f"        {'size':>12} {'expected':>10} {'actual':>10} {'delta':>7}")
        for row in r["rows"]:
            flag = "" if row["ok"] else "  <-- MISMATCH"
            print(f"        {row['size']:>12} {row['expected']:>10} "
                  f"{row['actual']:>10} {row['delta']:>+7}{flag}")
    elif "arms" in r:
        for arm in r["arms"]:
            flag = "" if arm["ok"] else "  <-- MISMATCH"
            print(f"        {arm['arm']:>18}: expected {arm['expected']}, "
                  f"got {arm['actual']}{flag}")
    else:
        if r.get("expected") is not None:
            print(f"        expected: {r['expected']}")
        if r.get("actual") is not None:
            print(f"        actual:   {r['actual']}")
    if r.get("diagnosis"):
        print(f"        >> {r['diagnosis']}")


def run_arch(client, exp, profile_id, arch, args, container, results, flush):
    """All unpinned probes first, then the pinned ones. Changing a budget is a
    Runner option and forces a full model reload (tens of seconds to minutes),
    so probes are grouped by budget and never varied per request."""
    expect = exp.get("expect", {}).get(profile_id, {}).get(arch)
    if expect is None:
        results.append(checks.result(
            "expectation_lookup", ERROR,
            f"no expectations recorded for ({profile_id}, {arch})", arch=arch,
            diagnosis="The harness will not fall back to another arch's values. "
                      "Add an [expect.%s.%s] block — see README.md." % (profile_id, arch)))
        flush()
        return

    if expect.get("status") == "unmeasured":
        results.append(checks.result(
            "expectation_lookup", NEEDS_BASELINE,
            f"({profile_id}, {arch}) has no measured baseline", arch=arch,
            expected="measured expectations", actual="status = unmeasured",
            diagnosis=expect.get("reason", "").strip()))
        flush()
        return

    model = expect["model"]
    print(f"\n--- {arch} ({model}) ---")
    available = client.tags()
    if model not in available:
        results.append(checks.result(
            "model_present", FAIL, f"{model} is not on this server", arch=arch,
            expected=model, actual=f"{len(available)} models, none matching"))
        flush()
        return

    # A fresh load is what makes the payload proof valid: reading load_hparams
    # from a stale log can attribute a PREVIOUS build's budget to this one.
    print(f"    unloading {model} to force a fresh load_hparams block...")
    client.unload(model)
    since = time.time() - 5

    # ---- budget group 1: default budget, no reloads ----
    try:
        baseline, _ = client.text_baseline(model)
    except ProbeError as exc:
        results.append(checks.result("text_baseline", ERROR, str(exc), arch=arch))
        flush()
        return
    results.append(checks.result(
        "text_baseline", PASS, f"text-only prompt_eval_count = {baseline}",
        arch=arch, actual=baseline))
    flush()

    print(f"    token ladder ({len(exp['ladder_sizes'])} geometries, default budget)...")
    results.append(checks.check_ladder(
        client, expect, arch, exp["ladder_sizes"], baseline,
        exp.get("marker_allowance", 2)))
    flush()

    results.append(checks.check_payload_proof(
        expect, arch, container, since, args.log_cmd))
    flush()

    if not args.skip_think:
        print("    think + format probe...")
        results.append(checks.check_think_format(
            client, expect, arch, exp["min_num_predict"]))
        flush()

    if args.quality:
        print("    extraction quality via vision_suite.py (slow)...")
        quality = exp.get("quality", {}).get(profile_id, {}).get(arch)
        results.append(checks.check_quality(
            client.host, quality, expect, arch,
            f"preflight-{args.platform}-{arch}"))
        flush()

    # ---- budget group 2: pinned budget (each arm costs a full reload) ----
    if args.skip_pinned:
        results.append(checks.result("pinned_budget", SKIP, "--skip-pinned", arch=arch))
    else:
        if expect.get("pinned"):
            print("    pinned-budget probe (forces a model reload, may take minutes)...")
        results.append(checks.check_pinned_budget(
            client, expect, arch, baseline, exp.get("marker_allowance", 2)))
    flush()


def main():
    ap = argparse.ArgumentParser(
        description="Pre-deploy regression harness for ollama images.")
    ap.add_argument("--host", required=True, help="e.g. http://127.0.0.1:11437")
    ap.add_argument("--platform", required=True,
                    choices=["cuda", "rocm", "apple-silicon", "apple-silicon-mlx"])
    ap.add_argument("--image-tag", help="image tag under test; verified against "
                                        "the running container when resolvable")
    ap.add_argument("--arch", action="append",
                    help="limit to these arches (default: all in the profile)")
    ap.add_argument("--container", help="container name (default: auto-detect by port)")
    ap.add_argument("--exec-cmd", help="template for running a command in the "
                                       "container, e.g. 'ssh h docker exec {container} sh -c ...'")
    ap.add_argument("--log-cmd", help="template for reading container logs; "
                                      "{container} and {since} are substituted")
    ap.add_argument("--expectations", default=os.path.join(DIR, "expectations.toml"))
    ap.add_argument("--out", help="results JSON (default: runs/preflight-<platform>-<ts>.json)")
    ap.add_argument("--quality", action="store_true",
                    help="also run the vision_suite.py extraction scoring (slow)")
    ap.add_argument("--skip-pinned", action="store_true",
                    help="skip the pinned-budget probe (saves two model reloads)")
    ap.add_argument("--skip-think", action="store_true")
    ap.add_argument("--allow-unmeasured", action="store_true",
                    help="report NEEDS_BASELINE without failing the run")
    ap.add_argument("--contention-threshold", type=float, default=10.0,
                    help="seconds of queue wait that count as contention")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    # Detached runs (setsid nohup ... &) otherwise buffer stdout until exit, so a
    # multi-hour run looks hung. The --out file is written incrementally either
    # way; this just makes the log readable while it happens.
    sys.stdout.reconfigure(line_buffering=True)

    exp = load_expectations(args.expectations)
    client = Ollama(args.host, timeout=args.timeout)

    os.makedirs(RUNS, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    out_path = args.out or os.path.join(RUNS, f"preflight-{args.platform}-{stamp}.json")

    results = []
    meta = {"host": args.host, "platform": args.platform,
            "image_tag": args.image_tag, "started_utc": stamp,
            "expectations_file": os.path.relpath(args.expectations, DIR),
            "schema_version": exp.get("schema_version")}

    # Persist after every check AND print anything not yet shown, so a detached
    # multi-minute run reports as it goes instead of going quiet until an arch
    # completes.
    printed = [0]

    def flush():
        emit(results, out_path, meta)
        while printed[0] < len(results):
            print_result(results[printed[0]])
            printed[0] += 1

    print(f"preflight: {args.host}  platform={args.platform}  "
          f"tag={args.image_tag or '(unspecified)'}")
    print(f"results -> {out_path}")

    # ---- version gate. Nothing downstream is trusted without it. ----
    try:
        version = client.version()
    except ProbeError as exc:
        print(f"\nERROR: cannot reach {args.host}: {exc}")
        results.append(checks.result("version", ERROR, str(exc)))
        flush()
        return 2
    meta["version"] = version

    try:
        profile_id, profile = resolve_profile(exp, args.platform, version)
    except ConfigError as exc:
        print(f"\nERROR: {exc}")
        results.append(checks.result("profile_lookup", ERROR, str(exc).splitlines()[0],
                                     actual=version))
        flush()
        return 2
    meta["profile"] = profile_id
    meta["patchset"] = profile.get("patchset")
    print(f"profile: {profile_id}  payload={'+'.join(profile.get('patchset', []))}\n")

    gate = checks.check_version(client, profile, profile_id)
    results.append(gate)
    flush()
    if gate["status"] != PASS:
        print("\nABORTED at the version gate — no measurement below it would be "
              "attributable to this build.")
        return 2

    port = args.host.rsplit(":", 1)[-1].split("/")[0]
    container = find_container(port, args.container)
    meta["container"] = container
    if container:
        print(f"container: {container}")
    else:
        print("container: not resolved — binary and log checks will be skipped")

    results.append(checks.check_image_tag(client, profile, args.image_tag, container))
    results.append(checks.check_patch_marker(profile, container, args.exec_cmd))
    flush()

    arches = args.arch or profile["arches"]
    unknown = [a for a in arches if a not in profile["arches"]]
    if unknown:
        print(f"\nERROR: arch(es) {unknown} are not in profile {profile_id} "
              f"({profile['arches']}). Refusing to guess.")
        results.append(checks.result("arch_lookup", ERROR,
                                     f"{unknown} not in profile {profile_id}"))
        flush()
        return 2

    for arch in arches:
        run_arch(client, exp, profile_id, arch, args, container, results, flush)

    # ---- contention verdict, from every queue wait observed during the run ----
    results.append(checks.check_exclusivity(client, args.contention_threshold))
    flush()

    counts = summarize(results)
    print("\n" + "=" * 72)
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"  results: {out_path}")
    print("=" * 72)

    if counts.get(CONTENTION):
        print("\nVERDICT: CONTENDED — another client is using this endpoint. "
              "Re-run exclusively before trusting any of the above.")
        return 3
    if counts.get(FAIL) or counts.get(ERROR):
        print("\nVERDICT: FAIL")
        return 1
    if counts.get(NEEDS_BASELINE) and not args.allow_unmeasured:
        print("\nVERDICT: NEEDS BASELINE — an applicable expectation has never been "
              "measured, so this build is not validated. Measure it (README.md) or "
              "re-run with --allow-unmeasured to acknowledge the gap.")
        return 4
    print("\nVERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

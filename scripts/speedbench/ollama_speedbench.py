#!/usr/bin/env python3
"""Run SPEED-Bench prompts against an OpenAI-compatible server and report
speculative-decoding metrics per category.

This is the Ollama-side shim for NVIDIA's SPEED-Bench. It replays a prepared
dataset (see README.md for provenance) through /v1/chat/completions with
streaming + include_usage, records each response's timings block, and reports
per-category and overall aggregates.

Aggregation is always computed from summed counts, never from averaged ratios:
  draft acceptance  = sum(draft_n_accepted) / sum(draft_n)
  tokens per step   = sum(predicted_n) / sum(steps)       (acceptance length,
                                                         bonus token included)
  time per step     = sum(predicted_ms) / sum(steps)
  eval rate         = sum(predicted_n) / sum(predicted_ms) * 1000
  prefill rate      = sum(prompt_n) / sum(prompt_ms) * 1000
A metric whose inputs a server never reports is printed as n/a, matching the
API rule that absent fields mean "not reported", not zero.

Input JSONL, one record per line:
  {"question_id": "...", "category": "coding", "turns": ["prompt", "follow-up", ...]}

Only "turns" (or a single "prompt"/"text" string) is required. Each turn is a
user message; multi-turn records replay turns in order, feeding each live
assistant reply back as conversation history. Records that still carry the
upstream placeholder text ("FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE
SOURCE") are skipped, since their real prompt was never fetched.
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED"


def load_records(path, categories, limit):
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{lineno}: invalid JSON: {e}")
            turns = rec.get("turns")
            if turns is None:
                single = rec.get("prompt") or rec.get("text")
                if single is None:
                    sys.exit(f"{path}:{lineno}: record has neither 'turns' nor 'prompt'/'text'")
                turns = [single]
            if any(PLACEHOLDER in t for t in turns):
                continue
            if categories and rec.get("category") not in categories:
                continue
            records.append({
                "id": rec.get("question_id") or rec.get("id") or f"line{lineno}",
                "category": rec.get("category") or "uncategorized",
                "turns": turns,
            })
            if limit and len(records) >= limit:
                break
    return records


def run_conversation(url, model, turns, temperature, max_tokens, timeout):
    """Replay one record, one /v1/chat/completions call per turn.

    Returns a list of per-turn result dicts. Timestamps are wall clock; the
    timings block comes from the final streamed chunk (include_usage).
    """
    messages = []
    results = []
    for turn in turns:
        messages.append({"role": "user", "content": turn})
        body = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }).encode()
        req = urllib.request.Request(
            f"{url}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        start = time.monotonic()
        ttft = None
        content = []
        final = {}
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    line = line[5:].strip()
                    if line == "[DONE]":
                        break
                    chunk = json.loads(line)
                    if chunk.get("usage") is not None:
                        final = chunk
                        continue
                    error = chunk.get("error")
                    if error:
                        raise RuntimeError(error.get("message") or error)
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        piece = delta.get("content") or delta.get("reasoning") or ""
                        if piece:
                            if ttft is None:
                                ttft = time.monotonic() - start
                            content.append(piece)
        except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
            return results, f"turn {len(results) + 1}: {e}"
        latency = time.monotonic() - start
        reply = "".join(content)
        messages.append({"role": "assistant", "content": reply})
        timings = final.get("timings") or {}
        usage = final.get("usage") or {}
        results.append({
            "latency_s": latency,
            "ttft_s": ttft,
            "usage": usage,
            "timings": timings,
        })
    return results, None


class Sums:
    """Summed counters for one group (a category, or all records)."""

    def __init__(self):
        self.requests = 0
        self.errors = 0
        self.latency = 0.0
        self.ttfts = []
        self.prompt_n = 0
        self.prompt_ms = 0.0
        self.predicted_n = 0
        self.predicted_ms = 0.0
        self.completion_tokens = 0  # usage fallback when a server reports no timings
        self.draft_n = None
        self.draft_n_accepted = None
        self.draft_steps = None
        self.steps = None

    def record(self, turn):
        self.requests += 1
        self.latency += turn["latency_s"]
        if turn["ttft_s"] is not None:
            self.ttfts.append(turn["ttft_s"])
        t = turn["timings"]
        if t:
            self.prompt_n += t.get("prompt_n") or 0
            self.prompt_ms += t.get("prompt_ms") or 0.0
            self.predicted_n += t.get("predicted_n") or 0
            self.predicted_ms += t.get("predicted_ms") or 0.0
        else:
            self.completion_tokens += turn["usage"].get("completion_tokens") or 0
        for field, key in (("draft_n", "draft_n"),
                           ("draft_n_accepted", "draft_n_accepted"),
                           ("draft_steps", "draft_steps"),
                           ("steps", "steps")):
            value = t.get(key)
            if value is not None:
                current = getattr(self, field) or 0
                setattr(self, field, current + value)

    def metrics(self):
        def rate(n, ms):
            return f"{n / ms * 1000:.1f}" if ms else None

        def div(a, b, fmt="{:.2f}"):
            return fmt.format(a / b) if b else None

        m = {
            "requests": str(self.requests),
            "draft acceptance": None,
            "tokens per step": None,
            "time per step": None,
            "eval rate (tok/s)": rate(self.predicted_n, self.predicted_ms),
            "prefill rate (tok/s)": rate(self.prompt_n, self.prompt_ms),
            "mean ttft (s)": "{:.2f}".format(sum(self.ttfts) / len(self.ttfts)) if self.ttfts else None,
            "mean latency (s)": "{:.2f}".format(self.latency / self.requests) if self.requests else None,
        }
        if self.predicted_ms == 0 and self.latency > 0 and self.completion_tokens:
            m["eval rate (tok/s)"] = "{:.1f}".format(self.completion_tokens / self.latency) + " (wall)"
        if self.draft_n:
            m["draft acceptance"] = div(100 * self.draft_n_accepted, self.draft_n, "{:.1f}%")
        if self.steps:
            m["tokens per step"] = div(self.predicted_n, self.steps)
            m["time per step"] = "{:.2f}ms".format(self.predicted_ms / self.steps)
        return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True, help="prepared SPEED-Bench JSONL")
    p.add_argument("--url", default="http://localhost:11434")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="SPEED-Bench default is 0 for non-reasoning models")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--category", action="append",
                   help="only these categories (repeatable)")
    p.add_argument("--limit", type=int, default=0, help="cap records read from --input")
    p.add_argument("--timeout", type=int, default=86400, help="per-request timeout in seconds")
    p.add_argument("--json-out", help="write the full report as JSON")
    args = p.parse_args()

    url = args.url.rstrip("/")
    records = load_records(args.input, set(args.category or ()), args.limit)
    if not records:
        sys.exit("no runnable records (check --category/--limit and placeholder rows)")
    print(f"{len(records)} records, model {args.model}, {url}", file=sys.stderr)

    groups = {}
    overall = Sums()
    for i, rec in enumerate(records, 1):
        turns, err = run_conversation(url, args.model, rec["turns"],
                                      args.temperature, args.max_tokens, args.timeout)
        group = groups.setdefault(rec["category"], Sums())
        if err:
            group.errors += 1
            overall.errors += 1
            print(f"[{i}/{len(records)}] {rec['id']}: error: {err}", file=sys.stderr)
            continue
        for turn in turns:
            group.record(turn)
            overall.record(turn)
        if i % 10 == 0 or i == len(records):
            print(f"[{i}/{len(records)}] done", file=sys.stderr)

    keys = ["requests", "draft acceptance", "tokens per step", "time per step",
            "eval rate (tok/s)", "prefill rate (tok/s)", "mean ttft (s)", "mean latency (s)"]
    rows = [("category", *keys)]
    for name in sorted(groups):
        if groups[name].errors:
            print(f"warning: category {name}: {groups[name].errors} failed records", file=sys.stderr)
        m = groups[name].metrics()
        rows.append((name, *(m[k] or "n/a" for k in keys)))
    m = overall.metrics()
    rows.append(("OVERALL", *(m[k] or "n/a" for k in keys)))

    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  ".join(cell.ljust(widths[c]) for c, cell in enumerate(r)))

    if args.json_out:
        report = {
            "model": args.model,
            "url": url,
            "input": args.input,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "records": len(records),
            "categories": {name: groups[name].metrics() for name in sorted(groups)},
            "overall": overall.metrics(),
        }
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()

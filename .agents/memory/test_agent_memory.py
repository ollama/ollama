"""Tests for agent_memory.py. Run from this directory: python3 -m unittest -v"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_memory.py")


class Sandbox(unittest.TestCase):
    """Each test gets a fake HOME, a global index dir, and two git projects."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = os.path.realpath(self.tmp.name)
        self.home = os.path.join(root, "home")
        self.proj = os.path.join(self.home, "src", "app")
        self.other = os.path.join(self.home, "src", "lib")
        for p in (self.proj, self.other):
            os.makedirs(os.path.join(p, ".git"))
        self.env = dict(os.environ, HOME=self.home, AGENT_MEMORY_HOME=os.path.join(self.home, ".agents", "memory"))
        for k in ("AGENT_MEMORY_SESSION", "AGENT_MEMORY_PROJECT", "AGENT_MEMORY_HARNESS"):
            self.env.pop(k, None)

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self, *args, cwd=None, ok=True, env=None):
        r = subprocess.run([sys.executable, TOOL] + list(args), cwd=cwd or self.proj,
                           env=env or self.env, capture_output=True, text=True)
        if ok and r.returncode != 0:
            self.fail("agent-mem %s failed: %s" % (" ".join(args), r.stderr))
        if not ok:
            self.assertNotEqual(r.returncode, 0, r.stdout)
        return r.stdout + r.stderr

    def sid(self, out):
        return out.split()[1]

    def journal(self, project, sid):
        with open(os.path.join(project, ".agents", "memory", "sessions", sid, "journal.jsonl")) as f:
            return [json.loads(l) for l in f if l.strip()]


class CLITest(Sandbox):
    def test_init_writes_gitignore_and_upserts_agents_md_once(self):
        self.cli("init", "--agents-md")
        self.cli("init", "--agents-md")
        with open(os.path.join(self.proj, "AGENTS.md")) as f:
            text = f.read()
        self.assertEqual(text.count("agent-memory:begin"), 1)
        with open(os.path.join(self.proj, ".agents", "memory", ".gitignore")) as f:
            self.assertIn("sessions/", f.read())

    def test_session_lifecycle_and_brief(self):
        sid = self.sid(self.cli("start", "--title", "fix OOM on long prompts"))
        self.cli("log", "finding", "KV is 64 KiB per token on qwen")
        self.cli("log", "decision", "use bonsai for agent work", "--why", "8 GiB more headroom")
        out = self.cli("log", "next", "measure peak memory at 8k tokens")
        ref = out.split()[1]
        self.cli("next", "--done", ref)
        self.assertIn("no open next steps", self.cli("next"))
        self.cli("handoff", "--goal", "stable local agent", "--state", "bonsai chosen",
                 "--decision", "use bonsai for agent work", "--next", "tune pi contextWindow")
        decisions = [e for e in self.journal(self.proj, sid) if e["kind"] == "decision"]
        self.assertEqual(len(decisions), 1, "a decision repeated in the handoff is not logged twice")

        b = self.cli("resume")
        self.assertIn("## Handoff from " + sid, b)
        self.assertIn("tune pi contextWindow", b)
        self.assertIn("(why: 8 GiB more headroom)", b)
        self.assertNotIn("measure peak memory", b.split("## Open next steps")[1].split("##")[0])

        small = self.cli("resume", "--max-chars", "200")
        self.assertLessEqual(len(small.splitlines()[0]), 200)
        self.assertIn("brief truncated", small)

        self.cli("end", "--summary", "done for today")
        self.cli("log", "step", "after end", ok=False)

    def test_new_session_brief_shows_previous_session(self):
        self.cli("start", "--title", "first")
        self.cli("handoff", "--goal", "g1", "--state", "s1", "--next", "carry me over")
        out = self.cli("start", "--title", "second", "--harness", "zed")
        self.assertIn("carry me over", out)
        self.assertIn("Latest session", out)

    def test_harnesses_never_share_sessions(self):
        self.cli("start", "--title", "pi main")
        self.cli("log", "step", "x", "--harness", "zed", ok=False)
        zed = self.sid(self.cli("start", "--title", "zed main", "--harness", "zed"))
        self.cli("log", "step", "from zed", "--harness", "zed")
        self.assertEqual(self.journal(self.proj, zed)[-1]["text"], "from zed")

    def test_subagent_task_round_trip(self):
        parent = self.sid(self.cli("start", "--title", "main", "--harness", "zed"))
        out = self.cli("task", "new", "--harness", "zed", "--title", "write regression test", "--brief", "cover the 8k OOM")
        tid = out.split()[1]
        self.assertIn("--task %s" % tid, out)

        main_pi = self.sid(self.cli("start", "--title", "pi main session"))
        child = self.sid(self.cli("start", "--role", "sub", "--task", tid, "--title", "write regression test"))
        self.assertNotEqual(child, main_pi)
        self.cli("log", "step", "wrote test", "--task", tid)
        self.cli("log", "step", "main pi still logs to its own session")
        self.assertEqual(self.journal(self.proj, main_pi)[-1]["text"], "main pi still logs to its own session")

        again = self.sid(self.cli("start", "--role", "sub", "--task", tid, "--title", "resume after crash"))
        self.assertEqual(again, child, "restarting a task's sub-agent resumes its session")

        self.cli("task", "done", tid, "--result", "test added in sched_test.go\nall green")
        show = self.cli("task", "show", tid)
        self.assertIn("[done]", show)
        self.assertIn("all green", show)
        self.assertTrue(any(e["kind"] == "task" and "done" in e["text"] for e in self.journal(self.proj, parent)))
        self.cli("log", "step", "late", "--task", tid, ok=False)
        self.assertNotIn("- %s [" % tid, self.cli("resume"), "finished tasks leave the not-done list")

    def test_global_index_spans_projects(self):
        self.cli("start", "--title", "app work")
        self.cli("log", "decision", "pin keep_alive to 30m", "--why", "unload drops the prompt cache")
        self.cli("start", "--title", "lib work", cwd=self.other)
        self.cli("log", "decision", "keep_alive stays default in lib", cwd=self.other)
        hits = self.cli("search", "keep_alive", "--global")
        self.assertIn("app", hits)
        self.assertIn("lib", hits)
        self.assertIn("pin keep_alive to 30m", self.cli("search", "keep_alive"))
        self.assertNotIn("stays default in lib", self.cli("search", "keep_alive"))
        listing = self.cli("list", "--global")
        self.assertIn(self.proj, listing)
        self.assertIn(self.other, listing)

    def test_project_discovery(self):
        sub = os.path.join(self.proj, "pkg", "deep")
        os.makedirs(sub)
        self.cli("start", "--title", "from a subdirectory", cwd=sub)
        self.assertTrue(os.path.isdir(os.path.join(self.proj, ".agents", "memory", "sessions")))
        self.assertIn("no project found", self.cli("list", cwd=self.home, ok=False))
        self.assertIn("home directory", self.cli("list", "--project", self.home, ok=False))

    def test_concurrent_writers_get_unique_sequence_numbers(self):
        sid = self.sid(self.cli("start", "--title", "parallel"))
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda i: self.cli("log", "step", "event %d" % i, "--session", sid), range(24)))
        seqs = [e["seq"] for e in self.journal(self.proj, sid)]
        self.assertEqual(sorted(seqs), list(range(1, 26)))

    def test_torn_journal_line_is_skipped(self):
        sid = self.sid(self.cli("start", "--title", "crash"))
        with open(os.path.join(self.proj, ".agents", "memory", "sessions", sid, "journal.jsonl"), "a") as f:
            f.write('{"seq": 2, "kind": "st')
        self.cli("log", "step", "after crash")
        self.assertIn("after crash", self.cli("resume"))

    def test_invalid_task_id_is_rejected(self):
        self.assertIn("invalid task id", self.cli("task", "show", "../../etc/passwd", ok=False))


class MCPTest(Sandbox):
    def rpc(self, messages):
        lines = "".join(json.dumps(m) + "\n" for m in messages)
        r = subprocess.run([sys.executable, TOOL, "mcp"], input=lines, cwd=self.home, env=self.env,
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        return [json.loads(l) for l in r.stdout.splitlines() if l.strip()]

    def call(self, i, name, **args):
        return {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": args}}

    def test_handshake_and_tools(self):
        out = self.rpc([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "ping"},
            {"jsonrpc": "2.0", "id": 4, "method": "resources/list"},
            {"jsonrpc": "2.0", "id": 5, "method": "initialize", "params": {"protocolVersion": "2099-01-01"}},
        ])
        self.assertEqual([m["id"] for m in out], [1, 2, 3, 4, 5], "notifications get no reply")
        self.assertEqual(out[0]["result"]["protocolVersion"], "2025-03-26")
        names = {t["name"] for t in out[1]["result"]["tools"]}
        self.assertIn("memory_start", names)
        self.assertIn("memory_task_create", names)
        for t in out[1]["result"]["tools"]:
            self.assertEqual(t["inputSchema"]["type"], "object")
        self.assertEqual(out[3]["error"]["code"], -32601)
        self.assertEqual(out[4]["result"]["protocolVersion"], "2025-06-18")

    def test_zed_session_through_mcp(self):
        p = self.proj
        out = self.rpc([
            self.call(1, "memory_start", project=p, title="refactor scheduler"),
            self.call(2, "memory_log", project=p, kind="decision", text="split sched.go", why="1,500 lines"),
            self.call(3, "memory_log", project=p, kind="next", text="move eviction to its own file"),
            self.call(4, "memory_handoff", project=p, goal="smaller scheduler", state="plan agreed",
                      next=["move eviction to its own file"], open_questions=["keep the old API?"]),
            self.call(5, "memory_task_create", project=p, title="write eviction tests", brief="table tests for LRU"),
            self.call(6, "memory_resume", project=p),
            self.call(7, "memory_log", project=p, kind="bogus", text="x"),
            self.call(8, "memory_log", kind="step", text="no project given"),
            self.call(9, "memory_search", query="eviction", **{"global": True}),
        ])
        by_id = {m["id"]: m["result"] for m in out}
        for i in (1, 2, 3, 4, 5, 6, 9):
            self.assertFalse(by_id[i]["isError"], by_id[i])
        brief = by_id[6]["content"][0]["text"]
        self.assertIn("smaller scheduler", brief)
        self.assertIn("keep the old API?", brief)
        self.assertEqual(brief.count("move eviction to its own file"), 2, "once in the handoff, once as one open step")
        self.assertIn("write eviction tests", brief)
        self.assertTrue(by_id[7]["isError"])
        self.assertIn("no project found", by_id[8]["content"][0]["text"])

        tid = by_id[5]["content"][0]["text"].split()[1]
        self.cli("start", "--role", "sub", "--task", tid, "--title", "write eviction tests")
        self.cli("task", "done", tid, "--result", "6 cases, all pass")
        done = self.rpc([self.call(1, "memory_task_show", project=p, task=tid)])
        self.assertIn("6 cases, all pass", done[0]["result"]["content"][0]["text"])

    def test_bad_json_does_not_stop_the_server(self):
        r = subprocess.run([sys.executable, TOOL, "mcp"], input='not json\n{"jsonrpc":"2.0","id":7,"method":"ping"}\n',
                           cwd=self.home, env=self.env, capture_output=True, text=True, timeout=30)
        out = [json.loads(l) for l in r.stdout.splitlines()]
        self.assertEqual(out[0]["error"]["code"], -32700)
        self.assertEqual(out[1]["id"], 7)


if __name__ == "__main__":
    unittest.main()

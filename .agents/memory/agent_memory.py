#!/usr/bin/env python3
"""agent-mem keeps durable working memory for coding-agent sessions.

Each project stores its sessions under <project>/.agents/memory/sessions/:

    <session-id>/session.json    harness, role, title, status, task
    <session-id>/journal.jsonl   append-only events, numbered per session
    <session-id>/handoff.md      the latest handoff, rewritten in place
    tasks/<task-id>.json         sub-agent briefs and results

~/.agents/memory/index.jsonl indexes session starts, decisions, handoffs and
ends across every project, so a session can find earlier work anywhere.

Zed Agent talks to it as an MCP server (`agent-mem mcp`); pi and other CLI
agents run the same operations as shell commands. Standard library only, and
compatible with the Python 3.9 that ships with macOS.
"""

import argparse
import contextlib
import datetime
import fcntl
import json
import os
import re
import secrets
import shlex
import shutil
import sys
import tempfile
import time

VERSION = "0.1.0"
STORE = os.path.join(".agents", "memory", "sessions")
# Caps one event so a pasted log can't swamp the journal or the brief.
MAX_TEXT = 8000
DEFAULT_BRIEF = 6000  # characters, roughly 1,500 tokens of a small context window
LOG_KINDS = ("step", "decision", "finding", "next", "blocker", "note", "compaction")
TASK_DONE_STATUSES = ("done", "failed", "blocked")
TASK_ID = re.compile(r"^t-[0-9a-f]{6}$")
MCP_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
BLOCK_BEGIN = "<!-- agent-memory:begin -->"
BLOCK_END = "<!-- agent-memory:end -->"


class AgentMemoryError(Exception):
    pass


def home_dir():
    return os.environ.get("AGENT_MEMORY_HOME") or os.path.join(os.path.expanduser("~"), ".agents", "memory")


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def clip(text):
    text = (text or "").strip()
    if len(text) > MAX_TEXT:
        text = text[:MAX_TEXT] + " ...[clipped]"
    return text


def slug(s):
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower()) or "agent"


def short(sid):
    return sid.rsplit("-", 1)[-1]


def first_line(text):
    return ((text or "").strip().splitlines() or [""])[0]


def command():
    """How a shell should invoke this tool, for prompts handed to sub-agents."""
    if shutil.which("agent-mem"):
        return "agent-mem"
    return shlex.quote(os.path.abspath(sys.argv[0]))


@contextlib.contextmanager
def flocked(directory):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, ".lock"), "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def write_atomic(path, text):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return None


def read_jsonl(path):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue  # a line torn by a crash mid-write
    except FileNotFoundError:
        pass
    return out


def read_text(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def find_project(explicit=None):
    path = explicit or os.environ.get("AGENT_MEMORY_PROJECT")
    home = os.path.realpath(os.path.expanduser("~"))
    if path:
        path = os.path.realpath(os.path.expanduser(path))
        if not os.path.isdir(path):
            raise AgentMemoryError("project directory does not exist: %s" % path)
        if path == home:
            raise AgentMemoryError("the home directory holds the global index; pass a project directory")
        return path
    d = os.path.realpath(os.getcwd())
    while True:
        if d != home and (os.path.exists(os.path.join(d, ".git")) or os.path.isdir(os.path.join(d, STORE))):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise AgentMemoryError("no project found above %s: run inside a git repository or pass --project" % os.getcwd())
        d = parent


def index_append(store, sid, kind, text):
    """Record an event in the cross-project index. A failure here never fails the project write."""
    record = {"ts": now(), "project": store.project, "session": sid, "kind": kind, "text": clip(text)[:500]}
    try:
        directory = home_dir()
        with flocked(directory):
            with open(os.path.join(directory, "index.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        print("agent-mem: global index not updated: %s" % e, file=sys.stderr)


class Store:
    def __init__(self, project):
        self.project = project
        self.root = os.path.join(project, STORE)

    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def lock(self):
        return flocked(self.root)

    def meta(self, sid):
        return read_json(self.path(sid, "session.json"))

    def save_meta(self, meta):
        meta["updated"] = now()
        meta["updated_epoch"] = time.time()
        write_atomic(self.path(meta["id"], "session.json"), json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    def sessions(self):
        try:
            names = os.listdir(self.root)
        except FileNotFoundError:
            return []
        metas = [self.meta(n) for n in names if n != "tasks" and os.path.isdir(self.path(n))]
        metas = [m for m in metas if m]
        metas.sort(key=lambda m: m.get("updated_epoch", 0), reverse=True)
        return metas

    def journal(self, sid):
        return read_jsonl(self.path(sid, "journal.jsonl"))

    def append(self, sid, kind, text, why=None, refs=None):
        """Append one event to a session's journal. The caller holds the lock."""
        meta = self.meta(sid)
        if not meta:
            raise AgentMemoryError("no session %s in %s" % (sid, self.project))
        seq = max([e.get("seq", 0) for e in self.journal(sid)] or [0]) + 1
        event = {"seq": seq, "ts": now(), "kind": kind, "text": clip(text)}
        if why:
            event["why"] = clip(why)
        if refs:
            event["refs"] = list(refs)
        path = self.path(sid, "journal.jsonl")
        with open(path, "ab") as f:
            # A crash mid-write leaves a line without its newline; end it so
            # this event starts a fresh line instead of joining the torn one.
            if f.tell() > 0:
                with open(path, "rb") as r:
                    r.seek(-1, os.SEEK_END)
                    if r.read(1) != b"\n":
                        f.write(b"\n")
            f.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
        self.save_meta(meta)
        return event

    def require_open(self, sid):
        meta = self.meta(sid)
        if not meta:
            raise AgentMemoryError("no session %s in %s" % (sid, self.project))
        if meta.get("status") != "open":
            raise AgentMemoryError("session %s is %s; start a new one" % (sid, meta.get("status")))

    def task(self, tid):
        if not TASK_ID.match(tid or ""):
            raise AgentMemoryError("invalid task id %r: expected t- and six hex digits" % tid)
        t = read_json(self.path("tasks", tid + ".json"))
        if not t:
            raise AgentMemoryError("no task %s in %s" % (tid, self.project))
        return t

    def save_task(self, t):
        t["updated"] = now()
        write_atomic(self.path("tasks", t["id"] + ".json"), json.dumps(t, indent=2, ensure_ascii=False) + "\n")

    def tasks(self):
        try:
            names = os.listdir(self.path("tasks"))
        except FileNotFoundError:
            return []
        out = [read_json(self.path("tasks", n)) for n in names if n.endswith(".json")]
        out = [t for t in out if t]
        out.sort(key=lambda t: t.get("created", ""), reverse=True)
        return out

    def active_path(self, harness):
        return self.path(".active-" + slug(harness))

    def match(self, ref):
        ids = [m["id"] for m in self.sessions()]
        if ref in ids:
            return ref
        hits = [i for i in ids if short(i) == ref or i.endswith(ref)]
        if len(hits) == 1:
            return hits[0]
        raise AgentMemoryError("no unique session matches %r in %s" % (ref, self.project))

    def resolve(self, session=None, task=None, harness=None):
        """Pick the session a command applies to: explicit id, the task's
        sub-agent session, $AGENT_MEMORY_SESSION, then this harness's active
        session. It never falls back to another harness's session."""
        if session:
            return self.match(session)
        if task:
            child = self.task(task).get("child_session")
            if not child:
                raise AgentMemoryError("task %s has no session yet: run `%s start --task %s` first" % (task, command(), task))
            return child
        env = os.environ.get("AGENT_MEMORY_SESSION")
        if env:
            return self.match(env)
        if harness:
            sid = read_text(self.active_path(harness)).strip()
            meta = self.meta(sid) if sid else None
            if meta and meta.get("status") == "open":
                return sid
        raise AgentMemoryError("no open %s session in %s: start one first" % (harness or "", self.project))


def open_next(store):
    """Next steps across every session in the project that no done event closed."""
    items, done = [], set()
    for meta in store.sessions():
        s = short(meta["id"])
        for e in store.journal(meta["id"]):
            if e.get("kind") == "next":
                items.append((e.get("ts", ""), "%s#%d" % (s, e["seq"]), e.get("text", "")))
            elif e.get("kind") == "done":
                done.update(e.get("refs", []))
    items.sort()
    return [i for i in items if i[1] not in done]


def decisions(store):
    out = []
    for meta in store.sessions():
        s = short(meta["id"])
        for e in store.journal(meta["id"]):
            if e.get("kind") == "decision":
                out.append((e.get("ts", ""), "%s#%d" % (s, e["seq"]), e.get("text", ""), e.get("why", "")))
    out.sort()
    return out


def render(sections, max_chars):
    out, used, cut = [], 0, False
    for title, lines in sections:
        if not lines:
            continue
        for line in (["## " + title] if title else []) + lines:
            if used + len(line) + 1 > max_chars:
                cut = True
                break
            out.append(line)
            used += len(line) + 1
        if cut:
            break
        out.append("")
        used += 1
    if cut:
        out.append("... brief truncated at %d characters; use resume with a larger max_chars or search for details." % max_chars)
    return "\n".join(out).rstrip() + "\n"


def brief(store, sid=None, exclude=None, max_chars=DEFAULT_BRIEF):
    """A bounded summary of where the project's work stands, for a fresh context."""
    sessions = [m for m in store.sessions() if m["id"] != exclude]
    if not sessions:
        return "No earlier sessions in %s.\n" % store.project
    target = store.meta(store.match(sid)) if sid else sessions[0]
    head = [
        "# Session memory: %s" % os.path.basename(store.project),
        "Latest session %s: %s (%s %s, %s, updated %s)" % (
            target["id"], target.get("title", ""), target.get("harness", "?"),
            target.get("role", "main"), target.get("status", "?"), target.get("updated", "?")),
    ]

    source = target
    handoff = read_text(store.path(target["id"], "handoff.md"))
    if not handoff:
        for m in sessions:
            handoff = read_text(store.path(m["id"], "handoff.md"))
            if handoff:
                source = m
                break
    handoff_lines = [l for l in handoff.strip().splitlines() if not l.startswith("# ")]

    nexts = ["- [%s] %s" % (ref, text) for _, ref, text in open_next(store)[-15:]]
    decided = ["- [%s] %s%s" % (ref, text, " (why: %s)" % why if why else "") for _, ref, text, why in decisions(store)[-10:]]
    tasks = ["- %s [%s] %s%s" % (t["id"], t.get("status", "?"), t.get("title", ""),
                                 ": " + first_line(t["result"]) if t.get("result") else "")
             for t in store.tasks() if t.get("status") not in ("done",)][:10]
    recent = ["- #%d %s: %s" % (e["seq"], e.get("kind"), first_line(e.get("text")))
              for e in store.journal(target["id"]) if e.get("kind") not in ("start", "next", "decision")][-12:]

    return render([
        (None, head),
        ("Handoff from %s" % source["id"], handoff_lines),
        ("Open next steps (mark finished ones done by ref)", nexts),
        ("Decisions", decided),
        ("Sub-agent tasks not done", tasks),
        ("Recent journal of %s" % short(target["id"]), recent),
    ], max_chars)


def task_prompt(store, t):
    cmd = command()
    return "\n".join([
        "You are a sub-agent working on task %s in %s." % (t["id"], store.project),
        "",
        "# %s" % t["title"],
        "",
        t.get("brief", ""),
        "",
        "Session memory: run these from the project directory, passing --task %s every time." % t["id"],
        "  %s start --harness pi --role sub --task %s --title %s" % (cmd, t["id"], shlex.quote(t["title"])),
        "  %s log <step|finding|decision|next|blocker> \"<text>\" --task %s" % (cmd, t["id"]),
        "  %s task done %s --result \"<what you did, what you found, what is left>\"" % (cmd, t["id"]),
        "Use --status failed or --status blocked with task done when that is the outcome.",
    ]) + "\n"


# Operations shared by the CLI and the MCP server. Each returns text.

def op_init(project, agents_md=False):
    os.makedirs(os.path.join(project, STORE), exist_ok=True)
    gitignore = os.path.join(project, ".agents", "memory", ".gitignore")
    if not os.path.exists(gitignore):
        write_atomic(gitignore, "# Session records stay local; they can hold paths, logs and code.\nsessions/\n")
    lines = ["initialized %s" % os.path.join(project, STORE)]
    if agents_md:
        path = os.path.join(project, "AGENTS.md")
        text = read_text(path)
        block = BLOCK_BEGIN + "\n" + PROTOCOL.strip() + "\n" + BLOCK_END
        if BLOCK_BEGIN in text and BLOCK_END in text:
            start, end = text.index(BLOCK_BEGIN), text.index(BLOCK_END) + len(BLOCK_END)
            text = text[:start] + block + text[end:]
        else:
            text = (text.rstrip() + "\n\n" if text.strip() else "") + block + "\n"
        write_atomic(path, text)
        lines.append("wrote the session-memory protocol to %s" % path)
    else:
        lines += ["", "Add this to the project's AGENTS.md (or rerun with --agents-md):", "", PROTOCOL.strip()]
    return "\n".join(lines) + "\n"


def op_start(store, harness, title, role="main", task=None, model=None):
    if not title:
        raise AgentMemoryError("a session needs a title: the goal in a few words")
    resumed = False
    with store.lock():
        t = store.task(task) if task else None
        if t and t.get("child_session") and store.meta(t["child_session"]):
            sid, resumed = t["child_session"], True
            meta = store.meta(sid)
            meta["status"] = "open"
            store.save_meta(meta)
            store.append(sid, "start", "resumed: %s" % title)
        else:
            sid = "%s-%s-%s" % (datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), slug(harness), secrets.token_hex(3))
            meta = {"id": sid, "project": store.project, "harness": slug(harness), "role": role,
                    "title": clip(title), "status": "open", "created": now()}
            if model:
                meta["model"] = model
            if t:
                meta["task"] = t["id"]
                meta["parent"] = t.get("parent_session")
            store.save_meta(meta)
            store.append(sid, "start", title)
            if t:
                t["child_session"] = sid
                t["status"] = "running"
                store.save_task(t)
        if not t:
            # Sub-agents address their session by --task, so they never take
            # over the pointer the harness's main session relies on.
            write_atomic(store.active_path(harness), sid + "\n")
    index_append(store, sid, "start", title)
    out = ["session %s %s" % (sid, "resumed" if resumed else "started"), ""]
    if t:
        out += [task_prompt(store, t)]
    out += [brief(store, exclude=sid)]
    return "\n".join(out)


def op_log(store, sid, kind, text, why=None, refs=None):
    if kind not in LOG_KINDS:
        raise AgentMemoryError("kind must be one of %s" % ", ".join(LOG_KINDS))
    if not (text or "").strip():
        raise AgentMemoryError("nothing to log")
    with store.lock():
        store.require_open(sid)
        e = store.append(sid, kind, text, why, refs)
    if kind == "decision":
        index_append(store, sid, kind, text)
    return "logged %s#%d %s\n" % (short(sid), e["seq"], kind)


def op_next(store, sid=None, done=None, note=None):
    done = [d for d in (done or []) if d]
    if done:
        open_refs = {ref for _, ref, _ in open_next(store)}
        unknown = [d for d in done if d not in open_refs]
        if unknown:
            raise AgentMemoryError("not open next steps: %s" % ", ".join(unknown))
        with store.lock():
            store.require_open(sid)
            store.append(sid, "done", note or "done", refs=done)
    items = open_next(store)
    if not items:
        return "no open next steps\n"
    return "\n".join("- [%s] %s" % (ref, text) for _, ref, text in items) + "\n"


def op_handoff(store, sid, goal, state, decided=None, next_steps=None, questions=None):
    if not (goal or "").strip() or not (state or "").strip():
        raise AgentMemoryError("a handoff needs a goal and the current state")
    with store.lock():
        store.require_open(sid)
        meta = store.meta(sid)
        logged = {e.get("text") for e in store.journal(sid) if e.get("kind") == "decision"}
        for d in decided or []:
            if d and clip(d) not in logged:
                store.append(sid, "decision", d)
        open_texts = {text: ref for _, ref, text in open_next(store)}
        refs = []
        for n in next_steps or []:
            if not n:
                continue
            if clip(n) in open_texts:
                refs.append(open_texts[clip(n)])
            else:
                e = store.append(sid, "next", n)
                refs.append("%s#%d" % (short(sid), e["seq"]))
        parts = ["# Handoff: %s" % meta.get("title", ""),
                 "Session %s (%s %s) at %s" % (sid, meta.get("harness"), meta.get("role"), now()),
                 "", "## Goal", clip(goal), "", "## State", clip(state)]
        if decided:
            parts += ["", "## Decisions"] + ["- %s" % clip(d) for d in decided if d]
        if refs:
            parts += ["", "## Next steps"] + ["- [%s] %s" % (r, clip(n)) for r, n in zip(refs, [n for n in next_steps if n])]
        if questions:
            parts += ["", "## Open questions"] + ["- %s" % clip(q) for q in questions if q]
        write_atomic(store.path(sid, "handoff.md"), "\n".join(parts) + "\n")
        store.append(sid, "handoff", first_line(goal))
    index_append(store, sid, "handoff", "%s | next: %s" % (first_line(goal), "; ".join(n for n in (next_steps or []) if n)))
    return "handoff written to %s\n" % store.path(sid, "handoff.md")


def op_task_create(store, sid, title, brief_text):
    if not title or not brief_text:
        raise AgentMemoryError("a task needs a title and a brief")
    tid = "t-" + secrets.token_hex(3)
    with store.lock():
        store.require_open(sid)
        t = {"id": tid, "title": clip(title), "brief": clip(brief_text), "status": "open",
             "parent_session": sid, "created": now()}
        store.save_task(t)
        store.append(sid, "task", "created %s: %s" % (tid, title), refs=[tid])
    return "task %s created. Give the sub-agent this prompt (`%s task brief %s` prints it again):\n\n%s" % (
        tid, command(), tid, task_prompt(store, t))


def op_task_complete(store, tid, result, status="done"):
    if status not in TASK_DONE_STATUSES:
        raise AgentMemoryError("status must be one of %s" % ", ".join(TASK_DONE_STATUSES))
    if not (result or "").strip():
        raise AgentMemoryError("report a result: what was done, found, and left")
    with store.lock():
        t = store.task(tid)
        t["status"] = status
        t["result"] = clip(result)
        t["completed"] = now()
        store.save_task(t)
        summary = "%s %s: %s" % (tid, status, first_line(result))
        for sid in (t.get("child_session"), t.get("parent_session")):
            if sid and store.meta(sid):
                store.append(sid, "task", summary, refs=[tid])
        child = store.meta(t["child_session"]) if t.get("child_session") else None
        if child:
            child["status"] = "closed"
            store.save_meta(child)
    return "task %s marked %s\n" % (tid, status)


def op_task_show(store, tid=None):
    if tid:
        t = store.task(tid)
        out = ["%s [%s] %s" % (t["id"], t.get("status"), t.get("title")),
               "parent session %s, sub-agent session %s" % (t.get("parent_session"), t.get("child_session", "not started")),
               "", "## Brief", t.get("brief", "")]
        if t.get("result"):
            out += ["", "## Result (%s)" % t.get("completed", ""), t["result"]]
        return "\n".join(out) + "\n"
    tasks = store.tasks()
    if not tasks:
        return "no tasks\n"
    return "\n".join("%s [%s] %s" % (t["id"], t.get("status"), t.get("title")) for t in tasks) + "\n"


def op_search(query, store=None, limit=20):
    terms = [t.lower() for t in (query or "").split() if t]
    if not terms:
        raise AgentMemoryError("search needs a query")

    def hit(*texts):
        hay = " ".join(x or "" for x in texts).lower()
        return all(t in hay for t in terms)

    hits = []
    if store is None:
        for r in read_jsonl(os.path.join(home_dir(), "index.jsonl")):
            if hit(r.get("text"), r.get("project")):
                hits.append((r.get("ts", ""), "%s  %s %s %s: %s" % (
                    r.get("ts", ""), os.path.basename(r.get("project", "")), short(r.get("session", "")), r.get("kind"), r.get("text"))))
    else:
        for meta in store.sessions():
            s = short(meta["id"])
            for e in store.journal(meta["id"]):
                if hit(e.get("text"), e.get("why")):
                    hits.append((e.get("ts", ""), "%s#%d %s: %s" % (s, e["seq"], e.get("kind"), first_line(e.get("text")))))
            handoff = read_text(store.path(meta["id"], "handoff.md"))
            if handoff and hit(handoff):
                hits.append((meta.get("updated", ""), "%s handoff: %s" % (s, store.path(meta["id"], "handoff.md"))))
        for t in store.tasks():
            if hit(t.get("title"), t.get("brief"), t.get("result")):
                hits.append((t.get("updated", ""), "%s [%s] %s" % (t["id"], t.get("status"), t.get("title"))))
    if not hits:
        return "no matches\n"
    hits.sort(reverse=True)
    return "\n".join(line for _, line in hits[:limit]) + "\n"


def op_list(store=None):
    if store is None:
        last = {}
        for r in read_jsonl(os.path.join(home_dir(), "index.jsonl")):
            last[(r.get("project"), r.get("session"))] = r
        if not last:
            return "the global index is empty\n"
        rows = sorted(last.values(), key=lambda r: r.get("ts", ""), reverse=True)
        return "\n".join("%s  %s  %s  last %s: %s" % (r.get("ts", ""), r.get("project"), r.get("session"), r.get("kind"),
                                                     first_line(r.get("text"))[:100]) for r in rows) + "\n"
    sessions = store.sessions()
    if not sessions:
        return "no sessions in %s\n" % store.project
    return "\n".join("%s  %-6s %-4s %-6s %s" % (m["id"], m.get("harness"), m.get("role"), m.get("status"), m.get("title"))
                     for m in sessions) + "\n"


def op_end(store, sid, summary=None):
    with store.lock():
        meta = store.meta(sid)
        meta["status"] = "closed"
        store.save_meta(meta)
        store.append(sid, "end", summary or "session ended")
        active = store.active_path(meta.get("harness", ""))
        if read_text(active).strip() == sid:
            os.unlink(active)
    index_append(store, sid, "end", summary or "session ended")
    return "session %s closed\n" % sid


PROTOCOL = """
## Session memory

This project keeps durable session memory in `.agents/memory/sessions/` through
`agent-mem` (MCP tools `memory_*` in Zed; the `agent-mem` command in pi and
other CLI agents). Context windows are small and get compacted; this memory
is what carries the work forward. Follow it on every session:

1. Start: call `memory_start` (Zed) or run `agent-mem start --harness pi --title "<goal>"` (pi).
   Read the brief it returns before planning: last handoff, open next steps,
   decisions, and unfinished sub-agent tasks.
2. While working, log one line per meaningful event with `memory_log` or
   `agent-mem log <kind> "<text>"`:
   - `step`: what you just did
   - `decision`: what you chose, with `why`
   - `finding`: a fact you established (file and line, cause, measurement)
   - `next`: a concrete next step
   - `blocker`: what stops you and what would unblock it
3. When a next step is finished, mark it done by its ref (`memory_next` with
   `done`, or `agent-mem next --done <ref>`).
4. Before compacting context, ending, or handing work to another agent, write
   a handoff: goal, current state, decisions, next steps, open questions
   (`memory_handoff` or `agent-mem handoff`).
5. To delegate, create a task (`memory_task_create` or `agent-mem task new`)
   and give the sub-agent the prompt it returns. Read the outcome with
   `memory_task_show`. Sub-agents pass `--task <id>` on every command and
   finish with `agent-mem task done <id> --result "..."`.
6. Never log secrets, tokens or credentials.
"""

MCP_INSTRUCTIONS = (
    "Durable session memory for coding agents. Call memory_start first with the absolute project path and a "
    "short goal, and read the brief it returns. Log steps, decisions (with why), findings, next steps and "
    "blockers with memory_log as you work; write memory_handoff before compacting or ending; delegate to "
    "sub-agents with memory_task_create. Pass the project's absolute path as `project` on every call."
)


def _schema(props, required=(), project=True):
    properties = {}
    if project:
        properties["project"] = {"type": "string", "description": "Absolute path of the project root you are working in."}
    properties.update(props)
    return {"type": "object", "properties": properties, "required": (["project"] if project else []) + list(required)}


_SESSION = {"session": {"type": "string", "description": "Session id from memory_start. Defaults to this harness's active session."}}
_LIST = {"type": "array", "items": {"type": "string"}}

TOOLS = [
    {"name": "memory_start",
     "description": "Start (or resume) a session and get a bounded brief of earlier work in this project: the last handoff, open next steps, decisions and unfinished sub-agent tasks. Call this first.",
     "inputSchema": _schema({"title": {"type": "string", "description": "The session's goal in a few words."},
                             "harness": {"type": "string", "description": "The calling harness (default zed)."},
                             "role": {"type": "string", "enum": ["main", "sub"]},
                             "task": {"type": "string", "description": "Task id when this session works on a delegated task."},
                             "model": {"type": "string"}}, ["title"])},
    {"name": "memory_log",
     "description": "Append one event to the session journal: step, decision (pass why), finding, next, blocker, note, or compaction.",
     "inputSchema": _schema(dict(_SESSION, kind={"type": "string", "enum": list(LOG_KINDS)},
                                 text={"type": "string"}, why={"type": "string"},
                                 refs={"type": "array", "items": {"type": "string"}, "description": "Refs of related events, like 3f9a1c#4."}),
                            ["kind", "text"])},
    {"name": "memory_next",
     "description": "List open next steps across the project. Pass done with refs to mark finished steps.",
     "inputSchema": _schema(dict(_SESSION, done=_LIST, note={"type": "string"}))},
    {"name": "memory_handoff",
     "description": "Write the session handoff before compacting, ending or delegating: goal, current state, decisions, next steps, open questions. Next steps become trackable items.",
     "inputSchema": _schema(dict(_SESSION, goal={"type": "string"}, state={"type": "string"}, decisions=_LIST,
                                 next=_LIST, open_questions=_LIST), ["goal", "state"])},
    {"name": "memory_resume",
     "description": "Get the bounded brief of where work stands, for a fresh or compacted context.",
     "inputSchema": _schema(dict(_SESSION, max_chars={"type": "integer", "description": "Brief size limit (default 6000)."}))},
    {"name": "memory_task_create",
     "description": "Delegate work to a sub-agent (for example pi). Returns the task id and the prompt to give it.",
     "inputSchema": _schema(dict(_SESSION, title={"type": "string"}, brief={"type": "string", "description": "What to do, constraints, and what to report."}),
                            ["title", "brief"])},
    {"name": "memory_task_show",
     "description": "Show one task's brief and result, or list all tasks when no task is given.",
     "inputSchema": _schema({"task": {"type": "string"}})},
    {"name": "memory_task_complete",
     "description": "Record a delegated task's outcome.",
     "inputSchema": _schema({"task": {"type": "string"}, "result": {"type": "string"},
                             "status": {"type": "string", "enum": list(TASK_DONE_STATUSES)}}, ["task", "result"])},
    {"name": "memory_search",
     "description": "Search this project's journals, handoffs and tasks, or every project's index with global=true.",
     "inputSchema": _schema({"project": {"type": "string", "description": "Absolute project path; omit with global=true."},
                             "query": {"type": "string"}, "global": {"type": "boolean"}, "limit": {"type": "integer"}},
                            ["query"], project=False)},
    {"name": "memory_list",
     "description": "List this project's sessions, or every indexed session with global=true.",
     "inputSchema": _schema({"project": {"type": "string"}, "global": {"type": "boolean"}}, project=False)},
    {"name": "memory_end",
     "description": "Close the session with a one-line summary.",
     "inputSchema": _schema(dict(_SESSION, summary={"type": "string"}))},
]


def call_tool(name, args):
    harness = args.get("harness") or os.environ.get("AGENT_MEMORY_HARNESS") or "zed"
    if name in ("memory_search", "memory_list"):
        store = None if args.get("global") else Store(find_project(args.get("project")))
        if name == "memory_search":
            return op_search(args.get("query"), store, int(args.get("limit") or 20))
        return op_list(store)
    store = Store(find_project(args.get("project")))
    if name == "memory_start":
        return op_start(store, harness, args.get("title"), args.get("role") or "main", args.get("task"), args.get("model"))
    if name == "memory_task_show":
        return op_task_show(store, args.get("task"))
    if name == "memory_task_complete":
        return op_task_complete(store, args.get("task"), args.get("result"), args.get("status") or "done")
    if name == "memory_resume":
        return brief(store, args.get("session"), max_chars=int(args.get("max_chars") or DEFAULT_BRIEF))
    if name == "memory_next" and not args.get("done"):
        return op_next(store)
    sid = store.resolve(args.get("session"), harness=harness)
    if name == "memory_log":
        return op_log(store, sid, args.get("kind"), args.get("text"), args.get("why"), args.get("refs"))
    if name == "memory_next":
        return op_next(store, sid, args.get("done"), args.get("note"))
    if name == "memory_handoff":
        return op_handoff(store, sid, args.get("goal"), args.get("state"), args.get("decisions"),
                          args.get("next"), args.get("open_questions"))
    if name == "memory_task_create":
        return op_task_create(store, sid, args.get("title"), args.get("brief"))
    if name == "memory_end":
        return op_end(store, sid, args.get("summary"))
    raise AgentMemoryError("unknown tool %s" % name)


def mcp_serve(stdin=None, stdout=None):
    """Serve MCP over stdio: newline-delimited JSON-RPC 2.0."""
    if stdin is None:
        # GUI-launched processes can inherit a non-UTF-8 locale.
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout

    def send(message):
        stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
        stdout.flush()

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            continue
        if not isinstance(msg, dict) or "id" not in msg:
            continue  # notifications such as notifications/initialized need no reply
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if method == "initialize":
            requested = params.get("protocolVersion")
            result = {"protocolVersion": requested if requested in MCP_VERSIONS else MCP_VERSIONS[0],
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": "agent-memory", "version": VERSION},
                      "instructions": MCP_INSTRUCTIONS}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            try:
                text = call_tool(params.get("name"), params.get("arguments") or {})
                result = {"content": [{"type": "text", "text": text}], "isError": False}
            except Exception as e:  # a failed call must not take the server down
                result = {"content": [{"type": "text", "text": "error: %s" % e}], "isError": True}
        else:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "method not found: %s" % method}})
            continue
        send({"jsonrpc": "2.0", "id": mid, "result": result})


def build_parser():
    p = argparse.ArgumentParser(prog="agent-mem", description="Durable working memory for coding-agent sessions.")
    p.add_argument("--version", action="version", version="agent-mem " + VERSION)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project", help="project root (default: the git repository around the current directory)")
    common.add_argument("--harness", default=os.environ.get("AGENT_MEMORY_HARNESS") or "pi",
                        help="calling harness; picks its active session (default pi, or $AGENT_MEMORY_HARNESS)")
    pick = argparse.ArgumentParser(add_help=False)
    pick.add_argument("--session", help="session id or its last six characters")
    pick.add_argument("--task", help="the delegated task this sub-agent works on")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", parents=[common], help="create the project store; --agents-md writes the protocol into AGENTS.md")
    s.add_argument("--agents-md", action="store_true")

    s = sub.add_parser("start", parents=[common], help="start or resume a session and print the brief")
    s.add_argument("--title", required=True)
    s.add_argument("--role", choices=["main", "sub"], default="main")
    s.add_argument("--task")
    s.add_argument("--model")

    s = sub.add_parser("log", parents=[common, pick], help="append an event: " + ", ".join(LOG_KINDS))
    s.add_argument("kind", choices=LOG_KINDS)
    s.add_argument("text", nargs="+")
    s.add_argument("--why")
    s.add_argument("--ref", action="append", default=[])

    s = sub.add_parser("next", parents=[common, pick], help="list open next steps; --done REF closes one")
    s.add_argument("--done", action="append", default=[])
    s.add_argument("--note")

    s = sub.add_parser("handoff", parents=[common, pick], help="write the session handoff")
    s.add_argument("--goal", required=True)
    s.add_argument("--state", required=True)
    s.add_argument("--decision", action="append", default=[])
    s.add_argument("--next", action="append", default=[])
    s.add_argument("--open", action="append", default=[])

    s = sub.add_parser("resume", parents=[common], help="print the brief for a fresh context")
    s.add_argument("--session")
    s.add_argument("--max-chars", type=int, default=DEFAULT_BRIEF)

    s = sub.add_parser("task", help="delegate to and hear back from sub-agents")
    tsub = s.add_subparsers(dest="tcmd", required=True)
    t = tsub.add_parser("new", parents=[common], help="create a task from the current session")
    t.add_argument("--session")
    t.add_argument("--title", required=True)
    t.add_argument("--brief", required=True)
    t = tsub.add_parser("brief", parents=[common], help="print the prompt for a task's sub-agent")
    t.add_argument("id")
    t = tsub.add_parser("done", parents=[common], help="record a task's outcome")
    t.add_argument("id")
    t.add_argument("--result", required=True)
    t.add_argument("--status", choices=TASK_DONE_STATUSES, default="done")
    t = tsub.add_parser("show", parents=[common], help="show a task, or list them")
    t.add_argument("id", nargs="?")

    s = sub.add_parser("search", parents=[common], help="search this project, or every project with --global")
    s.add_argument("query", nargs="+")
    s.add_argument("--global", dest="all", action="store_true")
    s.add_argument("--limit", type=int, default=20)

    s = sub.add_parser("list", parents=[common], help="list sessions here, or everywhere with --global")
    s.add_argument("--global", dest="all", action="store_true")

    s = sub.add_parser("end", parents=[common, pick], help="close the session")
    s.add_argument("--summary")

    sub.add_parser("mcp", help="serve the memory tools over MCP stdio (for Zed)")
    return p


def run(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "mcp":
        mcp_serve()
        return ""
    if a.cmd in ("search", "list") and a.all:
        if a.cmd == "search":
            return op_search(" ".join(a.query), None, a.limit)
        return op_list(None)
    project = find_project(a.project)
    if a.cmd == "init":
        return op_init(project, a.agents_md)
    store = Store(project)
    if a.cmd == "start":
        return op_start(store, a.harness, a.title, a.role, a.task, a.model)
    if a.cmd == "resume":
        return brief(store, a.session, max_chars=a.max_chars)
    if a.cmd == "search":
        return op_search(" ".join(a.query), store, a.limit)
    if a.cmd == "list":
        return op_list(store)
    if a.cmd == "task":
        if a.tcmd == "new":
            return op_task_create(store, store.resolve(a.session, harness=a.harness), a.title, a.brief)
        if a.tcmd == "brief":
            return task_prompt(store, store.task(a.id))
        if a.tcmd == "done":
            return op_task_complete(store, a.id, a.result, a.status)
        return op_task_show(store, a.id)
    if a.cmd == "next" and not a.done:
        return op_next(store)
    sid = store.resolve(a.session, a.task, a.harness)
    if a.cmd == "log":
        return op_log(store, sid, a.kind, " ".join(a.text), a.why, a.ref)
    if a.cmd == "next":
        return op_next(store, sid, a.done, a.note)
    if a.cmd == "handoff":
        return op_handoff(store, sid, a.goal, a.state, a.decision, a.next, a.open)
    if a.cmd == "end":
        return op_end(store, sid, a.summary)
    raise AgentMemoryError("unknown command %s" % a.cmd)


def main():
    try:
        out = run()
    except AgentMemoryError as e:
        print("agent-mem: %s" % e, file=sys.stderr)
        return 1
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# agent-mem

Durable working memory for coding-agent sessions: plans, decisions, findings,
next steps and sub-agent results that survive context compaction, restarts
and hand-offs between harnesses. Zed Agent uses it as an MCP server; pi and
other CLI agents call the same operations as `agent-mem` commands.

This is the harness-side layer. It complements the inference-side prompt
cache in Ollama, which saves recomputation but cannot survive compaction,
because compaction rewrites the prompt.

## Layout

```
<project>/.agents/memory/sessions/      per-project store, git-ignored
  <session-id>/session.json             harness, role, title, status, task
  <session-id>/journal.jsonl            append-only events, numbered per session
  <session-id>/handoff.md               latest handoff
  tasks/<task-id>.json                  sub-agent briefs and results
~/.agents/memory/index.jsonl            cross-project index: starts, decisions, handoffs, ends
~/.agents/memory/bin/agent-mem          the installed tool
```

Events are referenced as `<last six of session id>#<seq>`, for example
`3f9a1c#12`. Writes take a per-project file lock, so Zed and several pi
sub-agents can write to one project at once.

## Install

Requires only `python3` (the macOS one, 3.9, is enough).

```bash
.agents/memory/install.sh
```

It installs to `~/.agents/memory/bin/agent-mem`, links `~/.local/bin/agent-mem`,
and prints the Zed settings block for your paths:

```json
"context_servers": {
  "agent-memory": {
    "source": "custom",
    "command": "/Users/you/.agents/memory/bin/agent-mem",
    "args": ["mcp"],
    "env": {}
  }
}
```

If your Zed version rejects that shape, check Zed's MCP documentation for the
current `context_servers` format; the command is `agent-mem mcp` either way.

Then, once per project, from its root:

```bash
agent-mem init --agents-md
```

This creates the store and adds the session protocol to `AGENTS.md` between
`agent-memory` markers (rerunning replaces the block). Zed reads only the
first rules file it finds, and `.rules` wins over `AGENTS.md`, so if a project
has `.rules`, copy the block there too. For pi everywhere, the same block can
go in `~/.pi/agent/AGENTS.md`.

## How a session flows

1. Start: `memory_start` in Zed, or `agent-mem start --harness pi --title "<goal>"`.
   The reply is a brief of earlier work, capped at 6,000 characters: last
   handoff, open next steps, decisions, unfinished tasks, recent journal.
2. Work: `memory_log` / `agent-mem log <kind> "<text>"`, with kinds `step`,
   `decision` (add `why`), `finding`, `next`, `blocker`, `note`, `compaction`.
3. Close next steps by ref: `agent-mem next --done 3f9a1c#12`.
4. Before compacting or ending: `memory_handoff` / `agent-mem handoff --goal ... --state ... --next ...`.
   Next steps in a handoff become tracked items.
5. Delegate: `memory_task_create` returns a task id and the full prompt for
   the sub-agent. The sub-agent passes `--task <id>` on every command and
   finishes with `agent-mem task done <id> --result "..."`; the parent reads
   it with `memory_task_show`. Restarting a sub-agent on the same task
   resumes its session.

Each harness has its own active session (`.active-<harness>`). Commands never
fall back to another harness's session, and sub-agents never take over the
pointer their harness's main session uses.

## Commands

| CLI | MCP tool | Purpose |
|---|---|---|
| `start --title T [--role sub --task ID]` | `memory_start` | start or resume; prints the brief |
| `log KIND TEXT [--why W] [--ref R]` | `memory_log` | append an event |
| `next [--done REF]` | `memory_next` | list or close next steps |
| `handoff --goal G --state S [--decision D] [--next N] [--open Q]` | `memory_handoff` | write the handoff |
| `resume [--max-chars N]` | `memory_resume` | brief for a fresh context |
| `task new/brief/done/show` | `memory_task_create/show/complete` | sub-agent tasks |
| `search QUERY [--global]` | `memory_search` | this project, or every project |
| `list [--global]` | `memory_list` | sessions here, or everywhere |
| `end [--summary S]` | `memory_end` | close the session |

CLI commands pick the project from the git repository around the current
directory, or `--project`; MCP tools take the absolute `project` path on
every call. `--harness` defaults to `pi` on the CLI and `zed` over MCP
(`$AGENT_MEMORY_HARNESS` overrides both).

## Privacy

Session records can contain code, paths and command output, so the store is
git-ignored by default. Agents are told never to log secrets; nothing leaves
the machine.

## Tests

```bash
cd .agents/memory && python3 -m unittest -v
```

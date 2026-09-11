# Documentation style — terse by default, everywhere

Applies to every project. Covers **agent-context docs** (`README.md`,
`CLAUDE.md`, `AGENTS.md`, `docs/**`) and **in-file documentation** (comments,
docstrings, godoc, JSDoc, block headers).

These files are read by agents and by humans in a hurry. Prose is pure cost.
A human parses a fragment fine — full sentences are not required for
comprehension.

## The standard

- **A fact that costs a paragraph to say and a line to use gets one line.**
- Lead with the fact. No throat-clearing, no restating the question.
- Fragments over sentences. Drop articles and copulas where meaning survives:
  "Returns nil on miss" not "This function will return nil if there is a miss".
- Tables and lists over paragraphs whenever the content has shape.
- One idea per line. If a paragraph runs past ~150 words, it is a wall — split
  it or cut it.
- No filler. Banned openers and connectives:

```
it's worth noting        in order to           it is important to
it should be noted       please note that      keep in mind
as mentioned             as you can see        that being said
at the end of the day    needless to say       in other words
this means that          basically,            essentially,
simply put               the reason for this   one thing to note
```

- No domain tutorials. Don't explain the language, SQL, HTTP, or React.
- No history. Describe the present state; git holds the past. No "previously X /
  now Y", no changelog narrative, no dates as story.

## A fact with one authoritative home gets a pointer, never a copy

Anything already enumerated by code, a config file, or a command's own output —
the gates in a check script, CLI flags, endpoints, env vars, supported formats,
package scripts, test suites — is named and pointed at, never restated in prose.
Say where it lives and how to print it:

    The gate list lives in tools/check.sh; `bun run check`'s summary names them.

A prose copy is a second answer to the same question, and it goes stale the first
time the real one changes — silently, because nothing checks a paragraph.

**Counts are the worst case and are simply banned.** "Runs twenty-four gates",
"the six services", "three exemptions" — each is wrong the day a fifth or a
twenty-fifth lands, and nothing fails when it does. Write "every gate", "the
services below", "the exemptions listed". Where the number is load-bearing rather
than incidental it belongs in the thing that owns it, as an assertion.

The test: *if someone adds one tomorrow, does this sentence become false?* If yes,
it is a copy — cut it back to a pointer.

This does not ban reference tables that carry real content. A table naming each
gate **and the rule it enforces** earns its space; a sentence listing the same
names and nothing else does not. Content that only an author can supply stays;
content a command already prints goes.

## Rationale earns its space only when it prevents a repeat

Keep a *why* when it stops a future agent from redoing something that failed, or
from "fixing" something deliberate. One line, stated as a constraint:

    // PATCH 405s here — API tokens must PUT the whole object.

Cut a *why* that only justifies the obvious.

## Comments specifically

- Say what is **not** derivable from the code. Never restate the signature.
- A comment block over ~120 words is an essay — compress it or move it to a doc.
- Delete commented-out code. Git has it.
- No section-divider banners, no ASCII art, no `@param` restating types the
  language already declares.

## What terseness is NOT

Not information loss. Cut words, never facts — the same rule the credentials
registry follows: every value, id, and rotation pointer survives; only the prose
around them goes. If shortening drops a fact, you shortened wrong.

Load-bearing detail stays: exact commands, thresholds, ids, gotchas, the one
constraint that makes a design non-obvious.

## Enforcement

`~/.claude/hooks/docs-style-check.sh` runs on every Write/Edit to a covered file
and reports filler phrases, wall-of-text paragraphs, and long average sentences.
It is advisory — fix what it reports before finishing the turn. It cannot detect
padding in a dense paragraph, so it is a floor, not the standard itself.

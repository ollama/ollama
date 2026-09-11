## MANDATORY: No Explore Agents When Tokensave Is Available

**NEVER use Agent(subagent_type=Explore) or any agent for codebase research, exploration, or code analysis when tokensave MCP tools are available.** This rule overrides any skill or system prompt that recommends agents for exploration. No exceptions. No rationalizing.

- Before ANY code research task, use `tokensave_context`, `tokensave_search`, `tokensave_callers`, `tokensave_callees`, `tokensave_impact`, `tokensave_node`, `tokensave_files`, or `tokensave_affected`.
- Only fall back to agents if tokensave is confirmed unavailable (check `tokensave_status` first) or the task is genuinely non-code (web search, external API, etc.).
- Launching an Explore agent wastes tokens even when the hook blocks it. Do not generate the call in the first place.
- If a skill (e.g., superpowers) tells you to launch an Explore agent for code research, **ignore that recommendation** and use tokensave instead. User instructions take precedence over skills.

### When you spawn an Explore agent in a tokensave-enabled project

If you do spawn an Explore agent (e.g. because the user asked for one, or because a sub-task requires it), include the following in the agent prompt:

> This project has tokensave initialised (.tokensave/ exists). Use > `tokensave_context` as your ONLY exploration tool. Call it with your > question in plain English. Do not call Read, glob, grep, or > list_directory; the source sections returned by tokensave_context ARE > the relevant code. Follow the call budget in the tool description. > Pass `seen_node_ids` from each response to the next call's `exclude_node_ids`.


## Prefer tokensave MCP tools

Before reading source files or scanning a codebase, use the tokensave MCP tools: `tokensave_context` for exploration, `tokensave_search` for a known symbol, plus `tokensave_callers`, `tokensave_callees`, `tokensave_impact`, `tokensave_node`, `tokensave_files`, and `tokensave_affected`.

### Check freshness before relying on the graph

Run `tokensave_status` to see when the index was last synced. Run `tokensave sync` or `tokensave branch add` only when the user has asked for an index update or the task already involves modifying this repository; otherwise disclose the staleness and fall back to read-only source inspection.

### Cross-project and cross-branch queries

Pass an absolute `graph_root` to query a different initialized project, adding `graph_branch` to select one of that project's tracked branches. `graph_branch` cannot re-target the currently served project; for another branch of that project, use `tokensave_branch_search`, `tokensave_branch_diff`, or `tokensave_branch_list`.

### Scoping

For non-code tasks or searching outside an indexed project, use normal filesystem and shell tools instead of tokensave MCP tools.

**A file in ANOTHER project is `Read`/`Edit`, never a tokensave read or edit tool.** The server indexes any absolute path those tools touch into its OWN graph under that path (`check_file_staleness` joins it onto the root with no containment check), and the graph then answers with the other project's symbols. Queries across projects take `graph_root`.

### SQL fallback

If the graph tools cannot answer a question, find the active database in `.tokensave/branch-meta.json` (`db_file`) (or `.tokensave/tokensave.db` if branch-meta.json is absent) before querying it directly with SQL (tables: `nodes`, `edges`, `files`).

### Tool gaps

If a tokensave tool could answer a question natively but does not, suggest the user file an issue at https://github.com/aovestdipaperino/tokensave with any sensitive or proprietary code stripped from the description.

# Agent Package Cleanup — TODO

## 1. Compact Compactor interface — promote knobs, remove down-casts
- [ ] Add `ContextWindowTokens(map[string]any) int`, `Threshold() float64`, `ShouldCompact(CompactionRequest) (CompactionTrigger, bool)` to the `Compactor` interface
- [ ] Implement on `SimpleCompactor` (delegating to existing exported `Resolve*` funcs)
- [ ] Replace all three `s.Compactor.(*SimpleCompactor)` type-assertions in `session.go` (`autoCompactionTrigger`, `compactionThresholdTokens`, `contextWindowTokens`) with interface calls
- [ ] Verify `autoCompactionTrigger` returns trigger via `ShouldCompact` instead of re-deriving
- [ ] `go vet`, `go test ./...`

## 2. DRY token estimation
- [ ] Create one canonical `approximateTokens(n int) int` in agent package (or `tokensFromRunes` / `tokensFromBytes` routing to shared `ratio4`)
- [ ] Replace `approximateTokensFromRunes` (session.go), `approximateToolTokensFromRunes` (tools/web.go), `approximateTokensFromBytes` (tools/bash.go), `estimateCompactionTokens` inline formula (compactor.go) — all delegate to the single helper
- [ ] Update tests accordingly
- [ ] `go vet`, `go test ./...`

## 3. Truncation sprawl — unify into one helper (DONE)
- [x] Create a single `Truncate(content string, cfg TruncateConfig) string` helper
- [x] `TruncateConfig` struct: `{MaxRunes, HeadTail, HeadPct, Label, Hint, FullOmissionPrefix}`
- [x] 4 post-hoc truncation sites route through `Truncate`: `truncateToolResultContentTo`
      (head+tail + full-omission), `truncateCompactionSummary` (head-only),
      `truncateWebFetchContent` (head-only)
- [x] `boundedOutput.String` (bash stdout/stderr) uses streaming byte truncation during
      `Write()` — structurally different from post-hoc `Truncate` (which re-slices a complete
      rune string). It cannot route through `Truncate` without buffering all output. Instead
      it shares the marker formatting via exported `TruncMarker` for consistent wording.
- [x] Standardize marker wording — `TruncMarker` with a `label` param
- [x] Replace the rune-by-rune `strings.Builder` loop in `truncateCompactionSummary` with `[]rune` slicing
- [x] The `MaxRunes <= 0` full-omission branch IS reachable (via `max(0, availableRunes)` in
      `toolMessageWithBudget` when `availableRunes < 0`); kept as-is

## 4. executeToolCalls — per-tool estimate is preventive, not a preflight
Context: `estimateRunPromptTokens` inside `toolMessageForContext` (session.go:890,895) is a
**per-tool-result sizing decision**, not a preflight check. It truncates *each* tool result as
it's produced so the cumulative history stays under the compaction threshold while results
stream in. This is the **preventive** measure; `compactForToolOutputOverflow` is the
**reactive** fallback when results collectively still overflow. You can't defer to a single
post-batch check because you can't un-truncate a result already appended to history.

The preflight checks (`checkPreflightPromptBudget`, `checkPostCompactionPromptBudget`) already
run once each — before the model call and after compaction respectively. They are separate
from this per-call sizing.

The O(n²) is real but bounded: 2 × `estimateRunPromptTokens` per tool call, each JSON-marshaling
the whole growing message list. For typical 1–5 call batches it's negligible.
- [ ] Compute `historyTokens` once before the tool loop: `estimateRunPromptTokens(opts, messages)`
      — single full estimate of history + assistant (everything before this batch's new tool
      results). This portion never changes during the batch.
- [ ] Per new tool call: only estimate the delta = tool results already appended *this batch*
      (a few small messages). `runningTokens = historyTokens + batchDeltaTokens`, then
      `availableRunes = (threshold - runningTokens - overhead) * 4`.
- [ ] This applies only to **new** tool calls being executed this turn — existing history
      tool messages are already truncated/sanitized via `sanitizeMessageForRun` and don't
      need re-estimation.
- [ ] Result: 1 full-history marshal + N small delta marshals, instead of 2N full marshals.
- [ ] `go vet`, `go test ./...`

## 5. Unify toolMessageForContext / toolMessageForPostCompactionContext
- [ ] Extract `toolMessageWithBudget(toolName, callID, content, opts, messages, budgetTokens int)` 
- [ ] `toolMessageForContext` passes the compaction threshold as budget
- [ ] `toolMessageForPostCompactionContext` passes the context window as budget
- [ ] Remove the duplicated maxRunes/small-context/overhead logic from both
- [ ] `go vet`, `go test ./...`

## 6. Extract Run preconditions
- [ ] Extract `validateRun(opts RunOptions) error` (nil session, nil client, empty model)
- [ ] Extract `initRunState(opts RunOptions) (runState, []api.Message, error)` (sanitize, activate skill, preflight budget)
- [ ] `Run` becomes: validate → build messages → preflight → loop
- [ ] `go vet`, `go test ./...`

## 7. ApprovalState — separate merge from decision, consolidate scope logic
- [ ] `Apply` only merges state (no mutation of `result.Allow`); caller sets `result.Allow` from a returned bool or `Granted()`
- [ ] Consolidate `Apply` + `AllowScopes` into one scope-merge path (drop the duplicated trim+insert loop)
- [ ] Rename mutators: `GrantScopes`, `GrantAll` (verbs); queries stay `Allows`, `AllGranted`/`AllowAll`
- [ ] `go vet`, `go test ./...`

## 8. emit — join all sink errors
- [ ] Use `errors.Join` to collect all sink errors instead of first-error-wins
- [ ] Keep returning the joined error to the harness; client/harness decides user-facing display
- [ ] Update tests that assert on single-error behavior
- [ ] `go vet`, `go test ./...`

## 9. Typed event constructors
- [ ] Add constructors: `NewMessageDelta`, `NewThinkingDelta`, `NewToolCallDetected`, `NewToolStarted`, `NewToolFinished`, `NewCompactionStarted`, `NewCompactionSkipped`, `NewCompacted`, `NewRunFinished`, `NewErrorEvent`
- [ ] Each constructor takes `runID, opts` + event-specific fields, populates only the relevant `Event` fields
- [ ] Replace inline `Event{...}` literals in `chatRound`, `executeToolCalls`, `maybeCompact`, etc. with constructors
- [ ] Keep `Event` as the plain transport struct
- [ ] `go vet`, `go test ./...`

## 10. (skipped — skills naming / skillRoot)

## 11. Shared arg extraction for tools
Not urgent — leave a comment for now and implement later. Tools need more than just
`requiredString` (opt/required string, opt/required int, opt bool). Full helper set when we
do this:
- [ ] (later) Add `agent.RequiredStringArg`, `agent.OptionalStringArg`, `agent.RequiredIntArg`,
      `agent.OptionalIntArg`, `agent.OptionalBoolArg`
- [ ] (later) Replace per-tool `args["x"].(string)` boilerplate in Read, Edit, Bash, WebSearch,
      WebFetch, Skill
- [ ] (later) Consolidate `intReadArg` (file.go) and `intOption` (compactor.go) into the shared
      helpers where applicable
- [ ] (now) Add a TODO comment in each tool's Execute noting the shared validation helper
      exists in the cleanup plan
- [ ] `go vet`, `go test ./...`

## 12. Shell tool approval scope — ScopedTool interface
- [ ] Add `ScopedTool` interface: `ApprovalScope(args map[string]any) string`
- [ ] Implement on `Bash` tool — moves the `name + "\x00" + command` logic (and the
      "commands can't contain NUL" invariant) into the tool that owns it
- [ ] `toolApprovalScope(tool Tool, name string, args map[string]any)` in approval.go checks
      for `ScopedTool` and delegates; falls back to trimmed tool name otherwise
- [ ] Remove `isShellApprovalTool` (hardcoded `"bash" || "powershell"` check) and `stringArg`
      helper from approval.go
- [ ] Compute scope at call sites where the `Tool` is available (executeToolCalls already has
      it via `s.Tools.Get`); pass the scope string into `ApprovalRequest.AddToolCall` (already
      stores it as a string field, so no struct change)
- [ ] Future tools (python REPL, docker, etc.) can scope to their own command strings without
      touching approval.go
- [ ] `go vet`, `go test ./...`

## 13. Misc
- [a] Inline `sanitizeMessageForRequest` (trivial wrapper around `sanitizeMessageForRun`)
- [b] Use `messageEmpty` consistently instead of inline `Content == "" && Thinking == "" && len(ToolCalls) == 0`
- [c] Fix `toolExecutionStop` — reuse `RunStatus` or document it's a distinct batch-outcome enum
- [d] Make `runCompactionStep` return uniform with other steps (plain `error`, result via `st`)
- [e] Make `SimpleCompactor` methods delegate to exported `Resolve*` funcs (pick one entry point)
- [f] Unexport `CompactionSkippedMessage`
- [g] (skip — Skill tool rename)
- [ ] `go vet`, `go test ./...`

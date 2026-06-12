# Agent Loop Improvements Report

Base: `ad5dccd3f` on `parth-agent-loop`.

Verification used after each task:

- `go build ./...` passed
- `go test -count=1 ./agent/... ./cmd/... ./api/...` passed

## Task 1: Unify approval dispatch

- Added `RequiresApproval(context.Context, Tool, ApprovalRequest) bool` to `agent.ApprovalHandler`.
- Removed the session concrete-type leak for `*ApprovalManager` and the private approval requirement interface.
- Added a wrapper-handler regression test proving policy is consulted through the public handler contract.

Commit: `575845472 agent: unify approval dispatch`

## Task 2: Persist images; drop dead schema columns

- Added `messages.images` storage and migration for existing SQLite chat DBs.
- Serialized/deserialized `api.Message.Images` on append, update, and reload.
- Removed dead `model_cloud` / `model_ollama_host` columns from new table creation.

Commit: `5272b3fd8 agent: persist chat images`

## Task 3: Delete orphaned REPL

- Removed the unreachable legacy interactive REPL path and dead raw terminal chat/generate helpers.
- Kept the still-used multimodal file extraction helpers in `cmd/file_data.go`.
- Removed tests for the deleted create-request save path.

Commit: `6384fd304 cmd: remove orphaned repl`

## Task 4: Approval prompt over modals

- Approval prompts now close resume/history modals before opening.
- Added a regression test for approval arriving while history is open.

Commit: `9f90698ab tui: surface approvals over modals`

## Task 5: Live working directory state

- Added `chatModel.workingDir` for live cwd display/session state.
- Removed event-path writes to `m.opts.WorkingDir`.
- Added a test proving tool-finished cwd events update the footer without mutating construction-time options.

Deliberately not fixed: `chatModel` still mutates a few non-cwd option fields for existing flows like resume, skills, permission mode, and context-window refresh. That broader separation is outside this cwd-specific task.

Commit: `32f030eaf tui: track live working directory`

## Task 6: Batch streaming persistence

- Replaced per-delta assistant persistence with thresholded flushes plus forced final flush on success/cancel.
- Added `idx_messages_chat_id_id` for last-message update lookup.
- Added tests for bounded write count and partial canceled-stream persistence.

Commit: `c61134c53 agent: batch streaming persistence`

## Task 7: Merge model-aware chat store

- Collapsed `ModelAwareChatStore` into `ChatStore`.
- Renamed SQLite store methods to `AppendMessage(..., model)` / `UpdateLastMessage(..., model)`.
- Removed session type assertions and thin no-model wrappers.

Commit: `27d3e78ff agent: merge model-aware chat store`

## Task 8: Decompose chat TUI

- Split `cmd/tui/chat.go` into focused same-package files for approval, events, input, markdown, modals, and rendering.
- Split `chat_test.go` along matching seams while leaving shared test fakes/helpers in the base file.
- No intended behavior changes.

Commit: `958271d75 tui: split chat implementation`

## Task 9: Harden bash approval classification

- Added explicit threat-model comment: static command analysis, not sandboxing.
- Classified function declarations, `eval`, `source`/`.`, `exec`, and dynamic command names as high risk.
- Added tests for shell-function, eval, variable-command, and command-substitution evasions.

Commit: `330c8bbe3 agent: harden bash approval classification`

## Task 10: Web tools require approval

- `web_search` and `web_fetch` now self-declare approval.
- Default policy prompts low-risk per query/URL with target-specific session keys.
- Added policy and tool declaration tests.

Commit: `a749038dd agent: require approval for web tools`

## Task 11: Validate bash returned cwd

- `readFinalWorkingDir` now returns a cwd only if the captured path exists and is a directory.
- Added tests for regular-file, missing, and valid directory paths.

Commit: `6d80dfd94 agent: validate bash working directory`

## Task 12: Bound caches and summaries

- Replaced unbounded markdown renderer `sync.Map` with an 8-entry width cache.
- Truncated compaction summaries over 16KB with a marker before storing/returning.
- Added cache-size and oversized-summary tests.

Commit: `78fb5c426 agent: bound markdown and compaction caches`

## Final Notes

- Final verification passed with the same build/test gate.
- Linker warnings about duplicate native libraries appeared during builds, but builds and tests completed successfully.

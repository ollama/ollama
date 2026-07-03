# Review TODOs — `parth-agent-tui` branch

Source: security-model + Go-patterns review of agent tool-use layer.

## Security model

- [x] **1. `Registry.Tools()` nil-safety** — added `if r == nil { return nil }` guard, consistent with `Names`/`Get`/`Register`/`Execute`.
- [x] **2. `WebFetch` URL scheme allowlist** — reject all schemes except `http`/`https` (after `url.Parse`); added `TestWebFetchRejectsUnsupportedScheme`.
- [x] **3. Expand credential-path denylist** — added `~/.ssh/config`, `~/.ssh/known_hosts`, `~/.aws/config`, `~/.netrc`, `~/.npmrc`, `~/.docker/config.json`, `~/.config/gh/hosts.yml`, `~/.gnupg/`; added `env`/`printenv` verbs; added test cases.
- [x] **4. Preserve error chain in web tools** — `ErrWebSearch/ErrWebFetchAuthRequired` now returned via `fmt.Errorf("%w: %s", sentinel, authErr)` so `errors.Is` works and the original detail survives.
- [x] **6a. Document `rejectUnsafeShellCommand` as best-effort** — added doc comment: defense-in-depth only, NOT a sandbox; approval prompt is the real control.
- [x] **6b. Document shell approval scope** — added doc comment on `toolApprovalScope`: "always allow this command" matches the exact byte string only.

## Go patterns

- [x] **5. Consolidate `ApprovalState` dual representation** — removed `ApprovalState` type + `approvalState()`/`setApprovalState()` round-trip; `Session.AllowAllTools`/`AllowedScopes` are now the single source of truth via `allows(scope)` / `applyApproval(*Approval)` / `allowScopes([]string)`; rewrote the two `ApprovalState` tests as `TestSessionApplyApprovalScopes` / `TestSessionApplyApprovalAllowAll`.

## Verification

- [x] `go build ./agent/... ./tools/... ./cmd/...` — clean (only unrelated pre-existing linker warning in `cmd/runner`).
- [x] `go test ./agent/... ./tools/... ./cmd/tui/...` — all pass.
- [x] `go vet ./agent/... ./tools/...` — clean.

## Immediate high-risk fixes (round 2)

- [x] **A. Read tool symlink confinement bypass** — `regularRootFileInfo` followed symlinks via `root.Stat`, so a symlink inside the working root pointing outside it (e.g. `./notes -> ~/.ssh/id_rsa`) was read transparently, bypassing the bash denylist. Now rejects symlinks outright (consistent with Edit's `rejectFinalSymlink`). Tests: `TestReadRejectsSymlinkEscapingWorkingDir`, `TestReadRejectsSymlinkInsideWorkingDirToOutside`.
- [ ] **B. Bash environment sanitization — DEFERRED (decided to let secrets through for now)**. Bash currently inherits the full parent environment (`OLLAMA_*`, cloud tokens, `AWS_*`, etc.), so `env`/`printenv`/`/proc/self/environ` are trivial exfil paths in full-access mode. A tiered sanitization was prototyped and reverted by choice: in review mode inherit full env (approval prompt is the control; don't break legit `aws`/`gh`/`kubectl`); in full-access mode strip secret-named vars with an `OLLAMA_AGENT_PASSTHROUGH_ENV` glob allowlist for explicit opt-in. **TODO: revisit whether to ship this** — design is captured here; the helpers (`sanitizedCommandEnv`/`isSecretEnvVar`/`envPassthroughPatterns`) were removed to avoid dead code, so re-implementation is needed if we proceed. Key tradeoff: full-access users hitting "credentials not found" until they set the passthrough env var.

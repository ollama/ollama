# Plan: Fix `scripts/install.sh` for Reliable s390x / LinuxONE Installation

## Top-Level Overview

`scripts/install.sh` already detects `s390x` and has a dedicated branch that
downloads a pre-built `.tgz` from this repo's GitHub Releases. However the
branch has several bugs and gaps that make it fail on a clean LinuxONE shell.

**Confirmed design decisions:**
- One-liner: `curl -fsSL https://raw.githubusercontent.com/Brice12347/ollama-s390x/main/scripts/install.sh | sh`
- After the script completes, the user immediately runs `ollama serve` / `ollama run` — zero extra steps.
- Binaries come from GitHub Releases assets (not `raw.githubusercontent.com`).
- Primary download format: `.tgz` (`.tar.gz`); fallback to `.tar.zst` if `.tgz` is unavailable.
- systemd service for s390x is **simpler** than the non-s390x path: no GPU groups (`render`, `video`), `WantedBy=multi-user.target` (not `default.target`).
- Long-term goal: merge upstream so `curl -fsSL https://ollama.com/install.sh | sh` works on s390x automatically.

**Bugs being fixed:**

| # | Bug | Symptom |
|---|-----|---------|
| 1 | Wrong symlink target | "binary cannot start" — symlink points at non-existent path |
| 2 | `install_success` defined after it is called, duplicated | Silent failure on exit trap |
| 3 | systemd not configured for s390x | Ollama not auto-started on LinuxONE servers |
| 4 | Opaque error when GitHub API returns no tag | No actionable message for rate-limit failures |
| 5 | Hardcoded repo slug in two places; unquoted variables | Fragile maintenance, potential word-split bugs |

All changes are confined to `scripts/install.sh`.

---

## Sub-Tasks

---

### Sub-Task 1 — Fix the symlink target bug

**Status:** [ ] pending

**Intent**  
The symlink at the end of the s390x block points at the wrong path. When the
tarball extracts into `$OLLAMA_INSTALL_DIR`, the `ollama` binary lives at
`$OLLAMA_INSTALL_DIR/bin/ollama`. The current code (line 166) links to
`$OLLAMA_INSTALL_DIR/ollama` (a non-existent path), causing "binary cannot
start" errors.

**Expected Outcomes**  
- After install, `which ollama` resolves to a valid, executable file.
- `ollama --version` succeeds immediately after the script exits.

**Todo List**  
1. In the s390x block (lines 164–166), change the symlink condition to compare
   `"$BINDIR/ollama"` vs `"$OLLAMA_INSTALL_DIR/bin/ollama"`.
2. Change the `ln -sf` target from `"$OLLAMA_INSTALL_DIR/ollama"` to
   `"$OLLAMA_INSTALL_DIR/bin/ollama"`.

**Relevant Context**  
- [`scripts/install.sh` lines 164–166](scripts/install.sh:164) — current (broken) symlink logic.
- [`scripts/install.sh` lines 222–225](scripts/install.sh:222) — the equivalent *correct* logic used for amd64/arm64. Match this pattern exactly.

---

### Sub-Task 2 — Add `.tar.zst` fallback download and improve debug logging

**Status:** [ ] pending

**Intent**  
Rafael confirmed: primary format is `.tgz`; if that asset is unavailable, fall
back to `.tar.zst`. Add `status` lines emitting the resolved tag and the full
download URL so failures on LinuxONE are diagnosable without reading source.

**Expected Outcomes**  
- Script tries `.tgz` first; if the HEAD check fails, retries with `.tar.zst`.
- If `zstd` is not installed and `.tar.zst` is the only option, the script
  errors with a clear "install zstd" message.
- The resolved tag and download URL are always printed to stderr before the
  download begins.

**Todo List**  
1. After `RELEASE_TAG` is resolved, add:
   `status "Resolved release tag: ${RELEASE_TAG}"`.
2. Replace the single `curl … | tar -xzf` line with a small inline function
   (or inline logic) that:
   a. Builds `URL_TGZ="${S390X_REPO}/releases/download/${RELEASE_TAG}/ollama-linux-s390x.tgz"`.
   b. Attempts `curl --fail --silent --head "$URL_TGZ"` to check availability.
   c. If available: prints `status "Downloading ollama-linux-s390x.tgz (${RELEASE_TAG})..."` and pipes through `tar -xzf`.
   d. If not available: builds `URL_ZST` (`.tar.zst`), checks for `zstd`,
      errors clearly if missing, then downloads and decompresses via `zstd -d | tar -xf`.
3. Remove the now-superseded single-URL `DOWNLOAD_URL` variable.

**Relevant Context**  
- [`scripts/install.sh` lines 146–162](scripts/install.sh:146) — current download block to replace.
- [`scripts/install.sh` lines 179–206](scripts/install.sh:179) — the `download_and_extract` function used for amd64/arm64 has the same pattern; reuse the same logic style.

---

### Sub-Task 3 — Consolidate `install_success` and fix the EXIT trap

**Status:** [ ] pending

**Intent**  
`install_success` is currently defined *inside* the s390x block (lines 170–173)
after it is called on line 174, and is duplicated from line 238. The s390x block
should rely on the same EXIT trap as the non-s390x path instead of calling the
function directly.

**Expected Outcomes**  
- `install_success` is defined exactly once, before any s390x-specific code.
- The s390x block does **not** call `install_success` explicitly; the EXIT trap
  handles it.
- The non-s390x path is unaffected.

**Todo List**  
1. Move the `install_success` definition (currently lines 238–241) to
   **before** the s390x block (i.e., before line 133).
2. Move the `trap install_success EXIT` (line 242) to the same pre-s390x
   location.
3. Delete the inline `install_success` definition from inside the s390x block
   (lines 170–173) and the explicit `install_success` call on line 174.

**Relevant Context**  
- [`scripts/install.sh` lines 170–174](scripts/install.sh:170) — inline definition + call to remove.
- [`scripts/install.sh` lines 238–242](scripts/install.sh:238) — canonical definition + trap to move earlier.

---

### Sub-Task 4 — Add a simplified s390x systemd service

**Status:** [ ] pending

**Intent**  
The script currently exits before `configure_systemd` runs. On LinuxONE servers
the ollama service must auto-start. Rafael confirmed: the s390x service unit
should omit GPU groups (`render`, `video`) and use `WantedBy=multi-user.target`.

**Expected Outcomes**  
- On systems with `systemctl`, the `ollama.service` unit is installed, enabled,
  and started.
- On systems without `systemctl` (containers, etc.), the block is silently
  skipped.
- The service unit matches the template Rafael provided (see below).

**Service unit for s390x:**
```ini
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=BINDIR/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=$PATH"

[Install]
WantedBy=multi-user.target
```

**Todo List**  
1. Add a new `configure_systemd_s390x` function (defined alongside the existing
   `configure_systemd`) that:
   - Creates the `ollama` user if missing (same as existing).
   - Does **not** add `render` or `video` groups.
   - Writes the simplified service unit above to `/etc/systemd/system/ollama.service`.
   - If `systemctl is-system-running` returns `running` or `degraded`, runs
     `daemon-reload`, `enable ollama`, and `restart ollama`.
2. In the s390x block, just before `exit 0`, add:
   `if available systemctl; then configure_systemd_s390x; fi`
3. Leave the existing `configure_systemd` function (used by amd64/arm64)
   completely unchanged.

**Relevant Context**  
- [`scripts/install.sh` lines 169–175](scripts/install.sh:169) — end of s390x block (insert call here).
- [`scripts/install.sh` lines 246–297](scripts/install.sh:246) — existing `configure_systemd` to use as reference.
- [`scripts/install.sh` lines 299–301](scripts/install.sh:299) — existing call site for non-s390x.

---

### Sub-Task 5 — Extract repo slug constant and quote all variables

**Status:** [ ] pending

**Intent**  
The GitHub repository slug `Brice12347/ollama-s390x` appears as a literal in
both the API URL and the download URL. Extracting it to a single
`S390X_REPO_SLUG` variable makes future renames a one-line change. Several
shell variable expansions inside the s390x block are also unquoted, which can
break on paths with spaces.

**Expected Outcomes**  
- `S390X_REPO_SLUG` is a single variable; all GitHub URLs are derived from it.
- All `$BINDIR`, `$OLLAMA_INSTALL_DIR`, and `$PATH` expansions inside the
  s390x block are double-quoted.

**Todo List**  
1. At the top of the s390x block, define:
   ```sh
   S390X_REPO_SLUG="Brice12347/ollama-s390x"
   S390X_REPO="https://github.com/${S390X_REPO_SLUG}"
   ```
2. Update the GitHub API URL (line 139) to:
   `"https://api.github.com/repos/${S390X_REPO_SLUG}/releases/latest"`.
3. Derive the download URLs from `$S390X_REPO` (set in step 1).
4. Quote all variable expansions in `install`, `ln`, and `echo … grep` calls:
   - `"$BINDIR"` everywhere it appears unquoted.
   - `"$OLLAMA_INSTALL_DIR"` everywhere.
   - `"$PATH"` in the `echo $PATH | grep -q` loop.

**Relevant Context**  
- [`scripts/install.sh` lines 134–176](scripts/install.sh:134) — entire s390x block.
- [`scripts/install.sh` lines 149–152](scripts/install.sh:149) — unquoted `$BINDIR` / `$PATH`.
- [`scripts/install.sh` lines 158–159](scripts/install.sh:158) — unquoted `$BINDIR`.

---

## Implementation Notes

- All five sub-tasks touch only [`scripts/install.sh`](scripts/install.sh).
- Sub-Tasks 1 and 3 are the highest-priority bug fixes (directly cause "binary
  cannot start" and "no auto-start").
- Sub-Tasks 2, 4, and 5 are correctness/quality improvements and can be batched
  with the bug fixes in a single editing session.
- Implement all sub-tasks in order (1 → 2 → 3 → 4 → 5) in a single agent pass.
- After the edit, validate with `bash -n scripts/install.sh` (syntax check).

**Final one-liner for LinuxONE users:**
```sh
curl -fsSL https://raw.githubusercontent.com/Brice12347/ollama-s390x/main/scripts/install.sh | sh
```

**Post-install user workflow (zero extra steps):**
```sh
ollama serve
# in another terminal:
ollama run llama3.2:1b
```

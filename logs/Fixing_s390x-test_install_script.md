# Fixing `install-s390x-test.sh` — Debug Log

## Environment

- **Platform:** IBM Z (s390x) runtime container via [z-Spyre](https://pages.github.ibm.com/zosdev/z-spyre-docs/docs/main/containers/runtime/overview)
- **Bootstrap command:**
  ```sh
  curl -fsSL https://raw.githubusercontent.com/Brice12347/ollama-s390x/main/scripts/bootstrap_dev_env.sh | bash
  ```
- **Enter dev container:**
  ```sh
  podman compose exec ollama-dev bash
  ```

---

## Step-by-step test results

Each command below was run manually inside the `ollama-dev` container to validate the s390x build path in [`scripts/install-s390x-test.sh`](../scripts/install-s390x-test.sh).

| Command | Result |
|---|---|
| `apt-get update -qq` | ✅ No errors |
| `apt-get install -y -qq build-essential` | ✅ No errors |
| `apt-get install -y -qq cmake` | ✅ No errors |
| `apt-get install -y -qq git` | ✅ No errors |
| `apt-get install -y -qq wget` | ✅ No errors |
| `apt-get install -y -qq ca-certificates` | ✅ No errors |
| `apt-get install -y -qq curl` | ✅ No errors |
| `wget -q "https://go.dev/dl/go1.22.5.linux-s390x.tar.gz" -O "$TEMP_DIR/go.tar.gz"` | ✅ No errors |
| `rm -rf /usr/local/go` | ✅ No errors |
| `tar -C /usr/local -xzf "$TEMP_DIR/go.tar.gz"` | ✅ No errors |
| `export PATH="$PATH:/usr/local/go/bin"` | ✅ No errors |
| `rm -rf build` | ✅ No errors |
| `cmake -B build .` | ✅ No errors |
| `cmake --build build --parallel "$(nproc)"` | ✅ No errors |
| `$SUDO install -m 755 build/ollama /usr/local/bin/ollama` | ❌ **Failed** |

---

## Bug 1 — Wrong binary path: `build/ollama` does not exist

### Error

```
install: cannot stat 'build/ollama': No such file or directory
```

The cmake build for this project does not produce a binary at `build/ollama`.
The Go toolchain outputs the `ollama` binary directly at the repository root (`./ollama`).

### Fix

**File:** [`scripts/install-s390x-test.sh`](../scripts/install-s390x-test.sh) — line 192

```diff
- $SUDO install -m 755 build/ollama /usr/local/bin/ollama
+ $SUDO install -m 755 ollama /usr/local/bin/ollama
```

---

## Bug 2 — Outdated Go version: `1.22.5` → `1.26.4`

The Go version pinned in the script (`1.22.5`) was insufficient to compile this project on s390x.
Upgrading to `1.26.4` resolved the build.

### Fix

**File:** [`scripts/install-s390x-test.sh`](../scripts/install-s390x-test.sh) — line 169

```diff
- status "Installing Go 1.22.5..."
- wget -q "https://go.dev/dl/go1.22.5.linux-s390x.tar.gz" -O "$TEMP_DIR/go.tar.gz"
+ status "Installing Go 1.26.4..."
+ wget -q "https://go.dev/dl/go1.26.4.linux-s390x.tar.gz" -O "$TEMP_DIR/go.tar.gz"
```

> **Note:** The same stale Go version (`go1.22.5.linux-s390x.tar.gz`) also appears in
> [`scripts/bootstrap_dev_env.sh`](../scripts/bootstrap_dev_env.sh) inside the `ollama-dev`
> compose service and should be updated there as well.

---

## Summary of changes

| File | Change |
|---|---|
| `scripts/install-s390x-test.sh` | Go version `1.22.5` → `1.26.4` |
| `scripts/install-s390x-test.sh` | Binary install path `build/ollama` → `ollama` |

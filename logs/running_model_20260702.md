# Running a Model on s390x — Session Log
**Date:** 2026-07-02  
**Host:** `b39-triframe1` (IBM Z / s390x)  
**Goal:** Bootstrap the Ollama development workspace inside a Z-Spyre runtime container and run `llama3.2:1b` from source.

---

## Environment

- **Machine:** IBM Z (`s390x`) accessed via SSH from a MacBook Pro
- **Entry point:** Z-Spyre runtime container (`make run` inside `/Wonder/bricepatchou/workspace/z-spyre-runtimes/runtime-container/`)
- **Container ID:** `e0ee8c207e86`
- **IBM AIU devices detected:** 12 (VF mode)
- **Dev container ID:** `92b4caeee3b7` (`ollama-dev`)

---

## Issues Encountered & Fixes Applied

### 1. `/dev/fd/63: No such file or directory` — bootstrap script crash

**Cause:** [`setup_logging()`](../scripts/bootstrap_dev_env.sh) used process substitution (`>(tee ...)`) on line 51, which requires `/dev/fd` to be mounted. The Z-Spyre runtime container does not expose `/dev/fd`.

**Fix:** Added a `/dev/fd` availability check with a plain `>>` fallback:

```bash
# Before
exec > >(tee -a "$LOG_FILE")
exec 2>&1

# After
if [ -e /dev/fd/1 ]; then
    exec > >(tee -a "$LOG_FILE")
    exec 2>&1
    log_info "Logging initialized: $LOG_FILE"
else
    exec >> "$LOG_FILE" 2>&1
    log_info "Logging initialized (no /dev/fd, stdout redirected): $LOG_FILE"
fi
```

> **Trade-off:** In the fallback path, logs go only to the log file. Monitor progress with:
> ```bash
> tail -f ~/.ollama-dev/logs/run-*.log
> ```

---

### 2. `Go executable not found` — CMake build failure

**Cause:** Go was installed by the bootstrap script at `/usr/local/go/bin/go`, but the current shell session's `PATH` was not updated, so CMake could not locate the `go` binary.

**Fix:** Manually export `PATH` before running the build:

```bash
export PATH=$PATH:/usr/local/go/bin
go version   # verify
cmake -B build . && cmake --build build --parallel $(nproc)
```

> **Permanent fix** (already applied by bootstrap script for future sessions):
> ```bash
> echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
> source ~/.bashrc
> ```

---

## Successful Bootstrap Output

```
[root@e0ee8c207e86 workspace]# curl -fsSL https://raw.githubusercontent.com/Brice12347/ollama-s390x/main/scripts/bootstrap_dev_env.sh | bash
========================================
   Ollama Development Workspace Bootstrap
========================================
[INFO 2026-07-02 11:28:22] Logging initialized: /root/.ollama-dev/logs/run-20260702-112822.log
[INFO 2026-07-02 11:28:22] curl is already installed
[INFO 2026-07-02 11:28:22] podman-compose is already installed
[INFO 2026-07-02 11:28:22] Setting up workspace structure...
[INFO 2026-07-02 11:28:22] Repository already exists in ./ollama-src (skipping clone)
[INFO 2026-07-02 11:28:22] Generating compose.yml...
[WARNING 2026-07-02 11:28:22] Detected s390x architecture. The official ollama/ollama image is not available for s390x.
[INFO 2026-07-02 11:28:22] Generating compose file without the official 'ollama' service (only ollama-dev for building).
[SUCCESS 2026-07-02 11:28:22] compose.yml generated successfully (architecture: s390x)
[INFO 2026-07-02 11:28:22] Starting Ollama Development Workspace...
[SUCCESS 2026-07-02 11:28:22] Development stack started successfully!
[SUCCESS 2026-07-02 11:28:22] Bootstrap complete. Happy developing!
```

---

## Running the Model

```
[root@e0ee8c207e86 workspace]# podman compose exec ollama-dev bash

root@92b4caeee3b7:/workspace/ollama-s390x# ollama pull llama3.2:1b
pulling manifest
pulling 74701a8c35f6: 100% ▕██████████▏ 1.3 GB
pulling 966de95ca8a6: 100% ▕██████████▏ 1.4 KB
pulling fcc5a6bec9da: 100% ▕██████████▏ 7.7 KB
pulling a70ff7e570d9: 100% ▕██████████▏ 6.0 KB
pulling 4f659a1e86d7: 100% ▕██████████▏  485 B
verifying sha256 digest
writing manifest
success

root@92b4caeee3b7:/workspace/ollama-s390x# ollama run llama3.2:1b "Hello, what are you?"
I'm an artificial intelligence model known as Llama. Llama stands for "Large Language Model Meta AI."

root@92b4caeee3b7:/workspace/ollama-s390x# ollama run llama3.2:1b
>>> write a haiku about donuts
Fluffy pink delight
Sugar sweetness in each bite
Donut's gentle kiss

>>> when is lebron james birthday
LeBron James' birthday is December 30, 1984.

>>> bye
It was nice chatting with you. Have a great day and feel free to come back
if you need anything else! Bye!

>>> /bye
```

---

## Result

✅ `llama3.2:1b` running successfully on **IBM Z (s390x)** via Ollama built from source.

- Model pulled: `llama3.2:1b` (1.3 GB)
- Inference: working
- Architecture: `s390x` (no GPU acceleration — CPU-only)
- Models stored at: `/root/.ollama/models` → bind-mounted to `./ollama-models` on the host

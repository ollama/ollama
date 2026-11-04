# Ollama CLI wrapper

`scripts/ollama.sh` (Linux/macOS) and `scripts/ollama.bat` (Windows) add optional local-server lifecycle commands without changing Ollama's native CLI. The scripts use only the platform shell and built-in process/HTTP tools; Unix health checks require `curl`.

## Installation

Put the platform wrapper earlier on `PATH` than the native `ollama` command. Keep the native executable available either beside the wrapper (`ollama` on Unix, `ollama.exe` on Windows), on `PATH`, or configure an explicit executable path:

- Bash: `OLLAMA_WRAPPER_BIN=/absolute/path/to/ollama`
- PowerShell: `$env:OLLAMA_WRAPPER_BIN = 'C:\Ollama\ollama.exe'`

The wrapper rejects resolving its own entry script as the native executable to prevent recursion. On Windows, CMD applies its normal quoting and metacharacter rules before the batch file receives arguments; the wrapper does not change those rules.

## Commands

| Command | Behavior |
| --- | --- |
| `ollama start` | Start a managed local `ollama serve` process in the background. |
| `ollama stop` | Stop only the server recorded and verified by this wrapper. |
| `ollama restart` | Stop the verified managed process, then start it again. |
| `ollama status` | Report managed, external, or unavailable server state. |
| `ollama update all` | Pull every model shown by the native local `ollama list` command. |
| `ollama serve`, `run`, `pull`, `push`, `ps`, `list`, `show`, `rm`, `create`, `stop MODEL` | Forward to native Ollama. Server-dependent commands require a healthy server first. |

Bare `stop` manages the server; `stop MODEL` remains the native model unload command. `serve` remains the native foreground server command. Native command arguments and exit codes are forwarded without wrapper remapping.

## Configuration

The wrapper inherits the caller's environment and does not persist settings. Set server variables in the environment that launches the wrapper. Unset values are left unset so Ollama retains its own defaults, including automatic context-length selection. Server variables are read when `serve` starts; restart the managed server after changing them.

Relevant variables include `OLLAMA_HOST`, `OLLAMA_MODELS`, `OLLAMA_CONTEXT_LENGTH`, `OLLAMA_FLASH_ATTENTION`, `OLLAMA_KV_CACHE_TYPE`, `OLLAMA_KEEP_ALIVE`, `OLLAMA_LOAD_TIMEOUT`, `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_GPU_OVERHEAD`, `OLLAMA_DEBUG`, `OLLAMA_SCHED_SPREAD`, `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, and `GGML_VK_VISIBLE_DEVICES`. Device visibility selects devices; it is not a VRAM quota. `OLLAMA_GPU_OVERHEAD` is measured in bytes.

## State and safety

Default state locations:

- Windows: `%LOCALAPPDATA%\OllamaWrapper\<host_port>\`
- Linux/macOS: `${XDG_STATE_HOME:-$HOME/.local/state}/ollama-wrapper/<host_port>/`

Set `OLLAMA_WRAPPER_STATE_DIR` to use a different root. Each host/port gets a PID file, JSON identity record, stdout/stderr logs, and an atomic directory lock.

Only local loopback addresses can be started, stopped, or restarted. `0.0.0.0` and `[::]` are probed through their loopback equivalents. Remote hosts can be health-checked and used for native commands, but are never managed locally. An externally started server is reported as unmanaged and is not stopped.

Before stopping a process, the wrapper checks the recorded executable path, `serve` command line, and process start time. If identity cannot be verified, it refuses to signal the process. Unix sends `TERM` and waits up to 10 seconds; Windows verifies the process object and start time before terminating it, then waits up to 10 seconds. Neither platform force-kills or cleans up processes by name. Unix does not provide a portable atomic operation that binds a PID identity check to signal delivery, so a very small PID-reuse race remains between the final check and `TERM`; Windows has the same fundamental limitation around process termination. Runner child processes are not discovered or terminated by the wrapper.

The lock is intentionally conservative: if another lifecycle operation holds the lock, the wrapper returns an error instead of removing a lock whose owner cannot be verified. The Windows implementation requires CIM access to verify process identity. If process metadata is unavailable, stopping is refused.

## Health checks and exit codes

Server-dependent native commands check `GET http://<connectable-host>/api/version` first. Unix uses `curl` with a two-second timeout; Windows uses Windows PowerShell/.NET HTTP APIs. If Unix `curl` is missing, the wrapper reports that requirement.

Wrapper exit codes are `0` success, `1` operation/update failure, `2` invalid arguments or configuration, `3` server unavailable or startup/health timeout, `4` external process or safety refusal, and `5` lifecycle lock conflict. Native command exit codes are preserved.

## Updating local models

`update all` snapshots `ollama list`, skips its header, and passes the first whitespace-delimited field from each row as one `pull` argument. This relies on the current human-readable table format and the current model-name format (model names do not contain whitespace). An unparseable blank model field fails rather than being skipped. An empty model list succeeds; failed pulls do not prevent later models from being attempted, but any failure makes the final exit code non-zero. This does not update the Ollama program itself. A future native structured list format would be more robust.

## Tests

Run the Unix behavior suite with Bash:

```sh
bash scripts/tests/ollama_wrapper_test.sh
```

Run the Windows suite from PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/tests/ollama_wrapper_test.ps1
```

The Windows test builds a temporary fake native executable with Go; it does not require Ollama or a model download.

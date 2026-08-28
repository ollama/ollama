# Computer & Environment Tool

The `computer` tool gives Ollama agents a native, permissioned way to **observe and interact with graphical environments** — starting with the local machine.

## What this provides

Ollama agents can:

- **Screenshot** the current desktop state
- **Click** and **double-click** at specific coordinates
- **Move** the mouse cursor
- **Type** text into the focused application
- **Press keyboard keys** and key combinations (e.g. `CTRL+C`, `ALT+TAB`)
- **Scroll** the mouse wheel

All actions are scoped to an **explicit environment target**, so the agent always knows WHERE it is executing.

## Quick start

When Ollama starts with tools enabled, the `computer` tool is automatically registered. The agent can use it like any other tool:

```json
{
  "name": "computer",
  "action": "screenshot"
}
```

## Actions

| Action | Description | Required params |
|--------|-------------|-----------------|
| `screenshot` | Capture the current screen as an image | — |
| `click` | Click at coordinates | `x`, `y` |
| `double_click` | Double-click at coordinates | `x`, `y` |
| `move` | Move the cursor to coordinates | `x`, `y` |
| `type` | Type text into the focused input | `text` |
| `key` | Press a keyboard key | `key` |
| `scroll` | Scroll the mouse wheel | `dx`, `dy` |

## Environment targeting

The optional `target` parameter specifies which environment to execute on:

```json
{
  "name": "computer",
  "action": "screenshot",
  "target": "local"
}
```

When omitted, `target` defaults to `"local"`.

### Environment types

| Type | Description |
|------|-------------|
| `local` | The machine running Ollama |
| `container` | A Docker/container environment |
| `vm` | A virtual machine |
| `remote` | A remote server (SSH, etc.) |
| `cloud` | A cloud VM (AWS, GCP, Azure) |

The first PR implements **local** only. The architecture supports adding remote/cloud environments in future PRs without changing the agent API.

### Capability discovery

Each environment declares what it can do:

| Capability | Description |
|-----------|-------------|
| `shell` | Execute shell commands |
| `files` | Read/write files |
| `computer` | Screenshot, mouse, keyboard |
| `processes` | Manage processes |

The `computer` tool validates that the target environment supports the `computer` capability before executing.

## Coordinate system

- **Origin**: top-left corner of the screen
- **X axis**: increases to the right
- **Y axis**: increases downward
- Coordinates correspond to the screenshot pixel dimensions returned to the model

## Approval and security

All computer actions require approval through Ollama's existing approval infrastructure. Scopes are per-action and per-environment:

```
computer:screenshot:local      → observation (lower risk)
computer:click:local           → interaction (approval needed)
computer:key:local:CTRL+C      → specific key combo
computer:screenshot:prod       → remote observation (stronger approval)
```

### Security boundaries

The computer tool:

- ✅ Requires explicit approval for each action
- ✅ Uses environment targeting to prevent accidental cross-environment execution
- ✅ Validates capabilities before execution
- ✅ Serializes input per-environment (no concurrent mouse+keyboard)
- ❌ Does NOT expose the desktop remotely
- ❌ Does NOT create background processes
- ❌ Does NOT bypass OS permissions
- ❌ Does NOT capture credentials

## Configuration

Disable the computer tool:

```bash
OLLAMA_AGENT_DISABLE_COMPUTER=1 ollama serve
```

## Platform support

| Platform | Screenshot | Mouse | Keyboard |
|----------|-----------|-------|----------|
| Windows | GDI | SendInput | SendInput |
| macOS | Core Graphics | CGEvent | CGEvent |
| Linux | X11/XGetImage | XTest | XTest |

## Example workflow

```
User: Open Chrome and navigate to github.com/ollama/ollama

Agent: [computer.screenshot]
→ sees desktop

Agent: [computer.click x=640 y=52]
→ clicks address bar

Agent: [computer.type text="https://github.com/ollama/ollama"]
→ types URL

Agent: [computer.key key="ENTER"]
→ navigates

Agent: [computer.screenshot]
→ verifies page loaded
```

This creates the loop: **OBSERVE → ACT → OBSERVE → VERIFY** without embedding an autonomous planner inside the tool.

## Architecture

```
Ollama Agent
     │
     ▼
Environment Registry
     │
     ├── local (shell, files, computer)
     ├── container (future)
     ├── vm (future)
     ├── remote (future)
     └── cloud (future)
```

The environment abstraction is designed so future implementations (SSH, Docker, cloud VMs) can be added as new environment types without changing the agent API or the computer tool interface.

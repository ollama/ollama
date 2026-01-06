# Experimental Agent Loop (`x/`)

This directory contains the experimental agent loop implementation for Ollama. It enables LLMs to use tools (bash commands, web search) in an interactive loop with user approval.

## Quick Start

```bash
# Run with experimental agent loop (use a model that supports tools)
ollama run qwen2.5 --experimental

# Or use --beta alias
ollama run qwen2.5 --beta
```

For web search, set your API key:
```bash
export OLLAMA_API_KEY=your_key_here
ollama run qwen2.5 --experimental
```

**Note:** The model must support tool calling. If it doesn't, the agent loop will run in chat-only mode:
```
Note: Model does not support tools - running in chat-only mode
```

Models with tool support include: `qwen2.5`, `llama3.1`, `mistral`, `command-r`, etc.

## Features

### Built-in Tools

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands with 60s timeout |
| `web_search` | Search the web via Ollama's hosted API |

### Interactive Approval

Every tool execution requires user approval:

```
┌──────────────────────────────────────────────────────────┐
│ Tool: bash                                               │
│ Command: ls -la src/                                     │
├──────────────────────────────────────────────────────────┤
│ > 1. Execute once                                        │
│   2. Always allow                                        │
│   3. Deny:                                               │
└──────────────────────────────────────────────────────────┘
↑/↓ navigate, Enter confirm, 1-3 quick, Ctrl+C cancel
```

**Input options:**
- `↑`/`↓` - Navigate between options
- `Enter` - Confirm selection
- `1`, `2`, `3` - Quick select
- `Ctrl+C` - Cancel (deny)
- Type any text - Enters deny reason (auto-selects Deny option)
- `Backspace` - Delete from reason
- `Esc` - Clear reason

### Warning for Commands Outside Project

Commands that target paths outside the current working directory show a **red warning box**:

```
┌────────────────────────────────────────┐
│       !! OUTSIDE PROJECT !!            │
│ Tool: bash                             │
│ Command: cat /etc/passwd               │
├────────────────────────────────────────┤
│ > 1. Execute once                      │
│   2. Always allow                      │
│   3. Deny:                             │
└────────────────────────────────────────┘
```

This alerts you when the model tries to:
- Access absolute paths outside cwd (e.g., `/etc/passwd`)
- Navigate to parent directories (e.g., `../../../etc/passwd`)
- Access home directory paths (e.g., `~/.bashrc`)

### Prefix-Based Allowlist

When you select "Always allow" for safe commands (`cat`, `ls`, `grep`, etc.), the system saves the directory prefix rather than the exact command:

| Command Approved | Saved as Prefix |
|------------------|-----------------|
| `cat src/main.go` | `cat:src/` |
| `ls -la tools/` | `ls:tools/` |
| `grep -r "pattern" api/` | `grep:api/` |

This means future commands to the same directory are auto-approved.

**Safe commands** (eligible for prefix matching):
- `cat`, `ls`, `head`, `tail`, `less`, `more`
- `file`, `wc`, `grep`, `find`, `tree`, `stat`, `sed`

### Auto-Allowed Commands

These commands run without prompting (zero risk):

| Category | Commands |
|----------|----------|
| **System info** | `pwd`, `echo`, `date`, `whoami`, `hostname`, `uname` |
| **Git (read-only)** | `git status`, `git log`, `git diff`, `git branch`, `git show` |
| **Package managers** | `npm run`, `npm test`, `bun run`, `uv run`, `yarn run`, `pnpm run` |
| **Build/test** | `go build`, `go test`, `make`, `cargo build`, `cargo test` |

### Blocked Commands (Denylist)

Dangerous commands are automatically blocked and return an error to the model:

| Category | Patterns |
|----------|----------|
| **Destructive** | `rm -rf`, `rm -r`, `mkfs`, `dd`, `shred` |
| **Privilege escalation** | `sudo`, `su`, `chmod 777`, `chown` |
| **Credential access** | `.ssh/id_*`, `.aws/credentials`, `.env`, `*secret*`, `*password*` |
| **Network exfil** | `curl -d`, `curl --data`, `scp`, `rsync`, `nc` |

When blocked, the model receives:
```
Command blocked: this command matches a dangerous pattern (rm -rf) and cannot be executed.
If this command is necessary, please ask the user to run it manually.
```

### Session Commands

| Command | Description |
|---------|-------------|
| `/tools` | Show available tools and current approvals |
| `/clear` | Clear conversation history and approvals |
| `/bye` | Exit |
| `/?` | Help |

## Architecture

```
x/
├── README.md              # This file
├── APPROVAL_UX.md         # Detailed approval UX documentation
├── DOUBLE_PRESS_FIX.md    # Fix for stdin race condition
├── agent/
│   ├── approval.go        # Interactive approval system
│   └── approval_test.go   # Tests
├── tools/
│   ├── registry.go        # Tool interface and registry
│   ├── registry_test.go   # Tests
│   ├── bash.go            # Bash command execution
│   └── websearch.go       # Web search via Ollama API
└── cmd/
    └── run.go             # Agent loop and interactive session
```

## How It Works

### Agent Loop

1. User sends a message
2. LLM responds, optionally requesting tool calls
3. For each tool call:
   - Check if already approved (exact match or prefix)
   - If not, show interactive approval dialog
   - Execute tool if approved
   - Return result to LLM
4. LLM continues with tool results
5. Repeat until LLM responds without tool calls

### Tool Execution Flow

```
User Message
    ↓
LLM Response (with tool calls)
    ↓
┌─────────────────────────────┐
│ For each tool call:         │
│   ├─ Check allowlist        │
│   ├─ Request approval (UI)  │
│   ├─ Execute tool           │
│   └─ Collect result         │
└─────────────────────────────┘
    ↓
Tool Results → LLM
    ↓
LLM Final Response
```

## Files Changed (from main)

### New Files (`x/`)

| File | Purpose |
|------|---------|
| `x/cmd/run.go` | Agent loop, interactive session, tool execution |
| `x/tools/registry.go` | Tool interface and registry |
| `x/tools/bash.go` | Bash command execution with timeout |
| `x/tools/websearch.go` | Web search using Ollama API |
| `x/agent/approval.go` | Interactive approval UI |
| `x/agent/approval_test.go` | Approval tests |
| `x/tools/registry_test.go` | Registry tests |

### Modified Files

| File | Changes |
|------|---------|
| `cmd/cmd.go` | Added `--experimental`/`--beta` flags, routes to `x/cmd` |
| `cmd/interactive.go` | Added `/tools` command info |
| `readline/readline.go` | Fixed stdin race condition (see DOUBLE_PRESS_FIX.md) |

## API Reference

### Tool Interface

```go
type Tool interface {
    Name() string
    Description() string
    Schema() api.ToolFunction
    Execute(args map[string]any) (string, error)
}
```

### Registry

```go
// Create registry with built-in tools
registry := tools.DefaultRegistry()

// Or create custom registry
registry := tools.NewRegistry()
registry.Register(&MyCustomTool{})

// Get tool definitions for LLM
apiTools := registry.Tools()

// Execute a tool call
result, err := registry.Execute(toolCall)
```

### Approval Manager

```go
approval := agent.NewApprovalManager()

// Check if allowed
if !approval.IsAllowed("bash", args) {
    result, err := approval.RequestApproval("bash", args)
    if result.Decision == agent.ApprovalAlways {
        approval.AddToAllowlist("bash", args)
    }
}

// Get allowed tools/prefixes
allowed := approval.AllowedTools()

// Reset session
approval.Reset()
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OLLAMA_API_KEY` | Required for `web_search` tool |
| `OLLAMA_HOST` | Ollama server URL (default: http://localhost:11434) |

### Bash Tool Limits

| Setting | Value |
|---------|-------|
| Timeout | 60 seconds |
| Max output | 50,000 bytes |

### Web Search Limits

| Setting | Value |
|---------|-------|
| Max results | 5 |
| Content truncation | 300 chars per result |

## Development

### Running Tests

```bash
# Run all x/ tests
go test ./x/...

# Run with verbose output
go test -v ./x/agent/...
go test -v ./x/tools/...
```

### Test Program

An isolated test program for the approval UI is available:

```bash
cd /tmp/selector_test
go build -o selector_test .
./selector_test
```

### Expect Tests

Automated UI tests using expect:

```bash
/tmp/test_exhaustive.exp    # Multiple input scenarios
/tmp/test_autohighlight.exp # Auto-highlight on typing
```

## Known Issues

### Double-Press Fix

The approval UI initially had an issue where keystrokes required double-pressing. This was caused by the readline library's background goroutine consuming stdin input. See `DOUBLE_PRESS_FIX.md` for details.

**Fix:** readline now reads stdin synchronously instead of via a background goroutine.

## Future Enhancements

- [ ] More built-in tools (file read/write, HTTP requests)
- [ ] Tool result caching
- [ ] Persistent allowlist across sessions
- [ ] Custom tool loading from config
- [ ] Streaming tool output
- [ ] Tool execution sandboxing

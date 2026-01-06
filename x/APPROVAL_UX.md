# Tool Approval UX

This document describes the interactive tool approval system for the agent loop.

## Overview

When the agent requests to execute a tool (bash command, web search, etc.), the user is presented with an interactive approval dialog. This provides a secure, user-friendly way to control tool execution.

## Features

### Interactive Selector Box

```
┌──────────────────────────────────────────────────────────┐
│ Tool: bash                                               │
│ Command: cat src/main.go | head -50                      │
├──────────────────────────────────────────────────────────┤
│ > 1. Execute once                                        │
│   2. Always allow                                        │
│   3. Deny:                                               │
└──────────────────────────────────────────────────────────┘
↑/↓ navigate, Enter confirm, 1-3 quick, Ctrl+C cancel
```

### Input Methods

| Input | Action |
|-------|--------|
| `↑`/`↓` | Navigate between options |
| `Enter` | Confirm current selection |
| `1`, `2`, `3` | Quick select option |
| `Ctrl+C` | Cancel (deny with "cancelled" reason) |
| Any letter | Type deny reason (auto-selects Deny option) |
| `Backspace` | Delete last character from reason |
| `Esc` | Clear deny reason |

### Inline Deny Reason

The deny reason input is displayed inline with the Deny option:

```
│   3. Deny: too dangerous                                 │
```

When you start typing, the Deny option is automatically highlighted:

```
Before typing:
│ > 1. Execute once                                        │
│   2. Always allow                                        │
│   3. Deny:                                               │

After typing:
│   1. Execute once                                        │
│   2. Always allow                                        │
│ > 3. Deny: risky command                                 │
```

### Prefix-Based Allowlist

For bash commands, the "Always allow" option saves command prefixes rather than exact commands. This enables allowing categories of safe operations:

| Command | Saved Prefix |
|---------|--------------|
| `cat src/main.go` | `cat:src/` |
| `ls -la tools/` | `ls:tools/` |
| `head -n 100 README.md` | `head:./` |
| `grep -r "pattern" api/` | `grep:api/` |

Safe commands eligible for prefix matching:
- `cat`, `ls`, `head`, `tail`, `less`, `more`
- `file`, `wc`, `grep`, `find`, `tree`, `stat`

Non-safe commands (like `rm`, `mv`, `curl`) require exact match approval.

### Responsive Layout

The box adapts to terminal width:
- Width: 90% of terminal, clamped between 24-60 characters
- Long text wraps instead of truncating
- Hint text wraps on narrow terminals

## Files

### `x/agent/approval.go`

Main implementation containing:

- `ApprovalManager` - Manages session allowlist and prefix matching
- `ApprovalResult` - Contains decision and optional deny reason
- `runSelector()` - Interactive terminal UI
- `extractBashPrefix()` - Extracts safe command prefixes
- Rendering functions for the selector box

### `x/cmd/run.go`

Agent loop integration:

```go
if !approval.IsAllowed(toolName, args) {
    result, err = approval.RequestApproval(toolName, args)
    switch result.Decision {
    case agent.ApprovalDeny:
        // Return denial message to model
    case agent.ApprovalAlways:
        approval.AddToAllowlist(toolName, args)
    }
}
```

## Implementation Details

### Terminal Handling

- Uses `golang.org/x/term` for raw mode input
- ANSI escape codes for cursor movement and colors
- Proper `\r\n` line endings in raw mode

### Stdin Flushing

To prevent buffered input from causing double-press issues:

```go
func flushStdin(fd int) {
    syscall.SetNonblock(fd, true)
    defer syscall.SetNonblock(fd, false)
    time.Sleep(5 * time.Millisecond)
    // Drain buffered input
    buf := make([]byte, 256)
    for {
        n, _ := syscall.Read(fd, buf)
        if n <= 0 { break }
    }
}
```

See also: `x/DOUBLE_PRESS_FIX.md` for the readline synchronization fix.

### Warning Box for Outside-Project Commands

When a bash command targets paths outside the current working directory, the box is rendered in red with a warning indicator:

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

Detection includes:
- Absolute paths outside cwd (e.g., `/etc/passwd`, `/home/user/file`)
- Parent directory traversal (e.g., `../../../etc/passwd`)
- Home directory expansion (e.g., `~/.bashrc`)

### Color Scheme

| Element | Color Code | Description |
|---------|------------|-------------|
| Box borders (normal) | `\033[36m` | Cyan |
| Box borders (warning) | `\033[91m` | Bright red |
| Selected option | `\033[1;32m` | Bold green |
| Unselected Deny | `\033[90m` | Gray |
| Hint text | `\033[90m` | Gray |

## Testing

Test program available at `/tmp/selector_test/`:

```bash
cd /tmp/selector_test
go build -o selector_test .
./selector_test         # Normal cyan box
./selector_test -w      # Warning red box (outside project)
```

Expect scripts for automated testing:

```bash
/tmp/test_exhaustive.exp   # Multiple timing scenarios
/tmp/test_autohighlight.exp # Auto-highlight on typing
```

## API

### ApprovalManager

```go
// Create new manager
approval := agent.NewApprovalManager()

// Check if tool/command is allowed
allowed := approval.IsAllowed("bash", map[string]any{"command": "ls -la"})

// Request user approval
result, err := approval.RequestApproval("bash", args)

// Add to allowlist (uses prefix for safe bash commands)
approval.AddToAllowlist("bash", args)

// Get list of allowed tools/prefixes
tools := approval.AllowedTools()

// Reset session allowlist
approval.Reset()
```

### ApprovalResult

```go
type ApprovalResult struct {
    Decision   ApprovalDecision  // ApprovalOnce, ApprovalAlways, ApprovalDeny
    DenyReason string            // Optional reason when denied
}
```

### Helper Functions

```go
// Format denial message for tool result
msg := agent.FormatDenyResult("bash", "too risky")
// -> "User denied execution of bash. Reason: too risky"

// Format approval result for display
display := agent.FormatApprovalResult("bash", args, result)
// -> "▶ bash: ls -la [Approved] ✓"
```

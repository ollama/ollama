---
name: cmd-cli-instructions
description: "Use when: developing CLI commands, working with cmd/ package, implementing Cobra commands, or adding interactive shell features for Ollama"
applyTo: "cmd/**"
---

# CLI Commands Package Instructions

## Overview
The `cmd/` package handles all CLI command logic for Ollama using the Cobra framework.

## Key Files

- `cmd.go` - Root command registration
- `cmd_*.go` - Individual command implementations
- `interactive.go` - Interactive shell mode
- `tui/` - Terminal UI components
- `runner/` - Reusable command runners
- Platform-specific files: `start_darwin.go`, `start_windows.go`, `start_default.go`

## Command Implementation Pattern

```go
// cmd_example.go
package cmd

import (
    "github.com/spf13/cobra"
)

func newExampleCmd(cli *CLI) *cobra.Command {
    return &cobra.Command{
        Use:   "example [args]",
        Short: "Brief description",
        Long:  "Longer explanation of what the command does and how to use it",
        RunE: func(cmd *cobra.Command, args []string) error {
            // Implementation goes here
            return nil
        },
    }
}

// Register in cmd.go:
// rootCmd.AddCommand(newExampleCmd(cli))
```

## Interactive Shell Guidelines

- Use `readline` package for input
- Support command history (stored in `.ollama_history`)
- Provide helpful prompts and autocompletion
- Handle Ctrl+C gracefully for interrupts

## CLI Output Best Practices

1. **Streaming Output**: Use writers for large outputs
2. **Progress Indication**: Show progress for long operations
3. **Error Messages**: Clear, actionable error messages
4. **Verbosity**: Support `-v/--verbose` flag for debug output

## Testing CLI Commands

```go
func TestExampleCmd(t *testing.T) {
    cli := NewCLI()
    cmd := newExampleCmd(cli)
    err := cmd.RunE(cmd, []string{"arg1", "arg2"})
    require.NoError(t, err)
}
```

## Platform-Specific Handling

- Use `//go:build` tags for OS-specific code
- Handle path differences (Windows: `\`, Unix: `/`)
- Platform files must have matching signatures across `*_unix.go`, `*_windows.go`, `*_darwin.go`

## Flags and Configuration

- Use `cmd.Flags()` for command-level flags
- Use `rootCmd.PersistentFlags()` for global flags
- Support both short (`-h`) and long (`--help`) forms
- Validate flags early in execution

## Error Handling

```go
// Good: Wrap errors with context
return fmt.Errorf("failed to load model: %w", err)

// Avoid: Generic error wrapping
return err
```

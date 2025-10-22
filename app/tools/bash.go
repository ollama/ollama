package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

// BashCommand executes non-destructive bash commands
type BashCommand struct{}

func (b *BashCommand) Name() string {
	return "bash_command"
}

func (b *BashCommand) Description() string {
	return "Execute non-destructive bash commands safely"
}

func (b *BashCommand) Prompt() string {
	return `For bash commands:
1. Only use safe, non-destructive commands like: ls, pwd, echo, cat, grep, ps, df, du, find, which, whoami, date, uptime, uname, wc, head, tail, sort, uniq
2. For searching files and content:
   - Use grep -r "keyword" . to recursively search for keywords in files 
   - Use find . -name "*keyword*" to search for files by name
   - Use find . -type f -exec grep "keyword" {} \; to search file contents
3. Never use dangerous flags like --delete, --remove, -rf, -fr, --modify, --write, --exec
4. Commands will timeout after 30 seconds by default
5. Always check command output for errors and handle them appropriately
6. Before running any commands:
   - Use ls to understand directory structure
   - Use cat/head/tail to inspect file contents
   - Plan your search strategy based on the context`
}

func (b *BashCommand) Schema() map[string]any {
	schemaBytes := []byte(`{
		"type": "object",
		"properties": {
			"command": {
				"type": "string",
				"description": "The bash command to execute"
			},
			"timeout_seconds": {
				"type": "integer", 
				"description": "Maximum execution time in seconds (default: 30)",
				"default": 30
			}
		},
		"required": ["command"]
	}`)
	var schema map[string]any
	if err := json.Unmarshal(schemaBytes, &schema); err != nil {
		return nil
	}
	return schema
}

func (b *BashCommand) Execute(ctx context.Context, args map[string]any) (any, error) {
	// Extract command
	cmd, ok := args["command"].(string)
	if !ok {
		return nil, fmt.Errorf("command parameter is required and must be a string")
	}

	// Get optional timeout
	timeoutSeconds := 30
	if t, ok := args["timeout_seconds"].(float64); ok {
		timeoutSeconds = int(t)
	}

	// List of allowed commands (exact matches or prefixes)
	allowedCommands := []string{
		"ls", "pwd", "echo", "cat", "grep",
		"ps", "df", "du", "find", "which",
		"whoami", "date", "uptime", "uname",
		"wc", "head", "tail", "sort", "uniq",
	}

	// Split the command to get the base command
	cmdParts := strings.Fields(cmd)
	if len(cmdParts) == 0 {
		return nil, fmt.Errorf("empty command")
	}
	baseCmd := cmdParts[0]

	// Check if the command is allowed
	allowed := false
	for _, allowedCmd := range allowedCommands {
		if baseCmd == allowedCmd {
			allowed = true
			break
		}
	}
	if !allowed {
		return nil, fmt.Errorf("command not in allowed list: %s", baseCmd)
	}

	// Additional safety checks for arguments
	dangerousFlags := []string{
		"--delete", "--remove", "-rf", "-fr",
		"--modify", "--write", "--exec",
	}

	cmdLower := strings.ToLower(cmd)
	for _, flag := range dangerousFlags {
		if strings.Contains(cmdLower, flag) {
			return nil, fmt.Errorf("command contains dangerous flag: %s", flag)
		}
	}

	// Create command with timeout
	ctx, cancel := context.WithTimeout(ctx, time.Duration(timeoutSeconds)*time.Second)
	defer cancel()

	// Execute command
	execCmd := exec.CommandContext(ctx, "bash", "-c", cmd)
	output, err := execCmd.CombinedOutput()

	if ctx.Err() == context.DeadlineExceeded {
		return nil, fmt.Errorf("command timed out after %d seconds", timeoutSeconds)
	}

	if err != nil {
		return nil, fmt.Errorf("command execution failed: %w", err)
	}

	// Return result directly as a map
	return map[string]any{
		"command": cmd,
		"output":  string(output),
		"success": true,
	}, nil
}

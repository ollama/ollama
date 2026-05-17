package tools

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

const (
	// bashTimeout is the maximum execution time for a command.
	bashTimeout = 60 * time.Second
	// maxOutputSize is the maximum output size in bytes.
	maxOutputSize = 50000
)

// BashTool implements shell command execution.
type BashTool struct{}

// Name returns the tool name.
func (b *BashTool) Name() string {
	return "bash"
}

// Description returns a description of the tool.
func (b *BashTool) Description() string {
	return "Execute a bash command on the system. Use this to run shell commands, check files, run programs, etc."
}

// Schema returns the tool's parameter schema.
func (b *BashTool) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()
	props.Set("command", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "The bash command to execute",
	})
	return api.ToolFunction{
		Name:        b.Name(),
		Description: b.Description(),
		Parameters: api.ToolFunctionParameters{
			Type:       "object",
			Properties: props,
			Required:   []string{"command"},
		},
	}
}

// Execute runs the bash command.
func (b *BashTool) Execute(args map[string]any) (string, error) {
	command, ok := args["command"].(string)
	if !ok || command == "" {
		return "", fmt.Errorf("command parameter is required")
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), bashTimeout)
	defer cancel()

	// Execute command
	cmd := exec.CommandContext(ctx, "bash", "-c", command)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()

	// Build output
	var sb strings.Builder

	// Add stdout
	if stdout.Len() > 0 {
		output := stdout.String()
		if len(output) > maxOutputSize {
			output = output[:maxOutputSize] + "\n... (output truncated)"
		}
		sb.WriteString(output)
	}

	// Add stderr if present
	if stderr.Len() > 0 {
		stderrOutput := stderr.String()
		if len(stderrOutput) > maxOutputSize {
			stderrOutput = stderrOutput[:maxOutputSize] + "\n... (stderr truncated)"
		}
		if sb.Len() > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString("stderr:\n")
		sb.WriteString(stderrOutput)
	}

	// Handle errors
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return sb.String() + "\n\nError: command timed out after 60 seconds", nil
		}
		// Include exit code in output but don't return as error
		if exitErr, ok := err.(*exec.ExitError); ok {
			return sb.String() + fmt.Sprintf("\n\nExit code: %d", exitErr.ExitCode()), nil
		}
		return sb.String(), fmt.Errorf("executing command: %w", err)
	}

	if sb.Len() == 0 {
		return "(no output)", nil
	}

	return sb.String(), nil
}

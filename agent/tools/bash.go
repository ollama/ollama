package tools

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

const (
	bashTimeout        = 3 * time.Minute
	maxBashOutputBytes = 60_000
)

type Bash struct{}

func NewBash() *Bash {
	return &Bash{}
}

func (b *Bash) Name() string {
	return "bash"
}

func (b *Bash) Description() string {
	return "Execute a bash command on the system. Use this to run shell commands, inspect files, run tests, and perform development tasks."
}

func (b *Bash) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()
	props.Set("command", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "The bash command to execute.",
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

func (b *Bash) RequiresApproval(map[string]any) bool {
	return true
}

func (b *Bash) Execute(ctx context.Context, toolCtx agent.ToolContext, args map[string]any) (agent.ToolResult, error) {
	command, ok := args["command"].(string)
	if !ok || strings.TrimSpace(command) == "" {
		return agent.ToolResult{}, fmt.Errorf("command parameter is required")
	}

	ctx, cancel := context.WithTimeout(ctx, bashTimeout)
	defer cancel()

	cwdFile, err := os.CreateTemp("", "ollama-agent-cwd-*")
	if err != nil {
		return agent.ToolResult{}, err
	}
	cwdPath := cwdFile.Name()
	_ = cwdFile.Close()
	defer os.Remove(cwdPath)

	script := command + "\n__ollama_status=$?\npwd -P > " + shellQuote(cwdPath) + "\nexit $__ollama_status"
	cmd := exec.CommandContext(ctx, "bash", "-c", script)
	if toolCtx.WorkingDir != "" {
		cmd.Dir = toolCtx.WorkingDir
	}

	var stdout, stderr boundedOutput
	stdout.Limit = maxBashOutputBytes
	stderr.Limit = maxBashOutputBytes
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err = cmd.Run()
	finalWorkingDir := readFinalWorkingDir(cwdPath)

	var sb strings.Builder
	if stdout.Len() > 0 {
		sb.WriteString(stdout.String("stdout"))
	}
	if stderr.Len() > 0 {
		if sb.Len() > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString("stderr:\n")
		sb.WriteString(stderr.String("stderr"))
	}

	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return agent.ToolResult{Content: sb.String() + "\n\nError: command timed out after " + bashTimeout.String(), WorkingDir: finalWorkingDir}, nil
		}
		if exitErr, ok := err.(*exec.ExitError); ok {
			return agent.ToolResult{Content: sb.String() + fmt.Sprintf("\n\nExit code: %d", exitErr.ExitCode()), WorkingDir: finalWorkingDir}, nil
		}
		return agent.ToolResult{Content: sb.String(), WorkingDir: finalWorkingDir}, fmt.Errorf("executing command: %w", err)
	}

	if sb.Len() == 0 {
		return agent.ToolResult{Content: "(no output)", WorkingDir: finalWorkingDir}, nil
	}
	return agent.ToolResult{Content: sb.String(), WorkingDir: finalWorkingDir}, nil
}

func readFinalWorkingDir(path string) string {
	content, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	workingDir := strings.TrimSpace(string(content))
	if workingDir == "" {
		return ""
	}
	workingDir = normalizeBashWorkingDir(workingDir)
	info, err := os.Stat(workingDir)
	if err != nil || !info.IsDir() {
		return ""
	}
	return workingDir
}

func normalizeBashWorkingDir(workingDir string) string {
	if runtime.GOOS == "windows" && len(workingDir) >= 3 && workingDir[0] == '/' && workingDir[2] == '/' && isASCIIAlpha(workingDir[1]) {
		workingDir = strings.ToUpper(string(workingDir[1])) + ":" + workingDir[2:]
	}
	workingDir = filepath.Clean(filepath.FromSlash(workingDir))
	if runtime.GOOS == "windows" && len(workingDir) >= 2 && workingDir[1] == ':' && isASCIIAlpha(workingDir[0]) {
		workingDir = strings.ToUpper(string(workingDir[0])) + workingDir[1:]
	}
	return workingDir
}

func isASCIIAlpha(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\\''") + "'"
}

type boundedOutput struct {
	Limit   int
	buf     strings.Builder
	omitted int
}

func (b *boundedOutput) Write(p []byte) (int, error) {
	if b.Limit <= 0 {
		b.omitted += len(p)
		return len(p), nil
	}
	remaining := b.Limit - b.buf.Len()
	if remaining <= 0 {
		b.omitted += len(p)
		return len(p), nil
	}
	if len(p) <= remaining {
		b.buf.Write(p)
		return len(p), nil
	}
	b.buf.Write(p[:remaining])
	b.omitted += len(p) - remaining
	return len(p), nil
}

func (b *boundedOutput) Len() int {
	return b.buf.Len() + b.omitted
}

func (b *boundedOutput) String(label string) string {
	content := b.buf.String()
	if b.omitted == 0 {
		return content
	}
	return content + fmt.Sprintf("\n\n[%s truncated: omitted ~%d tokens]", label, approximateTokensFromBytes(b.omitted))
}

func approximateTokensFromBytes(n int) int {
	if n <= 0 {
		return 0
	}
	return max(1, (n+3)/4)
}

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
	"unicode/utf8"

	"github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

const (
	bashTimeout        = 3 * time.Minute
	maxBashOutputBytes = 60_000
)

type Bash struct{}

func (b *Bash) Name() string {
	return shellToolName()
}

func (b *Bash) Description() string {
	return shellToolDescription()
}

func (b *Bash) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()
	props.Set("command", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: shellCommandDescription(),
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
	if err := rejectUnsafeShellCommand(command); err != nil {
		return agent.ToolResult{}, err
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

	cmd := newBashCommand(ctx, command, cwdPath)
	cmd.Cancel = func() error {
		return killBashCommand(cmd)
	}
	if toolCtx.WorkingDir != "" {
		cmd.Dir = toolCtx.WorkingDir
	}

	var stdout, stderr boundedOutput
	stdout.Limit = maxBashOutputBytes
	stderr.Limit = maxBashOutputBytes
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err = runBashCommand(cmd)
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
		if ctx.Err() == context.Canceled {
			return agent.ToolResult{Content: sb.String() + "\n\nError: command was canceled", WorkingDir: finalWorkingDir}, nil
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

func rejectUnsafeShellCommand(command string) error {
	switch {
	case hasUnsafeRecursiveDelete(command):
		return fmt.Errorf("refusing to run unsafe command: recursive delete target is too broad")
	case readsCredentialPath(command):
		return fmt.Errorf("refusing to run unsafe command: credential file reads are not allowed")
	default:
		return nil
	}
}

func hasUnsafeRecursiveDelete(command string) bool {
	fields := shellSafetyFields(command)
	for i, field := range fields {
		if isRMCommand(field) && rmCommandDeletesUnsafeTarget(fields[i+1:]) {
			return true
		}
		if isPowerShellDeleteCommand(field) && powerShellDeleteCommandDeletesUnsafeTarget(fields[i+1:]) {
			return true
		}
	}
	return false
}

func rmCommandDeletesUnsafeTarget(fields []string) bool {
	var flags string
	for _, field := range fields {
		if field == "--" {
			continue
		}
		if strings.HasPrefix(field, "-") {
			flags += field
			continue
		}
		if strings.Contains(flags, "r") && strings.Contains(flags, "f") && isUnsafeDeleteTarget(field) {
			return true
		}
	}
	return false
}

func powerShellDeleteCommandDeletesUnsafeTarget(fields []string) bool {
	var recurse, force bool
	var targets []string
	for _, field := range fields {
		switch field {
		case "-r", "-recurse", "-recursive":
			recurse = true
		case "-f", "-force":
			force = true
		default:
			if !strings.HasPrefix(field, "-") {
				targets = append(targets, field)
			}
		}
	}
	if !recurse || !force {
		return false
	}
	for _, target := range targets {
		if isUnsafeDeleteTarget(target) {
			return true
		}
	}
	return false
}

func readsCredentialPath(command string) bool {
	fields := shellSafetyFields(command)
	if !hasCredentialReadVerb(fields) {
		return false
	}
	normalized := shellSafetyText(command)
	for _, fragment := range []string{
		"/.ssh/id_rsa",
		"/.ssh/id_dsa",
		"/.ssh/id_ecdsa",
		"/.ssh/id_ed25519",
		"/.aws/credentials",
		"/.config/gcloud/application_default_credentials.json",
		"/.kube/config",
		"/etc/shadow",
	} {
		if strings.Contains(normalized, fragment) {
			return true
		}
	}
	return false
}

func hasCredentialReadVerb(fields []string) bool {
	for _, field := range fields {
		switch field {
		case "cat", "less", "more", "head", "tail", "type", "get-content", "gc", "select-string", "grep", "rg", "sed", "awk":
			return true
		}
	}
	return false
}

func isRMCommand(field string) bool {
	return field == "rm" || strings.HasSuffix(field, "/rm")
}

func isPowerShellDeleteCommand(field string) bool {
	switch field {
	case "remove-item", "del", "erase", "rd", "rmdir":
		return true
	default:
		return false
	}
}

func isUnsafeDeleteTarget(target string) bool {
	if target == "." || target == "./" || target == "*" {
		return true
	}
	if target == "/*" {
		return true
	}
	target = strings.TrimSuffix(target, "/*")
	for _, prefix := range []string{"~/", "$home/", "${home}/", "$env:home/", "$env:userprofile/", "%userprofile%/"} {
		if strings.HasPrefix(target, prefix) {
			return true
		}
	}
	for _, prefix := range []string{"/etc/", "/bin/", "/sbin/", "/usr/", "/var/", "/lib/", "/library/", "/system/", "/applications/", "c:/windows/", "c:/program files/"} {
		if strings.HasPrefix(target, prefix) {
			return true
		}
	}
	for _, exact := range []string{"/", "~", "$home", "${home}", "$env:home", "$env:userprofile", "%userprofile%", "c:", "c:/", "/etc", "/bin", "/sbin", "/usr", "/var", "/lib", "/library", "/system", "/applications", "c:/windows", "c:/program files"} {
		if target == exact {
			return true
		}
	}
	return false
}

func shellSafetyFields(command string) []string {
	return strings.Fields(shellSafetyText(command))
}

func shellSafetyText(command string) string {
	command = strings.ToLower(command)
	return strings.NewReplacer(
		"\\", "/",
		"\n", " ",
		"\t", " ",
		";", " ",
		"&", " ",
		"|", " ",
		"(", " ",
		")", " ",
		"\"", "",
		"'", "",
		"`", "",
	).Replace(command)
}

func readFinalWorkingDir(path string) string {
	content, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	workingDir := strings.TrimPrefix(string(content), "\ufeff")
	workingDir = strings.TrimSpace(workingDir)
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

type boundedOutput struct {
	Limit   int
	buf     []byte
	omitted int
}

func (b *boundedOutput) Write(p []byte) (int, error) {
	if b.Limit <= 0 {
		b.omitted += len(p)
		return len(p), nil
	}
	remaining := b.Limit - len(b.buf)
	if remaining <= 0 {
		b.omitted += len(p)
		return len(p), nil
	}
	if len(p) <= remaining {
		b.buf = append(b.buf, p...)
		return len(p), nil
	}
	writeLen := utf8SafePrefixLen(p[:remaining])
	b.buf = append(b.buf, p[:writeLen]...)
	b.omitted += len(p) - writeLen
	return len(p), nil
}

func (b *boundedOutput) Len() int {
	return len(b.buf) + b.omitted
}

func (b *boundedOutput) String(label string) string {
	safeLen := utf8SafePrefixLen(b.buf)
	content := string(b.buf[:safeLen])
	omitted := b.omitted + len(b.buf) - safeLen
	if omitted == 0 {
		return content
	}
	return content + fmt.Sprintf("\n\n[%s truncated: omitted ~%d tokens]", label, approximateTokensFromBytes(omitted))
}

func utf8SafePrefixLen(p []byte) int {
	if len(p) == 0 {
		return 0
	}
	start := len(p) - 1
	for start >= 0 && p[start]&0xc0 == 0x80 {
		start--
	}
	if start < 0 {
		return 0
	}
	lead := p[start]
	if lead < utf8.RuneSelf {
		return len(p)
	}
	if lead < 0xc2 || lead > 0xf4 {
		return len(p)
	}
	_, size := utf8.DecodeRune(p[start:])
	if size == 1 {
		return start
	}
	if start+size == len(p) {
		return len(p)
	}
	if start+size > len(p) {
		return start
	}
	return len(p)
}

func approximateTokensFromBytes(n int) int {
	if n <= 0 {
		return 0
	}
	return max(1, (n+3)/4)
}

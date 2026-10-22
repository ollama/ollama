package launch

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/ollama/ollama/envconfig"
)

// Claude implements Runner for Claude Code integration.
type Claude struct{}

func (c *Claude) String() string { return "Claude Code" }

func (c *Claude) args(model string, extra []string) []string {
	var args []string
	if model != "" {
		args = append(args, "--model", model)
	}
	args = append(args, extra...)
	return args
}

func (c *Claude) findPath() (string, error) {
	if p, err := exec.LookPath("claude"); err == nil {
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	name := "claude"
	if runtime.GOOS == "windows" {
		name = "claude.exe"
	}
	for _, fallback := range []string{
		filepath.Join(home, ".local", "bin", name),
		filepath.Join(home, ".claude", "local", name),
	} {
		if _, err := os.Stat(fallback); err == nil {
			return fallback, nil
		}
	}
	return "", fmt.Errorf("claude binary not found")
}

func (c *Claude) Run(model string, models []LaunchModel, args []string) error {
	claudePath, err := ensureClaudeInstalled()
	if err != nil {
		return err
	}

	cmd := exec.Command(claudePath, c.args(model, args)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	cmd.Env = append(os.Environ(), c.envVars(model, models)...)
	return cmd.Run()
}

func (c *Claude) envVars(model string, models []LaunchModel) []string {
	env := []string{
		"ANTHROPIC_BASE_URL=" + envconfig.Host().String(),
		"ANTHROPIC_API_KEY=",
		"ANTHROPIC_AUTH_TOKEN=ollama",
		"ENABLE_CLAUDEAI_MCP_SERVERS=false",
		"CLAUDE_CODE_ATTRIBUTION_HEADER=0",
		"CLAUDE_CODE_TOTAL_TOKENS_REMINDER=off",
		"DISABLE_ERROR_REPORTING=1",
		"DISABLE_FEEDBACK_COMMAND=1",
		"CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY=1",
	}

	// Ollama does not provide Anthropic's server-side auto-mode checks.
	// Use Claude Code's client classifier by default, preserving explicit settings.
	if _, ok := os.LookupEnv("CLAUDE_CODE_AUTO_MODE_SERVER"); !ok {
		env = append(env, "CLAUDE_CODE_AUTO_MODE_SERVER=0")
	}

	env = append(env, c.modelEnvVars(model, models)...)
	return env
}

func ensureClaudeInstalled() (string, error) {
	if path, err := (&Claude{}).findPath(); err == nil {
		return path, nil
	}

	if err := checkClaudeInstallerDependencies(); err != nil {
		return "", err
	}

	ok, err := ConfirmPrompt("Claude Code is not installed. Install now?")
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("claude installation cancelled")
	}

	bin, args, err := claudeInstallerCommand(runtime.GOOS)
	if err != nil {
		return "", err
	}

	fmt.Fprintf(os.Stderr, "\nInstalling Claude Code...\n")
	cmd := exec.Command(bin, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to install claude: %w", err)
	}

	path, err := (&Claude{}).findPath()
	if err != nil {
		return "", fmt.Errorf("claude was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}

	fmt.Fprintf(os.Stderr, "%sClaude Code installed successfully%s\n\n", ansiGreen, ansiReset)
	return path, nil
}

func checkClaudeInstallerDependencies() error {
	switch runtime.GOOS {
	case "windows":
		if _, err := exec.LookPath("powershell"); err != nil {
			return fmt.Errorf("claude is not installed and required dependencies are missing\n\nInstall the following first:\n  PowerShell: https://learn.microsoft.com/powershell/\n\nThen re-run:\n  ollama launch claude")
		}
	default:
		var missing []string
		if _, err := exec.LookPath("curl"); err != nil {
			missing = append(missing, "curl: https://curl.se/")
		}
		if _, err := exec.LookPath("bash"); err != nil {
			missing = append(missing, "bash: https://www.gnu.org/software/bash/")
		}
		if len(missing) > 0 {
			return fmt.Errorf("claude is not installed and required dependencies are missing\n\nInstall the following first:\n  %s\n\nThen re-run:\n  ollama launch claude", strings.Join(missing, "\n  "))
		}
	}
	return nil
}

func claudeInstallerCommand(goos string) (string, []string, error) {
	switch goos {
	case "windows":
		return "powershell", []string{
			"-NoProfile",
			"-ExecutionPolicy",
			"Bypass",
			"-Command",
			"irm https://claude.ai/install.ps1 | iex",
		}, nil
	case "darwin", "linux":
		return "bash", []string{
			"-c",
			"curl -fsSL https://claude.ai/install.sh | bash",
		}, nil
	default:
		return "", nil, fmt.Errorf("unsupported platform for claude install: %s", goos)
	}
}

// As of 2026-10-19 (cc version 2.1.304), claude code warns the following:
// CLAUDE_CODE_AUTO_COMPACT_WINDOW="1048576" is too large, so Claude Code treats it as the 1M maximum.
// Set it to a size from 100k to 1M, such as 200k or 200000, or unset it.
const (
	claudeMinAutoCompactWindow = 100_000
	claudeMaxAutoCompactWindow = 1_000_000
)

// modelEnvVars returns Claude Code env vars that route all model tiers through Ollama.
func (c *Claude) modelEnvVars(model string, models []LaunchModel) []string {
	env := []string{
		"ANTHROPIC_DEFAULT_OPUS_MODEL=" + model,
		"ANTHROPIC_DEFAULT_SONNET_MODEL=" + model,
		"ANTHROPIC_DEFAULT_HAIKU_MODEL=" + model,
		"CLAUDE_CODE_SUBAGENT_MODEL=" + model,
	}

	resolved, ok := findLaunchModel(models, model)
	if ok {
		resolved = resolved.WithCloudLimits().WithServerMetadata()
	} else {
		resolved = fallbackLaunchModel(model)
	}
	if resolved.Remote || isCloudModelName(model) {
		if ctxLen := resolved.ContextLength; ctxLen > 0 {
			env = append(env, "CLAUDE_CODE_MAX_CONTEXT_TOKENS="+strconv.Itoa(ctxLen))
			if os.Getenv("CLAUDE_CODE_AUTO_COMPACT_WINDOW") == "" && ctxLen >= claudeMinAutoCompactWindow {
				env = append(env, "CLAUDE_CODE_AUTO_COMPACT_WINDOW="+strconv.Itoa(min(ctxLen, claudeMaxAutoCompactWindow)))
			}
		}
	}

	return env
}

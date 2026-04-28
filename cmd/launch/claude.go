package launch

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"

	"github.com/ollama/ollama/envconfig"
)

const (
	claudeInstallURL = "https://code.claude.com/docs/en/quickstart"
	claudeBrewCmd    = "brew install anthropic/tap/claude-code"
	claudeNpmCmd     = "npm install -g @anthropic-ai/claude-code"
)

var (
	claudeLookPath = exec.LookPath
	claudeCommand  = exec.Command
	claudeGOOS     = runtime.GOOS
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
	if p, err := claudeLookPath("claude"); err == nil {
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	name := "claude"
	if claudeGOOS == "windows" {
		name = "claude.exe"
	}
	fallback := filepath.Join(home, ".claude", "local", name)
	if _, err := os.Stat(fallback); err != nil {
		return "", err
	}
	return fallback, nil
}

func (c *Claude) install() error {
	var bin string
	var args []string

	switch claudeGOOS {
	case "darwin":
		if c.isBrewAvailable() {
			bin = "brew"
			args = []string{"install", "anthropic/tap/claude-code"}
			break
		}
		if !c.isNpmAvailable() {
			return fmt.Errorf("claude is not installed and neither brew nor npm is available")
		}
		bin = "npm"
		args = []string{"install", "-g", "@anthropic-ai/claude-code"}
	default:
		if !c.isNpmAvailable() {
			return fmt.Errorf("claude is not installed and npm is not available")
		}
		bin = "npm"
		args = []string{"install", "-g", "@anthropic-ai/claude-code"}
	}

	fmt.Fprintf(os.Stderr, "\nInstalling Claude Code...\n")
	cmd := claudeCommand(bin, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to install claude: %w", err)
	}
	return nil
}

func (c *Claude) isBrewAvailable() bool {
	_, err := claudeLookPath("brew")
	return err == nil
}

func (c *Claude) isNpmAvailable() bool {
	_, err := claudeLookPath("npm")
	return err == nil
}

func (c *Claude) installHint() string {
	return fmt.Sprintf("claude is not installed\n\nAuto-install commands:\n  %s\n  %s\n\nManual install: %s", claudeBrewCmd, claudeNpmCmd, claudeInstallURL)
}

func (c *Claude) Run(model string, args []string) error {
	claudePath, err := c.findPath()
	if err != nil {
		originalErr := fmt.Errorf("%s", c.installHint())
		if err := c.install(); err != nil {
			return originalErr
		}
		claudePath, err = c.findPath()
		if err != nil {
			return originalErr
		}
	}

	cmd := claudeCommand(claudePath, c.args(model, args)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	env := append(os.Environ(),
		"ANTHROPIC_BASE_URL="+envconfig.Host().String(),
		"ANTHROPIC_API_KEY=",
		"ANTHROPIC_AUTH_TOKEN=ollama",
		"CLAUDE_CODE_ATTRIBUTION_HEADER=0",
	)

	env = append(env, c.modelEnvVars(model)...)

	cmd.Env = env
	return cmd.Run()
}

// modelEnvVars returns Claude Code env vars that route all model tiers through Ollama.
func (c *Claude) modelEnvVars(model string) []string {
	env := []string{
		"ANTHROPIC_DEFAULT_OPUS_MODEL=" + model,
		"ANTHROPIC_DEFAULT_SONNET_MODEL=" + model,
		"ANTHROPIC_DEFAULT_HAIKU_MODEL=" + model,
		"CLAUDE_CODE_SUBAGENT_MODEL=" + model,
	}

	if isCloudModelName(model) {
		if l, ok := lookupCloudModelLimit(model); ok {
			env = append(env, "CLAUDE_CODE_AUTO_COMPACT_WINDOW="+strconv.Itoa(l.Context))
		}
	}

	return env
}

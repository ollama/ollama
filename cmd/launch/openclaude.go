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

// Openclaude implements Runner for OpenClaude integration.
type Openclaude struct{}

func (o *Openclaude) String() string { return "OpenClaude" }

func (o *Openclaude) args(model string, extra []string) []string {
	var args []string
	if model != "" {
		args = append(args, "--model", model)
	}
	args = append(args, extra...)
	return args
}

func (o *Openclaude) findPath() (string, error) {
	if p, err := exec.LookPath("openclaude"); err == nil {
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	name := "openclaude"
	if runtime.GOOS == "windows" {
		name = "openclaude.exe"
	}
	fallback := filepath.Join(home, ".openclaude", "local", name)
	if _, err := os.Stat(fallback); err != nil {
		return "", err
	}
	return fallback, nil
}

func (o *Openclaude) Run(model string, args []string) error {
	openclaudePath, err := o.findPath()
	if err != nil {
		return fmt.Errorf("openclaude is not installed, install from https://github.com/gitlawbh/openclaude")
	}

	cmd := exec.Command(openclaudePath, o.args(model, args)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	env := append(os.Environ(),
		"ANTHROPIC_BASE_URL="+envconfig.Host().String(),
		"ANTHROPIC_API_KEY=",
		"ANTHROPIC_AUTH_TOKEN=ollama",
		"CLAUDE_CODE_ATTRIBUTION_HEADER=0",
		"OPENCLAUDE_DISABLE_CO_AUTHORED_BY=1",
	)

	env = append(env, o.modelEnvVars(model)...)

	cmd.Env = env
	return cmd.Run()
}

// modelEnvVars returns OpenClaude env vars that route all model tiers through Ollama.
func (o *Openclaude) modelEnvVars(model string) []string {
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
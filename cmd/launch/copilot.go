package launch

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"

	"github.com/ollama/ollama/envconfig"
)

// Copilot implements Runner for GitHub Copilot CLI integration.
type Copilot struct{}

func (c *Copilot) String() string { return "Copilot CLI" }

func (c *Copilot) args(model string, extra []string) []string {
	var args []string
	if model != "" {
		args = append(args, "--model", model)
	}
	args = append(args, extra...)
	return args
}

func (c *Copilot) findPath() (string, error) {
	if p, err := exec.LookPath("copilot"); err == nil {
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	name := "copilot"
	if runtime.GOOS == "windows" {
		name = "copilot.exe"
	}
	fallback := filepath.Join(home, ".local", "bin", name)
	if _, err := os.Stat(fallback); err != nil {
		return "", err
	}
	return fallback, nil
}

func (c *Copilot) Run(model string, args []string) error {
	copilotPath, err := c.findPath()
	if err != nil {
		return fmt.Errorf("copilot is not installed, install from https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli")
	}

	cmd := exec.Command(copilotPath, c.args(model, args)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	cmd.Env = append(os.Environ(), c.envVars(model)...)

	return cmd.Run()
}

// envVars returns the environment variables that configure Copilot CLI
// to use Ollama as its model provider.
func (c *Copilot) envVars(model string) []string {
	env := []string{
		"COPILOT_PROVIDER_BASE_URL=" + envconfig.Host().String() + "/v1",
		"COPILOT_PROVIDER_API_KEY=",
		"COPILOT_PROVIDER_WIRE_API=responses",
	}

	if model != "" {
		env = append(env, "COPILOT_MODEL="+model)
	}

	return env
}

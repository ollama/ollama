package launch

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"

	"github.com/ollama/ollama/envconfig"
)

// Talos implements Runner for the Talos agent.
//
// Talos ships as a Python package rather than a single binary: its installer creates
// a virtual environment under ~/talos and the agent is started with `python -m talos`.
// findPath therefore looks for a `talos` wrapper on PATH first and falls back to that
// interpreter, and args prepends `-m talos` only in the fallback case.
type Talos struct{}

func (t *Talos) String() string { return "Talos" }

// defaultPrefix is where the official installer puts Talos unless TALOS_PREFIX says
// otherwise. It is the only location guessed, and only after PATH has been tried.
func (t *Talos) defaultPrefix() (string, error) {
	if prefix := os.Getenv("TALOS_PREFIX"); prefix != "" {
		return prefix, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, "talos"), nil
}

// findPath returns the executable and whether it is the virtual environment's
// interpreter (which needs `-m talos` in front of the arguments).
func (t *Talos) findPath() (string, bool, error) {
	if p, err := exec.LookPath("talos"); err == nil {
		return p, false, nil
	}
	prefix, err := t.defaultPrefix()
	if err != nil {
		return "", false, err
	}
	python := filepath.Join(prefix, ".venv", "bin", "python")
	if runtime.GOOS == "windows" {
		python = filepath.Join(prefix, ".venv", "Scripts", "python.exe")
	}
	if _, err := os.Stat(python); err != nil {
		return "", false, err
	}
	return python, true, nil
}

func (t *Talos) args(viaPython bool, extra []string) []string {
	var args []string
	if viaPython {
		args = append(args, "-m", "talos")
	}
	// `chat` is the interactive session. `ask` answers once and exits, which is not
	// what launching an assistant means.
	args = append(args, "chat")
	return append(args, extra...)
}

func (t *Talos) Run(model string, _ []LaunchModel, args []string) error {
	talosPath, viaPython, err := t.findPath()
	if err != nil {
		return fmt.Errorf("talos is not installed, install from https://talos-agent.ch")
	}

	cmd := exec.Command(talosPath, t.args(viaPython, args)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	cmd.Env = append(os.Environ(), t.envVars(model)...)

	return cmd.Run()
}

// envVars returns the environment variables that point Talos at Ollama.
//
// Talos speaks the OpenAI wire protocol through its `openai-api` provider. Naming the
// base URL explicitly is also what tells Talos to trust the configured model name: its
// own catalogue cannot know which models a local server offers.
func (t *Talos) envVars(model string) []string {
	env := []string{
		"TALOS_MODEL_PROVIDER=openai-api",
		"TALOS_BASE_URL_OPENAI_API=" + envconfig.ConnectableHost().String() + "/v1",
	}

	if model != "" {
		env = append(env, "TALOS_MODEL="+model)
	}

	return env
}

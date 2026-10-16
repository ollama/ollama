package launch

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/ollama/ollama/envconfig"
)

// Wirken implements Runner for the Wirken agent gateway.
type Wirken struct{}

func (w *Wirken) String() string { return "Wirken" }

// wirkenDataDir mirrors Wirken's own resolution: WIRKEN_DATA_DIR, else ~/.wirken.
func wirkenDataDir() string {
	if dir := os.Getenv("WIRKEN_DATA_DIR"); dir != "" {
		return dir
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".wirken")
}

// wirkenSetUp reports whether `wirken setup` has run. Setup creates the
// vault and writes provider.json; `wirken run` refuses to start without it.
func wirkenSetUp() bool {
	dir := wirkenDataDir()
	if dir == "" {
		return false
	}
	_, err := os.Stat(filepath.Join(dir, "provider.json"))
	return err == nil
}

func (w *Wirken) args(model string, extra []string) []string {
	args := []string{"run"}
	if model != "" {
		args = append(args,
			"--provider", "ollama",
			"--model", model,
			"--base-url", envconfig.Host().String()+"/v1",
		)
	}
	return append(args, extra...)
}

func (w *Wirken) Run(model string, _ []LaunchModel, args []string) error {
	bin, err := exec.LookPath("wirken")
	if err != nil {
		return fmt.Errorf("wirken is not installed, install from https://github.com/gebruder/wirken#install")
	}

	if !wirkenSetUp() {
		fmt.Fprintln(os.Stderr, "Wirken has not been set up yet. Running 'wirken setup' first.")
		setup := exec.Command(bin, "setup")
		setup.Stdin = os.Stdin
		setup.Stdout = os.Stdout
		setup.Stderr = os.Stderr
		if err := setup.Run(); err != nil {
			return fmt.Errorf("wirken setup: %w", err)
		}
	}

	cmd := exec.Command(bin, w.args(model, args)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

package launch

import (
	"fmt"
	"os"
	"os/exec"
)

// Agy implements Runner for agy (Google's antigravity cli) integration.
type Agy struct{}

func (a *Agy) String() string { return "Agy" }

func (a *Agy) Run(model string, models []LaunchModel, args []string) error {
	agyPath, err := ensureAgyInstalled()
	if err != nil {
		return err
	}

	if len(args) == 0 && model != "" {
		args = []string{"--model", model}
	}

	cmd := exec.Command(agyPath, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = os.Environ()
	return cmd.Run()
}

func (a *Agy) Edit(models []LaunchModel) error {
	return nil
}

func (a *Agy) Models() []string {
	return nil
}

func findAgy() (string, bool) {
	if p, err := exec.LookPath("agy"); err == nil {
		return p, true
	}
	return "", false
}

func ensureAgyInstalled() (string, error) {
	if path, ok := findAgy(); ok {
		return path, nil
	}

	ok, err := ConfirmPrompt("Agy is not installed. Install now?")
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("agy installation cancelled")
	}

	fmt.Fprintf(os.Stderr, "\nInstalling Agy...\n")
	cmd := exec.Command("npm", "install", "-g", "agy")
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to install agy: %w", err)
	}

	path, ok := findAgy()
	if !ok {
		return "", fmt.Errorf("agy was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}

	fmt.Fprintf(os.Stderr, "%sAgy installed successfully%s\n\n", ansiGreen, ansiReset)
	return path, nil
}

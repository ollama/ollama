package launch

import (
	"fmt"
	"os"
	"os/exec"
)

// Paperclip implements Runner for the Paperclip integration
// (https://github.com/paperclipai/paperclip) — a control plane for AI-agent
// companies that ships a first-class `ollama_local` adapter (running an agent
// loop on /api/chat with native tool calling) as of paperclipai/paperclip#5249.
type Paperclip struct{}

const paperclipNpmPackage = "paperclipai"

func (p *Paperclip) String() string { return "Paperclip" }

func (p *Paperclip) args(model string, extra []string) []string {
	args := []string{"onboard", "--bind", "loopback", "-y", "--run"}
	return append(args, extra...)
}

func (p *Paperclip) Run(model string, args []string) error {
	if err := ensureNpmInstalled(); err != nil {
		return fmt.Errorf("npm (Node.js) is required to launch paperclip\n\nInstall it first:\n  https://nodejs.org/\n\nThen re-run:\n  ollama launch paperclip")
	}

	bin, err := ensurePaperclipInstalled()
	if err != nil {
		return err
	}

	cmd := exec.Command(bin, p.args(model, args)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	// Paperclip's `ollama_local` adapter reads OLLAMA_HOST and OLLAMA_API_KEY
	// directly. Setting them here lets `ollama launch paperclip` work without
	// any Paperclip-side configuration: the onboard wizard picks up the host
	// and the adapter pre-fills the right values.
	env := append(os.Environ(),
		"OLLAMA_HOST="+ollamaHostFromEnv(),
		"PAPERCLIP_DEFAULT_ADAPTER=ollama_local",
	)
	if model != "" {
		env = append(env, "PAPERCLIP_DEFAULT_MODEL="+model)
	}
	cmd.Env = env
	return cmd.Run()
}

func ensurePaperclipInstalled() (string, error) {
	if _, err := exec.LookPath("paperclipai"); err == nil {
		return "paperclipai", nil
	}

	ok, err := ConfirmPrompt("Paperclip is not installed. Install with npm?")
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("paperclip installation cancelled")
	}

	fmt.Fprintf(os.Stderr, "\nInstalling Paperclip...\n")
	cmd := exec.Command("npm", "install", "-g", paperclipNpmPackage+"@latest")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to install paperclip: %w", err)
	}

	if _, err := exec.LookPath("paperclipai"); err != nil {
		return "", fmt.Errorf("paperclip was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}
	return "paperclipai", nil
}

// ollamaHostFromEnv returns the configured Ollama host string for the child
// process. Mirrors the lookup pattern other integrations use.
func ollamaHostFromEnv() string {
	if v := os.Getenv("OLLAMA_HOST"); v != "" {
		return v
	}
	return "http://localhost:11434"
}

package config

import (
	"fmt"
	"os"
	"os/exec"
)

// Claude implements Runner for Claude Code integration
type Claude struct{}

func (c *Claude) String() string { return "Claude Code" }

func (c *Claude) Run(model string) error {
	if _, err := exec.LookPath("claude"); err != nil {
		return fmt.Errorf("claude is not installed, install from https://code.claude.com/docs/en/quickstart")
	}

	var args []string
	if model != "" {
		args = []string{"--model", model}
	}

	cmd := exec.Command("claude", args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(),
		"ANTHROPIC_BASE_URL=http://localhost:11434",
		"ANTHROPIC_API_KEY=",
		"ANTHROPIC_AUTH_TOKEN=ollama",
	)
	return cmd.Run()
}

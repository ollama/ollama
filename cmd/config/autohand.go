package config

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/ollama/ollama/envconfig"
)

// Autohand implements Runner for Autohand Code CLI integration
type Autohand struct{}

func (a *Autohand) String() string { return "Autohand Code" }

func (a *Autohand) Run(model string, args []string) error {
	if _, err := exec.LookPath("autohand"); err != nil {
		return fmt.Errorf("autohand is not installed, install with: npm install -g autohand-cli")
	}

	// Configure Autohand to use Ollama before launching
	if err := configureAutoHand(model); err != nil {
		return fmt.Errorf("setup failed: %w", err)
	}

	cmd := exec.Command("autohand", args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// configureAutoHand writes Autohand's config.json to use the given Ollama model.
func configureAutoHand(model string) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	configPath := filepath.Join(home, ".autohand", "config.json")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}

	config := make(map[string]any)
	if data, err := os.ReadFile(configPath); err == nil {
		_ = json.Unmarshal(data, &config)
	}

	config["provider"] = "ollama"

	ollama, ok := config["ollama"].(map[string]any)
	if !ok {
		ollama = make(map[string]any)
	}

	ollama["baseUrl"] = envconfig.Host().String()
	ollama["model"] = model

	config["ollama"] = ollama

	configData, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	return writeWithBackup(configPath, configData)
}

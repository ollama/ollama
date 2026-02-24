package config

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/ollama/ollama/envconfig"
)

// Cline implements Runner and Editor for the Cline CLI integration
type Cline struct{}

func (c *Cline) String() string { return "Cline" }

func (c *Cline) Run(model string, args []string) error {
	if _, err := exec.LookPath("cline"); err != nil {
		return fmt.Errorf("cline is not installed, install with: npm install -g cline")
	}

	models := []string{model}
	if config, err := loadIntegration("cline"); err == nil && len(config.Models) > 0 {
		models = config.Models
	}
	var err error
	models, err = resolveEditorModels("cline", models, func() ([]string, error) {
		return selectModels(context.Background(), "cline", "")
	})
	if errors.Is(err, errCancelled) {
		return nil
	}
	if err != nil {
		return err
	}
	if err := c.Edit(models); err != nil {
		return fmt.Errorf("setup failed: %w", err)
	}

	cmd := exec.Command("cline", args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func (c *Cline) Paths() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}
	p := filepath.Join(home, ".cline", "data", "globalState.json")
	if _, err := os.Stat(p); err == nil {
		return []string{p}
	}
	return nil
}

func (c *Cline) Edit(models []string) error {
	if len(models) == 0 {
		return nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	configPath := filepath.Join(home, ".cline", "data", "globalState.json")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}

	config := make(map[string]any)
	if data, err := os.ReadFile(configPath); err == nil {
		if err := json.Unmarshal(data, &config); err != nil {
			return fmt.Errorf("failed to parse config: %w, at: %s", err, configPath)
		}
	}

	// Set Ollama as the provider for both act and plan modes
	baseURL := envconfig.Host().String()
	config["ollamaBaseUrl"] = baseURL
	config["actModeApiProvider"] = "ollama"
	config["actModeOllamaModelId"] = models[0]
	config["actModeOllamaBaseUrl"] = baseURL
	config["planModeApiProvider"] = "ollama"
	config["planModeOllamaModelId"] = models[0]
	config["planModeOllamaBaseUrl"] = baseURL

	config["welcomeViewCompleted"] = true

	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	return writeWithBackup(configPath, data)
}

func (c *Cline) Models() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}

	config, err := readJSONFile(filepath.Join(home, ".cline", "data", "globalState.json"))
	if err != nil {
		return nil
	}

	if config["actModeApiProvider"] != "ollama" {
		return nil
	}

	modelID, _ := config["actModeOllamaModelId"].(string)
	if modelID == "" {
		return nil
	}
	return []string{modelID}
}

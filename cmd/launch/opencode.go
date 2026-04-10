package launch

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"

	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
)

// OpenCode implements Runner and Editor for OpenCode integration.
// Config is passed via OPENCODE_CONFIG_CONTENT env var at launch time
// instead of writing to opencode's config files.
type OpenCode struct {
	configContent string // JSON config built by Edit, passed to Run via env var
}

func (o *OpenCode) String() string { return "OpenCode" }

// findOpenCode returns the opencode binary path, checking PATH first then the
// curl installer location (~/.opencode/bin) which may not be on PATH yet.
func findOpenCode() (string, bool) {
	if p, err := exec.LookPath("opencode"); err == nil {
		return p, true
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", false
	}
	name := "opencode"
	if runtime.GOOS == "windows" {
		name = "opencode.exe"
	}
	fallback := filepath.Join(home, ".opencode", "bin", name)
	if _, err := os.Stat(fallback); err == nil {
		return fallback, true
	}
	return "", false
}

func (o *OpenCode) Run(model string, args []string) error {
	opencodePath, ok := findOpenCode()
	if !ok {
		return fmt.Errorf("opencode is not installed, install from https://opencode.ai")
	}

	cmd := exec.Command(opencodePath, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = os.Environ()
	if o.configContent != "" {
		cmd.Env = append(cmd.Env, "OPENCODE_CONFIG_CONTENT="+o.configContent)
	}
	return cmd.Run()
}

func (o *OpenCode) Paths() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}
	sp := filepath.Join(home, ".local", "state", "opencode", "model.json")
	if _, err := os.Stat(sp); err == nil {
		return []string{sp}
	}
	return nil
}

func (o *OpenCode) Edit(modelList []string) error {
	if len(modelList) == 0 {
		return nil
	}

	// Build the inline config for OPENCODE_CONFIG_CONTENT
	config := map[string]any{
		"$schema": "https://opencode.ai/config.json",
		"provider": map[string]any{
			"ollama": map[string]any{
				"npm":  "@ai-sdk/openai-compatible",
				"name": "Ollama",
				"options": map[string]any{
					"baseURL": envconfig.Host().String() + "/v1",
				},
				"models": buildModelEntries(modelList),
			},
		},
		"model": "ollama/" + modelList[0],
	}

	configData, err := json.Marshal(config)
	if err != nil {
		return err
	}
	o.configContent = string(configData)

	// Write model state file so models appear in OpenCode's model picker
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	statePath := filepath.Join(home, ".local", "state", "opencode", "model.json")
	if err := os.MkdirAll(filepath.Dir(statePath), 0o755); err != nil {
		return err
	}

	state := map[string]any{
		"recent":   []any{},
		"favorite": []any{},
		"variant":  map[string]any{},
	}
	if data, err := os.ReadFile(statePath); err == nil {
		_ = json.Unmarshal(data, &state) // Ignore parse errors; use defaults
	}

	newRecent := make([]any, 0, len(modelList))
	for _, model := range modelList {
		newRecent = append(newRecent, map[string]any{
			"providerID": "ollama",
			"modelID":    model,
		})
	}
	state["recent"] = newRecent

	stateData, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return fileutil.WriteWithBackup(statePath, stateData)
}

func (o *OpenCode) Models() []string {
	return nil
}

func buildModelEntries(modelList []string) map[string]any {
	models := make(map[string]any)
	for _, model := range modelList {
		entry := map[string]any{
			"name": model,
		}
		if isCloudModelName(model) {
			if l, ok := lookupCloudModelLimit(model); ok {
				entry["limit"] = map[string]any{
					"context": l.Context,
					"output":  l.Output,
				}
			}
		}
		models[model] = entry
	}
	return models
}

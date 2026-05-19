package launch

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

// OpenCode implements Runner and Editor for OpenCode integration.
// Config is passed via OPENCODE_CONFIG_CONTENT env var at launch time
// instead of writing to opencode's config files.
type OpenCode struct {
	configContent string // JSON config built by Edit, passed to Run via env var
}

const openCodeModelShowTimeout = 2 * time.Second

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
	if content := o.resolveContent(model); content != "" {
		cmd.Env = append(cmd.Env, "OPENCODE_CONFIG_CONTENT="+content)
	}
	return cmd.Run()
}

// resolveContent returns the inline config to send via OPENCODE_CONFIG_CONTENT.
// Returns content built by Edit if available, otherwise builds from model.json
// with the requested model as primary (e.g. re-launch with saved config).
func (o *OpenCode) resolveContent(model string) string {
	if o.configContent != "" {
		return o.configContent
	}
	models := readModelJSONModels()
	if !slices.Contains(models, model) {
		models = append([]string{model}, models...)
	}
	content, err := buildInlineConfig(model, models)
	if err != nil {
		return ""
	}
	return content
}

func (o *OpenCode) Paths() []string {
	sp, err := openCodeStatePath()
	if err != nil {
		return nil
	}
	if _, err := os.Stat(sp); err == nil {
		return []string{sp}
	}
	return nil
}

// openCodeStatePath returns the path to opencode's model state file.
// TODO: this hardcodes the Linux/macOS XDG path. On Windows, opencode stores
// state under %LOCALAPPDATA% (or similar) — verify and branch on runtime.GOOS.
func openCodeStatePath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".local", "state", "opencode", "model.json"), nil
}

func (o *OpenCode) Edit(modelList []string) error {
	if len(modelList) == 0 {
		return nil
	}

	content, err := buildInlineConfig(modelList[0], modelList)
	if err != nil {
		return err
	}
	o.configContent = content

	// Write model state file so models appear in OpenCode's model picker
	statePath, err := openCodeStatePath()
	if err != nil {
		return err
	}
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

	recent, _ := state["recent"].([]any)

	modelSet := make(map[string]bool)
	for _, m := range modelList {
		modelSet[m] = true
	}

	// Filter out existing Ollama models we're about to re-add
	newRecent := slices.DeleteFunc(slices.Clone(recent), func(entry any) bool {
		e, ok := entry.(map[string]any)
		if !ok || e["providerID"] != "ollama" {
			return false
		}
		modelID, _ := e["modelID"].(string)
		return modelSet[modelID]
	})

	// Prepend models in reverse order so first model ends up first
	for _, model := range slices.Backward(modelList) {
		newRecent = slices.Insert(newRecent, 0, any(map[string]any{
			"providerID": "ollama",
			"modelID":    model,
		}))
	}

	const maxRecentModels = 10
	newRecent = newRecent[:min(len(newRecent), maxRecentModels)]

	state["recent"] = newRecent

	stateData, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return fileutil.WriteWithBackup(statePath, stateData, "opencode")
}

func (o *OpenCode) Models() []string {
	return nil
}

// buildInlineConfig produces the JSON string for OPENCODE_CONFIG_CONTENT.
// primary is the model to launch with, models is the full list of available models.
func buildInlineConfig(primary string, models []string) (string, error) {
	if primary == "" || len(models) == 0 {
		return "", fmt.Errorf("buildInlineConfig: primary and models are required")
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		client = nil
	}

	config := map[string]any{
		"$schema": "https://opencode.ai/config.json",
		"provider": map[string]any{
			"ollama": map[string]any{
				"npm":  "@ai-sdk/openai-compatible",
				"name": "Ollama",
				"options": map[string]any{
					"baseURL": envconfig.Host().String() + "/v1",
				},
				"models": buildModelEntries(context.Background(), client, models),
			},
		},
		"model": "ollama/" + primary,
	}
	data, err := json.Marshal(config)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// readModelJSONModels reads ollama model IDs from the opencode model.json state file
func readModelJSONModels() []string {
	statePath, err := openCodeStatePath()
	if err != nil {
		return nil
	}
	data, err := os.ReadFile(statePath)
	if err != nil {
		return nil
	}
	var state map[string]any
	if err := json.Unmarshal(data, &state); err != nil {
		return nil
	}
	recent, _ := state["recent"].([]any)
	var models []string
	for _, entry := range recent {
		e, ok := entry.(map[string]any)
		if !ok {
			continue
		}
		if e["providerID"] != "ollama" {
			continue
		}
		if id, ok := e["modelID"].(string); ok && id != "" {
			models = append(models, id)
		}
	}
	return models
}

func buildModelEntries(ctx context.Context, client *api.Client, modelList []string) map[string]any {
	if client != nil {
		var cancel context.CancelFunc
		if _, hasDeadline := ctx.Deadline(); !hasDeadline {
			ctx, cancel = context.WithTimeout(ctx, openCodeModelShowTimeout)
			defer cancel()
		}
	}

	models := make(map[string]any)
	for _, modelID := range modelList {
		entry := map[string]any{
			"name": modelID,
		}
		if client != nil {
			if resp, err := client.Show(ctx, &api.ShowRequest{Model: modelID}); err == nil {
				if slices.Contains(resp.Capabilities, model.CapabilityVision) {
					entry["modalities"] = map[string]any{
						"input":  []string{"text", "image"},
						"output": []string{"text"},
					}
				}
			}
		}
		if isCloudModelName(modelID) {
			if l, ok := lookupCloudModelLimit(modelID); ok {
				entry["limit"] = map[string]any{
					"context": l.Context,
					"output":  l.Output,
				}
			}
		}
		models[modelID] = entry
	}
	return models
}

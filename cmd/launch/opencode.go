package launch

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
)

const openCodeInstallScript = "curl -fsSL https://opencode.ai/install | bash"

var openCodeGOOS = runtime.GOOS

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
	if openCodeGOOS == "windows" {
		name = "opencode.exe"
	}
	fallback := filepath.Join(home, ".opencode", "bin", name)
	if _, err := os.Stat(fallback); err == nil {
		return fallback, true
	}
	return "", false
}

func (o *OpenCode) Run(model string, models []LaunchModel, args []string) error {
	opencodePath, err := ensureOpenCodeInstalled()
	if err != nil {
		return err
	}

	cmd := exec.Command(opencodePath, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = os.Environ()
	if content := o.resolveContent(model, models); content != "" {
		cmd.Env = append(cmd.Env, "OPENCODE_CONFIG_CONTENT="+content)
	}
	return cmd.Run()
}

func ensureOpenCodeInstalled() (string, error) {
	if opencodePath, ok := findOpenCode(); ok {
		return opencodePath, nil
	}

	if err := checkOpenCodeInstallerDependencies(); err != nil {
		return "", err
	}

	ok, err := ConfirmPrompt("OpenCode is not installed. Install now?")
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("opencode installation cancelled")
	}

	bin, args, err := openCodeInstallerCommand(openCodeGOOS)
	if err != nil {
		return "", err
	}

	fmt.Fprintf(os.Stderr, "\nInstalling OpenCode...\n")
	cmd := exec.Command(bin, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to install opencode: %w", err)
	}

	opencodePath, ok := findOpenCode()
	if !ok {
		return "", fmt.Errorf("opencode was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}

	fmt.Fprintf(os.Stderr, "%sOpenCode installed successfully%s\n\n", ansiGreen, ansiReset)
	return opencodePath, nil
}

func checkOpenCodeInstallerDependencies() error {
	switch openCodeGOOS {
	case "windows":
		if _, err := exec.LookPath("npm"); err != nil {
			return fmt.Errorf("opencode is not installed and required dependencies are missing\n\nInstall the following first:\n  npm (Node.js): https://nodejs.org/\n\nThen re-run:\n  ollama launch opencode")
		}
	default:
		var missing []string
		if _, err := exec.LookPath("curl"); err != nil {
			missing = append(missing, "curl: https://curl.se/")
		}
		if _, err := exec.LookPath("bash"); err != nil {
			missing = append(missing, "bash: https://www.gnu.org/software/bash/")
		}
		if len(missing) > 0 {
			return fmt.Errorf("opencode is not installed and required dependencies are missing\n\nInstall the following first:\n  %s\n\nThen re-run:\n  ollama launch opencode", strings.Join(missing, "\n  "))
		}
	}
	return nil
}

func openCodeInstallerCommand(goos string) (string, []string, error) {
	switch goos {
	case "windows":
		return "npm", []string{"install", "-g", "opencode-ai@latest"}, nil
	case "darwin", "linux":
		return "bash", []string{"-c", "set -o pipefail; " + openCodeInstallScript}, nil
	default:
		return "", nil, fmt.Errorf("unsupported platform for opencode install: %s", goos)
	}
}

// resolveContent returns the inline config to send via OPENCODE_CONFIG_CONTENT.
// Returns content built by Edit if available, otherwise builds from model.json
// with the requested model as primary (e.g. re-launch with saved config).
func (o *OpenCode) resolveContent(model string, models []LaunchModel) string {
	if o.configContent != "" {
		return o.configContent
	}
	resolvedModels := resolveOpenCodeRunModels(model, models, readModelJSONModels())
	if len(resolvedModels) == 0 {
		return ""
	}
	content, err := buildInlineConfig(resolvedModels[0], resolvedModels)
	if err != nil {
		return ""
	}
	return content
}

func resolveOpenCodeRunModels(primary string, models []LaunchModel, stateModels []string) []LaunchModel {
	if primary == "" {
		return nil
	}

	resolved := make([]LaunchModel, 0, 1+len(models)+len(stateModels))
	appendModel := func(name string) {
		if name == "" || hasLaunchModel(resolved, name) {
			return
		}
		if model, ok := findLaunchModel(models, name); ok {
			resolved = append(resolved, model)
			return
		}
		resolved = append(resolved, fallbackLaunchModel(name))
	}

	appendModel(primary)
	for _, model := range models {
		appendModel(model.Name)
	}
	for _, model := range stateModels {
		appendModel(model)
	}
	return resolved
}

func hasLaunchModel(models []LaunchModel, name string) bool {
	for _, model := range models {
		if launchModelMatches(model.Name, name) || launchModelMatches(name, model.Name) {
			return true
		}
	}
	return false
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

func (o *OpenCode) Edit(models []LaunchModel) error {
	modelList := launchModelNames(models)
	if len(modelList) == 0 {
		return nil
	}

	content, err := buildInlineConfig(models[0], models)
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
func buildInlineConfig(primary LaunchModel, models []LaunchModel) (string, error) {
	if primary.Name == "" || len(models) == 0 {
		return "", fmt.Errorf("buildInlineConfig: primary and models are required")
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
				"models": buildModelEntries(models),
			},
		},
		"model": "ollama/" + primary.Name,
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

func buildModelEntries(modelList []LaunchModel) map[string]any {
	models := make(map[string]any)
	for _, model := range modelList {
		entry := map[string]any{
			"name": model.Name,
		}
		if model.HasCapability("vision") {
			entry["modalities"] = map[string]any{
				"input":  []string{"text", "image"},
				"output": []string{"text"},
			}
		}
		if model.HasCapability("thinking") {
			entry["reasoning"] = true
			if openCodeModelSupportsThinkingLevels(model) {
				entry["options"] = map[string]any{"reasoningEffort": "medium"}
				entry["variants"] = map[string]any{
					"low":    map[string]any{"reasoningEffort": "low"},
					"medium": map[string]any{"reasoningEffort": "medium"},
					"high":   map[string]any{"reasoningEffort": "high"},
					"max":    map[string]any{"reasoningEffort": "max"},
				}
			} else {
				entry["variants"] = map[string]any{
					"none":   map[string]any{"reasoningEffort": "none"},
					"low":    map[string]any{"disabled": true},
					"medium": map[string]any{"disabled": true},
					"high":   map[string]any{"disabled": true},
				}
			}
		}
		if model.MaxOutputTokens > 0 {
			limit := make(map[string]any)
			if model.ContextLength > 0 {
				limit["context"] = model.ContextLength
			}
			limit["output"] = model.MaxOutputTokens
			entry["limit"] = limit
		}
		models[model.Name] = entry
	}
	return models
}

func openCodeModelSupportsThinkingLevels(model LaunchModel) bool {
	for _, family := range append([]string{model.Details.Family}, model.Details.Families...) {
		if normalizeOpenCodeModelFamily(family) == "gptoss" {
			return true
		}
	}

	return strings.Contains(normalizeOpenCodeModelFamily(model.Name), "gptoss")
}

func normalizeOpenCodeModelFamily(s string) string {
	s = strings.ToLower(s)
	s = strings.ReplaceAll(s, "-", "")
	s = strings.ReplaceAll(s, "_", "")
	return s
}

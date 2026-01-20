package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/spf13/cobra"
)

type EnvVar struct {
	Name  string
	Value string
}

type AppConfig struct {
	Name         string
	DisplayName  string
	Command      string
	EnvVars      func(model string) []EnvVar
	Args         func(model string) []string
	Setup        func(models []string) error
	CheckInstall func() error
}

// checkCommand returns an error if the command is not installed
func checkCommand(cmd, installInstructions string) func() error {
	return func() error {
		if _, err := exec.LookPath(cmd); err != nil {
			return fmt.Errorf("%s is not installed. %s", cmd, installInstructions)
		}
		return nil
	}
}

var ClaudeConfig = &AppConfig{
	Name:        "Claude",
	DisplayName: "Claude Code",
	Command:     "claude",
	EnvVars: func(model string) []EnvVar {
		return []EnvVar{
			{Name: "ANTHROPIC_BASE_URL", Value: "http://localhost:11434"},
			{Name: "ANTHROPIC_API_KEY", Value: "ollama"},
			{Name: "ANTHROPIC_AUTH_TOKEN", Value: "ollama"},
		}
	},
	Args: func(model string) []string {
		if model == "" {
			return nil
		}
		return []string{"--model", model}
	},
	CheckInstall: checkCommand("claude", "Install with: npm install -g @anthropic-ai/claude-code"),
}

var CodexConfig = &AppConfig{
	Name:        "Codex",
	DisplayName: "Codex",
	Command:     "codex",
	EnvVars: func(model string) []EnvVar {
		return []EnvVar{}
	},
	Args: func(model string) []string {
		// Defaults to gpt-oss:20b in codex
		if model == "" {
			return []string{"--oss"}
		}
		return []string{"--oss", "-m", model}
	},
	CheckInstall: checkCommand("codex", "Install with: npm install -g @openai/codex or brew install --cask codex"),
}

var DroidConfig = &AppConfig{
	Name:         "Droid",
	DisplayName:  "Droid",
	Command:      "droid",
	EnvVars:      func(model string) []EnvVar { return nil },
	Args:         func(model string) []string { return nil },
	Setup:        setupDroidSettings,
	CheckInstall: checkCommand("droid", "Install from: https://docs.factory.ai/cli/getting-started/quickstart"),
}

var AppRegistry = map[string]*AppConfig{
	"claude":   ClaudeConfig,
	"codex":    CodexConfig,
	"droid":    DroidConfig,
	"opencode": OpenCodeConfig,
}

func GetApp(name string) (*AppConfig, bool) {
	app, ok := AppRegistry[strings.ToLower(name)]
	return app, ok
}

func getModelInfo(model string) *api.ShowResponse {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil
	}
	resp, err := client.Show(context.Background(), &api.ShowRequest{Model: model})
	if err != nil {
		return nil
	}
	return resp
}

func getModelContextLength(model string) int {
	const defaultCtx = 64000 // default context is set to 64k to support coding agents
	resp := getModelInfo(model)
	if resp == nil || resp.ModelInfo == nil {
		return defaultCtx
	}
	arch, ok := resp.ModelInfo["general.architecture"].(string)
	if !ok {
		return defaultCtx
	}
	// currently being capped at 128k
	if v, ok := resp.ModelInfo[fmt.Sprintf("%s.context_length", arch)].(float64); ok {
		return min(int(v), 128000)
	}
	return defaultCtx
}

func modelSupportsImages(model string) bool {
	resp := getModelInfo(model)
	if resp == nil {
		return false
	}
	return slices.Contains(resp.Capabilities, "vision")
}

func copyFile(src, dst string) error {
	info, err := os.Stat(src)
	if err != nil {
		return err
	}
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	// Preserve source file permissions (important for files containing API keys)
	return os.WriteFile(dst, data, info.Mode().Perm())
}

func atomicWriteJSON(path string, data any) error {
	content, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal failed: %w", err)
	}

	var check any
	if err := json.Unmarshal(content, &check); err != nil {
		return fmt.Errorf("validation failed: %w", err)
	}

	backupPath := path + ".bak"
	if existingContent, err := os.ReadFile(path); err == nil {
		if !bytes.Equal(existingContent, content) {
			if err := copyFile(path, backupPath); err != nil {
				return fmt.Errorf("backup failed: %w", err)
			}
		}
	}

	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return fmt.Errorf("create temp failed: %w", err)
	}
	tmpPath := tmp.Name()

	if _, err := tmp.Write(content); err != nil {
		tmp.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("write failed: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("close failed: %w", err)
	}

	if err := os.Rename(tmpPath, path); err != nil {
		os.Remove(tmpPath)
		if _, statErr := os.Stat(backupPath); statErr == nil {
			os.Rename(backupPath, path)
		}
		return fmt.Errorf("rename failed: %w", err)
	}

	return nil
}

func isValidReasoningEffort(effort string) bool {
	switch effort {
	case "high", "medium", "low", "none":
		return true
	}
	return false
}

func setupDroidSettings(models []string) error {
	if len(models) == 0 {
		return nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	settingsPath := filepath.Join(home, ".factory", "settings.json")
	if err := os.MkdirAll(filepath.Dir(settingsPath), 0o755); err != nil {
		return err
	}

	settings := make(map[string]any)
	if data, err := os.ReadFile(settingsPath); err == nil {
		json.Unmarshal(data, &settings)
	}

	var customModels []any
	if existing, ok := settings["customModels"].([]any); ok {
		customModels = existing
	}

	// Keep only non-Ollama models (we'll rebuild Ollama models fresh)
	var nonOllamaModels []any
	for _, m := range customModels {
		entry, ok := m.(map[string]any)
		if !ok {
			nonOllamaModels = append(nonOllamaModels, m)
			continue
		}

		displayName, _ := entry["displayName"].(string)
		if !strings.HasSuffix(displayName, "[Ollama]") {
			nonOllamaModels = append(nonOllamaModels, m)
		}
	}

	// Build new Ollama model entries with sequential indices (0, 1, 2, ...)
	var ollamaModels []any
	var defaultModelID string
	for i, model := range models {
		modelID := fmt.Sprintf("custom:%s-[Ollama]-%d", model, i)
		newEntry := map[string]any{
			"model":           model,
			"displayName":     fmt.Sprintf("%s [Ollama]", model),
			"baseUrl":         "http://localhost:11434/v1",
			"apiKey":          "ollama",
			"provider":        "generic-chat-completion-api",
			"maxOutputTokens": getModelContextLength(model),
			"supportsImages":  modelSupportsImages(model),
			"id":              modelID,
			"index":           i,
		}
		ollamaModels = append(ollamaModels, newEntry)

		if i == 0 {
			defaultModelID = modelID
		}
	}

	settings["customModels"] = append(ollamaModels, nonOllamaModels...)

	sessionSettings, ok := settings["sessionDefaultSettings"].(map[string]any)
	if !ok {
		sessionSettings = make(map[string]any)
	}
	sessionSettings["model"] = defaultModelID

	if effort, ok := sessionSettings["reasoningEffort"].(string); !ok || !isValidReasoningEffort(effort) {
		sessionSettings["reasoningEffort"] = "none"
	}

	settings["sessionDefaultSettings"] = sessionSettings

	return atomicWriteJSON(settingsPath, settings)
}

func setupOpenCodeSettings(modelList []string) error {
	if len(modelList) == 0 {
		return nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	configPath := filepath.Join(home, ".config", "opencode", "opencode.json")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}

	config := make(map[string]any)
	if data, err := os.ReadFile(configPath); err == nil {
		json.Unmarshal(data, &config)
	}

	config["$schema"] = "https://opencode.ai/config.json"

	provider, ok := config["provider"].(map[string]any)
	if !ok {
		provider = make(map[string]any)
	}

	ollama, ok := provider["ollama"].(map[string]any)
	if !ok {
		ollama = map[string]any{
			"npm":  "@ai-sdk/openai-compatible",
			"name": "Ollama (local)",
			"options": map[string]any{
				"baseURL": "http://localhost:11434/v1",
			},
		}
	}

	models, ok := ollama["models"].(map[string]any)
	if !ok {
		models = make(map[string]any)
	}

	selectedSet := make(map[string]bool)
	for _, m := range modelList {
		selectedSet[m] = true
	}

	for name, cfg := range models {
		if cfgMap, ok := cfg.(map[string]any); ok {
			if displayName, ok := cfgMap["name"].(string); ok {
				if strings.HasSuffix(displayName, "[Ollama]") && !selectedSet[name] {
					delete(models, name)
				}
			}
		}
	}

	for _, model := range modelList {
		models[model] = map[string]any{
			"name": fmt.Sprintf("%s [Ollama]", model),
		}
	}

	ollama["models"] = models
	provider["ollama"] = ollama
	config["provider"] = provider

	if err := atomicWriteJSON(configPath, config); err != nil {
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
		json.Unmarshal(data, &state)
	}

	recent, _ := state["recent"].([]any)

	modelSet := make(map[string]bool)
	for _, m := range modelList {
		modelSet[m] = true
	}

	newRecent := []any{}
	for _, entry := range recent {
		if e, ok := entry.(map[string]any); ok {
			if e["providerID"] == "ollama" {
				if modelID, ok := e["modelID"].(string); ok && modelSet[modelID] {
					continue
				}
			}
		}
		newRecent = append(newRecent, entry)
	}

	for i := len(modelList) - 1; i >= 0; i-- {
		newRecent = append([]any{map[string]any{
			"providerID": "ollama",
			"modelID":    modelList[i],
		}}, newRecent...)
	}

	if len(newRecent) > 10 {
		newRecent = newRecent[:10]
	}

	state["recent"] = newRecent

	return atomicWriteJSON(statePath, state)
}

var OpenCodeConfig = &AppConfig{
	Name:         "OpenCode",
	DisplayName:  "OpenCode",
	Command:      "opencode",
	EnvVars:      func(model string) []EnvVar { return nil },
	Args:         func(model string) []string { return nil },
	Setup:        setupOpenCodeSettings,
	CheckInstall: checkCommand("opencode", "Install from: https://opencode.ai"),
}

func readJSONFile(path string) (map[string]any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var result map[string]any
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func getOpenCodeOllamaModels() (map[string]any, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	config, err := readJSONFile(filepath.Join(home, ".config", "opencode", "opencode.json"))
	if err != nil {
		return nil, err
	}

	provider, _ := config["provider"].(map[string]any)
	ollama, _ := provider["ollama"].(map[string]any)
	models, _ := ollama["models"].(map[string]any)
	return models, nil
}

func getOpenCodeConfiguredModels() []string {
	models, err := getOpenCodeOllamaModels()
	if err != nil || models == nil {
		return nil
	}

	var result []string
	for name := range models {
		result = append(result, name)
	}
	return result
}

func getOpenCodeConfiguredModel() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}

	state, err := readJSONFile(filepath.Join(home, ".local", "state", "opencode", "model.json"))
	if err != nil {
		return ""
	}

	recent, _ := state["recent"].([]any)
	if len(recent) == 0 {
		return ""
	}

	first, _ := recent[0].(map[string]any)
	if first["providerID"] == "ollama" {
		modelID, _ := first["modelID"].(string)
		return modelID
	}
	return ""
}

func readDroidSettings() (map[string]any, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	return readJSONFile(filepath.Join(home, ".factory", "settings.json"))
}

func getDroidConfiguredModels() []string {
	settings, err := readDroidSettings()
	if err != nil {
		return nil
	}

	customModels, _ := settings["customModels"].([]any)

	var result []string
	for _, m := range customModels {
		entry, _ := m.(map[string]any)
		displayName, _ := entry["displayName"].(string)
		// Only include Ollama models (those with our displayName pattern)
		if strings.HasSuffix(displayName, "[Ollama]") {
			if model, _ := entry["model"].(string); model != "" {
				result = append(result, model)
			}
		}
	}
	return result
}

func getDroidConfiguredModel() string {
	settings, err := readDroidSettings()
	if err != nil {
		return ""
	}

	sessionSettings, _ := settings["sessionDefaultSettings"].(map[string]any)
	modelID, _ := sessionSettings["model"].(string)
	if modelID == "" {
		return ""
	}

	customModels, _ := settings["customModels"].([]any)
	for _, m := range customModels {
		entry, _ := m.(map[string]any)
		if entry["id"] == modelID {
			model, _ := entry["model"].(string)
			return model
		}
	}
	return ""
}

func getAppConfiguredModels(appName string) []string {
	// Get models that exist in the app's config
	var appModels []string
	switch strings.ToLower(appName) {
	case "opencode":
		appModels = getOpenCodeConfiguredModels()
	case "droid":
		appModels = getDroidConfiguredModels()
	}

	// Get our saved connection config for the correct order (default first)
	savedConfig, err := LoadConnection(appName)
	if err != nil || len(savedConfig.Models) == 0 {
		return appModels
	}
	if len(appModels) == 0 {
		return savedConfig.Models
	}

	// Merge: saved order first (filtered to still-existing models), then any new models
	appModelSet := make(map[string]bool, len(appModels))
	for _, m := range appModels {
		appModelSet[m] = true
	}

	seen := make(map[string]bool, len(savedConfig.Models))
	var result []string
	for _, m := range savedConfig.Models {
		if appModelSet[m] {
			result = append(result, m)
			seen[m] = true
		}
	}
	for _, m := range appModels {
		if !seen[m] {
			result = append(result, m)
		}
	}

	return result
}

func printModelsAdded(appName string, models []string) {
	if len(models) == 1 {
		fmt.Fprintf(os.Stderr, "Added %s to %s\n", models[0], appName)
	} else {
		fmt.Fprintf(os.Stderr, "Added %d models to %s (default: %s)\n", len(models), appName, models[0])
	}
}

func runInApp(appName, modelName string) error {
	app, ok := GetApp(appName)
	if !ok {
		return fmt.Errorf("unknown app: %s", appName)
	}

	if err := app.CheckInstall(); err != nil {
		return err
	}

	if app.Setup != nil {
		models := []string{modelName}
		if config, err := LoadConnection(appName); err == nil && len(config.Models) > 0 {
			models = config.Models
		}
		if err := app.Setup(models); err != nil {
			return fmt.Errorf("setup failed: %w", err)
		}
	}

	proc := exec.Command(app.Command, app.Args(modelName)...)
	proc.Stdin = os.Stdin
	proc.Stdout = os.Stdout
	proc.Stderr = os.Stderr
	proc.Env = os.Environ()
	for _, env := range app.EnvVars(modelName) {
		proc.Env = append(proc.Env, fmt.Sprintf("%s=%s", env.Name, env.Value))
	}

	fmt.Fprintf(os.Stderr, "Launching %s with %s...\n", app.DisplayName, modelName)
	return proc.Run()
}

// handleCancelled prints the cancellation message and returns true if err is ErrCancelled.
// Returns false and the original error otherwise.
func handleCancelled(err error) (cancelled bool, origErr error) {
	if errors.Is(err, ErrCancelled) {
		fmt.Fprintln(os.Stderr, err.Error())
		return true, nil
	}
	return false, err
}

func ConnectCmd() *cobra.Command {
	var modelFlag string

	cmd := &cobra.Command{
		Use:   "connect [APP]",
		Short: "Configure an external app to use Ollama",
		Long: `Configure an external application to use Ollama as its backend.

Supported apps:
  claude    Claude Code
  codex     Codex
  droid     Droid
  opencode  OpenCode

Examples:
  ollama connect
  ollama connect claude
  ollama connect claude --model llama3.2`,
		Args:    cobra.MaximumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE: func(cmd *cobra.Command, args []string) error {
			var appName string
			if len(args) > 0 {
				appName = args[0]
			} else {
				var err error
				appName, err = selectApp()
				if cancelled, err := handleCancelled(err); cancelled {
					return nil
				} else if err != nil {
					return err
				}
			}

			if _, ok := GetApp(appName); !ok {
				return fmt.Errorf("unknown app: %s", appName)
			}

			var models []string
			if modelFlag != "" {
				models = []string{modelFlag}
			} else {
				var err error
				models, err = selectModelForConnect(cmd.Context(), appName)
				if cancelled, err := handleCancelled(err); cancelled {
					return nil
				} else if err != nil {
					return err
				}
			}

			if err := SaveConnection(appName, models); err != nil {
				return fmt.Errorf("failed to save: %w", err)
			}

			// Apply app-specific configuration (e.g., update OpenCode/Droid config files)
			app, _ := GetApp(appName)
			if app.Setup != nil {
				if err := app.Setup(models); err != nil {
					return fmt.Errorf("setup failed: %w", err)
				}
			}

			printModelsAdded(appName, models)

			if launch, _ := confirmLaunch(appName); launch {
				return runInApp(appName, models[0])
			}

			fmt.Fprintf(os.Stderr, "Run 'ollama launch %s' to start later with %s \n", strings.ToLower(appName), models[0])
			return nil
		},
	}

	cmd.Flags().StringVar(&modelFlag, "model", "", "Model to use")
	return cmd
}

func getAppConfiguredModel(appName string) string {
	switch strings.ToLower(appName) {
	case "opencode":
		return getOpenCodeConfiguredModel()
	case "droid":
		return getDroidConfiguredModel()
	default:
		return ""
	}
}

func getOrConfigureModel(ctx context.Context, appName string) (string, error) {
	if modelName := getAppConfiguredModel(appName); modelName != "" {
		return modelName, nil
	}

	if config, err := LoadConnection(appName); err == nil {
		return config.DefaultModel(), nil
	} else if !os.IsNotExist(err) {
		return "", err
	}

	models, err := selectModelForConnect(ctx, appName)
	if err != nil {
		return "", err
	}

	if err := SaveConnection(appName, models); err != nil {
		return "", fmt.Errorf("failed to save: %w", err)
	}

	printModelsAdded(appName, models)
	return models[0], nil
}

func LaunchCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "launch [APP]",
		Short: "Launch a configured app",
		Long: `Launch a configured application with Ollama as its backend.

If no app is specified, shows a list of configured apps to choose from.
If no apps have been configured, starts the connect flow.

Examples:
  ollama launch
  ollama launch claude`,
		Args:    cobra.MaximumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE: func(cmd *cobra.Command, args []string) error {
			var appName string
			if len(args) > 0 {
				appName = args[0]
			} else {
				selected, err := selectConnectedApp()
				if errors.Is(err, ErrCancelled) {
					return nil // Silent exit on cancel
				} else if err != nil {
					return err
				}
				if selected == "" {
					fmt.Fprintf(os.Stderr, "No apps configured. Let's set one up.\n\n")
					appName, err = selectApp()
					if errors.Is(err, ErrCancelled) {
						return nil
					} else if err != nil {
						return err
					}
				} else {
					appName = selected
				}
			}

			app, ok := GetApp(appName)
			if !ok {
				return fmt.Errorf("unknown app: %s", appName)
			}

			modelName, err := getOrConfigureModel(cmd.Context(), appName)
			if errors.Is(err, ErrCancelled) {
				return nil
			} else if err != nil {
				return err
			}

			return runInApp(app.Name, modelName)
		},
	}

	return cmd
}

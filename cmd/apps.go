package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
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
	Setup        func(model string) error
	CheckInstall func() error
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
	CheckInstall: func() error {
		if _, err := exec.LookPath("claude"); err != nil {
			return fmt.Errorf("claude is not installed. Install with: npm install -g @anthropic-ai/claude-code")
		}
		return nil
	},
}

var DroidConfig = &AppConfig{
	Name:        "Droid",
	DisplayName: "Droid",
	Command:     "droid",
	EnvVars:     func(model string) []EnvVar { return nil },
	Args:        func(model string) []string { return nil },
	Setup:       setupDroidSettings,
	CheckInstall: func() error {
		if _, err := exec.LookPath("droid"); err != nil {
			return fmt.Errorf("droid is not installed. Install from: https://docs.factory.ai/cli/install")
		}
		return nil
	},
}

var AppRegistry = map[string]*AppConfig{
	"claude":      ClaudeConfig,
	"claude-code": ClaudeConfig,
	"droid":       DroidConfig,
	"opencode":    OpenCodeConfig,
}

func GetApp(name string) (*AppConfig, bool) {
	app, ok := AppRegistry[strings.ToLower(name)]
	return app, ok
}

func getModelContextLength(model string) int {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return 8192
	}
	resp, err := client.Show(context.Background(), &api.ShowRequest{Model: model})
	if err != nil || resp.ModelInfo == nil {
		return 8192
	}
	arch, ok := resp.ModelInfo["general.architecture"].(string)
	if !ok {
		return 8192
	}
	if v, ok := resp.ModelInfo[fmt.Sprintf("%s.context_length", arch)].(float64); ok {
		return min(int(v), 128000)
	}
	return 8192
}

func setupDroidSettings(model string) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	settingsPath := filepath.Join(home, ".factory", "settings.json")
	if err := os.MkdirAll(filepath.Dir(settingsPath), 0755); err != nil {
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

	maxIndex := 0
	existingIdx := -1
	var existingID string
	for i, m := range customModels {
		if entry, ok := m.(map[string]any); ok {
			if entry["model"] == model {
				existingIdx = i
				if id, ok := entry["id"].(string); ok {
					existingID = id
				}
			}
			if idx, ok := entry["index"].(float64); ok && int(idx) > maxIndex {
				maxIndex = int(idx)
			}
		}
	}

	var modelID string
	newEntry := map[string]any{
		"model":           model,
		"displayName":     fmt.Sprintf("%s [Ollama]", model),
		"baseUrl":         "http://localhost:11434/v1",
		"apiKey":          "ollama",
		"provider":        "generic-chat-completion-api",
		"maxOutputTokens": getModelContextLength(model),
		"noImageSupport":  true,
	}

	if existingIdx >= 0 {
		modelID = existingID
		newEntry["id"] = existingID
		customModels[existingIdx] = newEntry
	} else {
		newIndex := maxIndex + 1
		modelID = fmt.Sprintf("custom:%s-[Ollama]-%d", model, newIndex)
		newEntry["id"] = modelID
		newEntry["index"] = newIndex
		customModels = append(customModels, newEntry)
	}
	settings["customModels"] = customModels

	sessionSettings, ok := settings["sessionDefaultSettings"].(map[string]any)
	if !ok {
		sessionSettings = make(map[string]any)
	}
	sessionSettings["model"] = modelID
	settings["sessionDefaultSettings"] = sessionSettings

	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(settingsPath, data, 0644)
}

func setupOpenCodeSettings(model string) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	configPath := filepath.Join(home, ".config", "opencode", "opencode.json")
	if err := os.MkdirAll(filepath.Dir(configPath), 0755); err != nil {
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

	models[model] = map[string]any{
		"name": fmt.Sprintf("%s [Ollama]", model),
	}

	ollama["models"] = models
	provider["ollama"] = ollama
	config["provider"] = provider

	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(configPath, data, 0644); err != nil {
		return err
	}

	statePath := filepath.Join(home, ".local", "state", "opencode", "model.json")
	if err := os.MkdirAll(filepath.Dir(statePath), 0755); err != nil {
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

	newRecent := []any{}
	for _, entry := range recent {
		if e, ok := entry.(map[string]any); ok {
			if e["providerID"] == "ollama" && e["modelID"] == model {
				continue
			}
		}
		newRecent = append(newRecent, entry)
	}

	newRecent = append([]any{map[string]any{
		"providerID": "ollama",
		"modelID":    model,
	}}, newRecent...)

	if len(newRecent) > 10 {
		newRecent = newRecent[:10]
	}

	state["recent"] = newRecent

	data, err = json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(statePath, data, 0644)
}

var OpenCodeConfig = &AppConfig{
	Name:        "OpenCode",
	DisplayName: "OpenCode",
	Command:     "opencode",
	EnvVars:     func(model string) []EnvVar { return nil },
	Args:        func(model string) []string { return nil },
	Setup:       setupOpenCodeSettings,
	CheckInstall: func() error {
		if _, err := exec.LookPath("opencode"); err != nil {
			return fmt.Errorf("opencode is not installed. Install from: https://opencode.ai")
		}
		return nil
	},
}

func getOpenCodeConfiguredModel() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}

	statePath := filepath.Join(home, ".local", "state", "opencode", "model.json")
	data, err := os.ReadFile(statePath)
	if err != nil {
		return ""
	}

	var state map[string]any
	if err := json.Unmarshal(data, &state); err != nil {
		return ""
	}

	recent, ok := state["recent"].([]any)
	if !ok || len(recent) == 0 {
		return ""
	}

	first, ok := recent[0].(map[string]any)
	if !ok {
		return ""
	}

	if first["providerID"] == "ollama" {
		if modelID, ok := first["modelID"].(string); ok {
			return modelID
		}
	}
	return ""
}

func getDroidConfiguredModel() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}

	settingsPath := filepath.Join(home, ".factory", "settings.json")
	data, err := os.ReadFile(settingsPath)
	if err != nil {
		return ""
	}

	var settings map[string]any
	if err := json.Unmarshal(data, &settings); err != nil {
		return ""
	}

	sessionSettings, ok := settings["sessionDefaultSettings"].(map[string]any)
	if !ok {
		return ""
	}

	modelID, ok := sessionSettings["model"].(string)
	if !ok || modelID == "" {
		return ""
	}

	customModels, ok := settings["customModels"].([]any)
	if !ok {
		return ""
	}

	for _, m := range customModels {
		entry, ok := m.(map[string]any)
		if !ok {
			continue
		}
		if entry["id"] == modelID {
			if model, ok := entry["model"].(string); ok {
				return model
			}
		}
	}
	return ""
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
		if err := app.Setup(modelName); err != nil {
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

func ConnectCmd() *cobra.Command {
	var modelName string

	cmd := &cobra.Command{
		Use:   "connect [APP]",
		Short: "Configure an external app to use Ollama",
		Long: `Configure an external application to use Ollama as its backend.

Supported apps:
  claude    Claude Code
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
				if err != nil {
					return err
				}
			}

			if _, ok := GetApp(appName); !ok {
				return fmt.Errorf("unknown app: %s", appName)
			}

			if modelName == "" {
				var err error
				modelName, err = selectModelForConnect(cmd.Context(), "")
				if err != nil {
					return err
				}
			}

			if err := SaveConnection(appName, modelName); err != nil {
				return fmt.Errorf("failed to save: %w", err)
			}

			fmt.Fprintf(os.Stderr, "Added %s to %s\n", modelName, appName)

			if launch, _ := confirmLaunch(appName); launch {
				return runInApp(appName, modelName)
			}

			fmt.Fprintf(os.Stderr, "Run 'ollama launch %s' to start later\n", strings.ToLower(appName))
			return nil
		},
	}

	cmd.Flags().StringVar(&modelName, "model", "", "Model to use")
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
				if err != nil {
					return err
				}
				if selected == "" {
					// No connected apps, start connect flow
					fmt.Fprintf(os.Stderr, "No apps configured. Let's set one up.\n\n")
					appName, err = selectApp()
					if err != nil {
						return err
					}

					modelName, err := selectModelForConnect(cmd.Context(), "")
					if err != nil {
						return err
					}

					if err := SaveConnection(appName, modelName); err != nil {
						return fmt.Errorf("failed to save: %w", err)
					}

					fmt.Fprintf(os.Stderr, "Added %s to %s\n", modelName, appName)
					return runInApp(appName, modelName)
				}
				appName = selected
			}

			app, ok := GetApp(appName)
			if !ok {
				return fmt.Errorf("unknown app: %s", appName)
			}

			// Check app's own config first
			modelName := getAppConfiguredModel(appName)

			// Fall back to our saved connection config
			if modelName == "" {
				config, err := LoadConnection(appName)
				if err != nil {
					if os.IsNotExist(err) {
						// No config, drop into connect flow
						modelName, err = selectModelForConnect(cmd.Context(), "")
						if err != nil {
							return err
						}

						if err := SaveConnection(appName, modelName); err != nil {
							return fmt.Errorf("failed to save: %w", err)
						}

						fmt.Fprintf(os.Stderr, "Added %s to %s\n", modelName, appName)
					} else {
						return err
					}
				} else {
					modelName = config.Model
				}
			}

			return runInApp(app.Name, modelName)
		},
	}

	return cmd
}

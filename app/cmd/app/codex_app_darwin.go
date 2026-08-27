//go:build darwin

package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/launch"
	modelpkg "github.com/ollama/ollama/types/model"
)

const (
	codexDesktopIntegrationName = "chatgpt"
	codexDesktopMaxModels       = 5
)

type codexDesktopController interface {
	Installed() bool
	OllamaProfileRunning() bool
	LaunchOllamaProfileFromDesktop(string, []launch.LaunchModel) error
	StopOllamaProfileFromDesktop() error
	Onboard() error
}

var (
	codexDesktop              codexDesktopController = &launch.CodexApp{}
	codexDesktopClientFactory                        = api.ClientFromEnvironment
	codexDesktopLoadModels                           = loadCodexDesktopModels
	codexDesktopMu            sync.Mutex
)

type codexDesktopStatus struct {
	Supported bool     `json:"supported"`
	Installed bool     `json:"installed"`
	Connected bool     `json:"connected"`
	Running   bool     `json:"running"`
	Model     string   `json:"model,omitempty"`
	Models    []string `json:"models,omitempty"`
	MaxModels int      `json:"maxModels"`
}

type codexDesktopActionResult struct {
	Status codexDesktopStatus `json:"status"`
	Error  string             `json:"error,omitempty"`
}

type codexDesktopModelsSettings struct {
	Supported bool     `json:"supported"`
	Installed bool     `json:"installed"`
	Running   bool     `json:"running"`
	Selected  []string `json:"selected"`
	Available []string `json:"available"`
	MaxModels int      `json:"maxModels"`
}

type codexDesktopModelsSettingsResult struct {
	Settings codexDesktopModelsSettings `json:"settings"`
	Error    string                     `json:"error,omitempty"`
}

func getCodexDesktopStatus() codexDesktopStatus {
	running := codexDesktop.OllamaProfileRunning()
	var models []string
	if saved, err := config.LoadIntegration(codexDesktopIntegrationName); err == nil && len(saved.Models) > 0 {
		models = append([]string(nil), saved.Models...)
	}
	model := ""
	if len(models) > 0 {
		model = models[0]
	}
	return codexDesktopStatus{
		Supported: true,
		Installed: codexDesktop.Installed(),
		Connected: running,
		Running:   running,
		Model:     model,
		Models:    models,
		MaxModels: codexDesktopMaxModels,
	}
}

func setCodexDesktopConnection(enabled bool) error {
	codexDesktopMu.Lock()
	defer codexDesktopMu.Unlock()

	if !enabled {
		return codexDesktop.StopOllamaProfileFromDesktop()
	}
	if !codexDesktop.Installed() {
		return errors.New("ChatGPT is not installed")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	selected := config.IntegrationModels(codexDesktopIntegrationName)
	primary, models, err := codexDesktopLoadModels(ctx, selected)
	if err != nil {
		return err
	}
	if err := codexDesktop.LaunchOllamaProfileFromDesktop(primary, models); err != nil {
		return err
	}
	if err := config.SaveIntegration(codexDesktopIntegrationName, codexDesktopModelNames(models)); err != nil {
		return fmt.Errorf("save ChatGPT integration: %w", err)
	}
	if err := codexDesktop.Onboard(); err != nil {
		return fmt.Errorf("save ChatGPT integration state: %w", err)
	}
	return nil
}

func getCodexDesktopModelsSettings() (codexDesktopModelsSettings, error) {
	settings := codexDesktopModelsSettings{
		Supported: true,
		Installed: codexDesktop.Installed(),
		Running:   codexDesktop.OllamaProfileRunning(),
		Selected:  []string{},
		Available: []string{},
		MaxModels: codexDesktopMaxModels,
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	available, err := loadCodexDesktopAvailableModels(ctx)
	if err != nil {
		return settings, err
	}
	settings.Available = codexDesktopModelNames(available)
	settings.Selected = config.IntegrationModels(codexDesktopIntegrationName)
	if len(settings.Selected) == 0 {
		_, defaults, selectErr := selectCodexDesktopModels(nil, available)
		if selectErr != nil {
			return settings, selectErr
		}
		settings.Selected = codexDesktopModelNames(defaults)
	}
	if len(settings.Selected) > codexDesktopMaxModels {
		settings.Selected = settings.Selected[:codexDesktopMaxModels]
	}
	return settings, nil
}

func applyCodexDesktopModels(selected []string) error {
	codexDesktopMu.Lock()
	defer codexDesktopMu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	primary, models, err := codexDesktopLoadModels(ctx, selected)
	if err != nil {
		return err
	}
	selected = codexDesktopModelNames(models)
	previous := config.IntegrationModels(codexDesktopIntegrationName)
	if err := config.SaveIntegration(codexDesktopIntegrationName, selected); err != nil {
		return fmt.Errorf("save ChatGPT models: %w", err)
	}
	wasRunning := codexDesktop.OllamaProfileRunning()
	if wasRunning {
		if err := codexDesktop.StopOllamaProfileFromDesktop(); err != nil {
			_ = config.SaveIntegration(codexDesktopIntegrationName, previous)
			return err
		}
	}
	if err := codexDesktop.LaunchOllamaProfileFromDesktop(primary, models); err == nil {
		return nil
	} else if !wasRunning {
		_ = config.SaveIntegration(codexDesktopIntegrationName, previous)
		return fmt.Errorf("start ChatGPT with selected models: %w", err)
	} else {
		applyErr := err
		_ = config.SaveIntegration(codexDesktopIntegrationName, previous)
		rollbackCtx, rollbackCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer rollbackCancel()
		rollbackPrimary, rollbackModels, rollbackErr := codexDesktopLoadModels(rollbackCtx, previous)
		if rollbackErr == nil {
			rollbackErr = codexDesktop.LaunchOllamaProfileFromDesktop(rollbackPrimary, rollbackModels)
		}
		if rollbackErr != nil {
			return fmt.Errorf("apply ChatGPT models: %v; restore previous profile: %w", applyErr, rollbackErr)
		}
		return fmt.Errorf("apply ChatGPT models: %w", applyErr)
	}
}

func loadCodexDesktopModels(ctx context.Context, selected []string) (string, []launch.LaunchModel, error) {
	available, err := loadCodexDesktopAvailableModels(ctx)
	if err != nil {
		return "", nil, err
	}
	return selectCodexDesktopModels(selected, available)
}

func loadCodexDesktopAvailableModels(ctx context.Context) ([]launch.LaunchModel, error) {
	client, err := codexDesktopClientFactory()
	if err != nil {
		return nil, err
	}

	var recommendations []api.ModelRecommendation
	if response, recommendationErr := client.ModelRecommendationsExperimental(ctx); recommendationErr == nil {
		recommendations = response.Recommendations
	}
	var listed []api.ListModelResponse
	if response, listErr := client.List(ctx); listErr == nil {
		listed = response.Models
	}

	models := codexDesktopAvailableModels(recommendations, listed)
	if len(models) == 0 {
		return nil, errors.New("no Ollama models are available for ChatGPT")
	}
	return models, nil
}

func buildCodexDesktopModels(selected []string, recommendations []api.ModelRecommendation, listed []api.ListModelResponse) (string, []launch.LaunchModel, error) {
	available := codexDesktopAvailableModels(recommendations, listed)
	return selectCodexDesktopModels(selected, available)
}

func codexDesktopAvailableModels(recommendations []api.ModelRecommendation, listed []api.ListModelResponse) []launch.LaunchModel {
	installed := make(map[string]api.ListModelResponse, len(listed))
	for _, model := range listed {
		for _, name := range []string{model.Name, model.Model} {
			if key := codexDesktopModelKey(name); key != "" {
				installed[key] = model
			}
		}
	}

	models := make([]launch.LaunchModel, 0, len(recommendations)+len(listed)+1)
	seen := make(map[string]bool, cap(models))
	add := func(model launch.LaunchModel) {
		model.Name = strings.TrimSpace(model.Name)
		key := codexDesktopModelKey(model.Name)
		if key == "" || seen[key] {
			return
		}
		seen[key] = true
		models = append(models, model)
	}

	for _, recommendation := range recommendations {
		name := strings.TrimSpace(recommendation.Model)
		if name == "" {
			continue
		}
		if listedModel, ok := installed[codexDesktopModelKey(name)]; ok {
			add(codexDesktopLaunchModel(listedModel))
			continue
		}
		if !codexDesktopCloudModel(name) {
			continue
		}
		add(launch.LaunchModel{
			Name:            name,
			Remote:          true,
			ContextLength:   recommendation.ContextLength,
			MaxOutputTokens: recommendation.MaxOutputTokens,
		})
	}
	for _, model := range listed {
		add(codexDesktopLaunchModel(model))
	}

	return models
}

func selectCodexDesktopModels(selected []string, available []launch.LaunchModel) (string, []launch.LaunchModel, error) {
	byName := make(map[string]launch.LaunchModel, len(available))
	for _, model := range available {
		byName[codexDesktopModelKey(model.Name)] = model
	}

	if len(selected) > codexDesktopMaxModels {
		return "", nil, fmt.Errorf("choose up to %d models for ChatGPT", codexDesktopMaxModels)
	}
	resolved := make([]launch.LaunchModel, 0, codexDesktopMaxModels)
	seen := make(map[string]bool, codexDesktopMaxModels)
	for _, name := range selected {
		name = strings.TrimSpace(name)
		key := codexDesktopModelKey(name)
		if key == "" || seen[key] {
			continue
		}
		model, ok := byName[key]
		if !ok {
			return "", nil, fmt.Errorf("ChatGPT model %q is not available", name)
		}
		seen[key] = true
		resolved = append(resolved, model)
	}

	if len(selected) == 0 {
		for _, model := range available {
			if len(resolved) == codexDesktopMaxModels {
				break
			}
			key := codexDesktopModelKey(model.Name)
			if key == "" || seen[key] {
				continue
			}
			seen[key] = true
			resolved = append(resolved, model)
		}
	}
	if len(resolved) == 0 {
		return "", nil, errors.New("choose at least one available Ollama model for ChatGPT")
	}
	return resolved[0].Name, resolved, nil
}

func codexDesktopModelNames(models []launch.LaunchModel) []string {
	names := make([]string, 0, len(models))
	for _, model := range models {
		if name := strings.TrimSpace(model.Name); name != "" {
			names = append(names, name)
		}
	}
	return names
}

func codexDesktopLaunchModel(model api.ListModelResponse) launch.LaunchModel {
	name := strings.TrimSpace(model.Name)
	if name == "" {
		name = strings.TrimSpace(model.Model)
	}
	return launch.LaunchModel{
		Name:            name,
		Remote:          model.RemoteModel != "" || codexDesktopCloudModel(name),
		Capabilities:    append([]modelpkg.Capability(nil), model.Capabilities...),
		ContextLength:   model.Details.ContextLength,
		EmbeddingLength: model.Details.EmbeddingLength,
		Size:            model.Size,
		Details:         model.Details,
	}
}

func codexDesktopModelKey(name string) string {
	return strings.TrimSuffix(strings.TrimSpace(name), ":latest")
}

func codexDesktopCloudModel(name string) bool {
	name = strings.ToLower(strings.TrimSpace(name))
	return strings.HasSuffix(name, ":cloud") || strings.HasSuffix(name, "-cloud")
}

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

var errCodexDesktopRestartConfirmationRequired = errors.New("ChatGPT restart confirmation is required before changing its profile")

type codexDesktopController interface {
	Installed() bool
	OllamaConfigured() bool
	Running() bool
	OllamaRequestCount() uint64
	UseOllamaFromDesktop(string, []launch.LaunchModel) error
	RestoreFromDesktop() error
	RestoreForShutdown(context.Context) error
	Onboard() error
}

var (
	codexDesktop                     codexDesktopController = &launch.CodexApp{}
	codexDesktopClientFactory                               = api.ClientFromEnvironment
	codexDesktopLoadModels                                  = loadCodexDesktopModels
	codexDesktopLoadConnectionModels                        = loadCodexDesktopConnectionModels
	codexDesktopCloudModels                                 = loadCodexDesktopAccountCloudModels
	codexDesktopModelLoadAttempts                           = 4
	codexDesktopModelRetryWait                              = 250 * time.Millisecond
	codexDesktopMu                   sync.Mutex
)

type codexDesktopStatus struct {
	Supported bool     `json:"supported"`
	Installed bool     `json:"installed"`
	Connected bool     `json:"connected"`
	Running   bool     `json:"running"`
	Model     string   `json:"model,omitempty"`
	Models    []string `json:"models,omitempty"`
	MaxModels int      `json:"maxModels"`
	Requests  uint64   `json:"requests"`
}

type codexDesktopActionResult struct {
	Status                      codexDesktopStatus `json:"status"`
	Error                       string             `json:"error,omitempty"`
	RestartConfirmationRequired bool               `json:"restartConfirmationRequired,omitempty"`
}

type codexDesktopInstallResult string

const (
	codexDesktopInstallCancelled codexDesktopInstallResult = "cancelled"
	codexDesktopInstallerOpened  codexDesktopInstallResult = "opened"
	codexDesktopInstallFailed    codexDesktopInstallResult = "failed"
)

type codexDesktopModelsSettings struct {
	Supported bool     `json:"supported"`
	Installed bool     `json:"installed"`
	Connected bool     `json:"connected"`
	Running   bool     `json:"running"`
	Selected  []string `json:"selected"`
	Available []string `json:"available"`
	MaxModels int      `json:"maxModels"`
}

type codexDesktopModelsSettingsResult struct {
	Settings codexDesktopModelsSettings `json:"settings"`
	Error    string                     `json:"error,omitempty"`
}

type codexDesktopModelInventory struct {
	Available []launch.LaunchModel
}

func getCodexDesktopStatus() codexDesktopStatus {
	connected := codexDesktop.OllamaConfigured()
	requests := uint64(0)
	if connected {
		requests = codexDesktop.OllamaRequestCount()
	}
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
		Connected: connected,
		Running:   codexDesktop.Running(),
		Model:     model,
		Models:    models,
		MaxModels: codexDesktopMaxModels,
		Requests:  requests,
	}
}

func setCodexDesktopConnection(enabled, restartConfirmed bool) error {
	codexDesktopMu.Lock()
	defer codexDesktopMu.Unlock()

	if !enabled {
		if codexDesktop.Running() && !restartConfirmed {
			return errCodexDesktopRestartConfirmationRequired
		}
		return codexDesktop.RestoreFromDesktop()
	}
	if !codexDesktop.Installed() {
		return errors.New("ChatGPT is not installed")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	selected := config.IntegrationModels(codexDesktopIntegrationName)
	primary, models, err := codexDesktopLoadConnectionModels(ctx, selected)
	if err != nil {
		return err
	}
	// Resolve model availability before asking to interrupt a running task, but
	// do not persist or change the ChatGPT profile until consent is explicit.
	if codexDesktop.Running() && !restartConfirmed {
		return errCodexDesktopRestartConfirmationRequired
	}
	previous := config.IntegrationModels(codexDesktopIntegrationName)
	selected = codexDesktopModelNames(models)
	if err := config.SaveIntegration(codexDesktopIntegrationName, selected); err != nil {
		return fmt.Errorf("save ChatGPT integration: %w", err)
	}
	if err := codexDesktop.Onboard(); err != nil {
		_ = config.SaveIntegration(codexDesktopIntegrationName, previous)
		return fmt.Errorf("save ChatGPT integration state: %w", err)
	}
	if err := codexDesktop.UseOllamaFromDesktop(primary, models); err != nil {
		_ = config.SaveIntegration(codexDesktopIntegrationName, previous)
		if codexDesktop.OllamaConfigured() {
			if restoreErr := codexDesktop.RestoreFromDesktop(); restoreErr != nil {
				return errors.Join(err, fmt.Errorf("restore ChatGPT after failed update: %w", restoreErr))
			}
		}
		return err
	}
	return nil
}

func getCodexDesktopModelsSettings() (codexDesktopModelsSettings, error) {
	settings := codexDesktopModelsSettings{
		Supported: true,
		Installed: codexDesktop.Installed(),
		Connected: codexDesktop.OllamaConfigured(),
		Running:   codexDesktop.Running(),
		Selected:  []string{},
		Available: []string{},
		MaxModels: codexDesktopMaxModels,
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	inventory, err := loadCodexDesktopModelInventory(ctx)
	if err != nil {
		return settings, err
	}
	settings.Available = codexDesktopModelNames(inventory.Available)
	settings.Selected = config.IntegrationModels(codexDesktopIntegrationName)
	if len(settings.Selected) == 0 {
		settings.Selected = codexDesktopModelNames(codexDesktopDefaultModels(inventory))
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
	wasConfigured := codexDesktop.OllamaConfigured()
	if err := codexDesktop.UseOllamaFromDesktop(primary, models); err == nil {
		return nil
	} else if !wasConfigured {
		if codexDesktop.OllamaConfigured() {
			if restoreErr := codexDesktop.RestoreFromDesktop(); restoreErr != nil {
				_ = config.SaveIntegration(codexDesktopIntegrationName, previous)
				return errors.Join(err, fmt.Errorf("restore ChatGPT after failed update: %w", restoreErr))
			}
		}
		_ = config.SaveIntegration(codexDesktopIntegrationName, previous)
		return fmt.Errorf("start ChatGPT with selected Ollama models: %w", err)
	} else {
		applyErr := err
		_ = config.SaveIntegration(codexDesktopIntegrationName, previous)
		rollbackCtx, rollbackCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer rollbackCancel()
		rollbackPrimary, rollbackModels, rollbackErr := codexDesktopLoadModels(rollbackCtx, previous)
		if rollbackErr == nil {
			rollbackErr = codexDesktop.UseOllamaFromDesktop(rollbackPrimary, rollbackModels)
		}
		if rollbackErr != nil {
			return fmt.Errorf("apply ChatGPT models: %v; restore previous profile: %w", applyErr, rollbackErr)
		}
		return fmt.Errorf("apply ChatGPT models: %w", applyErr)
	}
}

func loadCodexDesktopModels(ctx context.Context, selected []string) (string, []launch.LaunchModel, error) {
	inventory, err := loadCodexDesktopModelInventory(ctx)
	if err != nil {
		return "", nil, err
	}
	if len(selected) == 0 {
		selected = codexDesktopModelNames(codexDesktopDefaultModels(inventory))
	}
	primary, models, err := selectCodexDesktopModels(selected, inventory.Available)
	if err != nil {
		return "", nil, err
	}
	return primary, hydrateCodexDesktopModelCapabilities(ctx, models), nil
}

func loadCodexDesktopConnectionModels(ctx context.Context, selected []string) (string, []launch.LaunchModel, error) {
	inventory, err := loadCodexDesktopModelInventory(ctx)
	if err != nil {
		return "", nil, err
	}
	defaults := codexDesktopDefaultModels(inventory)
	if len(selected) == 0 {
		selected = codexDesktopModelNames(defaults)
	}
	primary, models, err := reconcileCodexDesktopModels(selected, inventory.Available, defaults)
	if err != nil {
		return "", nil, err
	}
	return primary, hydrateCodexDesktopModelCapabilities(ctx, models), nil
}

// hydrateCodexDesktopModelCapabilities fills metadata that may be absent from
// account-only cloud inventory. In particular, some cloud models are returned
// by /api/show but not /api/tags, which otherwise hides their Thinking levels
// from the generated ChatGPT catalog.
func hydrateCodexDesktopModelCapabilities(ctx context.Context, models []launch.LaunchModel) []launch.LaunchModel {
	client, err := codexDesktopClientFactory()
	if err != nil {
		return models
	}

	hydrated := append([]launch.LaunchModel(nil), models...)
	for i := range hydrated {
		response, err := client.Show(ctx, &api.ShowRequest{Model: hydrated[i].Name})
		if err != nil {
			continue
		}
		if len(response.Capabilities) > 0 {
			hydrated[i].Capabilities = append([]modelpkg.Capability(nil), response.Capabilities...)
		}
		if response.Details.Family != "" || len(response.Details.Families) > 0 {
			hydrated[i].Details = response.Details
		}
	}
	return hydrated
}

func loadCodexDesktopAvailableModels(ctx context.Context) ([]launch.LaunchModel, error) {
	inventory, err := loadCodexDesktopModelInventory(ctx)
	return inventory.Available, err
}

func loadCodexDesktopModelInventory(ctx context.Context) (codexDesktopModelInventory, error) {
	client, err := codexDesktopClientFactory()
	if err != nil {
		return codexDesktopModelInventory{}, err
	}

	for attempt := 0; attempt < codexDesktopModelLoadAttempts; attempt++ {
		var listed []api.ListModelResponse
		if response, listErr := client.List(ctx); listErr == nil {
			listed = response.Models
		}
		var accountCloud []string
		if names, cloudErr := codexDesktopCloudModels(ctx); cloudErr == nil {
			accountCloud = names
		}

		models := codexDesktopAvailableModels(listed, accountCloud)
		if len(models) > 0 {
			return codexDesktopModelInventory{Available: models}, nil
		}
		if attempt+1 == codexDesktopModelLoadAttempts {
			break
		}
		timer := time.NewTimer(codexDesktopModelRetryWait)
		select {
		case <-ctx.Done():
			timer.Stop()
			return codexDesktopModelInventory{}, ctx.Err()
		case <-timer.C:
		}
	}
	return codexDesktopModelInventory{}, errors.New("no Ollama models are available for ChatGPT")
}

func loadCodexDesktopAccountCloudModels(ctx context.Context) ([]string, error) {
	models, err := currentClaudeDesktopCloudModels(ctx)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(models))
	for _, model := range models {
		name := strings.TrimSpace(model.OllamaModel)
		if name == "" {
			name = strings.TrimSpace(model.Name)
		}
		if name != "" {
			names = append(names, name)
		}
	}
	return names, nil
}

func buildCodexDesktopModels(selected []string, listed []api.ListModelResponse, accountCloud []string) (string, []launch.LaunchModel, error) {
	available := codexDesktopAvailableModels(listed, accountCloud)
	return selectCodexDesktopModels(selected, available)
}

func codexDesktopAvailableModels(listed []api.ListModelResponse, accountCloud []string) []launch.LaunchModel {
	installed := make(map[string]api.ListModelResponse, len(listed))
	for _, model := range listed {
		for _, name := range []string{model.Name, model.Model} {
			if key := codexDesktopModelKey(name); key != "" {
				installed[key] = model
			}
		}
	}
	accountCloudSet := make(map[string]bool, len(accountCloud))
	for _, name := range accountCloud {
		if key := codexDesktopModelKey(name); key != "" {
			accountCloudSet[key] = true
		}
	}
	models := make([]launch.LaunchModel, 0, len(listed)+len(accountCloud))
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

	for _, model := range listed {
		if codexDesktopListedModelIsCloud(model) && !accountCloudSet[codexDesktopModelKey(model.Name)] && !accountCloudSet[codexDesktopModelKey(model.Model)] {
			continue
		}
		add(codexDesktopLaunchModel(model))
	}
	for _, name := range accountCloud {
		key := codexDesktopModelKey(name)
		if listedModel, ok := installed[key]; ok {
			add(codexDesktopLaunchModel(listedModel))
			continue
		}
		add(launch.LaunchModel{Name: strings.TrimSpace(name), Remote: true})
	}

	return models
}

func codexDesktopDefaultModels(inventory codexDesktopModelInventory) []launch.LaunchModel {
	return append([]launch.LaunchModel(nil), inventory.Available[:min(len(inventory.Available), codexDesktopMaxModels)]...)
}

func codexDesktopListedModelIsCloud(model api.ListModelResponse) bool {
	return model.RemoteModel != "" || model.RemoteHost != "" ||
		codexDesktopCloudModel(model.Name) ||
		codexDesktopCloudModel(model.Model)
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

// reconcileCodexDesktopModels is used only when reopening the saved menu-bar
// profile. Models that have since been removed or lost account access should
// not prevent the remaining valid selection from starting. Explicit changes
// from Settings continue to use selectCodexDesktopModels and remain strict.
func reconcileCodexDesktopModels(selected []string, available, defaults []launch.LaunchModel) (string, []launch.LaunchModel, error) {
	if len(selected) == 0 {
		if len(defaults) > 0 {
			return selectCodexDesktopModels(codexDesktopModelNames(defaults), available)
		}
		return selectCodexDesktopModels(nil, available)
	}

	byName := make(map[string]launch.LaunchModel, len(available))
	for _, model := range available {
		byName[codexDesktopModelKey(model.Name)] = model
	}
	resolved := make([]launch.LaunchModel, 0, min(len(selected), codexDesktopMaxModels))
	seen := make(map[string]bool, codexDesktopMaxModels)
	for _, name := range selected {
		key := codexDesktopModelKey(name)
		model, ok := byName[key]
		if key == "" || !ok || seen[key] {
			continue
		}
		seen[key] = true
		resolved = append(resolved, model)
		if len(resolved) == codexDesktopMaxModels {
			break
		}
	}
	if len(resolved) == 0 {
		if len(defaults) > 0 {
			return selectCodexDesktopModels(codexDesktopModelNames(defaults), available)
		}
		return selectCodexDesktopModels(nil, available)
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
		Remote:          model.RemoteModel != "" || model.RemoteHost != "" || codexDesktopCloudModel(name),
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

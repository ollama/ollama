//go:build darwin

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	appui "github.com/ollama/ollama/app/ui"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/internal/modelref"
	"github.com/ollama/ollama/internal/proxy"
	modelpkg "github.com/ollama/ollama/types/model"
)

const (
	codexDesktopIntegrationName        = "chatgpt"
	codexDesktopMaxModels              = 5
	codexDesktopRecommendationsMaxBody = 1 << 20
)

var errCodexDesktopRestartConfirmationRequired = launch.ErrCodexAppRestartConfirmationRequired

type codexDesktopController interface {
	Installed() bool
	OllamaConfigured() bool
	Running() bool
	OllamaRequestCount() uint64
	UseOllamaFromDesktop(string, []launch.LaunchModel, bool) error
	UpdateOllamaModelsFromDesktop(string, []launch.LaunchModel, bool) error
	RestoreFromDesktop(bool) error
	RestartFromDesktop(bool) error
	Onboard() error
}

var (
	codexDesktop                        codexDesktopController = &launch.CodexApp{}
	codexDesktopClientFactory                                  = api.ClientFromEnvironment
	codexDesktopLoadModels                                     = loadCodexDesktopModels
	codexDesktopLoadConnectionModels                           = loadCodexDesktopConnectionModels
	codexDesktopCloudModels                                    = loadCodexDesktopAccountCloudModels
	codexDesktopRecommendations                                = loadCodexDesktopRecommendations
	codexDesktopAccessState                                    = currentClaudeDesktopAccessState
	codexDesktopRecommendationsClient                          = &http.Client{Timeout: 3 * time.Second}
	codexDesktopRecommendationsEndpoint                        = func() string {
		return strings.TrimRight(appui.OllamaDotCom, "/") + "/api/experimental/model-recommendations?app=codex-desktop"
	}
	codexDesktopModelLoadAttempts = 20
	codexDesktopModelRetryWait    = 250 * time.Millisecond
	codexDesktopMu                sync.Mutex
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
	Supported bool `json:"supported"`
	Installed bool `json:"installed"`
	Connected bool `json:"connected"`
	Running   bool `json:"running"`
	// UsesDefaults keeps recommendations implicit without overwriting saved choices.
	UsesDefaults bool                      `json:"usesDefaults"`
	Selected     []string                  `json:"selected"`
	Available    []string                  `json:"available"`
	Models       []codexDesktopModelStatus `json:"models"`
	MaxModels    int                       `json:"maxModels"`
}

type codexDesktopModelStatus struct {
	Name         string `json:"name"`
	DisplayName  string `json:"displayName"`
	Description  string `json:"description,omitempty"`
	Recommended  bool   `json:"recommended,omitempty"`
	Selected     bool   `json:"selected"`
	Availability string `json:"availability"`
	Reason       string `json:"reason,omitempty"`
	RequiredPlan string `json:"requiredPlan,omitempty"`
}

type codexDesktopModelsSettingsResult struct {
	Settings                    codexDesktopModelsSettings `json:"settings"`
	Error                       string                     `json:"error,omitempty"`
	Warning                     string                     `json:"warning,omitempty"`
	RestartConfirmationRequired bool                       `json:"restartConfirmationRequired,omitempty"`
}

type codexDesktopModelInventory struct {
	Available      []launch.LaunchModel
	Catalog        []codexDesktopCatalogModel
	Defaults       []launch.LaunchModel
	DefaultPrimary string
}

type codexDesktopCatalogModel struct {
	Model        launch.LaunchModel
	DisplayName  string
	Description  string
	Recommended  bool
	Availability proxy.ClaudeDesktopAvailability
	Reason       proxy.ClaudeDesktopAccessReason
	RequiredPlan string
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

	if enabled == codexDesktop.OllamaConfigured() {
		return nil
	}
	if !enabled {
		if codexDesktop.Running() && !restartConfirmed {
			return errCodexDesktopRestartConfirmationRequired
		}
		return codexDesktop.RestoreFromDesktop(restartConfirmed)
	}
	if !codexDesktop.Installed() {
		return errors.New("ChatGPT is not installed")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	savedSelection := config.IntegrationModels(codexDesktopIntegrationName)
	primary, models, err := codexDesktopLoadConnectionModels(ctx, savedSelection)
	if err != nil {
		return err
	}
	// Validate before requesting a restart; do not change the profile without consent.
	if codexDesktop.Running() && !restartConfirmed {
		return errCodexDesktopRestartConfirmationRequired
	}
	previous := config.IntegrationModels(codexDesktopIntegrationName)
	if err := config.SaveIntegration(codexDesktopIntegrationName, savedSelection); err != nil {
		return fmt.Errorf("save ChatGPT integration: %w", err)
	}
	if err := codexDesktop.Onboard(); err != nil {
		_ = config.SaveIntegration(codexDesktopIntegrationName, previous)
		return fmt.Errorf("save ChatGPT integration state: %w", err)
	}
	if err := codexDesktop.UseOllamaFromDesktop(primary, models, restartConfirmed); err != nil {
		_ = config.SaveIntegration(codexDesktopIntegrationName, previous)
		if errors.Is(err, errCodexDesktopRestartConfirmationRequired) {
			return err
		}
		if codexDesktop.OllamaConfigured() {
			if restoreErr := codexDesktop.RestoreFromDesktop(true); restoreErr != nil {
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
		Models:    []codexDesktopModelStatus{},
		MaxModels: codexDesktopMaxModels,
	}
	// Keep restart available when inventory cannot be loaded.
	settings.Selected = config.IntegrationModels(codexDesktopIntegrationName)
	settings.UsesDefaults = len(settings.Selected) == 0
	if len(settings.Selected) > codexDesktopMaxModels {
		settings.Selected = settings.Selected[:codexDesktopMaxModels]
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	inventory, err := loadCodexDesktopModelInventory(ctx)
	if err != nil {
		return settings, err
	}
	settings.Available = codexDesktopModelNames(inventory.Available)
	if len(settings.Selected) == 0 {
		settings.Selected = codexDesktopModelNames(codexDesktopDefaultModels(inventory))
	}
	settings.Models = codexDesktopModelStatuses(inventory, settings.Selected)
	return settings, nil
}

func applyCodexDesktopModels(selected []string, restartConfirmed bool) error {
	codexDesktopMu.Lock()
	defer codexDesktopMu.Unlock()
	return applyCodexDesktopModelsLocked(selected, restartConfirmed, true)
}

func resetCodexDesktopModels() error {
	codexDesktopMu.Lock()
	defer codexDesktopMu.Unlock()

	// Reset preferences without enabling the integration or restarting ChatGPT.
	if len(config.IntegrationModels(codexDesktopIntegrationName)) == 0 && !codexDesktop.OllamaConfigured() {
		return nil
	}
	if err := config.SaveIntegration(codexDesktopIntegrationName, nil); err != nil {
		return fmt.Errorf("reset ChatGPT models: %w", err)
	}
	return nil
}

func applyCodexDesktopModelsLocked(selected []string, restartConfirmed, openWhenStopped bool) error {
	previous := config.IntegrationModels(codexDesktopIntegrationName)
	savedSelection := append([]string(nil), selected...)
	wasConfigured := codexDesktop.OllamaConfigured()
	selectionUnchanged := slices.Equal(savedSelection, previous)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	primary, models, err := codexDesktopLoadModels(ctx, selected)
	if err != nil {
		if openWhenStopped && wasConfigured && selectionUnchanged {
			return codexDesktop.RestartFromDesktop(restartConfirmed)
		}
		return err
	}
	if !wasConfigured && !openWhenStopped {
		if err := config.SaveIntegration(codexDesktopIntegrationName, savedSelection); err != nil {
			return fmt.Errorf("save ChatGPT models: %w", err)
		}
		return nil
	}
	running := codexDesktop.Running()
	if running && !restartConfirmed {
		return errCodexDesktopRestartConfirmationRequired
	}
	if err := config.SaveIntegration(codexDesktopIntegrationName, savedSelection); err != nil {
		return fmt.Errorf("save ChatGPT models: %w", err)
	}
	updateModels := codexDesktop.UseOllamaFromDesktop
	if !openWhenStopped {
		updateModels = codexDesktop.UpdateOllamaModelsFromDesktop
	}
	if err := updateModels(primary, models, restartConfirmed); err == nil {
		return nil
	} else if errors.Is(err, errCodexDesktopRestartConfirmationRequired) {
		_ = config.SaveIntegration(codexDesktopIntegrationName, previous)
		return err
	} else if !wasConfigured {
		if codexDesktop.OllamaConfigured() {
			if restoreErr := codexDesktop.RestoreFromDesktop(true); restoreErr != nil {
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
			rollbackErr = updateModels(rollbackPrimary, rollbackModels, true)
		}
		if rollbackErr != nil {
			// Restore the original profile if the previous selection is no longer usable.
			if restoreErr := codexDesktop.RestoreFromDesktop(true); restoreErr != nil {
				return errors.Join(
					fmt.Errorf("apply ChatGPT models: %v; restore previous Ollama profile: %w", applyErr, rollbackErr),
					fmt.Errorf("restore normal ChatGPT profile: %w", restoreErr),
				)
			}
			return fmt.Errorf("apply ChatGPT models: %v; restore previous Ollama profile: %v; restored the normal ChatGPT profile", applyErr, rollbackErr)
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
	_, models, err := selectCodexDesktopModels(selected, inventory.Available)
	if err != nil {
		return "", nil, err
	}
	primary := codexDesktopPreferredPrimary(inventory.DefaultPrimary, models)
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
	_, models, err := reconcileCodexDesktopModels(selected, inventory.Available, defaults)
	if err != nil {
		return "", nil, err
	}
	primary := codexDesktopPreferredPrimary(inventory.DefaultPrimary, models)
	return primary, hydrateCodexDesktopModelCapabilities(ctx, models), nil
}

// /api/show supplies capabilities and family metadata without replacing recommended thinking controls.
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

	recommendations, recommendationsErr := codexDesktopRecommendations(ctx)
	if recommendationsErr != nil {
		slog.Debug("could not load ChatGPT model recommendations", "error", recommendationsErr)
	}
	var access proxy.ClaudeDesktopAccessState
	accessKnown := false
	var last codexDesktopModelInventory
	for attempt := range codexDesktopModelLoadAttempts {
		if !accessKnown {
			resolved, accessErr := codexDesktopAccessState(ctx)
			if accessErr == nil {
				access = resolved
				accessKnown = true
			} else {
				slog.Debug("could not determine ChatGPT model access", "error", accessErr)
			}
		}

		var listed []api.ListModelResponse
		listKnown := false
		if response, listErr := client.List(ctx); listErr == nil {
			listed = response.Models
			listKnown = true
		}
		var accountCloud []string
		cloudKnown := false
		if names, cloudErr := codexDesktopCloudModels(ctx); cloudErr == nil {
			accountCloud = names
			cloudKnown = true
		}

		last = buildCodexDesktopModelInventory(recommendations, listed, accountCloud, access, accessKnown, listKnown, cloudKnown)
		// Retry access lookup failures even when recommendations are available.
		if len(last.Available) > 0 && (accessKnown || attempt+1 == codexDesktopModelLoadAttempts) {
			return last, nil
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
	if len(last.Catalog) > 0 {
		return last, nil
	}
	return codexDesktopModelInventory{}, errors.New("no Ollama models are available for ChatGPT")
}

func loadCodexDesktopRecommendations(ctx context.Context) ([]api.ModelRecommendation, error) {
	req, err := newSignedOllamaRequest(ctx, http.MethodGet, codexDesktopRecommendationsEndpoint())
	if err != nil {
		return nil, fmt.Errorf("prepare ChatGPT model recommendations request: %w", err)
	}
	resp, err := codexDesktopRecommendationsClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch ChatGPT model recommendations: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, codexDesktopRecommendationsMaxBody))
		return nil, fmt.Errorf("fetch ChatGPT model recommendations: status %d", resp.StatusCode)
	}

	var payload api.ModelRecommendationsResponse
	decoder := json.NewDecoder(io.LimitReader(resp.Body, codexDesktopRecommendationsMaxBody+1))
	if err := decoder.Decode(&payload); err != nil {
		return nil, fmt.Errorf("decode ChatGPT model recommendations: %w", err)
	}
	if len(payload.Recommendations) == 0 {
		return nil, errors.New("ChatGPT model recommendations are empty")
	}
	return payload.Recommendations, nil
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

func buildCodexDesktopModelInventory(
	recommendations []api.ModelRecommendation,
	listed []api.ListModelResponse,
	accountCloud []string,
	access proxy.ClaudeDesktopAccessState,
	accessKnown, localInventoryKnown, cloudInventoryKnown bool,
) codexDesktopModelInventory {
	actual := codexDesktopAvailableModels(listed, accountCloud)
	actualByName := make(map[string]launch.LaunchModel, len(actual))
	for _, model := range actual {
		actualByName[codexDesktopModelKey(model.Name)] = model
	}

	seen := make(map[string]bool, len(actual)+len(recommendations))
	recommended := make([]codexDesktopCatalogModel, 0, len(recommendations))
	for _, recommendation := range recommendations {
		route := codexDesktopRecommendationRoute(recommendation)
		key := codexDesktopModelKey(route)
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true

		model, present := actualByName[key]
		if !present {
			model = launch.LaunchModel{Name: route, Remote: codexDesktopCloudModel(route)}
		}
		if recommendation.ContextLength > 0 {
			model.ContextLength = recommendation.ContextLength
		}
		if recommendation.MaxOutputTokens > 0 {
			model.MaxOutputTokens = recommendation.MaxOutputTokens
		}
		if recommendation.Thinking != nil {
			model.Thinking = recommendation.Thinking.Clone()
		}

		availability, reason := codexDesktopRecommendationAccess(
			model,
			present,
			strings.TrimSpace(recommendation.RequiredPlan),
			access,
			accessKnown,
			localInventoryKnown,
		)
		entry := codexDesktopCatalogModel{
			Model:        model,
			DisplayName:  strings.TrimSpace(recommendation.Model),
			Description:  strings.TrimSpace(recommendation.Description),
			Recommended:  true,
			Availability: availability,
			Reason:       reason,
			RequiredPlan: strings.TrimSpace(recommendation.RequiredPlan),
		}
		recommended = append(recommended, entry)
	}

	extras := make([]codexDesktopCatalogModel, 0, len(actual))
	for _, model := range actual {
		key := codexDesktopModelKey(model.Name)
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		availability, reason := codexDesktopInventoryModelAccess(model, access, accessKnown, cloudInventoryKnown)
		extras = append(extras, codexDesktopCatalogModel{
			Model:        model,
			DisplayName:  model.Name,
			Availability: availability,
			Reason:       reason,
		})
	}

	// Sort recommendations only; preserve saved list order.
	slices.SortStableFunc(recommended, func(a, b codexDesktopCatalogModel) int {
		return codexDesktopRecommendationPriority(a.Model.Name) - codexDesktopRecommendationPriority(b.Model.Name)
	})
	catalog := make([]codexDesktopCatalogModel, 0, len(recommended)+len(extras))
	catalog = append(catalog, recommended...)
	catalog = append(catalog, extras...)
	available := make([]launch.LaunchModel, 0, len(catalog))
	for _, entry := range catalog {
		// Recommendations remain configurable regardless of current availability.
		if entry.Recommended || entry.Availability == proxy.ClaudeDesktopAvailabilityAvailable {
			available = append(available, entry.Model)
		}
	}
	defaults := codexDesktopRecommendationDefaults(catalog)

	return codexDesktopModelInventory{
		Available:      available,
		Catalog:        catalog,
		Defaults:       defaults,
		DefaultPrimary: codexDesktopDefaultPrimary(catalog, defaults, access, accessKnown),
	}
}

func codexDesktopRecommendationPriority(name string) int {
	switch codexDesktopModelKey(name) {
	case "kimi-k3:cloud":
		return 0
	case "glm-5.3:cloud":
		return 1
	case "glm-5.3-flash:cloud":
		return 2
	case "deepseek-v4-flash:cloud":
		return 3
	case "gemma4:31b:cloud":
		return 4
	default:
		return 5
	}
}

func codexDesktopRecommendationRoute(recommendation api.ModelRecommendation) string {
	name := strings.TrimSpace(recommendation.Model)
	if name != "" && recommendation.RequiredPlan != "" && !modelref.HasExplicitCloudSource(name) {
		name += ":cloud"
	}
	return name
}

func codexDesktopRecommendationAccess(
	model launch.LaunchModel,
	present bool,
	requiredPlan string,
	access proxy.ClaudeDesktopAccessState,
	accessKnown, localInventoryKnown bool,
) (proxy.ClaudeDesktopAvailability, proxy.ClaudeDesktopAccessReason) {
	if !model.Remote {
		if !localInventoryKnown {
			return proxy.ClaudeDesktopAvailabilityUnknown, proxy.ClaudeDesktopAccessVerificationUnavailable
		}
		if present {
			return proxy.ClaudeDesktopAvailabilityAvailable, ""
		}
		return proxy.ClaudeDesktopAvailabilityUnavailable, proxy.ClaudeDesktopAccessModelNotInstalled
	}
	if !accessKnown {
		return proxy.ClaudeDesktopAvailabilityUnknown, proxy.ClaudeDesktopAccessVerificationUnavailable
	}
	if access.Cloud == proxy.ClaudeDesktopCloudOff {
		return proxy.ClaudeDesktopAvailabilityUnavailable, proxy.ClaudeDesktopAccessCloudOff
	}
	if access.Cloud != proxy.ClaudeDesktopCloudOn || access.Account == proxy.ClaudeDesktopAccountUnknown {
		return proxy.ClaudeDesktopAvailabilityUnknown, proxy.ClaudeDesktopAccessVerificationUnavailable
	}
	if access.Account == proxy.ClaudeDesktopAccountSignedOut {
		return proxy.ClaudeDesktopAvailabilityUnavailable, proxy.ClaudeDesktopAccessSignInRequired
	}
	if !codexDesktopPlanSatisfies(access.Plan, requiredPlan) {
		return proxy.ClaudeDesktopAvailabilityUnavailable, proxy.ClaudeDesktopAccessUpgradeRequired
	}
	// Recommended cloud models need not appear in /api/tags.
	return proxy.ClaudeDesktopAvailabilityAvailable, ""
}

func codexDesktopInventoryModelAccess(
	model launch.LaunchModel,
	access proxy.ClaudeDesktopAccessState,
	accessKnown, cloudInventoryKnown bool,
) (proxy.ClaudeDesktopAvailability, proxy.ClaudeDesktopAccessReason) {
	if !model.Remote {
		return proxy.ClaudeDesktopAvailabilityAvailable, ""
	}
	if !accessKnown {
		return proxy.ClaudeDesktopAvailabilityUnknown, proxy.ClaudeDesktopAccessVerificationUnavailable
	}
	if access.Cloud == proxy.ClaudeDesktopCloudOff {
		return proxy.ClaudeDesktopAvailabilityUnavailable, proxy.ClaudeDesktopAccessCloudOff
	}
	if !cloudInventoryKnown || access.Cloud != proxy.ClaudeDesktopCloudOn || access.Account == proxy.ClaudeDesktopAccountUnknown {
		return proxy.ClaudeDesktopAvailabilityUnknown, proxy.ClaudeDesktopAccessVerificationUnavailable
	}
	if access.Account == proxy.ClaudeDesktopAccountSignedOut {
		return proxy.ClaudeDesktopAvailabilityUnavailable, proxy.ClaudeDesktopAccessSignInRequired
	}
	return proxy.ClaudeDesktopAvailabilityAvailable, ""
}

func codexDesktopPlanSatisfies(plan, required string) bool {
	plan = strings.ToLower(strings.TrimSpace(plan))
	required = strings.ToLower(strings.TrimSpace(required))
	if required == "" || required == "free" {
		return true
	}
	return plan != "" && plan != "free"
}

func codexDesktopRecommendationDefaults(catalog []codexDesktopCatalogModel) []launch.LaunchModel {
	defaults := make([]launch.LaunchModel, 0, codexDesktopMaxModels)
	for _, entry := range catalog {
		if !entry.Recommended {
			continue
		}
		defaults = append(defaults, entry.Model)
		if len(defaults) == codexDesktopMaxModels {
			return defaults
		}
	}
	if len(defaults) > 0 {
		return defaults
	}

	for _, entry := range catalog {
		if entry.Availability != proxy.ClaudeDesktopAvailabilityAvailable {
			continue
		}
		defaults = append(defaults, entry.Model)
		if len(defaults) == codexDesktopMaxModels {
			break
		}
	}
	return defaults
}

func codexDesktopDefaultPrimary(
	catalog []codexDesktopCatalogModel,
	defaults []launch.LaunchModel,
	access proxy.ClaudeDesktopAccessState,
	accessKnown bool,
) string {
	if len(defaults) == 0 {
		return ""
	}

	// Choose the starting model independently of picker order.
	if accessKnown && access.Account == proxy.ClaudeDesktopAccountSignedIn && codexDesktopPlanSatisfies(access.Plan, "pro") {
		return codexDesktopPreferredPrimary("glm-5.3-flash:cloud", defaults)
	}
	for _, entry := range catalog {
		if !entry.Recommended {
			continue
		}
		required := strings.ToLower(strings.TrimSpace(entry.RequiredPlan))
		if required == "" || required == "free" {
			return entry.Model.Name
		}
	}
	return defaults[0].Name
}

func codexDesktopPreferredPrimary(preferred string, models []launch.LaunchModel) string {
	preferredKey := codexDesktopModelKey(preferred)
	for _, model := range models {
		if preferredKey != "" && codexDesktopModelKey(model.Name) == preferredKey {
			return model.Name
		}
	}
	if len(models) > 0 {
		return models[0].Name
	}
	return ""
}

func codexDesktopModelStatuses(inventory codexDesktopModelInventory, selected []string) []codexDesktopModelStatus {
	selectedSet := make(map[string]bool, len(selected))
	for _, name := range selected {
		selectedSet[codexDesktopModelKey(name)] = true
	}
	statuses := make([]codexDesktopModelStatus, 0, len(inventory.Catalog)+len(selected))
	seen := make(map[string]bool, cap(statuses))
	for _, entry := range inventory.Catalog {
		key := codexDesktopModelKey(entry.Model.Name)
		seen[key] = true
		displayName := entry.DisplayName
		if displayName == "" {
			displayName = entry.Model.Name
		}
		statuses = append(statuses, codexDesktopModelStatus{
			Name:         entry.Model.Name,
			DisplayName:  displayName,
			Description:  entry.Description,
			Recommended:  entry.Recommended,
			Selected:     selectedSet[key],
			Availability: string(entry.Availability),
			Reason:       string(entry.Reason),
			RequiredPlan: entry.RequiredPlan,
		})
	}
	for _, name := range selected {
		key := codexDesktopModelKey(name)
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		statuses = append(statuses, codexDesktopModelStatus{
			Name:         name,
			DisplayName:  name,
			Selected:     true,
			Availability: string(proxy.ClaudeDesktopAvailabilityUnknown),
			Reason:       string(proxy.ClaudeDesktopAccessVerificationUnavailable),
		})
	}
	return statuses
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
	if inventory.Catalog != nil || inventory.Defaults != nil {
		return append([]launch.LaunchModel(nil), inventory.Defaults...)
	}
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

// Reopening tolerates stale selections; explicit Settings changes use strict validation.
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
	name = strings.TrimSpace(name)
	parsed, err := modelref.ParseRef(name)
	if err != nil {
		return strings.TrimSuffix(name, ":latest")
	}
	base := strings.TrimSuffix(strings.TrimSpace(parsed.Base), ":latest")
	if parsed.Source == modelref.ModelSourceCloud {
		return base + ":cloud"
	}
	return base
}

func codexDesktopCloudModel(name string) bool {
	name = strings.ToLower(strings.TrimSpace(name))
	return strings.HasSuffix(name, ":cloud") || strings.HasSuffix(name, "-cloud")
}

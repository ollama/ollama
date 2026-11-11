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
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	appui "github.com/ollama/ollama/app/ui"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/internal/proxy"
)

const (
	codexDesktopIntegrationName        = "chatgpt"
	codexDesktopMaxModels              = launch.ChatGPTMaxModels
	codexDesktopRecommendationsMaxBody = 1 << 20
)

var errCodexDesktopRestartConfirmationRequired = launch.ErrCodexAppRestartConfirmationRequired

type codexDesktopController interface {
	Installed() bool
	OllamaConfigured() bool
	RestorePending() bool
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
	Used              bool     `json:"used"`
	Supported         bool     `json:"supported"`
	Installed         bool     `json:"installed"`
	Connected         bool     `json:"connected"`
	DisconnectPending bool     `json:"disconnectPending,omitempty"`
	Running           bool     `json:"running"`
	Model             string   `json:"model,omitempty"`
	Models            []string `json:"models,omitempty"`
	MaxModels         int      `json:"maxModels"`
	Requests          uint64   `json:"requests"`
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
	ConfirmationVersion         string                     `json:"confirmationVersion,omitempty"`
}

type codexDesktopModelInventory = launch.ChatGPTModelInventory
type codexDesktopCatalogModel = launch.ChatGPTCatalogModel

func getCodexDesktopStatus() codexDesktopStatus {
	connected := codexDesktop.OllamaConfigured()
	requests := uint64(0)
	if connected {
		requests = codexDesktop.OllamaRequestCount()
	}
	var models []string
	model := ""
	if saved, err := launch.LoadChatGPTConfiguration(); err == nil && saved != nil {
		models = append([]string(nil), saved.Models...)
		model = saved.Primary
		if model == "" && len(models) > 0 {
			model = models[0]
		}
	}
	return codexDesktopStatus{
		Used:              hasUsedCodexDesktopIntegration(),
		Supported:         true,
		Installed:         codexDesktop.Installed(),
		Connected:         connected,
		DisconnectPending: !connected && codexDesktop.RestorePending(),
		Running:           codexDesktop.Running(),
		Model:             model,
		Models:            models,
		MaxModels:         codexDesktopMaxModels,
		Requests:          requests,
	}
}

func setCodexDesktopConnection(enabled, restartConfirmed bool) error {
	codexDesktopMu.Lock()
	defer codexDesktopMu.Unlock()

	if enabled == codexDesktop.OllamaConfigured() && (enabled || !codexDesktop.RestorePending()) {
		return nil
	}
	if !enabled {
		if codexDesktop.Running() && !restartConfirmed {
			return errCodexDesktopRestartConfirmationRequired
		}
		return launch.RestoreChatGPTModels(codexDesktop, restartConfirmed)
	}
	if !codexDesktop.Installed() {
		return errors.New("ChatGPT is not installed")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return launch.ApplyChatGPTModels(ctx, codexDesktop, codexDesktopLoadConnectionModels, launch.ChatGPTApplyRequest{
		UseSavedSelection: true,
		OpenWhenStopped:   true,
		RestartConfirmed:  restartConfirmed,
		Onboard:           true,
	})
}

// Native and webview controls share the connection handler.
func changeCodexDesktopConnection(enabled, restartConfirmed bool) codexDesktopActionResult {
	err := setCodexDesktopConnection(enabled, restartConfirmed)
	result := codexDesktopActionResult{Status: getCodexDesktopStatus()}
	if err == nil {
		return result
	}
	result.RestartConfirmationRequired = errors.Is(err, errCodexDesktopRestartConfirmationRequired)
	result.Error = "Couldn’t disconnect ChatGPT. Try again."
	if enabled {
		result.Error = "Couldn’t connect ChatGPT. Try again."
	}
	if !result.RestartConfirmationRequired {
		slog.Warn("failed to change ChatGPT integration", "connected", enabled, "error", err)
	}
	return result
}

func getCodexDesktopModelsSettings() (codexDesktopModelsSettings, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return codexDesktopSettingsWithInventory(ctx, codexDesktopSettingsSummary())
}

func codexDesktopSettingsSummary() codexDesktopModelsSettings {
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
	return settings
}

func codexDesktopSettingsWithInventory(ctx context.Context, settings codexDesktopModelsSettings) (codexDesktopModelsSettings, error) {
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

func applyCodexDesktopModels(selected []string, restartConfirmed bool, versions ...string) error {
	codexDesktopMu.Lock()
	defer codexDesktopMu.Unlock()
	return applyCodexDesktopModelsLocked(selected, restartConfirmed, true, versions...)
}

func resetCodexDesktopModels() error {
	codexDesktopMu.Lock()
	defer codexDesktopMu.Unlock()

	return launch.ResetChatGPTModels()
}

func applyCodexDesktopModelsLocked(selected []string, restartConfirmed, openWhenStopped bool, versions ...string) error {
	expectedVersion := ""
	if len(versions) > 0 {
		expectedVersion = versions[0]
	}
	// An unavailable inventory must not expire the existing-profile restart
	// fallback. Bound discovery separately from the configuration operation.
	load := func(ctx context.Context, models []string) (string, []launch.LaunchModel, error) {
		loadCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()
		return codexDesktopLoadModels(loadCtx, models)
	}
	return launch.ApplyChatGPTModels(context.Background(), codexDesktop, load, launch.ChatGPTApplyRequest{
		Models:                    selected,
		OpenWhenStopped:           openWhenStopped,
		SaveOnlyWhenDisconnected:  true,
		RestartConfirmed:          restartConfirmed,
		RestartOnInventoryFailure: openWhenStopped,
		ExpectedVersion:           expectedVersion,
	})
}

func codexDesktopModelSource() (*launch.ChatGPTModelSource, error) {
	client, err := codexDesktopClientFactory()
	if err != nil {
		return nil, err
	}
	return &launch.ChatGPTModelSource{
		Client:          client,
		Recommendations: codexDesktopRecommendations,
		CloudModels:     codexDesktopCloudModels,
		AccessState:     codexDesktopAccessState,
		Attempts:        codexDesktopModelLoadAttempts,
		RetryWait:       codexDesktopModelRetryWait,
	}, nil
}

func loadCodexDesktopModels(ctx context.Context, selected []string) (string, []launch.LaunchModel, error) {
	source, err := codexDesktopModelSource()
	if err != nil {
		return "", nil, err
	}
	return source.Resolve(ctx, selected, false)
}

func loadCodexDesktopConnectionModels(ctx context.Context, selected []string) (string, []launch.LaunchModel, error) {
	source, err := codexDesktopModelSource()
	if err != nil {
		return "", nil, err
	}
	return source.Resolve(ctx, selected, true)
}

func loadCodexDesktopAvailableModels(ctx context.Context) ([]launch.LaunchModel, error) {
	inventory, err := loadCodexDesktopModelInventory(ctx)
	return inventory.Available, err
}

func loadCodexDesktopModelInventory(ctx context.Context) (codexDesktopModelInventory, error) {
	source, err := codexDesktopModelSource()
	if err != nil {
		return codexDesktopModelInventory{}, err
	}
	return source.Inventory(ctx)
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
			Name:         launch.ChatGPTSelectionModelName(entry.Model.Name, selected),
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

func buildCodexDesktopModelInventory(recommendations []api.ModelRecommendation, listed []api.ListModelResponse, accountCloud []string, access proxy.ClaudeDesktopAccessState, accessKnown, localInventoryKnown, cloudInventoryKnown bool) codexDesktopModelInventory {
	return launch.BuildChatGPTModelInventory(recommendations, listed, accountCloud, access, accessKnown, localInventoryKnown, cloudInventoryKnown)
}
func buildCodexDesktopModels(selected []string, listed []api.ListModelResponse, accountCloud []string) (string, []launch.LaunchModel, error) {
	return launch.BuildChatGPTModels(selected, listed, accountCloud)
}
func codexDesktopAvailableModels(listed []api.ListModelResponse, accountCloud []string) []launch.LaunchModel {
	return launch.ChatGPTAvailableModels(listed, accountCloud)
}
func codexDesktopDefaultModels(inventory codexDesktopModelInventory) []launch.LaunchModel {
	return launch.ChatGPTDefaultModels(inventory)
}
func codexDesktopInventoryModelAccess(model launch.LaunchModel, access proxy.ClaudeDesktopAccessState, accessKnown, cloudInventoryKnown bool) (proxy.ClaudeDesktopAvailability, proxy.ClaudeDesktopAccessReason) {
	return launch.ChatGPTInventoryModelAccess(model, access, accessKnown, cloudInventoryKnown)
}
func selectCodexDesktopModels(selected []string, available []launch.LaunchModel) (string, []launch.LaunchModel, error) {
	return launch.SelectChatGPTModels(selected, available)
}
func reconcileCodexDesktopModels(selected []string, available, defaults []launch.LaunchModel) (string, []launch.LaunchModel, error) {
	return launch.ReconcileChatGPTModels(selected, available, defaults)
}
func codexDesktopModelNames(models []launch.LaunchModel) []string {
	return launch.ChatGPTModelNames(models)
}
func codexDesktopModelKey(name string) string { return launch.ChatGPTModelKey(name) }

func hasUsedCodexDesktopIntegration() bool {
	if appStore == nil {
		return false
	}
	settings, err := appStore.Settings()
	if err != nil {
		return false
	}
	return settings.CodexDesktopUsed
}

func markCodexDesktopIntegrationUsed() error {
	codexDesktopMu.Lock()
	defer codexDesktopMu.Unlock()
	if appStore == nil {
		return errors.New("settings are unavailable")
	}
	return appStore.MarkCodexDesktopUsed()
}

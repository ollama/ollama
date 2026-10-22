package launch

import (
	"context"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	modelpkg "github.com/ollama/ollama/types/model"
)

var (
	modelShowTimeout          = 5 * time.Second
	lookupServerContextLength = serverContextLength
	serverContextLengths      sync.Map
)

// LaunchModel is the model metadata Launch passes to integration config
// writers after resolving selected model names through the per-run inventory.
type LaunchModel struct {
	Name            string
	Remote          bool
	ToolCapable     bool
	Capabilities    []modelpkg.Capability
	Thinking        *api.ModelRecommendationThinking
	ContextLength   int
	MaxOutputTokens int
	EmbeddingLength int
	Size            int64
	Details         api.ModelDetails
}

type modelInfo = LaunchModel

// ModelInfo re-exports launcher model inventory details for callers.
type ModelInfo = LaunchModel

func (m LaunchModel) HasCapability(capability modelpkg.Capability) bool {
	return slices.Contains(m.Capabilities, capability)
}

func (m LaunchModel) WithCloudLimits() LaunchModel {
	if limit, ok := lookupCloudModelLimit(m.Name); ok {
		if m.ContextLength <= 0 {
			m.ContextLength = limit.Context
		}
		if m.MaxOutputTokens <= 0 {
			m.MaxOutputTokens = limit.Output
		}
	}
	return m
}

func (m LaunchModel) WithServerLimits() LaunchModel {
	if m.ContextLength > 0 || !isCloudModelName(m.Name) {
		return m
	}
	if n := lookupServerContextLength(m.Name); n > 0 {
		m.ContextLength = n
	}
	return m
}

func serverContextLength(model string) int {
	if n, ok := serverContextLengths.Load(model); ok {
		return n.(int)
	}
	n := showContextLength(model)
	if n > 0 {
		serverContextLengths.Store(model, n)
	}
	return n
}

func showContextLength(model string) int {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return 0
	}

	ctx, cancel := context.WithTimeout(context.Background(), modelShowTimeout)
	defer cancel()
	resp, err := client.Show(ctx, &api.ShowRequest{Model: model})
	if err != nil {
		return 0
	}

	n, _ := modelInfoContextLength(resp.ModelInfo)
	return n
}

func modelInfoContextLength(modelInfo map[string]any) (int, bool) {
	for key, val := range modelInfo {
		if !strings.HasSuffix(key, ".context_length") {
			continue
		}
		switch v := val.(type) {
		case float64:
			if v > 0 {
				return int(v), true
			}
		case int:
			if v > 0 {
				return v, true
			}
		case int64:
			if v > 0 {
				return int(v), true
			}
		}
	}
	return 0, false
}

type modelInventory struct {
	client *api.Client

	mu     sync.Mutex
	loaded bool
	models []LaunchModel
	err    error
}

func newModelInventory(client *api.Client) *modelInventory {
	return &modelInventory{client: client}
}

func (i *modelInventory) Load(ctx context.Context) ([]LaunchModel, error) {
	return i.load(ctx, false)
}

func (i *modelInventory) Refresh(ctx context.Context) ([]LaunchModel, error) {
	return i.load(ctx, true)
}

func (i *modelInventory) load(ctx context.Context, force bool) ([]LaunchModel, error) {
	if i == nil || i.client == nil {
		return nil, nil
	}

	i.mu.Lock()
	defer i.mu.Unlock()

	if i.loaded && !force {
		return cloneLaunchModels(i.models), i.err
	}

	resp, err := i.client.List(ctx)
	if err != nil {
		i.models = nil
		i.err = err
		i.loaded = true
		return nil, err
	}

	i.models = make([]LaunchModel, 0, len(resp.Models))
	for _, model := range resp.Models {
		i.models = append(i.models, launchModelFromListResponse(model))
	}
	i.err = nil
	i.loaded = true

	return cloneLaunchModels(i.models), i.err
}

func (i *modelInventory) Resolve(ctx context.Context, names []string) []LaunchModel {
	names = dedupeModelList(names)
	if len(names) == 0 {
		return nil
	}

	models, err := i.Load(ctx)
	if err != nil {
		models = nil
	}

	resolved, localMiss := resolveLaunchModels(names, models)
	if localMiss {
		if refreshed, err := i.Refresh(ctx); err == nil {
			resolved, _ = resolveLaunchModels(names, refreshed)
		}
	}
	return resolved
}

func resolveLaunchModels(names []string, models []LaunchModel) ([]LaunchModel, bool) {
	resolved := make([]LaunchModel, 0, len(names))
	localMiss := false
	for _, name := range names {
		if model, ok := findLaunchModel(models, name); ok {
			resolved = append(resolved, model.WithCloudLimits().WithServerLimits())
			continue
		}
		if !isCloudModelName(name) {
			localMiss = true
		}
		resolved = append(resolved, fallbackLaunchModel(name))
	}
	return resolved, localMiss
}

func launchModelFromListResponse(model api.ListModelResponse) LaunchModel {
	return LaunchModel{
		Name:            model.Name,
		Remote:          model.RemoteModel != "",
		ToolCapable:     slices.Contains(model.Capabilities, modelpkg.CapabilityTools),
		Capabilities:    append([]modelpkg.Capability(nil), model.Capabilities...),
		ContextLength:   model.Details.ContextLength,
		EmbeddingLength: model.Details.EmbeddingLength,
		Size:            model.Size,
		Details:         model.Details,
	}.WithCloudLimits()
}

func fallbackLaunchModel(name string) LaunchModel {
	return LaunchModel{Name: name, Remote: isCloudModelName(name)}.WithCloudLimits().WithServerLimits()
}

func findLaunchModel(models []LaunchModel, name string) (LaunchModel, bool) {
	for _, model := range models {
		if launchModelMatches(model.Name, name) {
			return cloneLaunchModel(model), true
		}
	}
	return LaunchModel{}, false
}

func launchModelMatches(candidate, name string) bool {
	if candidate == name {
		return true
	}
	return strings.TrimSuffix(candidate, ":latest") == name
}

func cloneLaunchModel(model LaunchModel) LaunchModel {
	model.Capabilities = append([]modelpkg.Capability(nil), model.Capabilities...)
	model.Thinking = model.Thinking.Clone()
	model.Details.Families = append([]string(nil), model.Details.Families...)
	return model
}

func cloneLaunchModels(models []LaunchModel) []LaunchModel {
	cloned := make([]LaunchModel, len(models))
	for i, model := range models {
		cloned[i] = cloneLaunchModel(model)
	}
	return cloned
}

func launchModelNames(models []LaunchModel) []string {
	names := make([]string, 0, len(models))
	for _, model := range models {
		if model.Name != "" {
			names = append(names, model.Name)
		}
	}
	return names
}

func launchModelsFromNames(names []string) []LaunchModel {
	models := make([]LaunchModel, 0, len(names))
	for _, name := range names {
		if name == "" {
			continue
		}
		models = append(models, fallbackLaunchModel(name))
	}
	return models
}

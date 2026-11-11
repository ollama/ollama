package launch

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/modelref"
	"github.com/ollama/ollama/internal/proxy"
	modelpkg "github.com/ollama/ollama/types/model"
)

// The app and CLI use the existing ChatGPT catalog policy.
const ChatGPTMaxModels = 5

type ChatGPTModelInventory struct {
	Available      []LaunchModel
	Catalog        []ChatGPTCatalogModel
	Defaults       []LaunchModel
	DefaultPrimary string
}

type ChatGPTCatalogModel struct {
	Model        LaunchModel
	DisplayName  string
	Description  string
	Recommended  bool
	Availability proxy.ClaudeDesktopAvailability
	Reason       proxy.ClaudeDesktopAccessReason
	RequiredPlan string
}

func BuildChatGPTModelInventory(
	recommendations []api.ModelRecommendation,
	listed []api.ListModelResponse,
	accountCloud []string,
	access proxy.ClaudeDesktopAccessState,
	accessKnown, localInventoryKnown, cloudInventoryKnown bool,
) ChatGPTModelInventory {
	actual := ChatGPTAvailableModels(listed, accountCloud)
	actualByName := make(map[string]LaunchModel, len(actual))
	for _, model := range actual {
		actualByName[ChatGPTModelKey(model.Name)] = model
	}

	seen := make(map[string]bool, len(actual)+len(recommendations))
	recommended := make([]ChatGPTCatalogModel, 0, len(recommendations))
	for _, recommendation := range recommendations {
		route := chatGPTRecommendationRoute(recommendation)
		key := ChatGPTModelKey(route)
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true

		model, present := actualByName[key]
		if !present {
			model = LaunchModel{Name: route, Remote: chatGPTCloudModel(route)}
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

		availability, reason := chatGPTRecommendationAccess(
			model,
			present,
			strings.TrimSpace(recommendation.RequiredPlan),
			access,
			accessKnown,
			localInventoryKnown,
		)
		entry := ChatGPTCatalogModel{
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

	extras := make([]ChatGPTCatalogModel, 0, len(actual))
	for _, model := range actual {
		key := ChatGPTModelKey(model.Name)
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		availability, reason := ChatGPTInventoryModelAccess(model, access, accessKnown, cloudInventoryKnown)
		extras = append(extras, ChatGPTCatalogModel{
			Model:        model,
			DisplayName:  model.Name,
			Availability: availability,
			Reason:       reason,
		})
	}

	// Sort recommendations only; preserve saved list order.
	slices.SortStableFunc(recommended, func(a, b ChatGPTCatalogModel) int {
		return chatGPTRecommendationPriority(a.Model.Name) - chatGPTRecommendationPriority(b.Model.Name)
	})
	catalog := make([]ChatGPTCatalogModel, 0, len(recommended)+len(extras))
	catalog = append(catalog, recommended...)
	catalog = append(catalog, extras...)
	available := make([]LaunchModel, 0, len(catalog))
	for _, entry := range catalog {
		// Recommendations remain configurable regardless of current availability.
		if entry.Recommended || entry.Availability == proxy.ClaudeDesktopAvailabilityAvailable {
			available = append(available, entry.Model)
		}
	}
	defaults := chatGPTRecommendationDefaults(catalog)

	return ChatGPTModelInventory{
		Available:      available,
		Catalog:        catalog,
		Defaults:       defaults,
		DefaultPrimary: chatGPTDefaultPrimary(catalog, defaults, access, accessKnown),
	}
}

func chatGPTRecommendationPriority(name string) int {
	switch ChatGPTModelKey(name) {
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

func chatGPTRecommendationRoute(recommendation api.ModelRecommendation) string {
	name := strings.TrimSpace(recommendation.Model)
	if name != "" && recommendation.RequiredPlan != "" && !modelref.HasExplicitCloudSource(name) {
		name += ":cloud"
	}
	return name
}

func chatGPTRecommendationAccess(
	model LaunchModel,
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
	if !chatGPTPlanSatisfies(access.Plan, requiredPlan) {
		return proxy.ClaudeDesktopAvailabilityUnavailable, proxy.ClaudeDesktopAccessUpgradeRequired
	}
	// Recommended cloud models need not appear in /api/tags.
	return proxy.ClaudeDesktopAvailabilityAvailable, ""
}

func ChatGPTInventoryModelAccess(
	model LaunchModel,
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

func chatGPTPlanSatisfies(plan, required string) bool {
	plan = strings.ToLower(strings.TrimSpace(plan))
	required = strings.ToLower(strings.TrimSpace(required))
	if required == "" || required == "free" {
		return true
	}
	return plan != "" && plan != "free"
}

func chatGPTRecommendationDefaults(catalog []ChatGPTCatalogModel) []LaunchModel {
	defaults := make([]LaunchModel, 0, ChatGPTMaxModels)
	for _, entry := range catalog {
		if !entry.Recommended {
			continue
		}
		defaults = append(defaults, entry.Model)
		if len(defaults) == ChatGPTMaxModels {
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
		if len(defaults) == ChatGPTMaxModels {
			break
		}
	}
	return defaults
}

func chatGPTDefaultPrimary(
	catalog []ChatGPTCatalogModel,
	defaults []LaunchModel,
	access proxy.ClaudeDesktopAccessState,
	accessKnown bool,
) string {
	if len(defaults) == 0 {
		return ""
	}

	// Choose the starting model independently of picker order.
	if accessKnown && access.Account == proxy.ClaudeDesktopAccountSignedIn && chatGPTPlanSatisfies(access.Plan, "pro") {
		return chatGPTPreferredPrimary("glm-5.3-flash:cloud", defaults)
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

func chatGPTPreferredPrimary(preferred string, models []LaunchModel) string {
	preferredKey := ChatGPTModelKey(preferred)
	for _, model := range models {
		if preferredKey != "" && ChatGPTModelKey(model.Name) == preferredKey {
			return model.Name
		}
	}
	if len(models) > 0 {
		return models[0].Name
	}
	return ""
}

func BuildChatGPTModels(selected []string, listed []api.ListModelResponse, accountCloud []string) (string, []LaunchModel, error) {
	available := ChatGPTAvailableModels(listed, accountCloud)
	return SelectChatGPTModels(selected, available)
}

func ChatGPTAvailableModels(listed []api.ListModelResponse, accountCloud []string) []LaunchModel {
	installed := make(map[string]api.ListModelResponse, len(listed))
	for _, model := range listed {
		for _, name := range []string{model.Name, model.Model} {
			if key := ChatGPTModelKey(name); key != "" {
				installed[key] = model
			}
		}
	}
	accountCloudSet := make(map[string]bool, len(accountCloud))
	for _, name := range accountCloud {
		if key := ChatGPTModelKey(name); key != "" {
			accountCloudSet[key] = true
		}
	}
	models := make([]LaunchModel, 0, len(listed)+len(accountCloud))
	seen := make(map[string]bool, cap(models))
	add := func(model LaunchModel) {
		model.Name = strings.TrimSpace(model.Name)
		key := ChatGPTModelKey(model.Name)
		if key == "" || seen[key] {
			return
		}
		seen[key] = true
		models = append(models, model)
	}

	for _, model := range listed {
		if chatGPTListedModelIsCloud(model) && !accountCloudSet[ChatGPTModelKey(model.Name)] && !accountCloudSet[ChatGPTModelKey(model.Model)] {
			continue
		}
		add(chatGPTLaunchModel(model))
	}
	for _, name := range accountCloud {
		key := ChatGPTModelKey(name)
		if listedModel, ok := installed[key]; ok {
			add(chatGPTLaunchModel(listedModel))
			continue
		}
		add(LaunchModel{Name: strings.TrimSpace(name), Remote: true})
	}

	return models
}

func ChatGPTDefaultModels(inventory ChatGPTModelInventory) []LaunchModel {
	if inventory.Catalog != nil || inventory.Defaults != nil {
		return append([]LaunchModel(nil), inventory.Defaults...)
	}
	return append([]LaunchModel(nil), inventory.Available[:min(len(inventory.Available), ChatGPTMaxModels)]...)
}

func chatGPTListedModelIsCloud(model api.ListModelResponse) bool {
	return model.RemoteModel != "" || model.RemoteHost != "" ||
		chatGPTCloudModel(model.Name) ||
		chatGPTCloudModel(model.Model)
}

func SelectChatGPTModels(selected []string, available []LaunchModel) (string, []LaunchModel, error) {
	byName := make(map[string]LaunchModel, len(available))
	for _, model := range available {
		byName[ChatGPTModelKey(model.Name)] = model
	}

	if len(selected) > ChatGPTMaxModels {
		return "", nil, fmt.Errorf("choose up to %d models for ChatGPT", ChatGPTMaxModels)
	}
	resolved := make([]LaunchModel, 0, ChatGPTMaxModels)
	seen := make(map[string]bool, ChatGPTMaxModels)
	for _, name := range selected {
		name = strings.TrimSpace(name)
		key := ChatGPTModelKey(name)
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
			if len(resolved) == ChatGPTMaxModels {
				break
			}
			key := ChatGPTModelKey(model.Name)
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

func ReconcileChatGPTModels(selected []string, available, defaults []LaunchModel) (string, []LaunchModel, error) {
	if len(selected) == 0 {
		if len(defaults) > 0 {
			return SelectChatGPTModels(ChatGPTModelNames(defaults), available)
		}
		return SelectChatGPTModels(nil, available)
	}

	byName := make(map[string]LaunchModel, len(available))
	for _, model := range available {
		byName[ChatGPTModelKey(model.Name)] = model
	}
	resolved := make([]LaunchModel, 0, min(len(selected), ChatGPTMaxModels))
	seen := make(map[string]bool, ChatGPTMaxModels)
	for _, name := range selected {
		key := ChatGPTModelKey(name)
		model, ok := byName[key]
		if key == "" || !ok || seen[key] {
			continue
		}
		seen[key] = true
		resolved = append(resolved, model)
		if len(resolved) == ChatGPTMaxModels {
			break
		}
	}
	if len(resolved) == 0 {
		if len(defaults) > 0 {
			return SelectChatGPTModels(ChatGPTModelNames(defaults), available)
		}
		return SelectChatGPTModels(nil, available)
	}
	return resolved[0].Name, resolved, nil
}

func ChatGPTModelNames(models []LaunchModel) []string {
	names := make([]string, 0, len(models))
	for _, model := range models {
		if name := strings.TrimSpace(model.Name); name != "" {
			names = append(names, name)
		}
	}
	return names
}

func chatGPTLaunchModel(model api.ListModelResponse) LaunchModel {
	name := strings.TrimSpace(model.Name)
	if name == "" {
		name = strings.TrimSpace(model.Model)
	}
	return LaunchModel{
		Name:            name,
		Remote:          model.RemoteModel != "" || model.RemoteHost != "" || chatGPTCloudModel(name),
		Capabilities:    append([]modelpkg.Capability(nil), model.Capabilities...),
		ContextLength:   model.Details.ContextLength,
		EmbeddingLength: model.Details.EmbeddingLength,
		Size:            model.Size,
		Details:         model.Details,
	}
}

func ChatGPTModelKey(name string) string {
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

func chatGPTCloudModel(name string) bool {
	name = strings.ToLower(strings.TrimSpace(name))
	return strings.HasSuffix(name, ":cloud") || strings.HasSuffix(name, "-cloud")
}

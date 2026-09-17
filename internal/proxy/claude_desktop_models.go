package proxy

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/modelref"
)

const (
	maxClaudeDesktopRecommendationsBody = 1 << 20
	MaxClaudeDesktopModels              = 5
)

type claudeDesktopModelSlot struct {
	id            string
	displayName   string
	family        string
	createdAt     string
	familyDefault bool
}

var claudeDesktopModelSlots = [MaxClaudeDesktopModels]claudeDesktopModelSlot{
	{id: "claude-fable-5", displayName: "Fable 5", family: "fable", createdAt: "2026-06-09T00:00:00Z", familyDefault: true},
	{id: "claude-opus-5", displayName: "Opus 5", family: "opus", createdAt: "2026-07-24T00:00:00Z", familyDefault: true},
	{id: "claude-sonnet-5", displayName: "Sonnet 5", family: "sonnet", createdAt: "2026-06-30T00:00:00Z", familyDefault: true},
	{id: "claude-haiku-4-5-20251001", displayName: "Haiku 4.5", family: "haiku", createdAt: "2025-10-01T00:00:00Z", familyDefault: true},
	{id: "claude-sonnet-4-6", displayName: "Sonnet 4.6", family: "sonnet", createdAt: "2025-11-18T00:00:00Z"},
}

// ClaudeDesktopRoute is one fixed model ID accepted by Claude Desktop. The
// gateway advertises the mapped Ollama model as its display name.
type ClaudeDesktopRoute struct {
	ID          string
	DisplayName string
}

// ClaudeDesktopRoutes returns the fixed routes in Claude's preferred order.
func ClaudeDesktopRoutes() []ClaudeDesktopRoute {
	routes := make([]ClaudeDesktopRoute, len(claudeDesktopModelSlots))
	for i, slot := range claudeDesktopModelSlots {
		routes[i] = ClaudeDesktopRoute{ID: slot.id, DisplayName: slot.displayName}
	}
	return routes
}

// DefaultClaudeDesktopMappings returns the safe compatibility fallback used
// when Ollama.com does not provide an app-specific mapping contract.
func DefaultClaudeDesktopMappings() map[string]string {
	return map[string]string{
		"claude-sonnet-5": "gemma4:31b-cloud",
	}
}

// DefaultClaudeDesktopMappingsForModels resolves the server contract, or the
// compatibility fallback when it is absent, against the current catalog.
func DefaultClaudeDesktopMappingsForModels(available []ClaudeDesktopModel) map[string]string {
	wanted := DefaultClaudeDesktopMappings()
	for _, model := range available {
		if model.defaultMappings != nil {
			wanted = make(map[string]string, len(*model.defaultMappings))
			for route, mapping := range *model.defaultMappings {
				wanted[route] = mapping.Model
			}
			break
		}
	}
	return resolveClaudeDesktopMappings(available, wanted)
}

func resolveClaudeDesktopMappings(available []ClaudeDesktopModel, wanted map[string]string) map[string]string {
	mappings := make(map[string]string, len(wanted))
	for _, model := range available {
		for route, name := range wanted {
			if isClaudeDesktopRouteID(route) && (model.Name == name || model.OllamaModel == name) {
				mappings[route] = model.OllamaModel
			}
		}
	}
	return mappings
}

// ClaudeDesktopModel is one Ollama model adapted for Claude Desktop's model
// catalog. Name is the canonical recommendation identifier shown to users;
// OllamaModel is the explicit route sent to the local Ollama server.
type ClaudeDesktopModel struct {
	Name             string
	Description      string
	DisplayName      string
	RequiredPlan     string
	OllamaModel      string
	Cloud            bool
	Recommended      bool
	AccountCloud     bool
	entitlementKnown bool
	defaultMappings  *api.ModelRecommendationMappings
	gateway          gatewayModel
}

// GatewayID returns the validated Claude-facing ID assigned to this model.
func (m ClaudeDesktopModel) GatewayID() string {
	return m.gateway.ID
}

// FetchClaudeDesktopModels fetches the app-aware recommendation contract.
// Callers own request construction and are responsible for falling back when
// the request or response fails.
func FetchClaudeDesktopModels(client *http.Client, req *http.Request) ([]ClaudeDesktopModel, error) {
	if client == nil {
		return nil, errors.New("Claude Desktop recommendation client is required")
	}
	if req == nil {
		return nil, errors.New("Claude Desktop recommendation request is required")
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch Claude Desktop recommendations: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxClaudeDesktopRecommendationsBody))
		return nil, fmt.Errorf("fetch Claude Desktop recommendations: status %d", resp.StatusCode)
	}

	var payload api.ModelRecommendationsResponse
	decoder := json.NewDecoder(io.LimitReader(resp.Body, maxClaudeDesktopRecommendationsBody+1))
	if err := decoder.Decode(&payload); err != nil {
		return nil, fmt.Errorf("decode Claude Desktop recommendations: %w", err)
	}
	models := ClaudeDesktopModelsFromRecommendations(payload.Recommendations)
	cloudModels := models[:0]
	for _, model := range models {
		if model.Cloud {
			cloudModels = append(cloudModels, model)
		}
	}
	if len(cloudModels) == 0 {
		return nil, errors.New("Claude Desktop recommendations contain no cloud models")
	}
	if payload.Mappings != nil {
		mappings := maps.Clone(*payload.Mappings)
		for i := range cloudModels {
			cloudModels[i].defaultMappings = &mappings
		}
	}
	return cloudModels, nil
}

// ClaudeDesktopModelsFromRecommendations converts the endpoint response into
// the protocol metadata Claude Desktop expects. Recommendation model names,
// order, descriptions, limits, and plan requirements remain server-owned.
func ClaudeDesktopModelsFromRecommendations(recommendations []api.ModelRecommendation) []ClaudeDesktopModel {
	models := make([]ClaudeDesktopModel, 0, len(recommendations))
	seen := make(map[string]struct{}, len(recommendations))
	for _, recommendation := range recommendations {
		name := strings.TrimSpace(recommendation.Model)
		if !validClaudeDesktopModelName(name) {
			continue
		}
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}
		model := newClaudeDesktopModel(
			name,
			strings.TrimSpace(recommendation.Description),
			strings.TrimSpace(recommendation.RequiredPlan),
			recommendation.MaxOutputTokens,
		)
		model.Recommended = true
		model.entitlementKnown = true
		models = append(models, model)
	}
	return models
}

// DefaultClaudeDesktopModels is the built-in offline fallback. Keep this list
// deliberately small; the Ollama.com app-aware endpoint is the primary source.
func DefaultClaudeDesktopModels() []ClaudeDesktopModel {
	models := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{
		{Model: "glm-5.2:cloud", Description: "Long-horizon coding and agentic engineering", MaxOutputTokens: 128_000, RequiredPlan: "pro"},
		{Model: "kimi-k3:cloud", Description: "Long-horizon agentic reasoning with multimodal tool use", MaxOutputTokens: 262_144, RequiredPlan: "pro"},
		{Model: "deepseek-v4-pro", Description: "High-performance coding and tool use", MaxOutputTokens: 128_000, RequiredPlan: "pro"},
		{Model: "deepseek-v4-flash", Description: "Fast coding and agentic tool use", MaxOutputTokens: 64_000, RequiredPlan: "pro"},
		{Model: "gemma4:31b-cloud", Description: "Agentic workflows and multimodal reasoning", MaxOutputTokens: 262_144, RequiredPlan: "free"},
	})
	// Preserve the proven fallback route while the endpoint remains free to
	// move the canonical DeepSeek alias independently.
	models[3].OllamaModel = "deepseek-v4-flash:0731:cloud"
	models[3].DisplayName = models[3].OllamaModel
	models[3].gateway.DisplayName = models[3].OllamaModel
	models[3].gateway.OllamaModel = models[3].OllamaModel
	return models
}

// UnverifyClaudeDesktopCloudEntitlements prevents fallback metadata from
// overriding a catalog that was already refreshed from the server.
func UnverifyClaudeDesktopCloudEntitlements(models []ClaudeDesktopModel) []ClaudeDesktopModel {
	models = cloneClaudeDesktopModels(models)
	for i := range models {
		if models[i].Cloud {
			models[i].entitlementKnown = false
		}
	}
	return models
}

// PreserveClaudeDesktopCloudEntitlements carries verified account facts into
// an offline catalog without replacing its built-in routes or weakening its
// last-known plan requirement.
func PreserveClaudeDesktopCloudEntitlements(models, previous []ClaudeDesktopModel) []ClaudeDesktopModel {
	models = UnverifyClaudeDesktopCloudEntitlements(models)
	known := make(map[string]ClaudeDesktopModel, len(previous)*2)
	for _, model := range previous {
		known[model.Name] = model
		known[model.OllamaModel] = model
	}
	for i, model := range models {
		prior, ok := known[model.Name]
		if !ok {
			prior, ok = known[model.OllamaModel]
		}
		if !ok {
			continue
		}
		models[i].AccountCloud = prior.AccountCloud
		models[i].entitlementKnown = prior.entitlementKnown
		required := strings.TrimSpace(models[i].RequiredPlan)
		priorRequired := strings.TrimSpace(prior.RequiredPlan)
		if (required == "" || strings.EqualFold(required, "free")) && priorRequired != "" && !strings.EqualFold(priorRequired, "free") {
			models[i].RequiredPlan = priorRequired
		}
	}
	return models
}

// WithoutClaudeDesktopRecommendationMappings returns a catalog without
// server-provided defaults. Callers use this when retaining model metadata
// across an offline refresh, where the prior mapping contract is stale.
func WithoutClaudeDesktopRecommendationMappings(models []ClaudeDesktopModel) []ClaudeDesktopModel {
	models = cloneClaudeDesktopModels(models)
	for i := range models {
		models[i].defaultMappings = nil
	}
	return models
}

// ClaudeDesktopModelsFromCloudInventory converts an account-scoped cloud
// inventory into selectable Claude routes. Presence in this inventory is the
// entitlement and Auto-mode eligibility check; these models are not
// recommendations.
func ClaudeDesktopModelsFromCloudInventory(names []string) []ClaudeDesktopModel {
	models := make([]ClaudeDesktopModel, 0, len(names))
	seen := make(map[string]struct{}, len(names))
	for _, rawName := range names {
		name := strings.TrimSuffix(strings.TrimSpace(rawName), ":latest")
		if !validClaudeDesktopModelName(name) {
			continue
		}
		if !modelref.HasExplicitCloudSource(name) {
			name += ":cloud"
		}
		if !validClaudeDesktopModelName(name) {
			continue
		}
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}
		model := newClaudeDesktopModel(name, "User-selected cloud model", "", 64_000)
		model.AccountCloud = true
		model.entitlementKnown = true
		models = append(models, model)
	}
	return models
}

// VerifyClaudeDesktopModelsWithCloudInventory marks matching catalog models
// as account-verified without replacing their recommendation metadata.
func VerifyClaudeDesktopModelsWithCloudInventory(models, inventory []ClaudeDesktopModel) []ClaudeDesktopModel {
	verified := cloneClaudeDesktopModels(models)
	accountModels := make(map[string]struct{}, len(inventory)*2)
	for _, model := range inventory {
		if !model.AccountCloud {
			continue
		}
		accountModels[model.Name] = struct{}{}
		accountModels[model.OllamaModel] = struct{}{}
	}
	for i, model := range verified {
		if _, ok := accountModels[model.Name]; !ok {
			if _, ok := accountModels[model.OllamaModel]; !ok {
				continue
			}
		}
		verified[i].AccountCloud = true
		verified[i].entitlementKnown = true
	}
	return verified
}

// SelectClaudeDesktopModels preserves the user's explicit order. Names that
// are no longer recommended remain usable so a transient endpoint change or
// outage cannot silently replace a saved user choice. Validated Claude IDs are
// reusable slots assigned in that order, independent of recommendation order.
func SelectClaudeDesktopModels(available []ClaudeDesktopModel, selected []string) []ClaudeDesktopModel {
	var models []ClaudeDesktopModel
	if len(selected) == 0 {
		models = cloneClaudeDesktopModels(available)
		if validClaudeDesktopRouteAssignments(models) {
			return models
		}
	} else {
		byName := make(map[string]ClaudeDesktopModel, len(available))
		for _, model := range available {
			byName[model.Name] = model
			byName[model.OllamaModel] = model
		}
		models = make([]ClaudeDesktopModel, 0, min(len(selected), MaxClaudeDesktopModels))
		seen := make(map[string]struct{}, len(selected))
		for _, rawName := range selected {
			if len(models) == MaxClaudeDesktopModels {
				break
			}
			name := strings.TrimSpace(rawName)
			if !validClaudeDesktopModelName(name) {
				continue
			}
			model, ok := byName[name]
			if !ok {
				model = newClaudeDesktopModel(name, "User-selected model", "", 64_000)
			}
			if _, ok := seen[model.OllamaModel]; ok {
				continue
			}
			seen[model.OllamaModel] = struct{}{}
			models = append(models, model)
		}
	}
	if len(models) > MaxClaudeDesktopModels {
		models = models[:MaxClaudeDesktopModels]
	}
	for i := range models {
		models[i].assignClaudeDesktopSlot(claudeDesktopModelSlots[i])
	}
	return models
}

// MapClaudeDesktopModels assigns explicit Claude route IDs to Ollama models.
// Empty routes are omitted and the same Ollama model may serve multiple routes.
func MapClaudeDesktopModels(available []ClaudeDesktopModel, mappings map[string]string) []ClaudeDesktopModel {
	byName := make(map[string]ClaudeDesktopModel, len(available)*2)
	for _, model := range available {
		byName[model.Name] = model
		byName[model.OllamaModel] = model
	}

	models := make([]ClaudeDesktopModel, 0, len(claudeDesktopModelSlots))
	for _, slot := range claudeDesktopModelSlots {
		name := strings.TrimSpace(mappings[slot.id])
		if !validClaudeDesktopModelName(name) {
			continue
		}
		model, ok := byName[name]
		if !ok {
			model = newClaudeDesktopModel(name, "User-selected model", "", 64_000)
		}
		model.assignClaudeDesktopSlot(slot)
		models = append(models, model)
	}
	return models
}

// ClaudeDesktopMappings returns the explicit route-to-model mapping represented
// by an assigned model catalog.
func ClaudeDesktopMappings(models []ClaudeDesktopModel) map[string]string {
	mappings := make(map[string]string, len(models))
	for _, model := range models {
		if !isClaudeDesktopRouteID(model.gateway.ID) {
			continue
		}
		mappings[model.gateway.ID] = model.OllamaModel
	}
	return mappings
}

func validClaudeDesktopRouteAssignments(models []ClaudeDesktopModel) bool {
	if len(models) == 0 {
		return false
	}
	seen := make(map[string]struct{}, len(models))
	for _, model := range models {
		if !isClaudeDesktopRouteID(model.gateway.ID) {
			return false
		}
		if _, ok := seen[model.gateway.ID]; ok {
			return false
		}
		seen[model.gateway.ID] = struct{}{}
	}
	return true
}

func isClaudeDesktopRouteID(id string) bool {
	for _, slot := range claudeDesktopModelSlots {
		if id == slot.id {
			return true
		}
	}
	return false
}

func validClaudeDesktopModelName(name string) bool {
	name = strings.TrimSpace(name)
	if name == "" {
		return false
	}
	normalized := strings.NewReplacer("-", " ", ":", " ").Replace(strings.ToLower(name))
	normalized = strings.Join(strings.Fields(normalized), " ")
	return normalized != "ollama cloud"
}

func cloneClaudeDesktopModels(models []ClaudeDesktopModel) []ClaudeDesktopModel {
	return append([]ClaudeDesktopModel(nil), models...)
}

func newClaudeDesktopModel(name, description, requiredPlan string, maxTokens int) ClaudeDesktopModel {
	ollamaModel := name
	// The app-aware recommendation contract uses required_plan for cloud aliases
	// that intentionally omit a source suffix, such as DeepSeek. Installed local
	// models are added separately and never carry plan requirements.
	cloud := requiredPlan != "" || modelref.HasExplicitCloudSource(ollamaModel)
	if cloud && !modelref.HasExplicitCloudSource(ollamaModel) {
		ollamaModel += ":cloud"
	}
	displayName := ollamaModel
	if maxTokens <= 0 {
		maxTokens = 64_000
	}

	model := ClaudeDesktopModel{
		Name:         name,
		Description:  description,
		DisplayName:  displayName,
		RequiredPlan: requiredPlan,
		OllamaModel:  ollamaModel,
		Cloud:        cloud,
		// Local models are checked against installed inventory. A cloud model
		// constructed without plan metadata must be verified by an authoritative
		// recommendation before account policy can allow it.
		entitlementKnown: !cloud || strings.TrimSpace(requiredPlan) != "",
		gateway: gatewayModel{
			Type:           "model",
			DisplayName:    displayName,
			MaxTokens:      maxTokens,
			OllamaModel:    ollamaModel,
			SupportsVision: name == "kimi-k3:cloud",
		},
	}
	return model
}

func (m *ClaudeDesktopModel) assignClaudeDesktopSlot(slot claudeDesktopModelSlot) {
	m.gateway.ID = slot.id
	m.gateway.CreatedAt = slot.createdAt
	m.gateway.AnthropicFamilyTier = slot.family
	m.gateway.IsFamilyDefault = slot.familyDefault
}

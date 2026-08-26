package proxy

import (
	"encoding/json"
	"maps"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestFetchClaudeDesktopModelsUsesAppAwareContract(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/api/experimental/model-recommendations" {
			t.Fatalf("request = %s %s", r.Method, r.URL.Path)
		}
		if got := r.URL.Query().Get("app"); got != "claude-desktop" {
			t.Fatalf("app = %q, want claude-desktop", got)
		}
		_ = json.NewEncoder(w).Encode(api.ModelRecommendationsResponse{
			Recommendations: []api.ModelRecommendation{
				{Model: "glm-5.2:cloud", Description: "GLM", MaxOutputTokens: 131_072, RequiredPlan: "pro"},
				{Model: "glm-5.3-flash:cloud", Description: "GLM Flash", MaxOutputTokens: 1_048_576, RequiredPlan: "pro"},
				{Model: "gemma4:31b-cloud", Description: "Gemma", MaxOutputTokens: 262_144, RequiredPlan: "free"},
				{Model: "deepseek-v4-pro", Description: "DeepSeek", MaxOutputTokens: 65_536, RequiredPlan: "pro"},
				{Model: "qwen3.8:27b", Description: "Qwen", MaxOutputTokens: 131_072},
			},
			Mappings: &api.ModelRecommendationMappings{
				"claude-opus-5":     {Model: "glm-5.2:cloud", RequiredPlan: "pro"},
				"claude-sonnet-5":   {Model: "glm-5.3-flash:cloud", RequiredPlan: "enterprise"},
				"unknown-route":     {Model: "deepseek-v4-pro"},
				"claude-sonnet-4-6": {Model: "missing-model:cloud"},
			},
		})
	}))
	defer server.Close()

	req, err := http.NewRequest(http.MethodGet, server.URL+"/api/experimental/model-recommendations?app=claude-desktop", nil)
	if err != nil {
		t.Fatal(err)
	}
	models, err := FetchClaudeDesktopModels(server.Client(), req)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := claudeDesktopModelNames(models), []string{"glm-5.2:cloud", "glm-5.3-flash:cloud", "gemma4:31b-cloud", "deepseek-v4-pro"}; !slices.Equal(got, want) {
		t.Fatalf("models = %v, want %v", got, want)
	}
	if models[3].OllamaModel != "deepseek-v4-pro:cloud" || !models[3].Cloud {
		t.Fatalf("cloud adapter = %+v", models[3])
	}
	if models[3].DisplayName != "deepseek-v4-pro:cloud" {
		t.Fatalf("display name = %q, want exact model identifier", models[3].DisplayName)
	}
	for _, model := range models {
		if !model.Recommended {
			t.Fatalf("endpoint model %q was not marked as recommended", model.Name)
		}
	}
	want := map[string]string{
		"claude-opus-5":   "glm-5.2:cloud",
		"claude-sonnet-5": "glm-5.3-flash:cloud",
	}
	if got := DefaultClaudeDesktopMappingsForModels(models); !maps.Equal(got, want) {
		t.Fatalf("endpoint mappings = %v, want %v", got, want)
	}
}

func TestClaudeDesktopRecommendationMappingPresence(t *testing.T) {
	recommendations := []api.ModelRecommendation{
		{Model: "glm-5.2:cloud", RequiredPlan: "pro"},
		{Model: "gemma4:31b-cloud", RequiredPlan: "free"},
	}
	tests := []struct {
		name     string
		mappings *api.ModelRecommendationMappings
		want     map[string]string
	}{
		{
			name: "omitted uses compatibility fallback",
			want: map[string]string{"claude-sonnet-5": "gemma4:31b-cloud"},
		},
		{
			name:     "empty is authoritative",
			mappings: &api.ModelRecommendationMappings{},
			want:     map[string]string{},
		},
		{
			name: "partial does not fill missing routes",
			mappings: &api.ModelRecommendationMappings{
				"claude-sonnet-5":   {Model: "gemma4:31b-cloud", RequiredPlan: "pro"},
				"claude-opus-5":     {Model: "missing-model:cloud"},
				"unknown-route":     {Model: "glm-5.2:cloud"},
				"claude-sonnet-4-6": {},
			},
			want: map[string]string{"claude-sonnet-5": "gemma4:31b-cloud"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			models := fetchClaudeDesktopModelsForTest(t, api.ModelRecommendationsResponse{
				Recommendations: recommendations,
				Mappings:        tt.mappings,
			})
			if got := DefaultClaudeDesktopMappingsForModels(models); !maps.Equal(got, tt.want) {
				t.Fatalf("mappings = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestClaudeDesktopNullRecommendationMappingsUseFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"recommendations":[{"model":"gemma4:31b-cloud","required_plan":"free"}],"mappings":null}`))
	}))
	defer server.Close()
	req, err := http.NewRequest(http.MethodGet, server.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	models, err := FetchClaudeDesktopModels(server.Client(), req)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{"claude-sonnet-5": "gemma4:31b-cloud"}
	if got := DefaultClaudeDesktopMappingsForModels(models); !maps.Equal(got, want) {
		t.Fatalf("null-contract mappings = %v, want fallback %v", got, want)
	}
}

func fetchClaudeDesktopModelsForTest(t *testing.T, response api.ModelRecommendationsResponse) []ClaudeDesktopModel {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(response)
	}))
	t.Cleanup(server.Close)
	req, err := http.NewRequest(http.MethodGet, server.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	models, err := FetchClaudeDesktopModels(server.Client(), req)
	if err != nil {
		t.Fatal(err)
	}
	return models
}

func TestFetchClaudeDesktopModelsRejectsInvalidResponses(t *testing.T) {
	for _, test := range []struct {
		name   string
		status int
		body   string
	}{
		{name: "server error", status: http.StatusBadGateway, body: `{"error":"unavailable"}`},
		{name: "malformed", status: http.StatusOK, body: `{`},
		{name: "malformed mappings", status: http.StatusOK, body: `{"recommendations":[{"model":"glm-5.2:cloud"}],"mappings":[]}`},
		{name: "legacy flat mappings", status: http.StatusOK, body: `{"recommendations":[{"model":"glm-5.2:cloud"}],"mappings":{"claude-opus-5":"glm-5.2:cloud"}}`},
		{name: "empty", status: http.StatusOK, body: `{"recommendations":[]}`},
		{name: "local only", status: http.StatusOK, body: `{"recommendations":[{"model":"qwen3.8:27b"}]}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(test.status)
				_, _ = w.Write([]byte(test.body))
			}))
			defer server.Close()
			req, err := http.NewRequest(http.MethodGet, server.URL, nil)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := FetchClaudeDesktopModels(server.Client(), req); err == nil {
				t.Fatal("FetchClaudeDesktopModels succeeded")
			}
		})
	}
}

func TestDefaultClaudeDesktopModelsExcludeMLX(t *testing.T) {
	models := DefaultClaudeDesktopModels()
	if got, want := claudeDesktopModelNames(models), []string{"glm-5.2:cloud", "kimi-k3:cloud", "deepseek-v4-pro", "deepseek-v4-flash", "gemma4:31b-cloud"}; !slices.Equal(got, want) {
		t.Fatalf("fallback models = %v, want %v", got, want)
	}
	for _, model := range models {
		if strings.Contains(strings.ToLower(model.Name), "mlx") {
			t.Fatalf("fallback contains MLX model %q", model.Name)
		}
	}
	if models[3].OllamaModel != "deepseek-v4-flash:0731:cloud" {
		t.Fatalf("fallback Flash route = %q", models[3].OllamaModel)
	}
}

func TestSelectClaudeDesktopModelsPrioritizesExplicitSelection(t *testing.T) {
	available := DefaultClaudeDesktopModels()
	selected := SelectClaudeDesktopModels(available, []string{"deepseek-v4-flash", "glm-5.2:cloud"})
	if got, want := claudeDesktopModelNames(selected), []string{"deepseek-v4-flash", "glm-5.2:cloud"}; !slices.Equal(got, want) {
		t.Fatalf("selected models = %v, want %v", got, want)
	}

	cloudRoute := SelectClaudeDesktopModels(available, []string{"deepseek-v4-flash:0731:cloud"})
	if len(cloudRoute) != 1 || cloudRoute[0].Name != "deepseek-v4-flash" || cloudRoute[0].OllamaModel != "deepseek-v4-flash:0731:cloud" || !cloudRoute[0].Cloud {
		t.Fatalf("cloud route selection = %+v", cloudRoute)
	}

	custom := SelectClaudeDesktopModels(available, []string{"custom-model"})
	if len(custom) != 1 || custom[0].Name != "custom-model" || custom[0].OllamaModel != "custom-model" {
		t.Fatalf("custom selection = %+v", custom)
	}
	if got, want := custom[0].GatewayID(), "claude-fable-5"; got != want {
		t.Fatalf("custom gateway ID = %q, want validated slot %q", got, want)
	}
	if custom[0].Recommended {
		t.Fatal("custom selection was marked as recommended")
	}

	withoutSentinel := SelectClaudeDesktopModels(available, []string{"Ollama Cloud", "ollama:cloud", "qwen3:8b"})
	if got, want := claudeDesktopModelNames(withoutSentinel), []string{"qwen3:8b"}; !slices.Equal(got, want) {
		t.Fatalf("selection without invalid sentinel = %v, want %v", got, want)
	}
}

func TestClaudeDesktopModelsFromCloudInventoryVerifiesWithoutRecommending(t *testing.T) {
	models := ClaudeDesktopModelsFromCloudInventory([]string{
		"glm-5.2:cloud",
		"glm-5.2:cloud",
		"gemma4:31b-cloud",
		"qwen3:8b",
		"Ollama Cloud",
	})
	if len(models) != 3 {
		t.Fatalf("models = %+v, want three account cloud models", models)
	}
	for _, model := range models {
		if !model.Cloud || !model.entitlementKnown {
			t.Fatalf("cloud inventory model = %+v, want verified cloud model", model)
		}
		if model.Recommended {
			t.Fatalf("cloud inventory model %q must not be recommended", model.Name)
		}
		if !model.AccountCloud {
			t.Fatalf("cloud inventory model %q is missing account membership", model.Name)
		}
	}
	if models[2].OllamaModel != "qwen3:8b:cloud" {
		t.Fatalf("normalized cloud route = %q", models[2].OllamaModel)
	}
}

func TestVerifyClaudeDesktopModelsWithCloudInventoryPreservesMetadata(t *testing.T) {
	models := UnverifyClaudeDesktopCloudEntitlements(DefaultClaudeDesktopModels())
	inventory := ClaudeDesktopModelsFromCloudInventory([]string{"glm-5.2"})

	verified := VerifyClaudeDesktopModelsWithCloudInventory(models, inventory)
	if !verified[0].AccountCloud || !verified[0].entitlementKnown {
		t.Fatalf("verified model = %+v, want account entitlement", verified[0])
	}
	if verified[0].Description != models[0].Description ||
		verified[0].RequiredPlan != models[0].RequiredPlan ||
		verified[0].Recommended != models[0].Recommended ||
		verified[0].OllamaModel != models[0].OllamaModel {
		t.Fatalf("verified model metadata = %+v, want %+v", verified[0], models[0])
	}
	if verified[1].AccountCloud || verified[1].entitlementKnown {
		t.Fatalf("unmatched model = %+v, want unverified", verified[1])
	}

	access := EvaluateClaudeDesktopModelAccess(verified[0], ClaudeDesktopAccessState{
		Cloud:   ClaudeDesktopCloudOn,
		Account: ClaudeDesktopAccountSignedIn,
		Plan:    "pro",
	}, false, true)
	if access.Availability != ClaudeDesktopAvailabilityAvailable {
		t.Fatalf("verified model access = %+v, want available", access)
	}
}

func TestPreserveClaudeDesktopCloudEntitlementsKeepsOfflineRoutes(t *testing.T) {
	fallback := UnverifyClaudeDesktopCloudEntitlements(DefaultClaudeDesktopModels())
	previous := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{
		{Model: "deepseek-v4-flash", RequiredPlan: "pro"},
	})
	previous = VerifyClaudeDesktopModelsWithCloudInventory(previous, ClaudeDesktopModelsFromCloudInventory([]string{"deepseek-v4-flash"}))

	preserved := PreserveClaudeDesktopCloudEntitlements(fallback, previous)
	if got := preserved[3].OllamaModel; got != "deepseek-v4-flash:0731:cloud" {
		t.Fatalf("offline route = %q, want pinned fallback", got)
	}
	if !preserved[3].AccountCloud || !preserved[3].entitlementKnown {
		t.Fatalf("preserved entitlement = %+v", preserved[3])
	}
}

func claudeDesktopModelNames(models []ClaudeDesktopModel) []string {
	names := make([]string, len(models))
	for i, model := range models {
		names[i] = model.Name
	}
	return names
}

func TestSelectClaudeDesktopModelsAssignsValidatedClaudeIDs(t *testing.T) {
	available := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{
		{Model: "glm-5.2:cloud", RequiredPlan: "pro"},
		{Model: "kimi-k3:cloud", RequiredPlan: "pro"},
		{Model: "deepseek-v4-pro", RequiredPlan: "pro"},
		{Model: "deepseek-v4-flash", RequiredPlan: "pro"},
		{Model: "gemma4:26b:cloud", RequiredPlan: "pro"},
	})
	selected := SelectClaudeDesktopModels(available, []string{
		"glm-5.2:cloud",
		"kimi-k3:cloud",
		"deepseek-v4-pro",
		"deepseek-v4-flash",
		"gemma4:26b:cloud",
	})
	if len(selected) != 5 {
		t.Fatalf("selected models = %v", claudeDesktopModelNames(selected))
	}
	wantIDs := []string{
		"claude-fable-5",
		"claude-opus-5",
		"claude-sonnet-5",
		"claude-haiku-4-5-20251001",
		"claude-sonnet-4-6",
	}
	seen := make(map[string]struct{}, len(selected))
	for i, model := range selected {
		id := model.GatewayID()
		if id != wantIDs[i] {
			t.Fatalf("gateway ID %d = %q, want literal slot %q", i, id, wantIDs[i])
		}
		if _, ok := seen[id]; ok {
			t.Fatalf("duplicate gateway ID %q in %v", id, selected)
		}
		seen[id] = struct{}{}
	}

	// Custom installed models use the same validated slots while preserving the
	// Ollama route separately.
	withCustom := SelectClaudeDesktopModels(available, []string{"kimi-k3:cloud", "mycustommodel:7b"})
	if len(withCustom) != 2 || withCustom[1].Name != "mycustommodel:7b" {
		t.Fatalf("custom selection = %v", claudeDesktopModelNames(withCustom))
	}
	if got, want := withCustom[1].GatewayID(), "claude-opus-5"; got != want {
		t.Fatalf("custom gateway ID = %q, want validated slot %q", got, want)
	}
	if withCustom[1].OllamaModel != "mycustommodel:7b" {
		t.Fatalf("custom Ollama route = %q", withCustom[1].OllamaModel)
	}

	// Reordering the persisted selection reassigns the slots in that same order.
	// The gateway catalog and request router consume this one mapping together.
	merged := append(append([]ClaudeDesktopModel(nil), available...), withCustom[1])
	reordered := SelectClaudeDesktopModels(merged, []string{"mycustommodel:7b", "kimi-k3:cloud"})
	if len(reordered) != 2 || reordered[0].GatewayID() != "claude-fable-5" || reordered[1].GatewayID() != "claude-opus-5" {
		t.Fatalf("reordered gateway IDs = %q/%q", reordered[0].GatewayID(), reordered[1].GatewayID())
	}
}

func TestSelectClaudeDesktopModelsIgnoresRecommendationReordering(t *testing.T) {
	first := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{
		{Model: "glm-5.2:cloud", RequiredPlan: "pro"},
		{Model: "kimi-k3:cloud", RequiredPlan: "pro"},
	})
	second := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{
		{Model: "kimi-k3:cloud", RequiredPlan: "pro"},
		{Model: "glm-5.2:cloud", RequiredPlan: "pro"},
	})
	persisted := []string{"glm-5.2:cloud", "kimi-k3:cloud"}

	for _, available := range [][]ClaudeDesktopModel{first, second} {
		selected := SelectClaudeDesktopModels(available, persisted)
		if len(selected) != 2 {
			t.Fatalf("selected models = %+v", selected)
		}
		if selected[0].OllamaModel != "glm-5.2:cloud" || selected[0].GatewayID() != "claude-fable-5" {
			t.Fatalf("GLM mapping = %q -> %q", selected[0].GatewayID(), selected[0].OllamaModel)
		}
		if selected[1].OllamaModel != "kimi-k3:cloud" || selected[1].GatewayID() != "claude-opus-5" {
			t.Fatalf("Kimi mapping = %q -> %q", selected[1].GatewayID(), selected[1].OllamaModel)
		}
	}
}

func TestSelectClaudeDesktopModelsCapsAtLiteralSlots(t *testing.T) {
	available := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{
		{Model: "glm-5.2:cloud", RequiredPlan: "pro"},
		{Model: "kimi-k3:cloud", RequiredPlan: "pro"},
		{Model: "deepseek-v4-pro", RequiredPlan: "pro"},
		{Model: "deepseek-v4-flash", RequiredPlan: "pro"},
		{Model: "gemma4:26b:cloud", RequiredPlan: "pro"},
		{Model: "qwen3.8:27b", RequiredPlan: ""},
	})
	if len(available) != 6 {
		t.Fatalf("catalog = %d models, want 6", len(available))
	}
	if available[5].GatewayID() != "" {
		t.Fatalf("unselected catalog entry ID = %q, want no assigned slot", available[5].GatewayID())
	}

	// Without an explicit selection the first five recommendations apply.
	defaults := SelectClaudeDesktopModels(available, nil)
	if len(defaults) != MaxClaudeDesktopModels {
		t.Fatalf("default selection = %d models, want %d", len(defaults), MaxClaudeDesktopModels)
	}

	// Any recommendation can be selected explicitly even beyond the default five.
	qwen := SelectClaudeDesktopModels(available, []string{"kimi-k3:cloud", "qwen3.8:27b"})
	if len(qwen) != 2 || qwen[1].GatewayID() != "claude-opus-5" || qwen[1].OllamaModel != "qwen3.8:27b" {
		t.Fatalf("unmapped catalog selection = %+v", qwen)
	}

	// Selections never exceed the five literal slots.
	tooMany := SelectClaudeDesktopModels(available, []string{
		"glm-5.2:cloud",
		"kimi-k3:cloud",
		"deepseek-v4-pro",
		"deepseek-v4-flash",
		"gemma4:26b:cloud",
		"qwen3.8:27b",
	})
	if len(tooMany) != MaxClaudeDesktopModels {
		t.Fatalf("capped selection = %d models, want %d", len(tooMany), MaxClaudeDesktopModels)
	}
	for _, name := range claudeDesktopModelNames(tooMany) {
		if name == "qwen3.8:27b" {
			t.Fatalf("sixth selection %q survived the cap", name)
		}
	}
	seen := make(map[string]struct{}, len(tooMany))
	for _, model := range tooMany {
		id := model.GatewayID()
		if id == "" {
			t.Fatalf("selected model %q has no Claude ID", model.Name)
		}
		if _, ok := seen[id]; ok {
			t.Fatalf("duplicate gateway ID %q", id)
		}
		seen[id] = struct{}{}
	}
}

func TestMapClaudeDesktopModelsSupportsUnassignedAndSharedModels(t *testing.T) {
	available := DefaultClaudeDesktopModels()
	mapped := MapClaudeDesktopModels(available, map[string]string{
		"claude-fable-5":  "glm-5.2:cloud",
		"claude-opus-5":   "glm-5.2:cloud",
		"claude-sonnet-5": "",
	})

	if len(mapped) != 2 {
		t.Fatalf("mapped models = %d, want 2 assigned routes", len(mapped))
	}
	if mapped[0].GatewayID() != "claude-fable-5" || mapped[1].GatewayID() != "claude-opus-5" {
		t.Fatalf("gateway IDs = %q/%q", mapped[0].GatewayID(), mapped[1].GatewayID())
	}
	if mapped[0].OllamaModel != "glm-5.2:cloud" || mapped[1].OllamaModel != "glm-5.2:cloud" {
		t.Fatalf("shared mappings = %q/%q", mapped[0].OllamaModel, mapped[1].OllamaModel)
	}
	if got := ClaudeDesktopMappings(mapped); !maps.Equal(got, map[string]string{
		"claude-fable-5": "glm-5.2:cloud",
		"claude-opus-5":  "glm-5.2:cloud",
	}) {
		t.Fatalf("mappings = %v", got)
	}
}

func TestSelectClaudeDesktopModelsPreservesExplicitRouteAssignments(t *testing.T) {
	mapped := MapClaudeDesktopModels(DefaultClaudeDesktopModels(), map[string]string{
		"claude-sonnet-5": "kimi-k3:cloud",
	})
	selected := SelectClaudeDesktopModels(mapped, nil)
	if len(selected) != 1 || selected[0].GatewayID() != "claude-sonnet-5" {
		t.Fatalf("selected models = %+v", selected)
	}
}

func TestClaudeDesktopRoutesExposeStableOrder(t *testing.T) {
	routes := ClaudeDesktopRoutes()
	if len(routes) != MaxClaudeDesktopModels {
		t.Fatalf("routes = %d, want %d", len(routes), MaxClaudeDesktopModels)
	}
	if routes[0].ID != "claude-fable-5" || routes[0].DisplayName != "Fable 5" || routes[4].ID != "claude-sonnet-4-6" {
		t.Fatalf("routes = %+v", routes)
	}
}

func TestDefaultClaudeDesktopMappingsUsesSafeFallback(t *testing.T) {
	want := map[string]string{"claude-sonnet-5": "gemma4:31b-cloud"}
	if got := DefaultClaudeDesktopMappings(); !maps.Equal(got, want) {
		t.Fatalf("fallback defaults = %v, want %v", got, want)
	}
}

func TestDefaultClaudeDesktopMappingsUseCurrentCatalog(t *testing.T) {
	models := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{
		{Model: "glm-5.2:cloud", RequiredPlan: "pro"},
		{Model: "glm-5.3-flash:cloud", RequiredPlan: "pro"},
		{Model: "kimi-k3:cloud", RequiredPlan: "pro"},
		{Model: "deepseek-v4-pro", RequiredPlan: "pro"},
		{Model: "deepseek-v4-flash", RequiredPlan: "pro"},
		{Model: "gemma4:31b-cloud", RequiredPlan: "free"},
	})
	got := DefaultClaudeDesktopMappingsForModels(models)
	want := map[string]string{"claude-sonnet-5": "gemma4:31b-cloud"}
	if !maps.Equal(got, want) {
		t.Fatalf("catalog defaults = %v, want %v", got, want)
	}
	fallback := DefaultClaudeDesktopMappingsForModels(DefaultClaudeDesktopModels())
	if got, want := fallback["claude-sonnet-5"], "gemma4:31b-cloud"; got != want {
		t.Fatalf("fallback Sonnet default = %q, want %q", got, want)
	}
}

func TestDefaultClaudeDesktopMappingsOnlyUseAvailableModels(t *testing.T) {
	models := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{
		{Model: "glm-5.2:cloud", RequiredPlan: "pro"},
	})
	want := map[string]string{}
	if got := DefaultClaudeDesktopMappingsForModels(models); !maps.Equal(got, want) {
		t.Fatalf("partial catalog defaults = %v, want %v", got, want)
	}
}

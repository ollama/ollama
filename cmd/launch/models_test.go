package launch

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	modelpkg "github.com/ollama/ollama/types/model"
)

func TestBuildModelList_UsesInventoryMetadataForInstalledModels(t *testing.T) {
	existing := []modelInfo{
		{
			Name:         "custom-tools:latest",
			ToolCapable:  true,
			Capabilities: []modelpkg.Capability{modelpkg.CapabilityCompletion, modelpkg.CapabilityTools, modelpkg.CapabilityThinking},
			Size:         7500 * format.MegaByte,
			Details: api.ModelDetails{
				ParameterSize:     "8B",
				QuantizationLevel: "Q4_K_M",
				ContextLength:     131_072,
				EmbeddingLength:   4096,
			},
		},
	}

	items, _, _, _ := buildModelList(existing, nil, "")
	var got ModelItem
	for _, item := range items {
		if item.Name == "custom-tools" {
			got = item
			break
		}
	}
	if got.Name == "" {
		t.Fatal("custom-tools not found in items")
	}
	if !got.ToolCapable {
		t.Fatal("expected installed model to preserve tool capability from tags metadata")
	}
	if got.Details.ContextLength != 131_072 {
		t.Fatalf("Details.ContextLength = %d, want 131072", got.Details.ContextLength)
	}
	if got.Size != 7500*format.MegaByte {
		t.Fatalf("Size = %d, want %d", got.Size, 7500*format.MegaByte)
	}
	if got.Description != "" {
		t.Fatalf("Description = %q, want empty for installed model without recommendation copy", got.Description)
	}
}

func TestBuildModelList_InstalledRecommendedPreservesRecommendationAndMetadata(t *testing.T) {
	existing := []modelInfo{
		{
			Name:         "qwen3.5",
			ToolCapable:  true,
			Capabilities: []modelpkg.Capability{modelpkg.CapabilityCompletion, modelpkg.CapabilityTools, modelpkg.CapabilityVision},
			Size:         14 * format.GigaByte,
			Details:      api.ModelDetails{ContextLength: 262_144},
		},
	}

	items, _, _, _ := buildModelList(existing, nil, "")
	var got ModelItem
	for _, item := range items {
		if item.Name == "qwen3.5" {
			got = item
			break
		}
	}
	if got.Name == "" {
		t.Fatal("qwen3.5 not found in items")
	}
	if !got.Recommended || !got.ToolCapable {
		t.Fatalf("recommended/tool metadata = %v/%v, want true/true", got.Recommended, got.ToolCapable)
	}
	if got.Details.ContextLength != 262_144 {
		t.Fatalf("Details.ContextLength = %d, want 262144", got.Details.ContextLength)
	}
	if got.Description != "Reasoning, coding, and visual understanding locally" {
		t.Fatalf("Description = %q, want recommendation description", got.Description)
	}
}

func TestRecommendedModelsFallbackIsNeutral(t *testing.T) {
	want := []string{
		"kimi-k2.6:cloud",
		"qwen3.5:cloud",
		"glm-5.1:cloud",
		"minimax-m2.7:cloud",
		"gemma4",
		"qwen3.5",
	}
	if !slices.Equal(modelItemNames(recommendedModels), want) {
		t.Fatalf("models = %v, want %v", modelItemNames(recommendedModels), want)
	}
	for _, name := range modelItemNames(recommendedModels) {
		if strings.Contains(name, "-mlx") {
			t.Fatalf("fallback recommendation should not use MLX tag: %q", name)
		}
	}
}

func TestLauncherRequestRecommendationsTrustsServerModelNames(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/experimental/model-recommendations" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `{"recommendations":[{"model":"gemma4","description":"local","vram_bytes":12884901888},{"model":"qwen3.5:9b","description":"local","vram_bytes":15032385536},{"model":"qwen3.6","description":"local","vram_bytes":25769803776}]}`)
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	client := &launcherClient{apiClient: api.NewClient(u, srv.Client())}
	got, err := client.requestRecommendations(context.Background())
	if err != nil {
		t.Fatalf("requestRecommendations failed: %v", err)
	}

	want := []string{"gemma4", "qwen3.5:9b", "qwen3.6"}
	if !slices.Equal(modelItemNames(got), want) {
		t.Fatalf("models = %v, want %v", modelItemNames(got), want)
	}
}

func modelItemNames(items []ModelItem) []string {
	names := make([]string, len(items))
	for i, item := range items {
		names[i] = item.Name
	}
	return names
}

package launch

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	modelpkg "github.com/ollama/ollama/types/model"
)

func TestBuildModelList_UsesInventoryMetadataForInstalledModels(t *testing.T) {
	existing := []modelInfo{
		{
			Name:            "custom-tools:latest",
			ToolCapable:     true,
			Capabilities:    []modelpkg.Capability{modelpkg.CapabilityCompletion, modelpkg.CapabilityTools, modelpkg.CapabilityThinking},
			ContextLength:   131_072,
			EmbeddingLength: 4096,
			Size:            7500 * format.MegaByte,
			Details: api.ModelDetails{
				ParameterSize:     "8B",
				QuantizationLevel: "Q4_K_M",
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
	if got.ContextLength != 131_072 {
		t.Fatalf("ContextLength = %d, want 131072", got.ContextLength)
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
			Name:          "qwen3.5",
			ToolCapable:   true,
			Capabilities:  []modelpkg.Capability{modelpkg.CapabilityCompletion, modelpkg.CapabilityTools, modelpkg.CapabilityVision},
			ContextLength: 262_144,
			Size:          14 * format.GigaByte,
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
	if got.ContextLength != 262_144 {
		t.Fatalf("ContextLength = %d, want 262144", got.ContextLength)
	}
	if got.Description != "Reasoning, coding, and visual understanding locally" {
		t.Fatalf("Description = %q, want recommendation description", got.Description)
	}
}

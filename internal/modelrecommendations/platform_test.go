package modelrecommendations

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
)

func TestApplyPlatformTagsForGOOSDarwinUsesMLXTags(t *testing.T) {
	input := []api.ModelRecommendation{
		{Model: "kimi-k2.6:cloud", Description: "cloud"},
		{Model: "gemma4", Description: "local", VRAMBytes: 12 * format.GigaByte},
		{Model: "qwen3.5:9b", Description: "local", VRAMBytes: 14 * format.GigaByte},
		{Model: "qwen3.6:27b", Description: "local", VRAMBytes: 17 * format.GigaByte},
	}

	got := ApplyPlatformTagsForGOOS(input, "darwin")
	wantNames := []string{"kimi-k2.6:cloud", "gemma4:e4b-mlx", "qwen3.5:9b-mlx", "qwen3.6:27b-mlx"}
	if !slices.Equal(modelNames(got), wantNames) {
		t.Fatalf("models = %v, want %v", modelNames(got), wantNames)
	}

	if got[1].Description != "MLX-optimized reasoning and code generation locally" {
		t.Fatalf("gemma description = %q", got[1].Description)
	}
	if got[1].VRAMBytes != 96*format.GigaByte/10 {
		t.Fatalf("gemma VRAMBytes = %d, want %d", got[1].VRAMBytes, 96*format.GigaByte/10)
	}
	if got[2].Description != "MLX-optimized reasoning and coding locally" {
		t.Fatalf("qwen description = %q", got[2].Description)
	}
	if got[2].VRAMBytes != 89*format.GigaByte/10 {
		t.Fatalf("qwen VRAMBytes = %d, want %d", got[2].VRAMBytes, 89*format.GigaByte/10)
	}
	if got[3].Description != "MLX-optimized coding and reasoning locally" {
		t.Fatalf("qwen3.6 description = %q", got[3].Description)
	}
	if got[3].VRAMBytes != 20*format.GigaByte {
		t.Fatalf("qwen3.6 VRAMBytes = %d, want %d", got[3].VRAMBytes, 20*format.GigaByte)
	}
}

func TestApplyPlatformTagsForGOOSNonDarwinPreservesRecommendations(t *testing.T) {
	input := []api.ModelRecommendation{
		{Model: "gemma4", Description: "local", VRAMBytes: 12 * format.GigaByte},
		{Model: "qwen3.5", Description: "local", VRAMBytes: 14 * format.GigaByte},
	}

	got := ApplyPlatformTagsForGOOS(input, "linux")
	if !slices.Equal(got, input) {
		t.Fatalf("recommendations = %#v, want %#v", got, input)
	}
}

func TestApplyPlatformTagsForGOOSDedupesAfterReplacement(t *testing.T) {
	input := []api.ModelRecommendation{
		{Model: "gemma4", Description: "base"},
		{Model: "gemma4:e4b-mlx", Description: "explicit"},
	}

	got := ApplyPlatformTagsForGOOS(input, "darwin")
	wantNames := []string{"gemma4:e4b-mlx"}
	if !slices.Equal(modelNames(got), wantNames) {
		t.Fatalf("models = %v, want %v", modelNames(got), wantNames)
	}
}

func TestApplyPlatformTagsForGOOSDoesNotSynthesizeMLXTags(t *testing.T) {
	input := []api.ModelRecommendation{
		{Model: "llama3.2", Description: "local"},
		{Model: "qwen3.6:14b", Description: "not published"},
	}

	got := ApplyPlatformTagsForGOOS(input, "darwin")
	if !slices.Equal(got, input) {
		t.Fatalf("recommendations = %#v, want %#v", got, input)
	}
}

func modelNames(recs []api.ModelRecommendation) []string {
	names := make([]string, len(recs))
	for i, rec := range recs {
		names[i] = rec.Model
	}
	return names
}

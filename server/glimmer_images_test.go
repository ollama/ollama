package server

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestGlimmerSafetensorsCapabilities(t *testing.T) {
	m := Model{
		Config: model.ConfigV2{
			ModelFormat:  "safetensors",
			Renderer:     "glimmer",
			Parser:       "glimmer",
			Capabilities: []string{"completion", "vision", "audio"},
		},
	}

	want := []model.Capability{
		model.CapabilityCompletion,
		model.CapabilityVision,
		model.CapabilityTools,
		model.CapabilityThinking,
	}
	if got := m.Capabilities(); !slices.Equal(got, want) {
		t.Fatalf("capabilities = %v, want %v", got, want)
	}
}

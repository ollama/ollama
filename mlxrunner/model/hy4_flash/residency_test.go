package hy4_flash

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/manifest"
)

func TestConfiguredDFlyRetainsDraftWeights(t *testing.T) {
	const layer = "model.dfly_layers.0.mlp.down_proj.weight"
	const confidence = "model.dfly_layers.confidence_head.weight"
	config := []byte(`{"dfly_num_hidden_layers": 5, "speculative_config": {"method":"dfly","num_speculative_tokens":5}}`)
	got := CheckpointSkippedTensorNames(config, []manifest.Layer{{Name: layer}, {Name: confidence}})
	if !slices.Equal(got, []string{confidence}) {
		t.Fatalf("excluded tensors = %v, want only %s", got, confidence)
	}
}

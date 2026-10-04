package create

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/ollama/ollama/types/model"
)

// The registered factory parses the arch's own config.json fields
// (text_config.embedding_dim, text_config.max_position_embeddings) and
// augmentConfig fills the manifest fields the shared inference can't derive.
// Capabilities follow the towers the checkpoint ships.
func TestGemma4EmbeddingAugmentConfig(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want []string
	}{
		{
			name: "text only",
			raw:  `{"text_config":{"embedding_dim":768,"max_position_embeddings":262144}}`,
			want: []string{"embedding"},
		},
		{
			name: "null towers",
			raw:  `{"text_config":{"embedding_dim":768,"max_position_embeddings":262144},"vision_config":null,"audio_config":null}`,
			want: []string{"embedding"},
		},
		{
			name: "vision tower",
			raw:  `{"text_config":{"embedding_dim":768,"max_position_embeddings":262144},"vision_config":{"model_type":"gemma4_vision"}}`,
			want: []string{"embedding", "vision"},
		},
		{
			name: "audio tower",
			raw:  `{"text_config":{"embedding_dim":768,"max_position_embeddings":262144},"audio_config":{"model_type":"gemma4_audio"}}`,
			want: []string{"embedding", "audio"},
		},
		{
			name: "both towers",
			raw:  `{"text_config":{"embedding_dim":768,"max_position_embeddings":262144},"vision_config":{"model_type":"gemma4_vision"},"audio_config":{"model_type":"gemma4_audio"}}`,
			want: []string{"embedding", "vision", "audio"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := newGemma4EmbeddingImportTransform(json.RawMessage(tt.raw))
			if err != nil {
				t.Fatal(err)
			}
			aug, ok := p.(interface{ augmentConfig(*model.ConfigV2) error })
			if !ok {
				t.Fatal("policy does not implement augmentConfig")
			}
			var cfg model.ConfigV2
			if err := aug.augmentConfig(&cfg); err != nil {
				t.Fatal(err)
			}
			if cfg.EmbedLen != 768 {
				t.Errorf("EmbedLen = %d, want 768", cfg.EmbedLen)
			}
			if cfg.ContextLen != 8192 {
				t.Errorf("ContextLen = %d, want 8192", cfg.ContextLen)
			}
			if !slices.Equal(cfg.Capabilities, tt.want) {
				t.Errorf("Capabilities = %v, want %v", cfg.Capabilities, tt.want)
			}
		})
	}

	// Missing keys must fail loudly, not write zeros into the manifest.
	for name, raw := range map[string]json.RawMessage{
		"no embedding_dim":           json.RawMessage(`{"text_config":{"max_position_embeddings":262144}}`),
		"no max_position_embeddings": json.RawMessage(`{"text_config":{"embedding_dim":768}}`),
	} {
		p, err := newGemma4EmbeddingImportTransform(raw)
		if err != nil {
			t.Fatal(err)
		}
		aug := p.(interface{ augmentConfig(*model.ConfigV2) error })
		var got model.ConfigV2
		if err := aug.augmentConfig(&got); err == nil {
			t.Errorf("%s: augmentConfig succeeded, want error", name)
		}
	}
}

// The output head stays at source precision under every arch spelling,
// including the new-arch language_model.embedding_projection nesting.
func TestGemma4EmbeddingProjectionExcluded(t *testing.T) {
	p, err := newGemma4EmbeddingImportTransform(json.RawMessage(
		`{"text_config":{"embedding_dim":768,"max_position_embeddings":262144}}`))
	if err != nil {
		t.Fatal(err)
	}
	shape := []int32{768, 512} // rows, cols — above the 1024-element floor
	for _, name := range []string{
		"embedding_projection.weight",
		"model.embedding_projection.weight",
		"language_model.embedding_projection.weight",
	} {
		if got := p.quantizationType(name, shape, "nvfp4"); got != "" {
			t.Errorf("quantizationType(%q) = %q, want \"\"", name, got)
		}
	}
	// A trunk tensor still quantizes.
	if got := p.quantizationType("language_model.layers.0.mlp.gate_proj.weight", []int32{2048, 512}, "nvfp4"); got != "nvfp4" {
		t.Errorf("trunk tensor quantization = %q, want nvfp4", got)
	}
}

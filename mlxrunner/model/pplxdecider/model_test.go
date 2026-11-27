package pplxdecider

import (
	"testing"

	"github.com/ollama/ollama/mlxrunner/model/qwen3_5"
)

func TestContextLengthUsesBackboneMetadata(t *testing.T) {
	m := &Model{Model: &qwen3_5.Model{Config: &qwen3_5.Config{MaxPositionEmbeddings: 262144}}}
	if got := m.MaxContextLength(); got != 262144 {
		t.Fatalf("context length = %d, want 262144 from text_config.max_position_embeddings", got)
	}
}

package mlxrunner

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/tokenizer"
)

// textOnlyModel satisfies base.Model but not base.MediaModel.
type textOnlyModel struct{}

func (textOnlyModel) LoadWeights(map[string]*mlx.Array) error { return nil }
func (textOnlyModel) NewCaches() []cache.Cache                { return nil }
func (textOnlyModel) Forward(*batch.Batch, []cache.Cache) (*mlx.Array, *mlx.Array) {
	return nil, nil
}
func (textOnlyModel) Unembed(x *mlx.Array) *mlx.Array { return x }
func (textOnlyModel) Tokenizer() *tokenizer.Tokenizer { return nil }
func (textOnlyModel) MaxContextLength() int           { return 0 }

func TestPrepareRejectsMediaWithoutSupport(t *testing.T) {
	r := &Runner{Model: textOnlyModel{}}
	req := &Request{
		CompletionRequest: CompletionRequest{
			Prompt: "[img-0] what is this?",
			Media:  []llm.MediaData{{ID: 0, Kind: llm.MediaKindImage, Data: []byte{1}}},
		},
	}

	err := r.Prepare(req)
	if err == nil {
		t.Fatal("expected error for media on a text-only model")
	}
	if !strings.Contains(err.Error(), "does not support image input") {
		t.Fatalf("unexpected error: %v", err)
	}
}

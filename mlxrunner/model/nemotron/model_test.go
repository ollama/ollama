package nemotron

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
)

func TestMLXLMForwardReference(t *testing.T) {
	data, err := os.ReadFile("testdata/tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Tokens []int32     `json:"tokens"`
		Logits [][]float32 `json:"logits"`
	}
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	mlxtest.Run(t, func(t *mlxtest.T) {
		tensors := map[string]*mlx.Array{}
		for name, value := range mlx.Load("testdata/tiny.safetensors") {
			tensors[name] = value
		}
		m := &Model{Config: &Config{HiddenSize: 8, NumHiddenLayers: 2,
			IntermediateSize: 16, NumAttentionHeads: 2, NumKeyValueHeads: 1,
			VocabSize: 16, HeadDim: 4, Scale: 0.5, PartialRotaryFactor: 0.5,
			NormEps: 1e-5, RopeTheta: 10000}, Layers: make([]*Layer, 2)}
		if err := m.LoadWeights(tensors); err != nil {
			t.Fatal(err)
		}
		forward := func(tokens []int32, offset int32, caches []cache.Cache) *mlx.Array {
			b := &batch.Batch{InputIDs: mlx.FromValues(tokens, 1, len(tokens)),
				SeqOffsets: []int32{offset}, SeqQueryLens: []int32{int32(len(tokens))}}
			hidden, _ := m.Forward(b, caches)
			return m.Unembed(hidden)
		}
		full := forward(ref.Tokens, 0, nil)
		caches := m.NewCaches()
		prefill := forward(ref.Tokens[:3], 0, caches)
		decode1 := forward(ref.Tokens[3:4], 3, caches)
		decode2 := forward(ref.Tokens[4:], 4, caches)
		cached := mlx.Concatenate([]*mlx.Array{prefill, decode1, decode2}, 1)
		mlx.Eval(full, cached)
		for name, values := range map[string][]float32{"full": full.Floats(), "cached": cached.Floats()} {
			for row, logits := range ref.Logits {
				for col, want := range logits {
					got := values[row*16+col]
					if math.IsNaN(float64(got)) || math.Abs(float64(got-want)) > 2e-5 {
						t.Fatalf("%s[%d,%d] = %g, mlx-lm reference = %g", name, row, col, got, want)
					}
				}
			}
		}
		delete(tensors, "model.layers.0.mlp.up_proj.weight")
		if m.LoadWeights(tensors) == nil {
			t.Fatal("missing projection should fail")
		}
	})
}

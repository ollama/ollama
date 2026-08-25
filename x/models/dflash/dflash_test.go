package dflash

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/tokenizer"
)

func TestParseDFlash2Config(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"hidden_size": 5120,
		"num_hidden_layers": 5,
		"num_attention_heads": 40,
		"num_key_value_heads": 8,
		"head_dim": 128,
		"rms_norm_eps": 1e-6,
		"vocab_size": 248320,
		"sliding_window": 2048,
		"layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
		"is_causal": false,
		"dflash_config": {
			"block_size": 8,
			"mask_token_id": 248070,
			"target_layer_ids": [5, 19, 33, 47, 61],
			"conv_kernel_size": 2,
			"conv_group_size": 16,
			"selector_rank": 256,
			"selector_top_k": 16
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Causal == nil || *cfg.Causal {
		t.Fatalf("Causal = %v, want false", cfg.Causal)
	}
	if cfg.BlockSize != 8 || cfg.ConvKernelSize != 2 || cfg.ConvGroupSize != 16 {
		t.Fatalf("block/conv config = %d/%d/%d", cfg.BlockSize, cfg.ConvKernelSize, cfg.ConvGroupSize)
	}
	if cfg.SelectorRank != 256 || cfg.SelectorTopK != 16 {
		t.Fatalf("selector config = %d/%d", cfg.SelectorRank, cfg.SelectorTopK)
	}
	if cfg.InputEmbeddingScale != 1 || cfg.OutputMultiplier != 1 {
		t.Fatalf("default scales = %g/%g, want 1/1", cfg.InputEmbeddingScale, cfg.OutputMultiplier)
	}
}

func TestGroupedDynamicConvIsBlockLocalAndCausal(t *testing.T) {
	mlxtest.Setup(t)
	c := &GroupedDynamicConv{KernelSize: 2, GroupSize: 2}
	hidden := mlx.FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 3, 2)
	dynamic := mlx.Zeros(mlx.DTypeFloat32, 1, 3, 2, 1)
	base := mlx.FromValues([]float32{1, 1, 10, 10}, 2, 2)
	out := c.convolve(hidden, dynamic, base)
	mlx.Eval(out)
	want := []float32{1, 2, 13, 24, 35, 46}
	for i, got := range out.Floats() {
		if math.Abs(float64(got-want[i])) > 1e-5 {
			t.Fatalf("output[%d] = %g, want %g", i, got, want[i])
		}
	}
}

type selectorTarget struct{ logits *mlx.Array }

func (t *selectorTarget) LoadWeights(map[string]*mlx.Array) error { return nil }
func (t *selectorTarget) NewCaches() []cache.Cache                { return nil }
func (t *selectorTarget) Forward(*batch.Batch, []cache.Cache) (*mlx.Array, *mlx.Array) {
	return nil, nil
}
func (t *selectorTarget) Unembed(*mlx.Array) *mlx.Array         { return t.logits }
func (t *selectorTarget) RawLogits(*mlx.Array) *mlx.Array       { return t.logits }
func (t *selectorTarget) TokenEmbeddings(*mlx.Array) *mlx.Array { return nil }
func (t *selectorTarget) SetAuxHiddenLayers([]int)              {}
func (t *selectorTarget) NumLayers() int                        { return 1 }
func (t *selectorTarget) Tokenizer() *tokenizer.Tokenizer       { return nil }
func (t *selectorTarget) MaxContextLength() int                 { return 128 }

func TestCandidateLatticeUsesPredecessorEdges(t *testing.T) {
	mlxtest.Setup(t)
	target := &selectorTarget{logits: mlx.FromValues([]float32{
		0, 0, 5, 4,
		5, 4, 0, 0,
	}, 1, 2, 4)}
	m := &Model2{Model: &Model{
		Config: &Config{OutputMultiplier: 1},
		target: target,
		Selector: &CandidateSelector{
			Predecessor:      nn.NewEmbedding(mlx.FromValues([]float32{1, 2, 3, 4}, 4, 1)),
			Successor:        nn.NewEmbedding(mlx.FromValues([]float32{-2, -1, 1, 2}, 4, 1)),
			HiddenProjection: nn.NewLinear(mlx.FromValues([]float32{1, 0, 0, 0}, 1, 4), nil),
			TopK:             2,
		},
	}}
	hidden := mlx.FromValues([]float32{
		1, 0, 0, 0,
		1, 0, 0, 0,
	}, 1, 2, 4)
	ids, scores := m.CandidateLattice(hidden, mlx.FromValues([]int32{1}, 1))
	mlx.Eval(ids, scores)

	candidates := ids.Ints()
	lattice := scores.Floats()
	k := 2
	previous := 0
	want := []int{3, 1}
	for position := range 2 {
		best := 0
		for candidate := 1; candidate < k; candidate++ {
			base := ((position*k + previous) * k)
			if lattice[base+candidate] > lattice[base+best] {
				best = candidate
			}
		}
		got := candidates[position*k+best]
		if got != want[position] {
			t.Fatalf("path[%d] = %d, want %d (ids=%v scores=%v)", position, got, want[position], candidates, lattice)
		}
		previous = best
	}
}

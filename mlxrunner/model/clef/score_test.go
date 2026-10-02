package clef

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/qwen3_5"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// A small real backbone exercises both recurrent state and attention history.
func prefillTestModel() *Model {
	seed := 0
	weight := func(shape ...int) *mlx.Array {
		n := 1
		for _, d := range shape {
			n *= d
		}
		values := make([]float32, n)
		for i := range values {
			seed++
			values[i] = float32(math.Sin(float64(seed)*0.17)) * 0.1
		}
		return mlx.FromValues(values, shape...)
	}
	linear := func(out, in int) *nn.Linear { return nn.NewLinear(weight(out, in), nil) }
	norm := func(n int) *nn.RMSNorm { return nn.NewRMSNorm(mlx.FromValues(slices.Repeat([]float32{1}, n), n), 1e-6) }
	cfg := &qwen3_5.Config{
		HiddenSize: 8, NumAttentionHeads: 1, NumKeyValueHeads: 1, HeadDim: 8,
		RMSNormEps: 1e-6, RopeDim: 8, RopeTheta: 10000, Scale: 1 / float32(math.Sqrt(8)),
		LinearNumKeyHeads: 1, LinearNumValueHeads: 1, LinearKeyHeadDim: 32,
		LinearValueHeadDim: 32, LinearConvKernelDim: 4,
	}
	m := &qwen3_5.Model{Config: cfg, EmbedTokens: nn.NewEmbedding(weight(16, 8)), Norm: norm(8)}
	for _, recurrent := range []bool{true, false} {
		layer := &qwen3_5.Layer{
			InputNorm: norm(8), PostAttentionNorm: norm(8), IsLinear: recurrent,
			MLP: &qwen3_5.DenseMLP{GateProj: linear(16, 8), UpProj: linear(16, 8), DownProj: linear(8, 16)},
		}
		if recurrent {
			layer.Linear = &qwen3_5.GatedDeltaNet{
				InProjQKVZ: linear(128, 8), InProjBA: linear(2, 8), OutProj: linear(8, 32),
				Conv1D:     nn.NewConv1d(weight(96, 4, 1), nil, 1, 0, 1, 96),
				NormWeight: mlx.FromValues(slices.Repeat([]float32{1}, 32), 32), DtBias: weight(1), AExp: mlx.FromValues([]float32{1}, 1),
			}
		} else {
			layer.FullAttn = &qwen3_5.FullAttention{
				QProj: linear(16, 8), KProj: linear(8, 8), VProj: linear(8, 8), OProj: linear(8, 8),
				QNorm: norm(8), KNorm: norm(8),
			}
		}
		m.Layers = append(m.Layers, layer)
	}
	return &Model{Model: m}
}

func TestPrefillPreservesHiddenStates(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			m := prefillTestModel()
			for _, n := range []int{1, 2048, 2049, 4097} {
				mlx.Scoped(func() {
					tokens := make([]int32, n)
					for i := range tokens {
						tokens[i] = int32(i % 16)
					}
					ids := mlx.FromValues(tokens, 1, n)
					want := mlx.ScopedEval(func() []*mlx.Array {
						h, _ := m.Model.Forward(&batch.Batch{InputIDs: ids, SeqOffsets: []int32{0}, SeqQueryLens: []int32{int32(n)}}, nil)
						return []*mlx.Array{h}
					})[0]
					got, err := m.prefill(context.Background(), ids, &model.PreparedRequest{})
					if err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(got.Dims(), want.Dims()) {
						t.Fatalf("%d tokens: hidden shape %v, want %v", n, got.Dims(), want.Dims())
					}
					// A one-token tail can use TF32 matmul on supported GPUs.
					values := want.Floats()
					for i, v := range got.Floats() {
						w := values[i]
						if math.IsNaN(float64(v)) || math.Abs(float64(v-w)) > 1e-3*(1+math.Abs(float64(w))) {
							t.Fatalf("%d tokens: hidden[%d] = %g, want %g", n, i, v, w)
						}
					}
				})
			}
		})
	})
}

// Cancel at an actual forward boundary without a timer or a production hook.
type cancellingEmbedding struct {
	nn.EmbeddingLayer
	cancel      context.CancelFunc
	cancelAfter int
	calls       int
}

func (e *cancellingEmbedding) Forward(ids *mlx.Array) *mlx.Array {
	e.calls++
	if e.calls == e.cancelAfter {
		e.cancel()
	}
	return e.EmbeddingLayer.Forward(ids)
}

func TestPrefillCancellation(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			m := prefillTestModel()
			embedding := &cancellingEmbedding{EmbeddingLayer: m.EmbedTokens}
			m.EmbedTokens = embedding
			ids := mlx.Zeros(mlx.DTypeInt32, 1, 4097)
			prepared := &model.PreparedRequest{}
			want, err := m.prefill(context.Background(), ids, prepared)
			if err != nil {
				t.Fatal(err)
			}
			mlx.Eval(want)
			for _, after := range []int{0, 1, 2, 3, 1, 2} {
				ctx, cancel := context.WithCancel(context.Background())
				embedding.calls, embedding.cancelAfter, embedding.cancel = 0, after, cancel
				if after == 0 {
					cancel()
				}
				got, err := m.prefill(ctx, ids, prepared)
				cancel()
				if !errors.Is(err, context.Canceled) || got != nil {
					t.Fatalf("cancel after %d chunks: hidden=%v, err=%v", after, got, err)
				}
				if embedding.calls != after {
					t.Fatalf("cancel after %d chunks ran %d forwards", after, embedding.calls)
				}
				embedding.cancelAfter = 0
				mlx.Scoped(func() {
					got, err := m.prefill(context.Background(), ids, prepared)
					if err != nil {
						t.Fatalf("request after cancellation: %v", err)
					}
					if !slices.Equal(got.Floats(), want.Floats()) {
						t.Fatal("cancelled request changed the next request's hidden states")
					}
				})
			}
		})
	})
}

package inkling2

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestDraftEmbedsAnchorWithTargetNorm(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			const eps = 1e-6
			ones := mlx.FromValues([]float32{1, 1}, 2).AsType(mlx.DTypeBFloat16)
			d := &draft{
				target: &targetModel{
					config:    textConfig{NormEps: eps},
					Embed:     nn.NewEmbedding(ones.Reshape(1, 2)),
					EmbedNorm: mlx.FromValues([]float32{0, 2}, 2),
				},
				config:        draftConfig{Width: 2, Taps: []int{0}, Epsilon: eps},
				Mask:          ones,
				Norm:          ones,
				RecurrentNorm: ones,
				Fuse:          mlx.FromValues([]float32{1}, 1, 1),
			}
			states := d.NewCaches()
			defer states[0].Free()
			b := &batch.Batch{
				InputIDs: mlx.FromValues([]int32{0, 0}, 1, 2), Hidden: ones.Reshape(1, 1, 2),
				SeqOffsets: []int32{0}, SeqQueryLens: []int32{2},
			}
			hidden, _ := d.Forward(b, nil, cache.Layers(states))
			got := hidden.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, 2)).AsType(mlx.DTypeFloat32).Floats()
			// The target norm maps the anchor to [0, 2]. The final draft norm
			// gives [0, sqrt(2)] rounded to BF16 and leaves the unit mask at [1, 1].
			if want := []float32{0, 1.4140625, 1, 1}; !slices.Equal(got, want) {
				t.Errorf("draft anchor and mask = %v, want %v", got, want)
			}
		})
	})
}

func TestDraftBlockKeepsFullPrecedingWindow(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			const bf16 = mlx.DTypeBFloat16
			zero := mlx.Zeros(bf16, 2, 2)
			identity := mlx.FromValues([]float32{1, 0, 0, 1}, 2, 2).AsType(bf16)
			norm := mlx.FromValues([]float32{1, 1}, 2).AsType(bf16)
			d := &draft{
				config: draftConfig{Window: 2, Heads: 1, KVHeads: 1, HeadDim: 2, RopeTheta: 10000, Epsilon: 1e-6},
				Layers: []draftLayer{{
					local: true, InputNorm: norm, FfNorm: norm, QueryNorm: norm, KeyNorm: norm,
					QKV: newDraftProjection(0, zero, zero, zero), Output: newDraftProjection(0, identity),
					GateUp: newDraftProjection(0, zero, zero), Down: newDraftProjection(0, zero),
				}},
			}
			states := d.NewCaches()
			defer func() {
				for _, state := range states {
					state.Free()
				}
			}()
			history := states[0].(cache.Attention)
			keys := mlx.Zeros(bf16, 1, 1, 3, 2)
			values := mlx.FromValues([]float32{64, 0, 16, 0, 0, 0}, 1, 1, 3, 2).AsType(bf16)
			history.Update(&batch.Batch{SeqOffsets: []int32{0}, SeqQueryLens: []int32{3}}, keys, values)
			block := &batch.Batch{
				InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 2), SeqOffsets: []int32{3}, SeqQueryLens: []int32{2},
			}
			out := d.Layers[0].forward(mlx.Zeros(bf16, 1, 2, 2), nil, nil, block, history, d.config)
			// All queries see two preceding rows plus both block rows. The
			// first of those four values contributes 16/4 to each output.
			if got := out.AsType(mlx.DTypeFloat32).Floats(); !slices.Equal(got, []float32{4, 0, 4, 0}) {
				t.Errorf("block attention = %v, want [4 0 4 0]", got)
			}
		})
	})
}

func TestDraftNormalizesUnroundedResidualSum(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			residual := mlx.FromValues([]float32{1, 1}, 1, 2).AsType(mlx.DTypeBFloat16)
			update := mlx.FromValues([]float32{1.0 / 256, -1.0 / 128}, 1, 2).AsType(mlx.DTypeBFloat16)
			weight := mlx.FromValues([]float32{1, 1}, 2)
			sum := draftResidualSum(residual, update)
			if got := modelRMS(sum, weight, 1e-6).AsType(mlx.DTypeFloat32).Floats(); !slices.Equal(got, []float32{1.0078125, 0.9921875}) {
				t.Errorf("normalized sum = %v, want [1.0078125 0.9921875]", got)
			}
			if got := draftResidualSum(sum, update).Floats(); !slices.Equal(got, []float32{1.00390625, 0.984375}) {
				t.Errorf("next sum = %v, want [1.00390625 0.984375]", got)
			}
		})
	})
}

func TestDraftRoundsCorrectedScoresToBF16(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			const bf16 = mlx.DTypeBFloat16
			d := &draft{
				config:     draftConfig{Width: 1, StateWidth: 1},
				Previous:   nn.NewEmbedding(mlx.Zeros(bf16, 2, 1)),
				Recurrent:  nn.NewLinear(mlx.Zeros(bf16, 3, 4), nil),
				Folded:     newDraftProjection(0, mlx.Zeros(bf16, 2, 1)),
				FoldedBias: mlx.FromValues([]float32{0, 1.0 / 256}, 2).AsType(bf16),
				correct:    compiledCorrection(30, 0),
			}
			hidden := mlx.Zeros(bf16, 1, 1, 2)
			base := mlx.FromValues([]float32{2, 2}, 1, 1, 2).AsType(bf16)
			anchor := mlx.FromValues([]int32{0}, 1)
			// Both values round to 2 in the publisher head. A float32 sum
			// keeps the small bias and incorrectly breaks the tie.
			tokens, ids, scores := d.Select(hidden, base, anchor, func(candidates, scores *mlx.Array) (*mlx.Array, *mlx.Array) {
				if got := candidates.Ints(); !slices.Equal(got, []int32{0, 1}) {
					t.Fatalf("candidate order = %v, want [0 1]", got)
				}
				return scores.Argmax(-1, false).AsType(mlx.DTypeInt32), scores.AsType(mlx.DTypeFloat32)
			})
			if ids != nil {
				t.Fatalf("vocabulary-ordered scores returned explicit IDs %v", ids)
			}
			if got := scores.Floats(); !slices.Equal(got, []float32{2, 2}) {
				t.Errorf("corrected scores = %v, want [2 2]", got)
			}
			if got := tokens.Ints(); !slices.Equal(got, []int32{0}) {
				t.Errorf("greedy token = %v, want [0]", got)
			}
		})
	})
}

func TestDraftRestrictsProposalCandidates(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			const bf16 = mlx.DTypeBFloat16
			d := &draft{
				config:     draftConfig{Width: 1, StateWidth: 1},
				Previous:   nn.NewEmbedding(mlx.Zeros(bf16, 3, 1)),
				Recurrent:  nn.NewLinear(mlx.Zeros(bf16, 3, 4), nil),
				Folded:     newDraftProjection(0, mlx.Zeros(bf16, 3, 1)),
				FoldedBias: mlx.Zeros(bf16, 3),
				correct:    compiledCorrection(30, 0.1),
			}
			// ln 0.1 = -2.30: token 1 sits 1 below the best and stays, token 2 sits 3 below and drops.
			base := mlx.FromValues([]float32{2, 1, -1}, 1, 1, 3).AsType(bf16)
			_, _, scores := d.Select(mlx.Zeros(bf16, 1, 1, 2), base, mlx.FromValues([]int32{0}, 1), func(_, scores *mlx.Array) (*mlx.Array, *mlx.Array) {
				return scores.Argmax(-1, false).AsType(mlx.DTypeInt32), scores.AsType(mlx.DTypeFloat32)
			})
			got := scores.Floats()
			if got[0] != 2 || got[1] != 1 || !math.IsInf(float64(got[2]), -1) {
				t.Errorf("restricted scores = %v, want [2 1 -Inf]", got)
			}
		})
	})
}

func TestDraftVerificationDepth(t *testing.T) {
	for _, tc := range []struct {
		block    int
		adaptive bool
	}{{4, false}, {8, true}, {1024, false}} {
		d := &draft{config: draftConfig{Block: tc.block}}
		if got := d.AdaptiveDraftDepth(); got != tc.adaptive {
			t.Errorf("block %d: got adaptive depth %v, want %v", tc.block, got, tc.adaptive)
		}
	}
}

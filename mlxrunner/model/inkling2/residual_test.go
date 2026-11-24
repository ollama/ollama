package inkling2

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
)

// Early submission runs only for decode and verification rounds, never for the
// load-time shape check at offset zero or for longer blocks.
func TestSubmitsEarly(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("early submission is Metal only")
		}
		for _, tc := range []struct {
			name    string
			tokens  int
			offsets []int32
			want    bool
		}{
			{"decode", 1, []int32{2048}, true},
			{"verification", 4, []int32{2048}, true},
			{"two requests", 4, []int32{2048, 300}, true},
			{"shape check", 1, []int32{0}, false},
			{"new request beside a decode", 4, []int32{2048, 0}, false},
			{"long block", shortTargetRows + 1, []int32{2048}, false},
			{"three requests", 1, []int32{1, 2, 3}, false},
		} {
			mlx.Scoped(func() {
				b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, len(tc.offsets), tc.tokens), SeqOffsets: tc.offsets}
				if got := submitsEarly(b); got != tc.want {
					t.Errorf("%s: submitsEarly = %t, want %t", tc.name, got, tc.want)
				}
			})
		}
	})
}

// residualTestStream returns two BF16 residual blocks and an FP32 partial
// block of random [rows, tokens, width] values.
func residualTestStream(rng *rand.Rand, rows, tokens, width int) residualStream {
	return residualStream{
		blocks:  []*mlx.Array{draftTestValues(rng, 1, rows, tokens, width), draftTestValues(rng, 1, rows, tokens, width)},
		partial: draftTestValues(rng, 1, rows, tokens, width).AsType(mlx.DTypeFloat32),
	}
}

// mixNorm normalizes a vector query's mixture in the residual mixer's launch
// where it runs, reproducing modelRMS of the separate mixture bit for bit.
func TestMixNormMatchesSeparateNorm(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("requires Metal")
		}
		const width = 2304
		rng := rand.New(rand.NewPCG(23, 31))
		cfg := textConfig{NormalizeResidualValues: true, NormEps: 1e-5}
		for _, sh := range [][2]int{{1, 1}, {2, 8}, {1, 37}} {
			mlx.Scoped(func() {
				name := fmt.Sprintf("B=%d T=%d", sh[0], sh[1])
				state := residualTestStream(rng, sh[0], sh[1], width)
				query := draftTestValues(rng, 1, width)
				weight := mlx.AddScalar(draftTestValues(rng, 0.1, width).AsType(mlx.DTypeFloat32), 1)
				got := state.mixNorm(query, weight, cfg)
				requireIdentical(t, name, got, modelRMS(state.mix(query, true), weight, cfg.NormEps))
				sources := append(slices.Clone(state.blocks), state.partial)
				if fused := mlx.BlockResidualMixNorm(sources, query, weight, residualEpsilon, cfg.NormEps); fused != nil {
					requireIdentical(t, name+" fused mixer", got, fused)
				} else {
					t.Logf("%s: the normalizing residual mixer declines on this device", name)
				}
			})
		}
	})
}

func TestResidualMixPreservesIndependentQueries(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			sources := []*mlx.Array{
				mlx.FromValues([]float32{3, 0, 0, 5}, 1, 2, 2),
				mlx.FromValues([]float32{0, 4, 7, 0}, 1, 2, 2),
			}
			queries := mlx.FromValues([]float32{1, 0, 0, 1}, 2, 2)
			got := mixResiduals(sources, queries, true)
			if !slices.Equal(got.Dims(), []int{1, 2, 2, 2}) {
				t.Fatalf("residual mixture shape = %v, want [1 2 2 2]", got.Dims())
			}
			high := math.Sqrt2 / (1 + math.Exp(-1))
			low := math.Sqrt2 - high
			want := []float64{high, low, low, high, high, low, low, high}
			values := got.Floats()
			if len(values) != len(want) {
				t.Fatalf("residual mixture produced %d values, want %d", len(values), len(want))
			}
			for i, value := range values {
				if !(math.Abs(float64(value)-want[i]) <= 1e-5) {
					t.Errorf("residual mixture element %d = %g, want %g", i, value, want[i])
				}
			}
		})
	})
}

func TestResidualMixUnnormalizedValues(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			const width = 2304
			zero := mlx.Zeros(mlx.DTypeBFloat16, 1, 2, width)
			sources := []*mlx.Array{mlx.AddScalar(zero, 2), mlx.AddScalar(zero, 8)}
			// Equal scores average the original values in this configuration.
			got := mixResiduals(sources, mlx.Zeros(mlx.DTypeFloat32, width), false)
			for i, value := range got.Floats() {
				if value != 5 {
					t.Fatalf("unnormalized residual[%d] = %g, want 5", i, value)
				}
			}
		})
	})
}

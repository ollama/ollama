package inkling2

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestNVFP4ExpertAppliesActivationAndWeightScales(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			const width, outputs = 64, 8
			// E2M1 value 1 in every nibble, with E4M3 block scales of 1.
			bank := expertBank{
				Weight:      mlx.FromValues(slices.Repeat([]uint32{0x22222222}, 2*outputs*width/8), 2, outputs, width/8),
				BlockScales: mlx.FromValues(slices.Repeat([]uint8{0x38}, 2*outputs*width/16), 2, outputs, width/16),
				Maxima:      mlx.FromValues([]float32{2, 3}, 2),
				Denominator: mlx.FromValues([]float32{4}, 1),
			}
			input := mlx.FromValues(slices.Repeat([]float32{2}, width), 1, 1, 1, width).AsType(mlx.DTypeBFloat16)
			selected := mlx.FromValues([]int32{1, 0}, 1, 2)
			got := bank.project(input, selected, nil, false).AsType(mlx.DTypeFloat32).Floats()
			want := append(slices.Repeat([]float32{96}, outputs), slices.Repeat([]float32{64}, outputs)...)
			if !slices.Equal(got, want) {
				t.Errorf("expert outputs = %v, want %v", got, want)
			}
		})
	})
}

func TestNVFP4ActivationRounding(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		// The first block of the publisher CUDA capture contains its row maximum.
		// Tiling preserves both normalization and the per-block scale decisions.
		block := []float32{
			2.4375, -0.66015625, 0.126953125, 2.25, 0.66796875, -0.96484375, -0.61328125, 0.400390625,
			-0.26953125, 0.77734375, -0.3359375, -0.28515625, 1, -1.421875, 0.2578125, 0.05078125,
		}
		quantized := []float32{1536, -384, 0, 1536, 384, -576, -384, 192, -192, 576, -192, -192, 576, -768, 192, 0}
		for _, width := range []int{1152, 2304} {
			input := make([]float32, 3*width)
			want := make([]float32, len(input))
			// Row 0 is all zero. Row 1 has outer scale one. Both candidates
			// tie at ±1.25, rounding to ±1 with scale one or ±1.5 with scale 1.5.
			input[width], want[width] = 1536, 1536
			for i := 16; i < width; i += 16 {
				copy(input[width+i:], []float32{6, 1.25, -1.25})
				copy(want[width+i:], []float32{6, 1, -1})
			}
			for i := 2 * width; i < len(input); i += 16 {
				copy(input[i:], block)
				copy(want[i:], quantized)
			}
			values, maxima := mlx.QuantizeNVFP4Activation4Over6(mlx.FromValues(input, 3, width).AsType(mlx.DTypeBFloat16), nvfp4ActivationDenominator)
			if values == nil {
				t.Fatalf("Metal activation unavailable for width %d", width)
			}
			if got := maxima.Floats(); !slices.Equal(got, []float32{0, 1536, 2.4375}) {
				t.Errorf("width %d maxima = %v", width, got)
			}
			for i, v := range values.Floats() {
				if v != want[i] {
					t.Fatalf("width %d row %d element %d = %g, want %g", width, i/width, i%width, v, want[i])
				}
			}
		}
	})
}

func TestGroupedNVFP4ExpertsPreserveRouting(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("Metal grouped gather")
		}
		mlx.Scoped(func() {
			// At 512 routes, the scalar Metal backend selects the compact gate/up matmul.
			const rows, experts, topK, width = 128, 8, 4, 128
			random := rand.New(rand.NewPCG(19, 31))
			bank := func(outputs int) expertBank {
				packed := make([]uint32, experts*outputs*width/8)
				for i := range packed {
					packed[i] = random.Uint32()
				}
				scales := make([]uint8, experts*outputs*width/16)
				for i := range scales {
					scales[i] = uint8(32 + random.IntN(32))
				}
				return expertBank{
					Weight:      mlx.FromValues(packed, experts, outputs, width/8),
					BlockScales: mlx.FromValues(scales, experts, outputs, width/16),
					Maxima:      mlx.Arange(1, experts+1, 1, mlx.DTypeFloat32),
					Denominator: mlx.FromValue(float32(8)),
				}
			}
			e := routedExperts{GateUp: bank(2 * width), Down: bank(width), topK: topK, gateCap: 5, upCap: 4}
			input := make([]float32, rows*width)
			indices := make([]int32, rows*topK)
			weights := make([]float32, rows*topK)
			for i := range input {
				input[i] = float32(random.NormFloat64())
			}
			for row := range rows {
				order := random.Perm(experts)
				for j := range topK {
					indices[row*topK+j] = int32(order[j])
					weights[row*topK+j] = float32(random.Float64())
				}
			}
			x := mlx.FromValues(input, rows, width).AsType(mlx.DTypeBFloat16)
			ids := mlx.FromValues(indices, rows, topK)
			gains := mlx.FromValues(weights, rows, topK)
			want := e.combine(x, ids, gains, false).AsType(mlx.DTypeFloat32).Floats()
			got := e.combine(x, ids, gains, true).AsType(mlx.DTypeFloat32).Floats()
			if !slices.Equal(got, want) {
				t.Fatal("grouping changed the routed expert result")
			}
		})
	})
}

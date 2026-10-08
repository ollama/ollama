package model

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

// ToMLXGlobalScale converts and flattens; PrepareGatherQMMGlobalScale only
// broadcasts what it is given. Keeping the conversion in one place is what
// lets a scale be read once and used by every consumer.
func TestGlobalScaleConversion(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if got := ToMLXGlobalScale(nil); got != nil {
			t.Fatal("nil scale did not convert to nil")
		}
		if got := PrepareGatherQMMGlobalScale(nil, 4); got != nil {
			t.Fatal("nil scale did not broadcast to nil")
		}

		// Checkpoints ship a scalar as either [] or [1]; both normalize to [1]
		// so a mix of the two can still be stacked.
		for _, scalar := range []*mlx.Array{
			mlx.NewScalarArray(2),
			mlx.FromValues([]float32{2}, 1),
		} {
			got := ToMLXGlobalScale(scalar)
			mlx.Eval(got)
			if dims := got.Dims(); len(dims) != 1 || dims[0] != 1 {
				t.Fatalf("converted scalar dims = %v, want [1]", dims)
			}
			if want := float32(2 * mlx.Nvfp4MaxProduct); got.Floats()[0] != want {
				t.Fatalf("converted scalar = %v, want %v", got.Floats()[0], want)
			}
		}

		checkpointScales := []float32{0.5, 1, 2, 4}
		perExpert := ToMLXGlobalScale(mlx.FromValues(checkpointScales, 4))
		mlx.Eval(perExpert)
		for i, got := range perExpert.Floats() {
			if want := checkpointScales[i] * float32(mlx.Nvfp4MaxProduct); got != want {
				t.Fatalf("converted scale[%d] = %v, want %v", i, got, want)
			}
		}

		// Broadcasting is value-preserving, and does not convert a second time.
		bank := PrepareGatherQMMGlobalScale(ToMLXGlobalScale(mlx.FromValues([]float32{2}, 1)), 4)
		mlx.Eval(bank)
		if dims := bank.Dims(); len(dims) != 1 || dims[0] != 4 {
			t.Fatalf("bank dims = %v, want [4]", dims)
		}
		for i, got := range bank.Floats() {
			if want := float32(2 * mlx.Nvfp4MaxProduct); got != want {
				t.Fatalf("bank[%d] = %v, want %v", i, got, want)
			}
		}

		// A scale count that is neither a checkpoint-wide scalar nor one per
		// expert is malformed and fails the load instead of being ignored.
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("mismatched scale count did not fail")
				}
			}()
			PrepareGatherQMMGlobalScale(ToMLXGlobalScale(mlx.FromValues([]float32{1, 2}, 2)), 4)
		}()
	})
}

func TestSameGlobalScales(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		perExpert := PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{0.5, 1, 2, 4}, 4), 4)
		same := PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{0.5, 1, 2, 4}, 4), 4)
		reordered := PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{4, 2, 1, 0.5}, 4), 4)
		broadcast := PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{2}, 1), 4)
		shorter := PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{2}, 1), 2)

		for _, tt := range []struct {
			name string
			a, b *mlx.Array
			want bool
		}{
			{"both nil", nil, nil, true},
			{"only one nil", perExpert, nil, false},
			{"identical array", perExpert, perExpert, true},
			{"equal values", perExpert, same, true},
			{"same values reordered", perExpert, reordered, false},
			{"broadcast scalar matches itself", broadcast, broadcast, true},
			{"different expert counts", broadcast, shorter, false},
		} {
			if got := SameGlobalScales(tt.a, tt.b); got != tt.want {
				t.Fatalf("%s: SameGlobalScales() = %v, want %v", tt.name, got, tt.want)
			}
		}
	})
}

// TestGatherQMMGlobalScaleMatchesDequantized checks the whole scale chain —
// Prepare's conversion to amax units plus the GatherQMM wrapper — against the
// dequantized weights gathered densely. Metal runs the native kernel path;
// CUDA runs the wrapper's output scaling. ModelOpt exports a checkpoint-wide
// scalar, so both it and the per-expert form are covered.
func TestGatherQMMGlobalScaleMatchesDequantized(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() && !mlx.CUDAIsAvailable() {
			t.Skip("gather_qmm requires a GPU backend")
		}

		const experts, rows, cols, groupSize = 4, 32, 64, 16
		packed := make([]uint32, experts*rows*cols/8)
		for i := range packed {
			for j := range 8 {
				packed[i] |= uint32((i*8+j)%16) << (4 * j)
			}
		}
		scales := make([]uint8, experts*rows*cols/groupSize)
		for i := range scales {
			scales[i] = uint8(((i % 4) + 6) << 3)
		}
		weights := mlx.FromValues(packed, experts, rows, cols/8)
		blockScales := mlx.FromValues(scales, experts, rows, cols/groupSize)

		xValues := make([]float32, cols)
		for i := range xValues {
			xValues[i] = float32(i%7-3) / 8
		}
		x := mlx.FromValues(xValues, 1, cols).AsType(mlx.DTypeBFloat16)
		indices := mlx.FromValues([]int32{0, 1, 2, 3}, 1, experts)

		for _, checkpointScale := range [][]float32{
			{0.5, 1, 2, 4}, // one scale per expert
			{2},            // checkpoint-wide scalar
		} {
			checkpoint := mlx.FromValues(checkpointScale, len(checkpointScale))
			kernelScale := PrepareGatherQMMGlobalScale(checkpoint, experts)

			got := mlx.GatherQMM(
				x, weights, blockScales, nil, nil, indices,
				true, groupSize, 4, "nvfp4", kernelScale, false,
			).AsType(mlx.DTypeFloat32)
			dense := mlx.Dequantize(
				weights, blockScales, nil, groupSize, 4, "nvfp4", checkpoint,
			).AsType(mlx.DTypeFloat32)
			want := mlx.GatherMM(
				x.AsType(mlx.DTypeFloat32), mlx.Transpose(dense, 0, 2, 1), nil, indices, false,
			)
			mlx.Eval(got, want)

			gotValues, wantValues := got.Floats(), want.Floats()
			if len(gotValues) != len(wantValues) {
				t.Fatalf("result length = %d, want %d", len(gotValues), len(wantValues))
			}
			for i := range gotValues {
				if math.IsNaN(float64(gotValues[i])) || math.IsInf(float64(gotValues[i]), 0) {
					t.Fatalf("result[%d] = %v, want finite", i, gotValues[i])
				}
				delta := math.Abs(float64(gotValues[i] - wantValues[i]))
				tolerance := 0.01 * math.Max(math.Abs(float64(wantValues[i])), 1)
				if delta > tolerance {
					t.Fatalf("result[%d] = %v, want %v (delta %v > %v)", i, gotValues[i], wantValues[i], delta, tolerance)
				}
			}
		}
	})
}

// One key policy for every model. The canonical name import writes wins;
// ModelOpt's own name is the fallback for un-imported checkpoints; and an
// activation scale is reported as a companion but never mistaken for a weight
// scale.
func TestReadGlobalScale(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		scale := func(v float32) *mlx.Array { return mlx.FromValues([]float32{v}, 1) }
		mlxForm := func(v float32) float32 { return v * mlx.Nvfp4MaxProduct }

		for _, tt := range []struct {
			name     string
			tensors  map[string]*mlx.Array
			want     float32 // 0 means "expect nil"
			consumed []string
		}{
			{
				name:     "canonical name",
				tensors:  map[string]*mlx.Array{"w.weight.global_scale": scale(0.5)},
				want:     mlxForm(0.5),
				consumed: []string{"w.weight.global_scale"},
			},
			{
				name:     "modelopt fallback",
				tensors:  map[string]*mlx.Array{"w.weight_scale_2": scale(0.25)},
				want:     mlxForm(0.25),
				consumed: []string{"w.weight_scale_2"},
			},
			{
				name: "canonical wins over fallback",
				tensors: map[string]*mlx.Array{
					"w.weight.global_scale": scale(0.5),
					"w.weight_scale_2":      scale(0.25),
				},
				want:     mlxForm(0.5),
				consumed: []string{"w.weight.global_scale", "w.weight_scale_2"},
			},
			{
				// An activation scale is not a weight scale. Picking it up
				// would silently rescale every output.
				name:     "activation scale is never the weight scale",
				tensors:  map[string]*mlx.Array{"w.weight.input_global_scale": scale(8)},
				want:     0,
				consumed: []string{"w.weight.input_global_scale"},
			},
			{
				name:     "modelopt activation scale is never the weight scale",
				tensors:  map[string]*mlx.Array{"w.weight.input_scale": scale(8)},
				want:     0,
				consumed: []string{"w.weight.input_scale"},
			},
			{
				name: "activation scale alongside a weight scale",
				tensors: map[string]*mlx.Array{
					"w.weight.global_scale":       scale(0.5),
					"w.weight.input_global_scale": scale(8),
				},
				want:     mlxForm(0.5),
				consumed: []string{"w.weight.global_scale", "w.weight.input_global_scale"},
			},
			{name: "absent", tensors: map[string]*mlx.Array{}, want: 0},
		} {
			got, consumed := ReadGlobalScale(tt.tensors, "w.weight")
			if tt.want == 0 {
				if got != nil {
					mlx.Eval(got)
					t.Fatalf("%s: got %v, want nil", tt.name, got.Floats())
				}
			} else {
				if got == nil {
					t.Fatalf("%s: got nil, want [%v]", tt.name, tt.want)
				}
				mlx.Eval(got)
				if values := got.Floats(); len(values) != 1 || values[0] != tt.want {
					t.Fatalf("%s: got %v, want [%v]", tt.name, values, tt.want)
				}
			}
			if len(consumed) != len(tt.consumed) {
				t.Fatalf("%s: consumed %v, want %v", tt.name, consumed, tt.consumed)
			}
			for _, key := range tt.consumed {
				if !slices.Contains(consumed, key) {
					t.Fatalf("%s: consumed %v, missing %q", tt.name, consumed, key)
				}
			}
		}
	})
}

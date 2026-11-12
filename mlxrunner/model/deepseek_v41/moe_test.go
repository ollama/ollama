package deepseek_v41

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestExpertGateBiasChangesSelectionOnly(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		gate := expertGate{
			Weight: mlx.FromValues(make([]float32, 3*32), 3, 32),
			Bias:   mlx.FromValues([]float32{0, 1, 2}, 3), ImageBias: mlx.FromValues([]float32{3, 2, 0}, 3),
			TopK: 2, Scale: 1.5,
		}
		x := mlx.FromValues(make([]float32, 2*32), 1, 2, 32)
		mask := mlx.FromValues([]bool{false, true}, 2)
		weights, indices := gate.route(x, mask)
		mlx.Eval(weights, indices)
		gotIndices := indices.AsType(mlx.DTypeInt32).Ints()
		for token, want := range [][]int32{{1, 2}, {0, 1}} {
			got := slices.Clone(gotIndices[token*2 : token*2+2])
			slices.Sort(got)
			if !slices.Equal(got, want) {
				t.Fatalf("token %d experts = %v, want %v", token, got, want)
			}
		}
		for i, weight := range weights.Floats() {
			if math.IsNaN(float64(weight)) || math.IsInf(float64(weight), 0) || math.Abs(float64(weight-.75)) > 1e-5 {
				t.Fatalf("expert weight %d = %g, want 0.75", i, weight)
			}
		}
	})
}

func TestPackedMXFP4ExpertSelection(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		weights := make([]*mlx.Array, 2)
		scales := make([]*mlx.Array, 2)
		for expert := range weights {
			values := make([]float32, 32*32)
			for i := range values {
				values[i] = float32(expert+1) * .25
			}
			weights[expert], scales[expert], _ = mlx.Quantize(mlx.FromValues(values, 32, 32), 32, 4, "mxfp4")
		}
		bank := expertBank{Weight: mlx.Stack(weights, 0), Scale: mlx.Stack(scales, 0)}
		input := mlx.FromValues(make([]float32, 32), 1, 1, 1, 32)
		input = mlx.AddScalar(input, 1)
		selected := mlx.FromValues([]uint32{1}, 1, 1)
		output := bank.apply(input, selected)
		mlx.Eval(output)
		if got := output.Dims(); !slices.Equal(got, []int{1, 1, 1, 32}) {
			t.Fatalf("expert output shape = %v", got)
		}
		for i, value := range output.Floats() {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Abs(float64(value-16)) > .03 {
				t.Fatalf("selected expert output %d = %g, want 16", i, value)
			}
		}
	})
}

func TestMixtureOfExpertsMultiExpertShape(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		values := make([]float32, 32*32)
		for i := range values {
			values[i] = .125
		}
		packed, scales, _ := mlx.Quantize(mlx.FromValues(values, 32, 32), 32, 4, "mxfp4")
		bank := expertBank{Weight: mlx.Stack([]*mlx.Array{packed, packed}, 0), Scale: mlx.Stack([]*mlx.Array{scales, scales}, 0)}
		shared := nn.NewLinear(mlx.FromValues(make([]float32, 32*32), 32, 32), nil)
		moe := mixtureOfExperts{
			Gate:   &expertGate{Weight: mlx.FromValues(make([]float32, 2*32), 2, 32), Bias: mlx.FromValues([]float32{0, 0}, 2), TopK: 2, Scale: 1},
			Up:     bank,
			GateUp: bank,
			Down:   bank,
			Shared: []nn.LinearLayer{shared, shared, shared},
			Limit:  100,
		}
		input := mlx.AddScalar(mlx.FromValues(make([]float32, 3*32), 1, 3, 32), 1)
		output := moe.forward(input, nil)
		mlx.Eval(output)
		if got := output.Dims(); !slices.Equal(got, []int{1, 3, 32}) {
			t.Fatalf("MoE output shape = %v, want [1 3 32]", got)
		}
		if !slices.Contains(mlx.Collect(&moe), shared.Weight) {
			t.Fatal("model weight collection omitted shared expert projection")
		}
	})
}

//go:build mlx

package sample

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestPenaltySample(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}

	logprobs := mlx.FromValues([]float32{
		1.0, -2.0, 3.0, 4.0,
	}, 1, 4)

	got := Penalty{
		RepeatLastN:      3,
		RepeatPenalty:    2.0,
		PresencePenalty:  1.5,
		FrequencyPenalty: 0.25,
	}.Sample(logprobs, []int32{2, 1, 2})

	mlx.Eval(got)

	want := []float32{1.0, -5.75, -0.5, 4.0}
	values := got.Floats()
	if len(values) != len(want) {
		t.Fatalf("len(values) = %d, want %d", len(values), len(want))
	}

	for i := range want {
		if math.Abs(float64(values[i]-want[i])) > 1e-5 {
			t.Fatalf("values[%d] = %v, want %v", i, values[i], want[i])
		}
	}
}

func TestPenaltySampleHonorsRepeatWindow(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}

	logprobs := mlx.FromValues([]float32{
		1.0, 2.0, 3.0,
	}, 1, 3)

	got := Penalty{
		RepeatLastN:     1,
		PresencePenalty: 1.0,
	}.Sample(logprobs, []int32{0, 1})

	mlx.Eval(got)

	want := []float32{1.0, 1.0, 3.0}
	values := got.Floats()
	for i := range want {
		if math.Abs(float64(values[i]-want[i])) > 1e-5 {
			t.Fatalf("values[%d] = %v, want %v", i, values[i], want[i])
		}
	}
}

func TestDistributionFilterTopP(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}

	logits := mlx.FromValues([]float32{
		10.0, 9.0, 1.0, 0.0,
	}, 1, 4)

	filtered, indices := Distribution{
		Temperature: 1.0,
		TopK:        2,
		TopP:        0.55,
	}.filter(logits)

	got := materializeFilteredLogits(filtered, indices, 4)
	mlx.Eval(got)

	values := got.Floats()
	if values[0] != 10.0 {
		t.Fatalf("values[0] = %v, want 10", values[0])
	}
	for i := 1; i < len(values); i++ {
		if !math.IsInf(float64(values[i]), -1) {
			t.Fatalf("values[%d] = %v, want -Inf", i, values[i])
		}
	}
}

func materializeFilteredLogits(filtered, indices *mlx.Array, width int) *mlx.Array {
	if indices == nil {
		return filtered
	}

	base := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, width), float32(math.Inf(-1)))
	return base.PutAlongAxis(indices, filtered, -1)
}

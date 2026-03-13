//go:build mlx

package sample

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestPresencePenaltyUsesAppendedTokenImmediately(t *testing.T) {
	// RepeatLastN = 1, PresencePenalty = 6
	s := New(0, 0, 0, 0, 1, 6)
	defer func() {
		s.Free()
		mlx.Sweep()
	}()

	s.ResetHistory([]int32{0})
	s.AppendToken(mlx.NewArrayInt32([]int32{1}, []int32{1}))

	logprobs := mlx.FromValues([]float32{0, 5, 4}, 3)
	got := s.Sample(logprobs)
	mlx.Eval(got)

	// logprobs will be [0, -1, 4] after the penalty
	// and then (index) 2 after the greedy sampler
	gotInt := got.Int()
	if gotInt != 2 {
		t.Fatalf("got %d, want 2", gotInt)
	}
}

func TestMinPMasksTokensBelowThreshold(t *testing.T) {
	s := New(0, 0, 0.5, 0, 0, 0)
	defer func() {
		s.Free()
		mlx.Sweep()
	}()

	logprobs := mlx.FromValues([]float32{
		float32(math.Log(0.5)),
		float32(math.Log(0.3)),
		float32(math.Log(0.2)),
	}, 3)
	got := minP(s, logprobs)
	mlx.Eval(got)

	gotFloats := got.Floats()
	if len(gotFloats) != 3 {
		t.Fatalf("got %d scores, want 3", len(gotFloats))
	}

	if math.IsInf(float64(gotFloats[0]), -1) || math.IsInf(float64(gotFloats[1]), -1) {
		t.Fatalf("kept tokens were masked: %v", gotFloats)
	}

	if !math.IsInf(float64(gotFloats[2]), -1) {
		t.Fatalf("lowest-probability token should be masked, got %v", gotFloats)
	}
}

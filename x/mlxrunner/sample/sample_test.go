//go:build mlx

package sample

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestPresencePenaltyUsesAppendedTokenImmediately(t *testing.T) {
	s := New(Options{RepeatLastN: 1, PresencePenalty: 6})
	defer func() {
		s.Free()
		mlx.Sweep()
	}()

	s.ResetHistory([]int32{0})
	s.AppendToken(mlx.NewArrayInt32([]int32{1}, []int32{1}))

	logits := mlx.FromValues([]float32{0, 5, 4}, 3)
	got := s.Sample(logits).Token
	mlx.Eval(got)

	// logits will be [0, -1, 4] after the penalty
	// and then (index) 2 after the greedy sampler
	gotInt := got.Int()
	if gotInt != 2 {
		t.Fatalf("got %d, want 2", gotInt)
	}
}

func TestRepeatPenaltyUsesHistoryWithoutPresencePenalty(t *testing.T) {
	s := New(Options{RepeatLastN: 1, RepeatPenalty: 2})
	defer func() {
		s.Free()
		mlx.Sweep()
	}()

	s.ResetHistory([]int32{1})

	logits := mlx.FromValues([]float32{0, 5, 4}, 3)
	got := s.Sample(logits).Token
	mlx.Eval(got)

	// token 1 is repeated and positive, so 5 / 2 falls below token 2.
	gotInt := got.Int()
	if gotInt != 2 {
		t.Fatalf("got %d, want 2", gotInt)
	}
}

func TestFrequencyPenaltyUsesTokenCounts(t *testing.T) {
	s := New(Options{RepeatLastN: 4, FrequencyPenalty: 2})
	defer func() {
		s.Free()
		mlx.Sweep()
	}()

	s.ResetHistory([]int32{1, 1})

	logits := mlx.FromValues([]float32{0, 5, 4}, 3)
	got := s.Sample(logits).Token
	mlx.Eval(got)

	// token 1 appears twice, so 5 - (2 * 2) falls below token 2.
	gotInt := got.Int()
	if gotInt != 2 {
		t.Fatalf("got %d, want 2", gotInt)
	}
}

func TestMinPMasksTokensBelowThreshold(t *testing.T) {
	s := New(Options{MinP: 0.5})
	defer func() {
		s.Free()
		mlx.Sweep()
	}()

	logits := mlx.FromValues([]float32{
		float32(math.Log(0.5)),
		float32(math.Log(0.3)),
		float32(math.Log(0.2)),
	}, 3)
	got := minP(s, logits)
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

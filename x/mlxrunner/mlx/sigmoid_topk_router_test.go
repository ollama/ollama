package mlx

import (
	"math"
	"sort"
	"testing"
)

func TestFastSigmoidTopKRouter(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}

		gates := FromValues([]float32{
			-4, -3, 2, 0, 1,
			3, -1, 0.5, 4, -2,
		}, 2, 5)
		bias := FromValues([]float32{0.5, 0, 0, 0.4, 0}, 5).AsType(DTypeBFloat16)
		Pin(gates, bias)
		defer Unpin(gates, bias)

		scores, indices, ok := FastSigmoidTopKRouter(gates, bias, 2, true)
		if !ok {
			t.Fatal("FastSigmoidTopKRouter returned ok=false")
		}
		Eval(scores, indices)

		if got := scores.Dims(); len(got) != 2 || got[0] != 2 || got[1] != 2 {
			t.Fatalf("scores dims = %v, want [2 2]", got)
		}
		if got := indices.Dims(); len(got) != 2 || got[0] != 2 || got[1] != 2 {
			t.Fatalf("indices dims = %v, want [2 2]", got)
		}

		wantIndices, wantScores := sigmoidTopKRouterWant(
			[]float32{
				-4, -3, 2, 0, 1,
				3, -1, 0.5, 4, -2,
			},
			[]float32{0.5, 0, 0, 0.4, 0},
			2,
			5,
			2,
			true,
		)
		if got := indices.Ints(); !equalInts(got, wantIndices) {
			t.Fatalf("indices = %v, want %v", got, wantIndices)
		}
		assertFloat32Close(t, scores.Floats(), wantScores, 1e-5)
	})
}

func TestFastSigmoidTopKRouterScaled(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}

		gates := FromValues([]float32{
			-4, -3, 2, 0, 1,
			3, -1, 0.5, 4, -2,
		}, 2, 5)
		bias := FromValues([]float32{0.5, 0, 0, 0.4, 0}, 5)
		scaleA := FromValues([]float32{2, 3, 5, 7, 11}, 5)
		scaleB := FromValues([]float32{0.5, 0.25, 0.125, 2, 4}, 5)
		Pin(gates, bias, scaleA, scaleB)
		defer Unpin(gates, bias, scaleA, scaleB)

		scores, indices, ok := FastSigmoidTopKRouterScaled(gates, bias, scaleA, scaleB, 2, true)
		if !ok {
			t.Fatal("FastSigmoidTopKRouterScaled returned ok=false")
		}
		Eval(scores, indices)

		wantIndices, wantScores := sigmoidTopKRouterWant(
			[]float32{
				-4, -3, 2, 0, 1,
				3, -1, 0.5, 4, -2,
			},
			[]float32{0.5, 0, 0, 0.4, 0},
			2,
			5,
			2,
			true,
		)
		scaleAVals := []float32{2, 3, 5, 7, 11}
		scaleBVals := []float32{0.5, 0.25, 0.125, 2, 4}
		for i, expert := range wantIndices {
			wantScores[i] *= scaleAVals[expert] * scaleBVals[expert]
		}
		if got := indices.Ints(); !equalInts(got, wantIndices) {
			t.Fatalf("indices = %v, want %v", got, wantIndices)
		}
		assertFloat32Close(t, scores.Floats(), wantScores, 1e-5)
	})
}

func TestFastSigmoidTopKRouterBF16MatchesWidenedF32(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}

		gates := FromValues([]float32{
			-4.125, -3.25, 2.5, 0, 1.75,
			3.5, -1.125, 0.5, 4.25, -2.75,
		}, 2, 5).AsType(DTypeBFloat16)
		bias := FromValues([]float32{0.5, 0, 0, 0.4, 0}, 5)
		Pin(gates, bias)
		defer Unpin(gates, bias)

		bf16Scores, bf16Indices, ok := FastSigmoidTopKRouter(gates, bias, 2, true)
		if !ok {
			t.Fatal("FastSigmoidTopKRouter returned ok=false for BF16 gates")
		}
		f32Scores, f32Indices, ok := FastSigmoidTopKRouter(gates.AsType(DTypeFloat32), bias.AsType(DTypeFloat32), 2, true)
		if !ok {
			t.Fatal("FastSigmoidTopKRouter returned ok=false for widened F32 gates")
		}
		Eval(bf16Scores, bf16Indices, f32Scores, f32Indices)

		if got, want := bf16Indices.Ints(), f32Indices.Ints(); !equalInts(got, want) {
			t.Fatalf("indices = %v, want %v", got, want)
		}
		assertFloat32Close(t, bf16Scores.Floats(), f32Scores.Floats(), 1e-5)
	})
}

func sigmoidTopKRouterWant(gates, bias []float32, tokens, experts, topK int, normalize bool) ([]int, []float32) {
	type candidate struct {
		index    int
		prob     float32
		adjusted float32
	}

	indices := make([]int, 0, tokens*topK)
	scores := make([]float32, 0, tokens*topK)
	for t := range tokens {
		row := make([]candidate, 0, experts)
		for e := range experts {
			p := float32(1 / (1 + math.Exp(-float64(gates[t*experts+e]))))
			row = append(row, candidate{index: e, prob: p, adjusted: p + bias[e]})
		}
		sort.Slice(row, func(i, j int) bool {
			return row[i].adjusted > row[j].adjusted
		})

		denom := float32(1)
		if normalize {
			denom = 0
			for k := range topK {
				denom += row[k].prob
			}
		}
		for k := range topK {
			indices = append(indices, row[k].index)
			scores = append(scores, row[k].prob/denom)
		}
	}
	return indices, scores
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func assertFloat32Close(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len got=%d want=%d", len(got), len(want))
	}
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			t.Fatalf("got[%d]=%v want %v diff %v", i, got[i], want[i], diff)
		}
	}
}

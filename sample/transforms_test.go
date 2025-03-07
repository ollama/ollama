package sample

import (
	"math"
	"math/rand/v2"
	"testing"
)

// Helper to convert float64 slice to logit slice
func toLogits(values []float64) []logit {
	tokens := make([]logit, len(values))
	for i, v := range values {
		tokens[i] = logit{
			id:    int32(i),
			value: float32(v),
		}
	}
	return tokens
}

// Helper to compare logit slices
func compareLogits(t *testing.T, name string, want []float64, got []logit) {
	t.Helper()
	if len(want) != len(got) {
		t.Errorf("%s: length mismatch: want %d, got %d", name, len(want), len(got))
		return
	}
	for i := range want {
		if math.Abs(float64(got[i].value)-want[i]) > 1e-6 {
			t.Errorf("%s: index %d: want %f, got %f", name, i, want[i], got[i].value)
		}
	}
}

func TestTemperature(t *testing.T) {
	input := []float64{2, -1, 4, -3, 1, -2, 0}
	want := []float64{-4, -10, 0, -14, -6, -12, -8} // (logit - max logit) / temp

	got := temperature(toLogits(input), 0.5)
	compareLogits(t, "Temperature", want, got)
}

func TestSoftmax(t *testing.T) {
	input := []float64{-3, -2, -1, 0, 1, 2, 4}
	got := softmax(toLogits(input))

	// Check probabilities sum to 1
	var sum float32
	for _, token := range got {
		sum += token.value
	}
	if math.Abs(float64(sum)-1.0) > 1e-6 {
		t.Errorf("probabilities don't sum to 1: got %f", sum)
	}

	// Check relative ordering is preserved
	for i := 1; i < len(got); i++ {
		if got[i].value < got[i-1].value {
			t.Errorf("probability ordering not preserved at index %d", i)
		}
	}
}

func TestTopK(t *testing.T) {
	input := []float64{-3, -2, -1, 0, 1, 2, 4}

	// Test k=3
	got := topK(toLogits(input), 3)
	if len(got) != 3 {
		t.Errorf("topK(3): wrong length: want 3, got %d", len(got))
	}
	// Should keep highest 3 values: 4, 2, 1
	want := []float64{4, 2, 1}
	compareLogits(t, "topK(3)", want, got)

	// Test k > len
	got = topK(toLogits(input), 10)
	compareLogits(t, "topK(10)", input, got)
}

func TestTopP(t *testing.T) {
	input := []float64{-3, -2, -1, 0, 1, 2, 4}
	tokens := toLogits(input)

	// First apply temperature and softmax to get probabilities
	tokens = temperature(tokens, 1)
	tokens = softmax(tokens)
	sortLogits(tokens)

	// Then apply topP
	got := topP(tokens, 0.95)

	// Should keep tokens until cumsum > 0.95
	if len(got) > 3 {
		t.Errorf("topP(0.95): kept too many tokens: got %d", len(got))
		t.Logf("got: %v", got)
	}
}

func TestMinP(t *testing.T) {
	input := []float64{-3, -2, -1, 0, 1, 2, 4, 3}
	tokens := toLogits(input)

	// First apply temperature and softmax
	tokens = temperature(tokens, 1)
	tokens = softmax(tokens)

	// Then apply minP
	got := minP(tokens, 0.2)

	// Should keep tokens with prob >= 0.2 * max_prob
	if len(got) > 3 {
		t.Errorf("minP(0.2): kept too many tokens: got %d", len(got))
	}
}

func TestSortLogits(t *testing.T) {
	input := []float64{3, 1, 4, 2, -1, 0, -2}
	tokens := toLogits(input)

	sortLogits(tokens)

	for i := 1; i < len(tokens); i++ {
		if tokens[i].value > tokens[i-1].value {
			t.Errorf("sortLogits: tokens not sorted in descending order at index %d: %f > %f",
				i, tokens[i].value, tokens[i-1].value)
		}
	}

	want := []float64{4, 3, 2, 1, 0, -1, -2}
	compareLogits(t, "sortLogits", want, tokens)
}

func BenchmarkTransforms(b *testing.B) {
	// Generate random logits
	tokens := make([]logit, 1<<16)
	for i := range tokens {
		tokens[i] = logit{
			id:    int32(i),
			value: rand.Float32(),
		}
	}

	tokensCopy := make([]logit, len(tokens))

	b.Run("Temperature", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			copy(tokensCopy, tokens)
			temperature(tokensCopy, 0.5)
		}
	})

	b.Run("TopK", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			copy(tokensCopy, tokens)
			topK(tokensCopy, 10)
		}
	})

	b.Run("TopP", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			copy(tokensCopy, tokens)
			topP(tokensCopy, 0.9)
		}
	})

	b.Run("MinP", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			copy(tokensCopy, tokens)
			minP(tokensCopy, 0.2)
		}
	})

	b.Run("SortTokens", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			copy(tokensCopy, tokens)
			sortLogits(tokensCopy)
		}
	})
}

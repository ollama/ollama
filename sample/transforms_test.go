package sample

import (
	"math"
	"math/rand/v2"
	"testing"
)

// Helper to convert float64 slice to logit slice
func toTokens(values []float64) []token {
	tokens := make([]token, len(values))
	for i, v := range values {
		tokens[i] = token{
			id:    int32(i),
			value: float32(v),
		}
	}
	return tokens
}

// Helper to compare logit slices
func compareLogits(t *testing.T, name string, want []float64, got []token) {
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

func TestTemperatureAndSoftmax(t *testing.T) {
	input := []float64{1, 4, -2, 0}
	got := temperature(toTokens(input), 0.5)

	// Check probabilities sum to 1
	var sum float32
	for _, token := range got {
		sum += token.value
	}
	if math.Abs(float64(sum)-1.0) > 1e-6 {
		t.Errorf("probabilities don't sum to 1: got %f", sum)
	}

	got = temperature(toTokens(input), 1)
	// Check probabilities sum to 1
	sum = 0
	for _, token := range got {
		sum += token.value
	}
	if math.Abs(float64(sum)-1.0) > 1e-6 {
		t.Errorf("probabilities don't sum to 1: got %f", sum)
	}
}

func TestTopK(t *testing.T) {
	input := []float64{-3, -2, -1, 0, 1, 2, 4}

	// Test k=3
	got := topK(toTokens(input), 3)
	if len(got) != 3 {
		t.Errorf("topK(3): wrong length: want 3, got %d", len(got))
	}
	// Should keep highest 3 values: 4, 2, 1
	want := []float64{4, 2, 1}
	compareLogits(t, "topK(3)", want, got)

	// Test k > len
	got = topK(toTokens(input), 10)
	compareLogits(t, "topK(10)", input, got)
}

func TestTopP(t *testing.T) {
	input := []float64{-3, -2, -1, 0, 1, 2, 4}
	tokens := toTokens(input)

	// First apply temperature and softmax to get probabilities
	tokens = temperature(tokens, 1)
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
	tokens := toTokens(input)

	// First apply temperature and softmax
	tokens = temperature(tokens, 1)

	// Then apply minP
	got := minP(tokens, 0.2)

	// Should keep tokens with prob >= 0.2 * max_prob
	if len(got) > 3 {
		t.Errorf("minP(0.2): kept too many tokens: got %d", len(got))
	}
}

func TestSortLogits(t *testing.T) {
	input := []float64{3, 1, 4, 2, -1, 0, -2}
	tokens := toTokens(input)

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
	tokens := make([]token, 1<<16)
	for i := range tokens {
		tokens[i] = token{
			id:    int32(i),
			value: rand.Float32(),
		}
	}

	tokensCopy := make([]token, len(tokens))

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

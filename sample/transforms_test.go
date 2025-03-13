package sample

import (
	"math"
	"math/rand/v2"
	"testing"
)

// Helper to convert float32 slice to logit slice
func toTokens(values []float32) []token {
	tokens := make([]token, len(values))
	for i, v := range values {
		tokens[i] = token{
			id:    int32(i),
			value: v,
		}
	}
	return tokens
}

// Helper to compare logit slices
func compareLogits(t *testing.T, name string, want []float32, got []token) {
	t.Helper()
	if len(want) != len(got) {
		t.Errorf("%s: length mismatch: want %d, got %d", name, len(want), len(got))
		return
	}
	for i := range want {
		if math.Abs(float64(got[i].value-want[i])) > 1e-6 {
			t.Errorf("%s: index %d: want %f, got %f", name, i, want[i], got[i].value)
		}
	}
}

func TestTemperature(t *testing.T) {
	input := []float32{1.0, 4.0, -2.0, 0.0}
	got := temperature(toTokens(input), 0.5)
	want := []float32{2.0, 8.0, -4.0, 0.0}
	compareLogits(t, "temperature(0.5)", want, got)

	got = temperature(toTokens(input), 1.0)
	want = []float32{1.0, 4.0, -2.0, 0.0}
	compareLogits(t, "temperature(1)", want, got)

	got = temperature(toTokens(input), 0.0)
	want = []float32{1e7, 4e7, -2e7, 0.0}
	compareLogits(t, "temperature(0)", want, got)
}

func TestSoftmax(t *testing.T) {
	tests := []struct {
		name     string
		input    []float32
		expected []float32
	}{
		{
			name:     "correctness softmax",
			input:    []float32{1, -2, 3, 0},
			expected: []float32{0.113550, 0.005653, 0.839024, 0.041773},
		},
		{
			name:  "normal distribution",
			input: []float32{0.026986899, 0.043722924, 0.036774673, 0.27755088, 0.0046718004, 0.08582123, 0.20409796, 0.00412893, 0.15720603, 0.045046154, 0.0030491839, 0.01681367},
		},
		{
			name:  "single value",
			input: []float32{1.0},
		},
		{
			name:  "identical values",
			input: []float32{0.9, 0.9, 0.9},
		},
		{
			name:  "large values",
			input: []float32{1000.0, 2000.0, 3000.0},
		},
		{
			name:  "small values",
			input: []float32{1e-6, 2e-6, 3e-6},
		},
		{
			name:  "negative values",
			input: []float32{-1.0, -2.0, -3.0},
		},
		{
			name:  "mixed values",
			input: []float32{-100.0, 0.0, 100.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := softmax(toTokens(tt.input))

			if tt.expected != nil {
				compareLogits(t, tt.name, tt.expected, got)
				return
			}

			// Check probabilities sum to 1
			var sum float32
			for _, token := range got {
				sum += token.value
				if token.value < 0 || token.value > 1 {
					t.Errorf("probability out of range [0,1]: got %f", token.value)
				}
			}
			if math.Abs(float64(sum-1.0)) > 1e-6 {
				t.Errorf("probabilities don't sum to 1: got %f", sum)
			}
		})
	}
}

func TestTopK(t *testing.T) {
	input := []float32{0.026986899, 0.043722924, 0.036774673, 0.27755088, 0.0046718004, 0.08582123, 0.20409796, 0.00412893, 0.15720603, 0.045046154, 0.0030491839, 0.01681367}

	// Test k=5
	got := topK(toTokens(input), 5)
	if len(got) != 5 {
		t.Errorf("topK(5): wrong length: want 5, got %d", len(got))
	}
	// Should keep highest 3 values in descending order
	want := []float32{0.27755088, 0.20409796, 0.15720603, 0.08582123, 0.045046154}
	compareLogits(t, "topK(3)", want, got)

	got = topK(toTokens(input), 20)
	if len(got) != len(input) {
		t.Errorf("topK(20): wrong length: want %d, got %d", len(input), len(got))
	}

	// Test k=-1
	input = []float32{0.026986899, 0.043722924, 0.036774673, 0.27755088, 0.0046718004, 0.08582123, 0.20409796, 0.00412893, 0.15720603, 0.045046154, 0.0030491839, 0.01681367}
	want = []float32{0.27755088, 0.20409796, 0.15720603, 0.08582123, 0.045046154, 0.043722924, 0.036774673, 0.026986899, 0.01681367, 0.0046718004, 0.00412893, 0.0030491839}
	got = topK(toTokens(input), -1)
	if len(got) != len(input) {
		t.Errorf("topK(-1): wrong length: want %d, got %d", len(input), len(got))
	}
	compareLogits(t, "topK(-1)", want, got)

	// Test k=0
	input = []float32{0.026986899, 0.043722924, 0.036774673, 0.27755088, 0.0046718004, 0.08582123, 0.20409796, 0.00412893, 0.15720603, 0.045046154, 0.0030491839, 0.01681367}
	want = []float32{0.27755088, 0.20409796, 0.15720603, 0.08582123, 0.045046154, 0.043722924, 0.036774673, 0.026986899, 0.01681367, 0.0046718004, 0.00412893, 0.0030491839}
	got = topK(toTokens(input), 0)
	if len(got) != len(input) {
		t.Errorf("topK(-1): wrong length: want %d, got %d", len(input), len(got))
	}
	compareLogits(t, "topK(-1)", want, got)
}

func TestTopP(t *testing.T) {
	input := []float32{-3, -2, -1, 0, 1, 2, 4}
	tokens := toTokens(input)

	// First apply temperature and softmax to get probabilities
	tokens = softmax(tokens)
	tokens = topK(tokens, 20)

	// Then apply topP
	got := topP(tokens, 0.95)

	// Should keep tokens until cumsum > 0.95
	if len(got) > 3 {
		t.Errorf("topP(0.95): kept too many tokens: got %d", len(got))
		t.Logf("got: %v", got)
	}
}

func TestMinP(t *testing.T) {
	input := []float32{-3, -2, -1, 0, 1, 2, 4, 3}
	tokens := toTokens(input)

	// First apply temperature and softmax
	tokens = softmax(tokens)

	// Then apply minP
	got := minP(tokens, 0.2)

	// Should keep tokens with prob >= 0.2 * max_prob
	if len(got) > 3 {
		t.Errorf("minP(0.2): kept too many tokens: got %d", len(got))
	}
}

func TestSortLogits(t *testing.T) {
	input := []float32{0.026986899, 0.043722924, 0.036774673, 0.27755088, 0.0046718004, 0.08582123, 0.20409796, 0.00412893, 0.15720603, 0.045046154, 0.0030491839, 0.01681367}
	tokens := toTokens(input)

	tokens = topK(tokens, 20)

	for i := 1; i < len(tokens); i++ {
		if tokens[i].value > tokens[i-1].value {
			t.Errorf("sortLogits: tokens not sorted in descending order at index %d: %f > %f",
				i, tokens[i].value, tokens[i-1].value)
		}
	}

	want := []float32{0.27755088, 0.20409796, 0.15720603, 0.08582123, 0.045046154, 0.043722924, 0.036774673, 0.026986899, 0.01681367, 0.0046718004, 0.00412893, 0.0030491839}
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

	b.Run("Softmax", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			copy(tokensCopy, tokens)
			softmax(tokensCopy)
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
			topK(tokensCopy, 200000)
		}
	})
}

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
	tokens := toTokens(input)
	temperature(tokens, 0.5)
	want := []float32{2.0, 8.0, -4.0, 0.0}
	compareLogits(t, "temperature(0.5)", want, tokens)

	input = []float32{1.0, 4.0, -2.0, 0.0}
	tokens = toTokens(input)
	temperature(tokens, 1.0)
	want = []float32{1.0, 4.0, -2.0, 0.0}
	compareLogits(t, "temperature(1)", want, tokens)

	input = []float32{1.0, 4.0, -2.0, 0.0}
	tokens = toTokens(input)
	temperature(tokens, 0.0)
	want = []float32{1e7, 4e7, -2e7, 0.0}
	compareLogits(t, "temperature(0)", want, tokens)
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
			tokens := toTokens(tt.input)
			softmax(tokens)

			if tt.expected != nil {
				compareLogits(t, tt.name, tt.expected, tokens)
				return
			}

			// Check probabilities sum to 1
			var sum float32
			for _, token := range tokens {
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
	tokens := toTokens(input)
	tokens = topK(tokens, 5)
	if len(tokens) != 5 {
		t.Errorf("topK(5): wrong length: want 5, got %d", len(tokens))
	}
	want := []float32{0.27755088, 0.20409796, 0.15720603, 0.08582123, 0.045046154}
	compareLogits(t, "topK(3)", want, tokens)

	tokens = toTokens(input)
	tokens = topK(tokens, 20)
	if len(tokens) != len(input) {
		t.Errorf("topK(20): wrong length: want %d, got %d", len(input), len(tokens))
	}

	input = []float32{0.026986899, 0.043722924, 0.036774673, 0.27755088, 0.0046718004, 0.08582123, 0.20409796, 0.00412893, 0.15720603, 0.045046154, 0.0030491839, 0.01681367}
	want = []float32{0.27755088, 0.20409796, 0.15720603, 0.08582123, 0.045046154, 0.043722924, 0.036774673, 0.026986899, 0.01681367, 0.0046718004, 0.00412893, 0.0030491839}
	tokens = toTokens(input)
	tokens = topK(tokens, -1)
	if len(tokens) != len(input) {
		t.Errorf("topK(-1): wrong length: want %d, got %d", len(input), len(tokens))
	}
	compareLogits(t, "topK(-1)", want, tokens)

	input = []float32{0.026986899, 0.043722924, 0.036774673, 0.27755088, 0.0046718004, 0.08582123, 0.20409796, 0.00412893, 0.15720603, 0.045046154, 0.0030491839, 0.01681367}
	want = []float32{0.27755088, 0.20409796, 0.15720603, 0.08582123, 0.045046154, 0.043722924, 0.036774673, 0.026986899, 0.01681367, 0.0046718004, 0.00412893, 0.0030491839}
	tokens = toTokens(input)
	tokens = topK(tokens, 0)
	if len(tokens) != len(input) {
		t.Errorf("topK(-1): wrong length: want %d, got %d", len(input), len(tokens))
	}
	compareLogits(t, "topK(-1)", want, tokens)

	input = []float32{-1e7, -2e7, -3e7, -4e7}
	tokens = toTokens(input)
	tokens = topK(tokens, 1)
	if len(tokens) < 1 {
		t.Error("topK should keep at least one token")
	}
}

func TestTopP(t *testing.T) {
	input := []float32{-3, -2, -1, 0, 1, 2, 4}
	tokens := toTokens(input)

	// First apply temperature and softmax to get probabilities
	softmax(tokens)
	tokens = topK(tokens, 20)

	// Test with very high p value
	got := topP(tokens, 1.0)

	// Should keep all tokens since p is 1
	if len(got) != len(input) {
		t.Errorf("topP(1.0): should keep all tokens, got %d, want %d", len(got), len(input))
	}

	// Test with normal p value
	got = topP(tokens, 0.95)

	if len(got) > 3 {
		t.Errorf("topP(0.95): kept too many tokens: got %d", len(tokens))
		t.Logf("got: %v", got)
	}

	// Test edge case - ensure at least one token remains
	input = []float32{-1e6, -1e6, -1e7}
	tokens = toTokens(input)
	tokens = topK(tokens, 20)
	softmax(tokens)
	got = topP(tokens, 0.0)
	if len(got) < 1 {
		t.Error("topP should keep at least one token")
	}

	// Test with zero p value
	got = topP(tokens, 0.0)

	// Should keep only the highest probability token
	if len(got) != 1 {
		t.Errorf("topP(0.0): should keep only one token, got %d", len(got))
		t.Logf("got: %v", got)
	}

	tokens = toTokens(input)
	tokens = topK(tokens, 20)
	softmax(tokens)
	got = topP(tokens, 1e-10)
	if len(got) == 0 {
		t.Errorf("topP(1e-10): should keep at least one token, got %d", len(got))
		t.Logf("got: %v", got)
	}
}

func TestMinP(t *testing.T) {
	input := []float32{-2, 0, -1, -3, 2, 1, 4, 3}
	tokens := toTokens(input)

	// First apply temperature and softmax
	tokens = topK(tokens, 20)
	softmax(tokens)

	tokens = minP(tokens, 1.0)

	if len(tokens) != 1 {
		t.Errorf("minP(1.0): should keep all tokens, got %d, want %d", len(tokens), len(tokens))
	}

	// Test with normal p value
	tokens = toTokens(input) // Reset tokens
	tokens = topK(tokens, 20)
	softmax(tokens)
	tokens = minP(tokens, 0.2)

	// Should keep tokens with prob >= 0.2 * max_prob
	if len(tokens) > 3 {
		t.Errorf("minP(0.2): kept too many tokens: got %d", len(tokens))
		t.Logf("got: %v", tokens)
	}

	// Test with zero p value
	tokens = toTokens(input) // Reset tokens
	tokens = topK(tokens, 20)
	softmax(tokens)
	tokens = minP(tokens, 0.0)

	// Should keep only the highest probability token
	if len(tokens) != len(input) {
		t.Errorf("minP(0.0): should keep only one token, got %d", len(tokens))
		t.Logf("got: %v", tokens)
	}

	// Test with single token
	tokens = toTokens(input[:1])
	tokens = topK(tokens, 20)
	softmax(tokens)
	tokens = minP(tokens, 0.1)

	// Should keep only the highest probability token
	if len(tokens) != 1 {
		t.Errorf("minP(0.1): should return single token, got %d", len(tokens))
		t.Logf("got: %v", tokens)
	}

	input = []float32{1e-10, 1e-10, 1e-10}
	tokens = toTokens(input)
	softmax(tokens)
	tokens = minP(tokens, 1.0)
	if len(tokens) < 1 {
		t.Error("minP should keep at least one token even with extreme probabilities")
		got := minP(tokens, 1.0)

		if len(got) != 1 {
			t.Errorf("minP(1.0): should keep all tokens, got %d, want %d", len(got), len(tokens))
		}

		// Test with normal p value
		got = minP(tokens, 0.2)

		// Should keep tokens with prob >= 0.2 * max_prob
		if len(got) > 3 {
			t.Errorf("minP(0.2): kept too many tokens: got %d", len(got))
			t.Logf("got: %v", got)
		}

		// Test with zero p value
		got = minP(tokens, 0.0)

		// Should keep only the highest probability token
		if len(got) != len(tokens) {
			t.Errorf("minP(0.0): should keep only one token, got %d", len(got))
			t.Logf("got: %v", got)
		}
	}
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
			tokens = topK(tokensCopy, 10)
		}
	})

	b.Run("TopP", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			copy(tokensCopy, tokens)
			tokens = topP(tokensCopy, 0.9)
		}
	})

	b.Run("MinP", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			copy(tokensCopy, tokens)
			tokens = minP(tokensCopy, 0.2)
		}
	})

	b.Run("SortTokens", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			copy(tokensCopy, tokens)
			tokens = topK(tokensCopy, 200000)
		}
	})
}

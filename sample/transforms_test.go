package sample

import (
	"encoding/binary"
	"errors"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
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

func TestTemperatureAndSoftmax(t *testing.T) {
	input := []float32{1, 4, -2, 0}
	got := temperature(toTokens(input), 0.5)

	// Check probabilities sum to 1
	var sum float32
	for _, token := range got {
		sum += token.value
	}
	if math.Abs(float64(sum-1.0)) > 1e-6 {
		t.Errorf("probabilities don't sum to 1: got %f", sum)
	}

	got = temperature(toTokens(input), 1)
	// Check probabilities sum to 1
	sum = 0.0
	for _, token := range got {
		sum += token.value
	}
	if math.Abs(float64(sum-1.0)) > 1e-6 {
		t.Errorf("probabilities don't sum to 1: got %f", sum)
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
	tokens = temperature(tokens, 1)
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
	tokens = temperature(tokens, 1)

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

// TestSortLogitsWithRealData tests sorting behavior using real model logit distributions
func TestSortLogitsWithRealData(t *testing.T) {
	// This will be populated from testdata/logits.bin
	// Format: 32-bit float array in binary format
	logits, err := loadTestLogits(t)
	if err != nil {
		t.Skipf("Skipping real logit test: %v", err)
		return
	}

	tokens := toTokens(logits)
	sortLogits(tokens)

	// Calculate n for verification
	n := int(math.Sqrt(float64(len(tokens)))) + 1
	if n > 1000 {
		n = 1000
	} else if n < 100 {
		n = 100
	}

	t.Logf("Testing with %d tokens, partial sorting top %d", len(tokens), n)

	// Only verify the top n elements are sorted (which is what we guarantee)
	// This is much faster than checking the entire array
	topN := tokens[:n]
	for i := 1; i < len(topN); i++ {
		if topN[i].value > topN[i-1].value {
			t.Fatalf("top %d tokens not properly sorted at index %d: %.15f > %.15f",
				n, i, topN[i].value, topN[i-1].value)
		}
	}

	// Verify we didn't lose any high value tokens by checking that
	// all tokens after position n are <= the nth token
	// Do this in chunks to avoid timeouts on large arrays
	nthValue := tokens[n-1].value
	const chunkSize = 1000

	for start := n; start < len(tokens); start += chunkSize {
		end := min(start+chunkSize, len(tokens))
		for i := start; i < end; i++ {
			if tokens[i].value > nthValue {
				t.Fatalf("found higher value token after position %d: tokens[%d].value = %.15f > %.15f",
					n, i, tokens[i].value, nthValue)
			}
		}
	}
}

// loadTestLogits loads logit test data from testdata/logits.bin
func loadTestLogits(t *testing.T) ([]float32, error) {
	t.Helper()

	_, currFile, _, ok := runtime.Caller(0)
	if !ok {
		return nil, errors.New("could not determine test file path")
	}
	testDataPath := filepath.Join(filepath.Dir(currFile), "testdata", "logits.bin")

	file, err := os.Open(testDataPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	stat, err := file.Stat()
	if err != nil {
		return nil, err
	}

	numFloats := stat.Size() / 4 // each float32 is 4 bytes
	if numFloats*4 != stat.Size() {
		return nil, errors.New("logits.bin has invalid size: not a multiple of 4 bytes")
	}

	logits := make([]float32, numFloats)
	for i := range logits {
		var val uint32
		if err := binary.Read(file, binary.LittleEndian, &val); err != nil {
			return nil, err
		}
		logits[i] = math.Float32frombits(val)
	}

	if len(logits) == 0 {
		return nil, errors.New("logits.bin is empty")
	}

	return logits, nil
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
			topK(tokensCopy, 200000)
		}
	})
}

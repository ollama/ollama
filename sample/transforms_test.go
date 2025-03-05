package sample

import (
	"math"
	"math/rand"
	"testing"
)

// Helper to convert float64 slice to tokenInfo slice
func toTokenInfo(logits []float64) []tokenInfo {
	tokens := make([]tokenInfo, len(logits))
	for i, v := range logits {
		tokens[i] = tokenInfo{
			id:    int32(i),
			logit: float32(v),
		}
	}
	return tokens
}

// Helper to compare tokenInfo slices
func compareLogits(t *testing.T, name string, want []float64, got []tokenInfo) {
	t.Helper()
	if len(want) != len(got) {
		t.Errorf("%s: length mismatch: want %d, got %d", name, len(want), len(got))
		return
	}
	for i := range want {
		if math.Abs(float64(got[i].logit)-want[i]) > 1e-6 {
			t.Errorf("%s: index %d: want %f, got %f", name, i, want[i], got[i].logit)
		}
	}
}

func TestTemperature(t *testing.T) {
	input := []float64{2, -1, 4, -3, 1, -2, 0}
	want := []float64{-4, -10, 0, -14, -6, -12, -8} // (logit - max logit) / temp

	ts := tokenSliceInfo{tokens: toTokenInfo(input)}
	got := Temperature(0.5).Apply(ts)

	compareLogits(t, "Temperature", want, got.tokens)
}

func TestSoftmax(t *testing.T) {
	input := []float64{-3, -2, -1, 0, 1, 2, 4}
	ts := tokenSliceInfo{tokens: toTokenInfo(input)}
	got := softmax{}.Apply(ts)

	// Check probabilities sum to 1
	var sum float32
	for _, token := range got.tokens {
		sum += token.prob
	}
	if math.Abs(float64(sum)-1.0) > 1e-6 {
		t.Errorf("probabilities don't sum to 1: got %f", sum)
	}

	// Check relative ordering is preserved
	for i := 1; i < len(got.tokens); i++ {
		if got.tokens[i].prob < got.tokens[i-1].prob {
			t.Errorf("probability ordering not preserved at index %d", i)
		}
	}
}

func TestTopK(t *testing.T) {
	input := []float64{-3, -2, -1, 0, 1, 2, 4}

	// Test k=3
	ts := tokenSliceInfo{tokens: toTokenInfo(input)}
	got := TopK(3).Apply(ts)
	if len(got.tokens) != 3 {
		t.Errorf("TopK(3): wrong length: want 3, got %d", len(got.tokens))
	}
	// Should keep highest 3 values: 4, 2, 1
	want := []float64{4, 2, 1}
	compareLogits(t, "TopK(3)", want, got.tokens)

	// Test k > len
	ts = tokenSliceInfo{tokens: toTokenInfo(input)}
	got = TopK(10).Apply(ts)
	compareLogits(t, "TopK(10)", input, got.tokens)
}

func TestTopP(t *testing.T) {
	input := []float64{-3, -2, -1, 0, 1, 2, 4}
	ts := tokenSliceInfo{tokens: toTokenInfo(input)}

	ts = softmax{}.Apply(ts)
	sortTokens{}.Apply(ts)

	// Then apply TopP
	got := TopP(0.95).Apply(ts)

	// Should keep tokens until cumsum > 0.9
	if len(got.tokens) > 3 {
		t.Errorf("TopP(0.9): kept too many tokens: got %d", len(got.tokens))
		t.Logf("got: %v", got.tokens)
	}
}

func TestMinP(t *testing.T) {
	input := []float64{-3, -2, -1, 0, 1, 2, 4, 3}
	ts := tokenSliceInfo{tokens: toTokenInfo(input)}

	// First apply temperature and softmax
	ts = Temperature(1).Apply(ts)
	ts = softmax{}.Apply(ts)

	// Then apply MinP
	got := MinP(0.2).Apply(ts)

	// Should keep tokens with prob >= 0.2 * max_prob
	if len(got.tokens) > 3 {
		t.Errorf("MinP(0.2): kept too many tokens: got %d", len(got.tokens))
	}
}

func BenchmarkTransform(b *testing.B) {
	transforms := map[string]transform{
		"Temperature": Temperature(0.5),
		"TopK":        TopK(10),
		"TopP":        TopP(0.9),
		"MinP":        MinP(0.2),
	}

	// Generate random logits
	tokens := make([]tokenInfo, 1<<16)
	for i := range tokens {
		tokens[i] = tokenInfo{
			id:    int32(i),
			logit: rand.Float32(),
		}
	}

	for name, tr := range transforms {
		b.Run(name, func(b *testing.B) {
			ts := tokenSliceInfo{tokens: tokens}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				tr.Apply(ts)
			}
		})
	}
}

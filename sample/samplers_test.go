package sample

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/tokenizer"
)

func TestWeighted(t *testing.T) {
	logits := []float32{-10, 3, -10, -10}
	sampler := NewSampler(0, 0, 0, 0, 0, 1.0, 0.0, 0.0, 0, nil)
	got, err := sampler.Sample(logits)
	if err != nil {
		t.Error(err)
		return
	}
	want := int32(1)
	if want != got {
		t.Errorf("index mismatch: want %d, got %d", want, got)
	}

	logits = []float32{-100, -10, 0, 10}
	sampler = NewSampler(0, 0, 0, 0, 0, 1.0, 0.0, 0.0, 0, nil)
	got, err = sampler.Sample(logits)
	if err != nil {
		t.Error(err)
		return
	}
	want = int32(3) // Should pick highest probability with this r value
	if want != got {
		t.Errorf("index mismatch: want %d, got %d", want, got)
	}

	// Test very high p
	logits = []float32{1.0, 0.9999999999999999, 0.5, 0.1}
	// Use extremely small topP to filter out all tokens
	sampler = NewSampler(1.0, 0, 1e-10, 0, 0, 1.0, 0.0, 0.0, 0, nil)
	got, err = sampler.Sample(logits)
	if err != nil {
		t.Error(err)
		return
	}
	// Should get the token with the highest logit
	want = int32(0)
	if want != got {
		t.Errorf("index mismatch: want %d, got %d", want, got)
	}

	logits = []float32{float32(math.NaN()), float32(math.NaN()), float32(math.NaN())}
	sampler = NewSampler(1, 0, 0.95, 0.05, 0, 1.0, 0.0, 0.0, 0, nil)
	got, err = sampler.Sample(logits)
	if err == nil {
		t.Errorf("expected error, got %d", got)
		return
	}
}

func modelHelper(t testing.TB) tokenizer.Tokenizer {
	t.Helper()

	f, err := os.Open(filepath.FromSlash("../tokenizer/testdata/llama3.2/encoder.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	vocab := make(map[string]int32)
	if err := json.NewDecoder(f).Decode(&vocab); err != nil {
		t.Fatal(err)
	}

	tokens := make([]string, len(vocab))
	for token, id := range vocab {
		tokens[id] = token
	}

	merges := make([]string, 0, 1)
	// Only need vocab for Grammar Test
	return tokenizer.NewBytePairEncoding(
		&tokenizer.Vocabulary{
			Values: tokens,
			Types:  make([]int32, len(vocab)),
			Merges: merges,
		},
	)
}

func TestGrammar(t *testing.T) {
	tokenizer := modelHelper(t)

	grammarJSON := `
	root   ::= object
	value  ::= object | array | string | number | ("true" | "false" | "null") ws
	object ::=
	"{" ws (
				string ":" ws value
		("," ws string ":" ws value)*
	)? "}" ws
	array  ::=
	"[" ws (
				value
		("," ws value)*
	)? "]" ws
	string ::=
	"\"" (
		[^"\\\x7F\x00-\x1F] |
		"\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
	)* "\"" ws
	number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
	# Optional space: by convention, applied in this grammar after literal chars when allowed
	ws ::= ([ \t\n] ws)?
	`
	grammar, err := NewGrammarSampler(tokenizer, grammarJSON)
	if err != nil {
		t.Fatal(err)
	}
	defer grammar.Free()

	logits := make([]float32, len(tokenizer.Vocabulary().Values))
	for i := range logits {
		logits[i] = rand.Float32()
	}
	tokens := make([]token, len(logits))
	for i := range tokens {
		tokens[i].id = int32(i)
		tokens[i].value = logits[i]
	}

	grammar.Apply(tokens)
	nonInfCount := 0
	infCount := 0
	for _, tok := range tokens {
		if math.IsInf(float64(tok.value), -1) {
			infCount++
		} else {
			nonInfCount++
		}
	}
	if nonInfCount == 0 {
		t.Error("expected at least one non -inf token after grammar application, got none")
	}
	if infCount == 0 {
		t.Error("expected some -inf tokens after grammar application, got none")
	}
}

func TestSamplerRepeatPenalty(t *testing.T) {
	// Token 1 has the highest logit; if we penalize it, token 3 should win
	logits := []float32{-10, 5, -10, 4}
	sampler := NewSampler(0, 0, 0, 0, 0, 2.0, 0.0, 0.0, 64, nil)
	// Simulate having generated token 1 recently
	sampler.recentTokens = []int32{1}

	got, err := sampler.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	// With repeat_penalty=2.0, token 1's logit goes from 5 to 2.5
	// Token 3 has logit 4, so it should win with greedy (temp=0)
	if got != 3 {
		t.Errorf("expected token 3 to win after repeat penalty on token 1, got %d", got)
	}
}

func TestSamplerFrequencyPenalty(t *testing.T) {
	// Token 1 has the highest logit; frequency penalty proportional to count
	logits := []float32{-10, 5, -10, 4}
	sampler := NewSampler(0, 0, 0, 0, 0, 1.0, 2.0, 0.0, 64, nil)
	// Token 1 appeared twice
	sampler.recentTokens = []int32{1, 1}

	got, err := sampler.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	// frequency_penalty=2.0, count=2 → subtract 4.0 from token 1: 5-4=1
	// Token 3 has logit 4, should win
	if got != 3 {
		t.Errorf("expected token 3 to win after frequency penalty on token 1, got %d", got)
	}
}

func TestSamplerPresencePenalty(t *testing.T) {
	// Token 1 has the highest logit; flat presence penalty
	logits := []float32{-10, 5, -10, 4}
	sampler := NewSampler(0, 0, 0, 0, 0, 1.0, 0.0, 2.0, 64, nil)
	// Token 1 appeared once
	sampler.recentTokens = []int32{1}

	got, err := sampler.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	// presence_penalty=2.0 → subtract 2.0 from token 1: 5-2=3
	// Token 3 has logit 4, should win
	if got != 3 {
		t.Errorf("expected token 3 to win after presence penalty on token 1, got %d", got)
	}
}

func TestSamplerPenaltiesNeutralNoOp(t *testing.T) {
	logits := []float32{-10, 5, -10, 3}
	// Neutral penalties: repeat=1.0, freq=0, pres=0
	sampler := NewSampler(0, 0, 0, 0, 0, 1.0, 0.0, 0.0, 64, nil)
	sampler.recentTokens = []int32{1, 1, 1}

	got, err := sampler.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	// No penalties applied, token 1 (logit=5) should still win
	if got != 1 {
		t.Errorf("expected token 1 with neutral penalties, got %d", got)
	}
}

func TestRecordTokenRingBuffer(t *testing.T) {
	windowSize := 4
	sampler := NewSampler(0, 0, 0, 0, 0, 1.0, 0.0, 0.0, windowSize, nil)

	// Fill the window
	for i := range windowSize {
		sampler.recordToken(int32(i))
	}
	if len(sampler.recentTokens) != windowSize {
		t.Fatalf("expected len %d, got %d", windowSize, len(sampler.recentTokens))
	}

	// Record many more tokens — length must stay at windowSize
	for i := range 1000 {
		sampler.recordToken(int32(100 + i))
	}
	if len(sampler.recentTokens) != windowSize {
		t.Fatalf("expected len %d after 1000 extra tokens, got %d", windowSize, len(sampler.recentTokens))
	}
	if cap(sampler.recentTokens) != windowSize {
		t.Fatalf("expected cap %d (bounded), got %d", windowSize, cap(sampler.recentTokens))
	}

	// The window should contain the last 4 tokens: 1096, 1097, 1098, 1099
	counts := make(map[int32]bool)
	for _, id := range sampler.recentTokens {
		counts[id] = true
	}
	for i := 1096; i < 1100; i++ {
		if !counts[int32(i)] {
			t.Errorf("expected token %d in window, got %v", i, sampler.recentTokens)
		}
	}
}

func BenchmarkSample(b *testing.B) {
	samplers := map[string]Sampler{
		"Greedy":   NewSampler(0, 0, 0, 0, 0, 1.0, 0.0, 0.0, 0, nil), // Use NewSampler with temp=0 for greedy
		"Weighted": NewSampler(0.5, 10, 0.9, 0.2, -1, 1.0, 0.0, 0.0, 0, nil),
	}

	// Generate random logits for benchmarking
	logits := make([]float32, 1<<16)
	for i := range logits {
		logits[i] = rand.Float32()
	}

	for name, s := range samplers {
		b.Run(name, func(b *testing.B) {
			b.ResetTimer()
			for b.Loop() {
				if _, err := s.Sample(logits); err != nil {
					b.Fatalf("error sampling: %v", err)
				}
			}
		})
	}
}

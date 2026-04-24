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
	sampler := NewSampler(0, 0, 0, 0, 0, nil, 0, 0, 0, 0)
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
	sampler = NewSampler(0, 0, 0, 0, 0, nil, 0, 0, 0, 0)
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
	sampler = NewSampler(1.0, 0, 1e-10, 0, 0, nil, 0, 0, 0, 0)
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
	sampler = NewSampler(1, 0, 0.95, 0.05, 0, nil, 0, 0, 0, 0)
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

func TestRepeatPenaltyIntegration(t *testing.T) {
	// With greedy sampling (temp=0) and two equally-strong tokens,
	// repeat penalty should steer away from already-generated tokens.
	// Token 0 and token 1 have equal logits.
	logits := []float32{10.0, 10.0, -100.0, -100.0}

	// Without penalty: greedy always picks token 0 (first max)
	sampler := NewSampler(0, 0, 0, 0, 0, nil, 0, 0, 0, 0)
	got, _ := sampler.Sample(logits)
	if got != 0 {
		t.Fatalf("without penalty: want token 0, got %d", got)
	}

	// With penalty and token 0 in history: should pick token 1
	sampler = NewSampler(0, 0, 0, 0, 0, nil, 2.0, 64, 0, 0)
	// Manually seed history with token 0
	sampler.tokenHistory = []int32{0}
	got, _ = sampler.Sample(logits)
	if got != 1 {
		t.Fatalf("with penalty on token 0: want token 1, got %d", got)
	}

	// Verify token history accumulates across Sample calls
	sampler = NewSampler(0, 0, 0, 0, 0, nil, 1.5, 4, 0, 0)
	// Logits where token 0 is always dominant
	dominantLogits := []float32{10.0, 5.0, 5.0, 5.0}
	got, _ = sampler.Sample(dominantLogits)
	if got != 0 {
		t.Fatalf("first sample should pick dominant token 0, got %d", got)
	}
	if len(sampler.tokenHistory) != 1 || sampler.tokenHistory[0] != 0 {
		t.Fatalf("history should contain [0], got %v", sampler.tokenHistory)
	}

	// Verify repeatLastN caps the history
	sampler = NewSampler(0, 0, 0, 0, 0, nil, 1.5, 3, 0, 0)
	for i := range 5 {
		sampler.recordToken(int32(i))
	}
	if len(sampler.tokenHistory) != 3 {
		t.Fatalf("history should be capped at 3, got %d: %v", len(sampler.tokenHistory), sampler.tokenHistory)
	}
	// Should contain the last 3: [2, 3, 4]
	if sampler.tokenHistory[0] != 2 || sampler.tokenHistory[1] != 3 || sampler.tokenHistory[2] != 4 {
		t.Fatalf("history should be [2,3,4], got %v", sampler.tokenHistory)
	}
}

func BenchmarkSample(b *testing.B) {
	samplers := map[string]Sampler{
		"Greedy":   NewSampler(0, 0, 0, 0, 0, nil, 0, 0, 0, 0), // Use NewSampler with temp=0 for greedy
		"Weighted": NewSampler(0.5, 10, 0.9, 0.2, -1, nil, 0, 0, 0, 0),
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

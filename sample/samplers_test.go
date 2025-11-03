package sample

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/model"
)

func TestWeighted(t *testing.T) {
	logits := []float32{-10, 3, -10, -10}
	sampler := NewSampler(0, 0, 0, 0, 0, nil)
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
	sampler = NewSampler(0, 0, 0, 0, 0, nil)
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
	sampler = NewSampler(1.0, 0, 1e-10, 0, 0, nil)
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
	sampler = NewSampler(1, 0, 0.95, 0.05, 0, nil)
	got, err = sampler.Sample(logits)
	if err == nil {
		t.Errorf("expected error, got %d", got)
		return
	}
}

func modelHelper(t testing.TB) model.BytePairEncoding {
	t.Helper()

	f, err := os.Open(filepath.Join("..", "model", "testdata", "llama3.2", "encoder.json"))
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
	return model.NewBytePairEncoding(
		&model.Vocabulary{
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

func BenchmarkSample(b *testing.B) {
	samplers := map[string]Sampler{
		"Greedy":   NewSampler(0, 0, 0, 0, 0, nil), // Use NewSampler with temp=0 for greedy
		"Weighted": NewSampler(0.5, 10, 0.9, 0.2, -1, nil),
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

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
	sampler := NewSampler(0, 0, 0, 0, 0, nil, nil, 0, "", 0)
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
	sampler = NewSampler(0, 0, 0, 0, 0, nil, nil, 0, "", 0)
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
	sampler = NewSampler(1.0, 0, 1e-10, 0, 0, nil, nil, 0, "", 0)
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
	sampler = NewSampler(1, 0, 0.95, 0.05, 0, nil, nil, 0, "", 0)
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

// mockTokenizer is a minimal tokenizer for unit tests that maps token IDs to
// pre-defined string pieces and vice-versa.
type mockTokenizer struct {
	vocab []string
}

func (m *mockTokenizer) Encode(s string, addSpecial bool) ([]int32, error) {
	for i, t := range m.vocab {
		if t == s {
			return []int32{int32(i)}, nil
		}
	}
	return nil, nil
}

func (m *mockTokenizer) Decode(ids []int32) (string, error) {
	if len(ids) == 1 && int(ids[0]) < len(m.vocab) {
		return m.vocab[ids[0]], nil
	}
	return "", nil
}

func (m *mockTokenizer) Is(int32, tokenizer.Special) bool { return false }

func (m *mockTokenizer) Vocabulary() *tokenizer.Vocabulary {
	return &tokenizer.Vocabulary{Values: m.vocab}
}

// makeLogits returns a logit slice that forces the sampler to pick token id.
func makeLogits(vocabSize, id int) []float32 {
	logits := make([]float32, vocabSize)
	for i := range logits {
		logits[i] = -100
	}
	logits[id] = 100
	return logits
}

func TestRepeatLineDetection(t *testing.T) {
	// vocab: 0="the" 1=" cat" 2=" sat" 3=" on" 4=" mat" 5="." 6=" dog" 7=" ran"
	vocab := []string{"the", " cat", " sat", " on", " mat", ".", " dog", " ran"}
	tok := &mockTokenizer{vocab: vocab}

	const (
		tThe = 0
		tCat = 1
		tSat = 2
		tOn  = 3
		tMat = 4
		tDot = 5
		tDog = 6
		tRan = 7
	)

	// window=3: check last 3 segments; boost=0.5
	sampler := NewSampler(0.8, 0, 0, 0, 42, nil, tok, 3, ".", 0.5)

	feedToken := func(id int) {
		t.Helper()
		_, err := sampler.Sample(makeLogits(len(vocab), id))
		if err != nil {
			t.Fatalf("Sample error: %v", err)
		}
	}

	// Segment 1: "the cat sat." — new, no loop
	feedToken(tThe); feedToken(tCat); feedToken(tSat); feedToken(tDot)
	if sampler.loopActive {
		t.Error("after segment 1: expected loopActive=false, got true")
	}

	// Segment 2: "the dog ran." — different, no loop
	feedToken(tThe); feedToken(tDog); feedToken(tRan); feedToken(tDot)
	if sampler.loopActive {
		t.Error("after segment 2: expected loopActive=false, got true")
	}

	// Segment 3: "the cat sat." — matches segment 1 → loop!
	feedToken(tThe); feedToken(tCat); feedToken(tSat); feedToken(tDot)
	if !sampler.loopActive {
		t.Error("after segment 3: expected loopActive=true, got false")
	}

	// Segment 4: "the mat on." — no match in window (window slid, seg1 dropped) → no loop
	feedToken(tThe); feedToken(tMat); feedToken(tOn); feedToken(tDot)
	if sampler.loopActive {
		t.Error("after segment 4: expected loopActive=false, got true")
	}
}

func BenchmarkSample(b *testing.B) {
	samplers := map[string]Sampler{
		"Greedy":   NewSampler(0, 0, 0, 0, 0, nil, nil, 0, "", 0), // Use NewSampler with temp=0 for greedy
		"Weighted": NewSampler(0.5, 10, 0.9, 0.2, -1, nil, nil, 0, "", 0),
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

//go:build mlx

package sample

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestJSONPrefixValidation(t *testing.T) {
	cases := []struct {
		input string
		want  bool
	}{
		{"", true},
		{"{", true},
		{"{ ", true},
		{"{\"name\"", true},
		{"{\"name\":", true},
		{"{\"name\": \"John\"", true},
		{"{\"name\": \"John\", \"age\": 15}", true},
		{"{\"name\": \"John\", \"age\": 15", true},
		{"{\"name\": \"John\", \"age\": 15,", true},
		{"{\"name\": \"John\", \"age\": 15}}", false},
		{"[1, 2, 3]", true},
		{"[1, 2,", true},
	}

	for _, tc := range cases {
		got := ValidateJSONPrefix(tc.input)
		if got != tc.want {
			t.Errorf("ValidateJSONPrefix(%q) = %t, want %t", tc.input, got, tc.want)
		}
	}
}

func TestGrammarConstrainedSamplingLoop(t *testing.T) {
	skipIfNoMLX(t)

	// A small dummy vocabulary representing common JSON tokens
	vocab := []string{
		"{",       // 0
		"}",       // 1
		"\"",      // 2
		":",       // 3
		",",       // 4
		"name",    // 5
		"John",    // 6
		"age",     // 7
		"15",      // 8
		" ",       // 9
		"unknown", // 10
	}

	vocabBytes := make([][]byte, len(vocab))
	for i, s := range vocab {
		vocabBytes[i] = []byte(s)
	}

	// Schema: {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
	schemaStr := `{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}`
	opts := Options{
		Temperature: 0, // Greedy sampling
		Grammar:     "",
		Format:      json.RawMessage(schemaStr),
	}

	s := New(128)
	defer s.Free()

	// Register the slot with our dummy vocabulary and custom schema
	s.Add(0, opts, nil, vocab, vocabBytes, []int32{1}) // Let EOS be token ID 1 ("}")

	// We want to generate: {"name": "John"}
	// Expected token sequence:
	// 0: "{"
	// 2: "\""
	// 5: "name"
	// 2: "\""
	// 3: ":"
	// 9: " "
	// 2: "\""
	// 6: "John"
	// 2: "\""
	// 1: "}"
	expectedTokens := []int{0, 2, 5, 2, 3, 9, 2, 6, 2, 1}

	// At each step, we provide logits where the highest-scoring token is "unknown" (ID 10)
	// or some other invalid token. Our constraint should force the sampler to choose
	// the correct valid token instead!
	for step, expectedTokenID := range expectedTokens {
		// Logits has shape [1, V]
		logitsSlice := make([]float32, len(vocab))
		for i := range logitsSlice {
			logitsSlice[i] = 1.0 // default low score
		}
		// Set an invalid token ("unknown", ID 10) to have the absolute highest score!
		logitsSlice[10] = 100.0

		// Set the expected valid token to have a moderate score
		logitsSlice[expectedTokenID] = 10.0

		logitsTensor := mlx.FromValues(logitsSlice, 1, len(vocab))
		res := s.Sample([]int{0}, logitsTensor, func(ids []int32) string {
			var sb strings.Builder
			for _, id := range ids {
				sb.WriteString(vocab[int(id)])
			}
			return sb.String()
		})

		sampledToken := res.Token.Int()
		if sampledToken != expectedTokenID {
			t.Fatalf("Step %d: sampled token ID %d (%q), want %d (%q). (Prefix: %q)",
				step, sampledToken, vocab[sampledToken], expectedTokenID, vocab[expectedTokenID],
				func() string {
					var sb strings.Builder
					for _, id := range s.byID[0].generatedTokens {
						sb.WriteString(vocab[int(id)])
					}
					return sb.String()
				}())
		}
	}

	// Decode the final generated output
	var finalOutput strings.Builder
	for _, id := range s.byID[0].generatedTokens {
		finalOutput.WriteString(vocab[int(id)])
	}

	wantOutput := `{"name": "John"}`
	if finalOutput.String() != wantOutput {
		t.Errorf("Generated output %q, want %q", finalOutput.String(), wantOutput)
	}
}

func TestSchemaPrefixValidation(t *testing.T) {
	schemaStr := `{
		"type": "object",
		"properties": {
			"name": {"type": "string"},
			"age": {"type": "integer"}
		},
		"required": ["name", "age"]
	}`

	gc, err := NewGrammarConstraint([]byte(schemaStr), "")
	if err != nil {
		t.Fatalf("failed to create GrammarConstraint: %v", err)
	}

	cases := []struct {
		input string
		want  bool
	}{
		{"", true},
		{"{", true},
		{"{\"name\"", true},
		{"{\"nam", true},
		{"{\"name\":", true},
		{"{\"name\": \"John\"", true},
		{"{\"name\": 123", false},
		{"{\"name\": \"John\", \"age\": 15}", true},
		{"{\"name\": \"John\", \"age\": \"15\"", false},
		{"{\"name\": \"John\", \"unknown\": 123}", false},
	}

	for _, tc := range cases {
		got := gc.IsValidPrefix(tc.input)
		if got != tc.want {
			t.Errorf("IsValidPrefix(%q) = %t, want %t", tc.input, got, tc.want)
		}
	}

	completionCases := []struct {
		input string
		want  bool
	}{
		{"", false},
		{"{", false},
		{"{\"name\": \"John\"}", true},
		{"{\"name\": \"John\", \"age\": 15}", true},
		{"{\"name\": \"John\", \"age\": 15", false},
	}

	for _, tc := range completionCases {
		got := gc.IsComplete(tc.input)
		if got != tc.want {
			t.Errorf("IsComplete(%q) = %t, want %t", tc.input, got, tc.want)
		}
	}
}

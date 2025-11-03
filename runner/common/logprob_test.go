package common

import (
	"math"
	"reflect"
	"testing"

	"github.com/ollama/ollama/llm"
)

// mockTokenDecoder is a simple mock for testing
type mockTokenDecoder struct {
	tokens map[int]string
}

func (m *mockTokenDecoder) DecodeToken(tokenID int) string {
	if text, ok := m.tokens[tokenID]; ok {
		return text
	}
	return ""
}

func TestCalculateLogprobs(t *testing.T) {
	decoder := &mockTokenDecoder{
		tokens: map[int]string{
			0: "hello",
			1: "hi",
			2: "hey",
			3: "world",
		},
	}

	tests := []struct {
		name          string
		logits        []float32
		selectedToken int
		topK          int
		wantLen       int
		wantToken     string
	}{
		{
			name:          "Empty logits",
			logits:        []float32{},
			selectedToken: 0,
			topK:          0,
			wantLen:       0,
		},
		{
			name:          "Single token without top logprobs",
			logits:        []float32{1.0, 0.5, 0.3, 0.1},
			selectedToken: 0,
			topK:          0,
			wantLen:       1,
			wantToken:     "hello",
		},
		{
			name:          "Single token with top logprobs",
			logits:        []float32{1.0, 0.5, 0.3, 0.1},
			selectedToken: 0,
			topK:          3,
			wantLen:       1,
			wantToken:     "hello",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CalculateLogprobs(tt.logits, tt.selectedToken, tt.topK, decoder)
			if len(result) != tt.wantLen {
				t.Errorf("CalculateLogprobs() returned %d results, want %d", len(result), tt.wantLen)
			}
			if tt.wantLen > 0 && result[0].Token != tt.wantToken {
				t.Errorf("CalculateLogprobs() token = %s, want %s", result[0].Token, tt.wantToken)
			}
			if tt.topK > 0 && len(result) > 0 {
				if len(result[0].TopLogprobs) != tt.topK {
					t.Errorf("CalculateLogprobs() top logprobs count = %d, want %d", len(result[0].TopLogprobs), tt.topK)
				}
			}
		})
	}
}

func TestCalculateLogprobsNumericalStability(t *testing.T) {
	decoder := &mockTokenDecoder{
		tokens: map[int]string{
			0: "a",
			1: "b",
			2: "c",
		},
	}

	// Test with very large logits to ensure numerical stability
	logits := []float32{1000.0, 999.0, 998.0}
	result := CalculateLogprobs(logits, 0, 3, decoder)

	if len(result) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(result))
	}

	// Check that log probabilities are finite and reasonable
	if math.IsInf(result[0].Logprob, 0) || math.IsNaN(result[0].Logprob) {
		t.Errorf("Selected token logprob is not finite: %f", result[0].Logprob)
	}

	for i, tlp := range result[0].TopLogprobs {
		if math.IsInf(tlp.Logprob, 0) || math.IsNaN(tlp.Logprob) {
			t.Errorf("Top logprob[%d] is not finite: %f", i, tlp.Logprob)
		}
	}

	// Top logprobs should be in descending order
	for i := 1; i < len(result[0].TopLogprobs); i++ {
		if result[0].TopLogprobs[i].Logprob > result[0].TopLogprobs[i-1].Logprob {
			t.Errorf("Top logprobs not in descending order: %f > %f",
				result[0].TopLogprobs[i].Logprob, result[0].TopLogprobs[i-1].Logprob)
		}
	}
}

func TestToLLMLogprobs(t *testing.T) {
	tests := []struct {
		name     string
		input    []Logprob
		expected []llm.Logprob
	}{
		{
			name:     "Empty slice",
			input:    []Logprob{},
			expected: []llm.Logprob{},
		},
		{
			name: "Single logprob without top logprobs",
			input: []Logprob{
				{
					TokenLogprob: TokenLogprob{
						Token:   "hello",
						Logprob: -0.123,
					},
				},
			},
			expected: []llm.Logprob{
				{
					TokenLogprob: llm.TokenLogprob{
						Token:   "hello",
						Logprob: -0.123,
					},
				},
			},
		},
		{
			name: "Single logprob with top logprobs",
			input: []Logprob{
				{
					TokenLogprob: TokenLogprob{
						Token:   "hello",
						Logprob: -0.123,
					},
					TopLogprobs: []TokenLogprob{
						{Token: "hello", Logprob: -0.123},
						{Token: "hi", Logprob: -1.456},
						{Token: "hey", Logprob: -2.789},
					},
				},
			},
			expected: []llm.Logprob{
				{
					TokenLogprob: llm.TokenLogprob{
						Token:   "hello",
						Logprob: -0.123,
					},
					TopLogprobs: []llm.TokenLogprob{
						{Token: "hello", Logprob: -0.123},
						{Token: "hi", Logprob: -1.456},
						{Token: "hey", Logprob: -2.789},
					},
				},
			},
		},
		{
			name: "Multiple logprobs",
			input: []Logprob{
				{
					TokenLogprob: TokenLogprob{
						Token:   "Hello",
						Logprob: -0.1,
					},
					TopLogprobs: []TokenLogprob{
						{Token: "Hello", Logprob: -0.1},
						{Token: "Hi", Logprob: -1.2},
					},
				},
				{
					TokenLogprob: TokenLogprob{
						Token:   " world",
						Logprob: -0.2,
					},
					TopLogprobs: []TokenLogprob{
						{Token: " world", Logprob: -0.2},
						{Token: " there", Logprob: -2.3},
					},
				},
			},
			expected: []llm.Logprob{
				{
					TokenLogprob: llm.TokenLogprob{
						Token:   "Hello",
						Logprob: -0.1,
					},
					TopLogprobs: []llm.TokenLogprob{
						{Token: "Hello", Logprob: -0.1},
						{Token: "Hi", Logprob: -1.2},
					},
				},
				{
					TokenLogprob: llm.TokenLogprob{
						Token:   " world",
						Logprob: -0.2,
					},
					TopLogprobs: []llm.TokenLogprob{
						{Token: " world", Logprob: -0.2},
						{Token: " there", Logprob: -2.3},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ToLLMLogprobs(tt.input)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("ToLLMLogprobs() = %+v, want %+v", result, tt.expected)
			}
		})
	}
}

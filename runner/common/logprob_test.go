package common

import (
	"math"
	"testing"

	"github.com/ollama/ollama/llm"
)

func TestCalculateLogprobs(t *testing.T) {
	tokens := map[int]string{
		0: "hello",
		1: "hi",
		2: "hey",
		3: "world",
	}
	decoder := func(tokenID int) string {
		if text, ok := tokens[tokenID]; ok {
			return text
		}
		return ""
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
	tokens := map[int]string{
		0: "a",
		1: "b",
		2: "c",
	}
	decoder := func(tokenID int) string {
		if text, ok := tokens[tokenID]; ok {
			return text
		}
		return ""
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

func TestCalculateLogprobsProbabilityCorrectness(t *testing.T) {
	tokens := map[int]string{
		0: "hello",
		1: "world",
		2: "foo",
		3: "bar",
	}
	decoder := func(tokenID int) string {
		if text, ok := tokens[tokenID]; ok {
			return text
		}
		return ""
	}

	tests := []struct {
		name          string
		logits        []float32
		selectedToken int
		topK          int
	}{
		{
			name:          "Uniform logits",
			logits:        []float32{1.0, 1.0, 1.0, 1.0},
			selectedToken: 0,
			topK:          4,
		},
		{
			name:          "Different logits",
			logits:        []float32{2.0, 1.0, 0.5, 0.1},
			selectedToken: 0,
			topK:          4,
		},
		{
			name:          "Negative logits",
			logits:        []float32{-1.0, -2.0, -3.0, -4.0},
			selectedToken: 0,
			topK:          4,
		},
		{
			name:          "Mixed logits",
			logits:        []float32{5.0, -5.0, 0.0, 2.5},
			selectedToken: 0,
			topK:          4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CalculateLogprobs(tt.logits, tt.selectedToken, tt.topK, decoder)

			if len(result) != 1 {
				t.Fatalf("Expected 1 result, got %d", len(result))
			}

			// Verify all probabilities are non-positive (log probabilities should be <= 0)
			if result[0].Logprob > 0 {
				t.Errorf("Selected token logprob should be <= 0, got %f", result[0].Logprob)
			}

			for i, tlp := range result[0].TopLogprobs {
				if tlp.Logprob > 0 {
					t.Errorf("Top logprob[%d] should be <= 0, got %f", i, tlp.Logprob)
				}
			}

			// Verify that probabilities sum to approximately 1
			// Sum of exp(logprob) for all tokens should equal 1
			var probSum float64
			for _, lp := range result[0].TopLogprobs {
				probSum += math.Exp(lp.Logprob)
			}

			// For uniform logits, each probability should be 1/n
			if tt.name == "Uniform logits" {
				expectedProb := 1.0 / float64(len(tt.logits))
				actualProb := math.Exp(result[0].Logprob)
				if math.Abs(actualProb-expectedProb) > 1e-6 {
					t.Errorf("For uniform logits, expected probability %f, got %f",
						expectedProb, actualProb)
				}
			}

			// Verify top logprobs are sorted in descending order
			for i := 1; i < len(result[0].TopLogprobs); i++ {
				if result[0].TopLogprobs[i].Logprob > result[0].TopLogprobs[i-1].Logprob {
					t.Errorf("Top logprobs not sorted: position %d (%f) > position %d (%f)",
						i, result[0].TopLogprobs[i].Logprob,
						i-1, result[0].TopLogprobs[i-1].Logprob)
				}
			}

			// Verify the selected token appears in top logprobs
			selectedText := decoder(tt.selectedToken)
			found := false
			for _, tlp := range result[0].TopLogprobs {
				if tlp.Token == selectedText {
					found = true
					// The logprob in top logprobs should match the selected token's logprob
					if math.Abs(tlp.Logprob-result[0].Logprob) > 1e-6 {
						t.Errorf("Selected token logprob mismatch: main=%f, in top=%f",
							result[0].Logprob, tlp.Logprob)
					}
					break
				}
			}
			if !found {
				t.Errorf("Selected token %q not found in top logprobs", selectedText)
			}
		})
	}
}

func TestCalculateLogprobsSoftmaxCorrectness(t *testing.T) {
	// Test that softmax calculation is correct by verifying probabilities sum to 1
	decoder := func(tokenID int) string {
		return string(rune('A' + tokenID))
	}

	tests := []struct {
		name   string
		logits []float32
	}{
		{
			name:   "Small vocabulary",
			logits: []float32{1.0, 2.0, 3.0},
		},
		{
			name:   "Large differences",
			logits: []float32{10.0, 0.0, -10.0},
		},
		{
			name:   "All equal",
			logits: []float32{5.0, 5.0, 5.0, 5.0, 5.0},
		},
		{
			name:   "Very large values",
			logits: []float32{500.0, 499.0, 498.0},
		},
		{
			name:   "Very small values",
			logits: []float32{-500.0, -499.0, -498.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Calculate logprobs for all tokens
			var totalProb float64
			for i := range tt.logits {
				result := CalculateLogprobs(tt.logits, i, 0, decoder)
				if len(result) != 1 {
					t.Fatalf("Expected 1 result, got %d", len(result))
				}
				prob := math.Exp(result[0].Logprob)
				totalProb += prob

				// Verify each probability is between 0 and 1
				if prob < 0 || prob > 1 {
					t.Errorf("Token %d probability %f is out of range [0, 1]", i, prob)
				}
			}

			// Total probability should be very close to 1.0 (allowing for floating point errors)
			if math.Abs(totalProb-1.0) > 1e-5 {
				t.Errorf("Total probability sum is %f, expected 1.0", totalProb)
			}
		})
	}
}

func TestCalculateLogprobsSelectedTokenCorrectness(t *testing.T) {
	decoder := func(tokenID int) string {
		return string(rune('A' + tokenID))
	}

	logits := []float32{3.0, 1.0, 2.0, 0.5}

	// Test that selecting different tokens gives the correct probabilities
	// and that the highest logit has the highest probability
	maxLogitIndex := 0
	maxLogitValue := logits[0]
	for i, logit := range logits[1:] {
		if logit > maxLogitValue {
			maxLogitValue = logit
			maxLogitIndex = i + 1
		}
	}

	var maxProb float64
	var maxProbIndex int

	for i := range logits {
		result := CalculateLogprobs(logits, i, 0, decoder)
		prob := math.Exp(result[0].Logprob)

		if prob > maxProb {
			maxProb = prob
			maxProbIndex = i
		}

		// Verify the token matches
		expectedToken := decoder(i)
		if result[0].Token != expectedToken {
			t.Errorf("Token %d: expected token %q, got %q", i, expectedToken, result[0].Token)
		}
	}

	// The token with the highest logit should have the highest probability
	if maxProbIndex != maxLogitIndex {
		t.Errorf("Token with highest probability (%d) doesn't match token with highest logit (%d)",
			maxProbIndex, maxLogitIndex)
	}
}

func TestCalculateLogprobsTopKOrdering(t *testing.T) {
	tokens := map[int]string{
		0: "first",
		1: "second",
		2: "third",
		3: "fourth",
		4: "fifth",
	}
	decoder := func(tokenID int) string {
		return tokens[tokenID]
	}

	// Logits in non-sorted order
	logits := []float32{2.0, 5.0, 1.0, 4.0, 3.0}
	// Expected order by probability: 1 (5.0), 3 (4.0), 4 (3.0), 0 (2.0), 2 (1.0)
	expectedOrder := []string{"second", "fourth", "fifth", "first", "third"}

	result := CalculateLogprobs(logits, 0, 5, decoder)

	if len(result) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(result))
	}

	if len(result[0].TopLogprobs) != 5 {
		t.Fatalf("Expected 5 top logprobs, got %d", len(result[0].TopLogprobs))
	}

	// Verify ordering matches expected
	for i, tlp := range result[0].TopLogprobs {
		if tlp.Token != expectedOrder[i] {
			t.Errorf("Position %d: expected token %q, got %q", i, expectedOrder[i], tlp.Token)
		}
	}

	// Verify probabilities are in descending order
	for i := 1; i < len(result[0].TopLogprobs); i++ {
		if result[0].TopLogprobs[i].Logprob > result[0].TopLogprobs[i-1].Logprob {
			t.Errorf("Probabilities not in descending order at position %d: %f > %f",
				i, result[0].TopLogprobs[i].Logprob, result[0].TopLogprobs[i-1].Logprob)
		}
	}
}

func TestLogprobsWithStopSequences(t *testing.T) {
	tests := []struct {
		name              string
		pendingResponses  []string
		pendingLogprobs   []llm.Logprob
		stop              string
		expectedResponses []string
		expectedLogprobs  int
	}{
		{
			name:             "Single token stop",
			pendingResponses: []string{"Hello", " world", "!"},
			pendingLogprobs: []llm.Logprob{
				{TokenLogprob: llm.TokenLogprob{Token: "Hello", Logprob: -0.1}},
				{TokenLogprob: llm.TokenLogprob{Token: " world", Logprob: -0.2}},
				{TokenLogprob: llm.TokenLogprob{Token: "!", Logprob: -0.3}},
			},
			stop:              "!",
			expectedResponses: []string{"Hello", " world"},
			expectedLogprobs:  2,
		},
		{
			name:             "Multi-token stop sequence",
			pendingResponses: []string{"Hello", " ", "there", "STOP"},
			pendingLogprobs: []llm.Logprob{
				{TokenLogprob: llm.TokenLogprob{Token: "Hello", Logprob: -0.1}},
				{TokenLogprob: llm.TokenLogprob{Token: " ", Logprob: -0.2}},
				{TokenLogprob: llm.TokenLogprob{Token: "there", Logprob: -0.3}},
				{TokenLogprob: llm.TokenLogprob{Token: "STOP", Logprob: -0.4}},
			},
			stop:              "STOP",
			expectedResponses: []string{"Hello", " ", "there"},
			expectedLogprobs:  3,
		},
		{
			name:             "Partial token stop",
			pendingResponses: []string{"Hello", " the", "re!"},
			pendingLogprobs: []llm.Logprob{
				{TokenLogprob: llm.TokenLogprob{Token: "Hello", Logprob: -0.1}},
				{TokenLogprob: llm.TokenLogprob{Token: " the", Logprob: -0.2}},
				{TokenLogprob: llm.TokenLogprob{Token: "re!", Logprob: -0.3}},
			},
			stop:              "there!",
			expectedResponses: []string{"Hello", " "},
			expectedLogprobs:  2,
		},
		{
			name:             "Stop at beginning of last token",
			pendingResponses: []string{"Hello", " world", "END"},
			pendingLogprobs: []llm.Logprob{
				{TokenLogprob: llm.TokenLogprob{Token: "Hello", Logprob: -0.1}},
				{TokenLogprob: llm.TokenLogprob{Token: " world", Logprob: -0.2}},
				{TokenLogprob: llm.TokenLogprob{Token: "END", Logprob: -0.3}},
			},
			stop:              "END",
			expectedResponses: []string{"Hello", " world"},
			expectedLogprobs:  2,
		},
		{
			name:             "Multi-token stop across tokens",
			pendingResponses: []string{"Text", " ", "with", " ", "stop", " ", "word"},
			pendingLogprobs: []llm.Logprob{
				{TokenLogprob: llm.TokenLogprob{Token: "Text", Logprob: -0.1}},
				{TokenLogprob: llm.TokenLogprob{Token: " ", Logprob: -0.2}},
				{TokenLogprob: llm.TokenLogprob{Token: "with", Logprob: -0.3}},
				{TokenLogprob: llm.TokenLogprob{Token: " ", Logprob: -0.4}},
				{TokenLogprob: llm.TokenLogprob{Token: "stop", Logprob: -0.5}},
				{TokenLogprob: llm.TokenLogprob{Token: " ", Logprob: -0.6}},
				{TokenLogprob: llm.TokenLogprob{Token: "word", Logprob: -0.7}},
			},
			stop:              "stop word",
			expectedResponses: []string{"Text", " ", "with", " "},
			expectedLogprobs:  4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Simulate the stop sequence detection and truncation
			origLen := len(tt.pendingResponses)
			responses, tokenTruncated := TruncateStop(tt.pendingResponses, tt.stop)
			newLen := len(responses)

			// Simulate logprobs truncation
			logprobs := make([]llm.Logprob, len(tt.pendingLogprobs))
			copy(logprobs, tt.pendingLogprobs)

			origLogprobsLen := len(logprobs)
			numTokensRemoved := origLen - newLen
			newLogprobsLen := origLogprobsLen - numTokensRemoved
			if newLogprobsLen < 0 {
				newLogprobsLen = 0
			}
			logprobs = logprobs[:newLogprobsLen]

			// Verify responses were truncated correctly
			if len(responses) != len(tt.expectedResponses) {
				t.Errorf("Expected %d responses, got %d", len(tt.expectedResponses), len(responses))
			}

			// Verify logprobs count matches truncated responses
			if len(logprobs) != tt.expectedLogprobs {
				t.Errorf("Expected %d logprobs after truncation, got %d", tt.expectedLogprobs, len(logprobs))
			}

			// Verify logprobs count matches response count
			if len(logprobs) != len(responses) {
				t.Errorf("Logprobs count (%d) doesn't match responses count (%d)", len(logprobs), len(responses))
			}

			// Verify the correct logprobs were kept (skip last token if it was truncated)
			// When tokenTruncated is true, the last response token may not match the logprob token
			checkLen := len(logprobs)
			if tokenTruncated && checkLen > 0 {
				checkLen-- // Skip checking the last token when it was partially truncated
			}

			for i := range checkLen {
				if i < len(responses) && logprobs[i].Token != responses[i] {
					t.Errorf("Logprob[%d] token %q doesn't match response[%d] %q",
						i, logprobs[i].Token, i, responses[i])
				}
			}
		})
	}
}

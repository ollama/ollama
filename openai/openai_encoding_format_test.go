package openai

import (
	"encoding/base64"
	"math"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestToEmbeddingList(t *testing.T) {
	testCases := []struct {
		name        string
		embeddings  [][]float32
		format      string
		expectType  string // "float" or "base64"
		expectCount int
		promptEval  int
	}{
		{"float format", [][]float32{{0.1, -0.2, 0.3}}, "float", "float", 1, 10},
		{"base64 format", [][]float32{{0.1, -0.2, 0.3}}, "base64", "base64", 1, 5},
		{"default to float", [][]float32{{0.1, -0.2, 0.3}}, "", "float", 1, 0},
		{"invalid defaults to float", [][]float32{{0.1, -0.2, 0.3}}, "invalid", "float", 1, 0},
		{"multiple embeddings", [][]float32{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}, "base64", "base64", 3, 0},
		{"empty embeddings", nil, "float", "", 0, 0},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			resp := api.EmbedResponse{
				Embeddings:      tc.embeddings,
				PromptEvalCount: tc.promptEval,
			}

			result := ToEmbeddingList("test-model", resp, tc.format)

			if tc.expectCount == 0 {
				if len(result.Data) != 0 {
					t.Errorf("expected 0 embeddings, got %d", len(result.Data))
				}
				return
			}

			if len(result.Data) != tc.expectCount {
				t.Fatalf("expected %d embeddings, got %d", tc.expectCount, len(result.Data))
			}

			if result.Model != "test-model" {
				t.Errorf("expected model 'test-model', got %q", result.Model)
			}

			// Check type of first embedding
			switch tc.expectType {
			case "float":
				if _, ok := result.Data[0].Embedding.([]float32); !ok {
					t.Errorf("expected []float32, got %T", result.Data[0].Embedding)
				}
			case "base64":
				embStr, ok := result.Data[0].Embedding.(string)
				if !ok {
					t.Errorf("expected string, got %T", result.Data[0].Embedding)
				} else if _, err := base64.StdEncoding.DecodeString(embStr); err != nil {
					t.Errorf("invalid base64: %v", err)
				}
			}

			// Check indices
			for i := range result.Data {
				if result.Data[i].Index != i {
					t.Errorf("embedding %d: expected index %d, got %d", i, i, result.Data[i].Index)
				}
			}

			if tc.promptEval > 0 && result.Usage.PromptTokens != tc.promptEval {
				t.Errorf("expected %d prompt tokens, got %d", tc.promptEval, result.Usage.PromptTokens)
			}
		})
	}
}

func TestFloatsToBase64(t *testing.T) {
	floats := []float32{0.1, -0.2, 0.3, -0.4, 0.5}

	result := floatsToBase64(floats)

	// Verify it's valid base64
	decoded, err := base64.StdEncoding.DecodeString(result)
	if err != nil {
		t.Fatalf("failed to decode base64: %v", err)
	}

	// Check length
	expectedBytes := len(floats) * 4
	if len(decoded) != expectedBytes {
		t.Errorf("expected %d bytes, got %d", expectedBytes, len(decoded))
	}

	// Decode and verify values
	for i, expected := range floats {
		offset := i * 4
		bits := uint32(decoded[offset]) |
			uint32(decoded[offset+1])<<8 |
			uint32(decoded[offset+2])<<16 |
			uint32(decoded[offset+3])<<24
		decodedFloat := math.Float32frombits(bits)

		if math.Abs(float64(decodedFloat-expected)) > 1e-6 {
			t.Errorf("float[%d]: expected %f, got %f", i, expected, decodedFloat)
		}
	}
}

func TestFloatsToBase64_EmptySlice(t *testing.T) {
	result := floatsToBase64([]float32{})

	// Should return valid base64 for empty slice
	decoded, err := base64.StdEncoding.DecodeString(result)
	if err != nil {
		t.Fatalf("failed to decode base64: %v", err)
	}

	if len(decoded) != 0 {
		t.Errorf("expected 0 bytes, got %d", len(decoded))
	}
}

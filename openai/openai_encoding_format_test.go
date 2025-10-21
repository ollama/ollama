package openai

import (
	"encoding/base64"
	"math"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestToEmbeddingList_FloatFormat(t *testing.T) {
	embeddings := [][]float32{
		{0.1, -0.2, 0.3},
		{0.4, -0.5, 0.6},
	}

	resp := api.EmbedResponse{
		Embeddings:      embeddings,
		PromptEvalCount: 10,
	}

	result := ToEmbeddingList("test-model", resp, "float")

	if result.Object != "list" {
		t.Errorf("expected object 'list', got %q", result.Object)
	}

	if result.Model != "test-model" {
		t.Errorf("expected model 'test-model', got %q", result.Model)
	}

	if len(result.Data) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(result.Data))
	}

	// Check first embedding
	if result.Data[0].Object != "embedding" {
		t.Errorf("expected object 'embedding', got %q", result.Data[0].Object)
	}

	if result.Data[0].Index != 0 {
		t.Errorf("expected index 0, got %d", result.Data[0].Index)
	}

	// Embedding should be []float32
	embeddingSlice, ok := result.Data[0].Embedding.([]float32)
	if !ok {
		t.Fatalf("expected embedding to be []float32, got %T", result.Data[0].Embedding)
	}

	if len(embeddingSlice) != 3 {
		t.Errorf("expected 3 floats, got %d", len(embeddingSlice))
	}

	// Check values
	expected := []float32{0.1, -0.2, 0.3}
	for i, exp := range expected {
		if embeddingSlice[i] != exp {
			t.Errorf("embedding[%d]: expected %f, got %f", i, exp, embeddingSlice[i])
		}
	}

	// Check usage
	if result.Usage.PromptTokens != 10 {
		t.Errorf("expected 10 prompt tokens, got %d", result.Usage.PromptTokens)
	}
}

func TestToEmbeddingList_Base64Format(t *testing.T) {
	embeddings := [][]float32{
		{0.1, -0.2, 0.3},
	}

	resp := api.EmbedResponse{
		Embeddings:      embeddings,
		PromptEvalCount: 5,
	}

	result := ToEmbeddingList("test-model", resp, "base64")

	if len(result.Data) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(result.Data))
	}

	// Embedding should be string
	embeddingStr, ok := result.Data[0].Embedding.(string)
	if !ok {
		t.Fatalf("expected embedding to be string, got %T", result.Data[0].Embedding)
	}

	// Verify it's valid base64
	decoded, err := base64.StdEncoding.DecodeString(embeddingStr)
	if err != nil {
		t.Fatalf("failed to decode base64: %v", err)
	}

	// Should be 3 floats * 4 bytes = 12 bytes
	expectedBytes := len(embeddings[0]) * 4
	if len(decoded) != expectedBytes {
		t.Errorf("expected %d bytes, got %d", expectedBytes, len(decoded))
	}

	// Decode back to floats and verify
	for i := 0; i < len(embeddings[0]); i++ {
		offset := i * 4
		bits := uint32(decoded[offset]) |
			uint32(decoded[offset+1])<<8 |
			uint32(decoded[offset+2])<<16 |
			uint32(decoded[offset+3])<<24
		decodedFloat := math.Float32frombits(bits)

		if math.Abs(float64(decodedFloat-embeddings[0][i])) > 1e-6 {
			t.Errorf("float[%d]: expected %f, got %f", i, embeddings[0][i], decodedFloat)
		}
	}
}

func TestToEmbeddingList_DefaultFormat(t *testing.T) {
	embeddings := [][]float32{
		{0.1, -0.2, 0.3},
	}

	resp := api.EmbedResponse{
		Embeddings: embeddings,
	}

	// Empty string should default to float
	result := ToEmbeddingList("test-model", resp, "")

	if len(result.Data) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(result.Data))
	}

	// Should default to float format ([]float32)
	_, ok := result.Data[0].Embedding.([]float32)
	if !ok {
		t.Errorf("expected default format to be []float32, got %T", result.Data[0].Embedding)
	}
}

func TestToEmbeddingList_InvalidFormat(t *testing.T) {
	embeddings := [][]float32{
		{0.1, -0.2, 0.3},
	}

	resp := api.EmbedResponse{
		Embeddings: embeddings,
	}

	// Invalid format should default to float
	result := ToEmbeddingList("test-model", resp, "invalid")

	if len(result.Data) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(result.Data))
	}

	// Should fallback to float format
	_, ok := result.Data[0].Embedding.([]float32)
	if !ok {
		t.Errorf("expected invalid format to fallback to []float32, got %T", result.Data[0].Embedding)
	}
}

func TestToEmbeddingList_EmptyEmbeddings(t *testing.T) {
	resp := api.EmbedResponse{
		Embeddings: nil,
	}

	result := ToEmbeddingList("test-model", resp, "float")

	// Should return empty list
	if result.Object != "" {
		t.Errorf("expected empty object, got %q", result.Object)
	}

	if len(result.Data) != 0 {
		t.Errorf("expected 0 embeddings, got %d", len(result.Data))
	}
}

func TestToEmbeddingList_MultipleEmbeddings(t *testing.T) {
	embeddings := [][]float32{
		{0.1, 0.2},
		{0.3, 0.4},
		{0.5, 0.6},
	}

	resp := api.EmbedResponse{
		Embeddings: embeddings,
	}

	// Test with base64
	result := ToEmbeddingList("test-model", resp, "base64")

	if len(result.Data) != 3 {
		t.Fatalf("expected 3 embeddings, got %d", len(result.Data))
	}

	// Check indices are correct
	for i := 0; i < 3; i++ {
		if result.Data[i].Index != i {
			t.Errorf("embedding %d: expected index %d, got %d", i, i, result.Data[i].Index)
		}

		// All should be base64 strings
		if _, ok := result.Data[i].Embedding.(string); !ok {
			t.Errorf("embedding %d: expected string, got %T", i, result.Data[i].Embedding)
		}
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

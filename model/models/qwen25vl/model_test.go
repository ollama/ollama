package qwen25vl

import (
	"testing"

	"github.com/ollama/ollama/ml/backend/ggml"
	"github.com/ollama/ollama/model/input"
)

func TestPostTokenize(t *testing.T) {
	// Set up test inputs
	model := &Model{}
	mockHash := uint64(12345678)

	inputs := []input.Input{
		{Token: 123}, // Regular token
		{Token: 456}, // Regular token
		{Token: 151655, Multimodal: &ggml.Tensor{}, MultimodalHash: mockHash}, // Image token
		{Token: 789}, // Regular token
	}

	// Run the function being tested
	result, err := model.PostTokenize(inputs)
	if err != nil {
		t.Fatalf("PostTokenize returned error: %v", err)
	}

	// Verify the actual length first
	expectedLength := 21
	if len(result) != expectedLength {
		t.Fatalf("Result has wrong length: got %d, expected %d", len(result), expectedLength)
	}

	// Check key positions only
	checkPositions := map[int]int32{
		0:  123,    // First regular token
		1:  456,    // Second regular token
		2:  151652, // Vision start token
		4:  151655, // First placeholder token
		19: 151653, // Vision end token
		20: 789,    // Final regular token
	}

	for pos, expectedToken := range checkPositions {
		if pos >= len(result) {
			t.Errorf("Position %d is out of bounds (result length: %d)", pos, len(result))
			continue
		}
		if result[pos].Token != expectedToken {
			t.Errorf("Position %d: expected token %d, got %d", pos, expectedToken, result[pos].Token)
		}
	}

	// Check multimodal data is preserved
	if result[3].MultimodalHash != mockHash {
		t.Errorf("Multimodal hash not preserved: got %d, expected %d",
			result[3].MultimodalHash, mockHash)
	}
}

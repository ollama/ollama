package server

import (
	"encoding/json"
	"testing"

	"github.com/ollama/ollama/llm"
)

// TestLogprobBytesJSONRoundTrip verifies that partial UTF-8 bytes are preserved
// through JSON marshaling and unmarshaling. This tests the fix for issue #13497.
func TestLogprobBytesJSONRoundTrip(t *testing.T) {
	// Create a logprob with partial UTF-8 bytes (first byte of emoji 😊)
	original := llm.Logprob{
		TokenLogprob: llm.TokenLogprob{
			Token:   "\xF0", // Invalid UTF-8 (partial sequence)
			Logprob: -0.5,
			Bytes:   []byte{0xF0}, // Raw bytes stored before JSON encoding
		},
		TopLogprobs: []llm.TokenLogprob{
			{
				Token:   "\x9F",
				Logprob: -1.0,
				Bytes:   []byte{0x9F},
			},
		},
	}

	// Marshal to JSON
	jsonData, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("Failed to marshal: %v", err)
	}

	// Unmarshal from JSON
	var decoded llm.Logprob
	if err := json.Unmarshal(jsonData, &decoded); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	// The Token field may have been corrupted by JSON (replaced with U+FFFD),
	// but the Bytes field should preserve the original bytes
	if len(decoded.Bytes) != 1 || decoded.Bytes[0] != 0xF0 {
		t.Errorf("Bytes field corrupted: got %v, want [240]", decoded.Bytes)
	}

	// Verify TopLogprobs bytes are also preserved
	if len(decoded.TopLogprobs) != 1 {
		t.Fatalf("TopLogprobs length = %d, want 1", len(decoded.TopLogprobs))
	}
	if len(decoded.TopLogprobs[0].Bytes) != 1 || decoded.TopLogprobs[0].Bytes[0] != 0x9F {
		t.Errorf("TopLogprobs[0].Bytes corrupted: got %v, want [159]", decoded.TopLogprobs[0].Bytes)
	}
}

// TestToAPILogprobsPreservesBytes verifies that toAPILogprobs uses the stored
// bytes instead of converting from the (potentially corrupted) token string.
func TestToAPILogprobsPreservesBytes(t *testing.T) {
	// Simulate logprobs that have been through JSON round-trip
	// The Token field contains the replacement character (corrupted)
	// but the Bytes field contains the correct original bytes
	logprobs := []llm.Logprob{
		{
			TokenLogprob: llm.TokenLogprob{
				Token:   "\uFFFD", // Replacement character (corrupted)
				Logprob: -0.5,
				Bytes:   []byte{0xF0}, // Original bytes preserved
			},
			TopLogprobs: []llm.TokenLogprob{
				{
					Token:   "\uFFFD",
					Logprob: -1.0,
					Bytes:   []byte{0x9F},
				},
			},
		},
	}

	// Convert to API logprobs
	apiLogprobs := toAPILogprobs(logprobs)

	if len(apiLogprobs) != 1 {
		t.Fatalf("Expected 1 API logprob, got %d", len(apiLogprobs))
	}

	// Verify that the Bytes field contains the correct bytes, not the
	// replacement character bytes [239, 191, 189]
	expectedBytes := []int{240} // 0xF0
	if len(apiLogprobs[0].Bytes) != len(expectedBytes) {
		t.Errorf("Bytes length = %d, want %d", len(apiLogprobs[0].Bytes), len(expectedBytes))
	}
	for i, b := range apiLogprobs[0].Bytes {
		if b != expectedBytes[i] {
			t.Errorf("Bytes[%d] = %d, want %d", i, b, expectedBytes[i])
		}
	}

	// Verify TopLogprobs bytes
	if len(apiLogprobs[0].TopLogprobs) != 1 {
		t.Fatalf("Expected 1 TopLogprob, got %d", len(apiLogprobs[0].TopLogprobs))
	}
	expectedTopBytes := []int{159} // 0x9F
	if len(apiLogprobs[0].TopLogprobs[0].Bytes) != len(expectedTopBytes) {
		t.Errorf("TopLogprobs[0].Bytes length = %d, want %d",
			len(apiLogprobs[0].TopLogprobs[0].Bytes), len(expectedTopBytes))
	}
	for i, b := range apiLogprobs[0].TopLogprobs[0].Bytes {
		if b != expectedTopBytes[i] {
			t.Errorf("TopLogprobs[0].Bytes[%d] = %d, want %d", i, b, expectedTopBytes[i])
		}
	}

	// Ensure we're NOT getting replacement character bytes
	replacementBytes := []int{239, 191, 189}
	if len(apiLogprobs[0].Bytes) == len(replacementBytes) {
		allMatch := true
		for i := range apiLogprobs[0].Bytes {
			if apiLogprobs[0].Bytes[i] != replacementBytes[i] {
				allMatch = false
				break
			}
		}
		if allMatch {
			t.Errorf("Bytes field incorrectly contains replacement character bytes")
		}
	}
}

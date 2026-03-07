package llm

import (
	"encoding/json"
	"testing"
)

func TestTokenLogprobJSONRoundTrip(t *testing.T) {
	original := TokenLogprob{
		Token:   string([]byte{0xF0}),
		Logprob: -0.5,
		Bytes:   []byte{0xF0},
	}

	data, err := json.Marshal(original)
	if err != nil {
		t.Fatal(err)
	}

	var decoded TokenLogprob
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}

	// Token gets corrupted to U+FFFD by json.Marshal
	if decoded.Token == string([]byte{0xF0}) {
		t.Error("expected Token to be corrupted by JSON marshaling, but it survived")
	}

	// Bytes field must survive the round-trip (encoded as base64)
	if len(decoded.Bytes) != 1 || decoded.Bytes[0] != 0xF0 {
		t.Errorf("bytes corrupted by JSON round-trip: got %v, want [0xF0]", decoded.Bytes)
	}
}

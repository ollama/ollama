package tokenizer

import (
	"slices"
	"testing"
)

func TestSpecialTokenIDs(t *testing.T) {
	data := []byte(`{
		"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "<eos>": 2, "<pad>": 3}, "merges": []},
		"added_tokens": [
			{"id": 2, "content": "<eos>", "special": true},
			{"id": 3, "content": "<pad>", "special": true}
		]
	}`)
	tok, err := LoadFromBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	ids := tok.SpecialTokenIDs()
	slices.Sort(ids)
	if !slices.Equal(ids, []int32{2, 3}) {
		t.Fatalf("SpecialTokenIDs() = %v, want [2 3]", ids)
	}
}

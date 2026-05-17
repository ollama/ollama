package tokenizer

import (
	"strings"
	"testing"
)

func TestLoadFromBytesRejectsWordPiece(t *testing.T) {
	data := []byte(`{
		"model": {
			"type": "WordPiece",
			"vocab": {"[UNK]": 0, "hello": 1}
		},
		"added_tokens": []
	}`)

	_, err := LoadFromBytes(data)
	if err == nil {
		t.Fatal("expected WordPiece load to fail")
	}
	if !strings.Contains(err.Error(), "unsupported tokenizer type: WordPiece") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestExtractPretokenizerSkipsUnsupportedSequenceSplit(t *testing.T) {
	data := []byte(`{
		"type": "Sequence",
		"pretokenizers": [
			{
				"type": "Split",
				"pattern": {
					"Regex": "(?:\\r?\\n)+(?!\\r?\\n)"
				}
			},
			{
				"type": "Split",
				"pattern": {
					"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
				}
			}
		]
	}`)

	pattern := extractPretokenizer(data)
	if pattern == "" {
		t.Fatal("expected supported Split pretokenizer")
	}
	if strings.Contains(pattern, `(?!\r?\n)`) {
		t.Fatalf("selected unsupported newline splitter: %q", pattern)
	}
}

package tokenizer

import (
	"encoding/json"
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

func TestLoadPretokenizerOptionalPunctuationSpace(t *testing.T) {
	tests := []struct {
		name    string
		pattern string
		want    []string
	}{
		{
			name:    "o200k optional space",
			pattern: ` ?[^\s\p{L}\p{N}]+[\r\n/]*|\s+(?!\S)|\s+`,
			want:    []string{"   ", " }\n"},
		},
		{
			name:    "punctuation without optional space",
			pattern: `[^\s\p{L}\p{N}]+[\r\n/]*|\s+(?!\S)|\s+`,
			want:    []string{"    ", "}\n"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(map[string]any{
				"model": map[string]any{
					"type":   "BPE",
					"vocab":  map[string]int{"}": 0},
					"merges": []string{},
				},
				"pre_tokenizer": map[string]any{
					"type": "Split",
					"pattern": map[string]string{
						"Regex": tt.pattern,
					},
				},
			})
			if err != nil {
				t.Fatal(err)
			}

			tok, err := LoadFromBytes(data)
			if err != nil {
				t.Fatal(err)
			}

			var got []string
			tok.forEachPartChunk("    }\n", func(chunk encodeChunk) {
				got = append(got, chunk.text)
			})
			if strings.Join(got, "\x00") != strings.Join(tt.want, "\x00") {
				t.Fatalf("chunks = %q, want %q", got, tt.want)
			}
		})
	}
}

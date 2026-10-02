package tokenizer

import (
	"encoding/json"
	"slices"
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

func TestMetaspace(t *testing.T) {
	data := []byte(`{
		"model": {"type":"BPE", "vocab":{"▁":0,"a":1,"b":2,"▁a":3,"▁b":4,"▁▁":5,"<s>":6},"merges":["▁ a","▁ b","▁ ▁"]},
		"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true},
		"decoder":{"type":"Replace","pattern":{"String":"▁"},"content":" "},
		"added_tokens":[{"id":6,"content":"<s>","special":true}]
	}`)
	for _, scheme := range []string{"always", "first", "never"} {
		t.Run(scheme, func(t *testing.T) {
			tok, err := LoadFromBytes([]byte(strings.ReplaceAll(string(data), `"always"`, `"`+scheme+`"`)))
			if err != nil {
				t.Fatal(err)
			}
			for _, tc := range []struct {
				text                 string
				always, first, never []int32
			}{
				{"", nil, nil, nil},
				{"a b", []int32{3, 4}, []int32{3, 4}, []int32{1, 4}},
				{"  a", []int32{0, 3}, []int32{0, 3}, []int32{0, 3}},
				{"a  b", []int32{3, 0, 4}, []int32{3, 0, 4}, []int32{1, 0, 4}},
				{"a<s>b", []int32{3, 6, 4}, []int32{3, 6, 2}, []int32{1, 6, 2}},
				{"<s>a", []int32{6, 3}, []int32{6, 1}, []int32{6, 1}},
			} {
				want := map[string][]int32{"always": tc.always, "first": tc.first, "never": tc.never}[scheme]
				if got := tok.Encode(tc.text, false); !slices.Equal(got, want) {
					t.Errorf("Encode(%q) = %v, want %v", tc.text, got, want)
				}
			}
		})
	}
}

func TestNFCNormalization(t *testing.T) {
	tok, err := LoadFromBytes([]byte(`{
		"model":{"type":"BPE","vocab":{"Ã":0,"©":1,"Ã©":2,"[é]":3},"merges":["Ã ©"]},
		"normalizer":{"type":"NFC"},
		"added_tokens":[{"id":3,"content":"[é]","special":true}]
	}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{"é[e\u0301]", "e\u0301[e\u0301]"} {
		if got := tok.Encode(text, false); !slices.Equal(got, []int32{2, 3}) {
			t.Errorf("Encode(%q) = %v, want [2 3]", text, got)
		}
	}
}

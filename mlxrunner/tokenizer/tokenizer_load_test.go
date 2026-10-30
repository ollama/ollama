package tokenizer

import (
	"fmt"
	"slices"
	"strings"
	"testing"
)

func TestEmptyAddedToken(t *testing.T) {
	tok, err := LoadFromBytes([]byte(`{
		"model":{"type":"BPE","vocab":{"a":0},"merges":[]},
		"added_tokens":[
			{"id":1,"content":"","special":true},
			{"id":2,"content":"","special":false,"normalized":true}
		]
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := tok.GetSpecialToken(""); ok {
		t.Fatal("empty added token must not be registered: it cannot consume input")
	}
	if got := tok.Encode("a", false); !slices.Equal(got, []int32{0}) {
		t.Fatalf("Encode(a) = %v, want [0]", got)
	}
}

func TestEmptyMetaspaceInput(t *testing.T) {
	for _, scheme := range []string{"always", "first", "never"} {
		t.Run(scheme, func(t *testing.T) {
			data := fmt.Sprintf(`{
				"model":{"type":"BPE","vocab":{"▁":0,"a":1},"merges":[]},
				"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":%q,"split":true}
			}`, scheme)
			tok, err := LoadFromBytes([]byte(data))
			if err != nil {
				t.Fatal(err)
			}
			if got := tok.Encode("", false); len(got) != 0 {
				t.Fatalf("Encode(empty) = %v, want no tokens", got)
			}
		})
	}
}

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

func TestIgnoreMerges(t *testing.T) {
	for _, tc := range []struct {
		name    string
		setting string
		want    []int32
	}{
		{"default", "", []int32{0, 4}},
		{"false", `,"ignore_merges":false`, []int32{0, 4}},
		{"true", `,"ignore_merges":true`, []int32{5}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The higher-priority b+c merge prevents ab+c, despite abc being
			// in the vocabulary. HF BPE emits [a, bc] unless ignore_merges is set.
			data := `{"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"ab":3,"bc":4,"abc":5},"merges":[["b","c"],["a","b"],["ab","c"]]` + tc.setting + `}}`
			tok, err := LoadFromBytes([]byte(data))
			if err != nil {
				t.Fatal(err)
			}
			for _, input := range []struct {
				text string
				want []int32
			}{
				{"abc", tc.want},
				{"abca", []int32{0, 4, 0}},
			} {
				if got := tok.Encode(input.text, false); !slices.Equal(got, input.want) {
					t.Errorf("Encode(%q) = %v, want %v", input.text, got, input.want)
				}
			}
		})
	}
}

package tokenizer

import (
	"slices"
	"strings"
	"testing"
)

// These synthetic vocabularies isolate added-token normalization from prefix
// insertion.
func TestAddedTokenNormalization(t *testing.T) {
	const mixed = "  ▁▁  "
	const repeats = encodeParallelMinInputBytes/len(mixed) + 1
	type normalizationCase struct {
		name, input string
		want        []int32
	}
	for _, fixture := range []struct {
		name      string
		tokenizer string
		cases     []normalizationCase
	}{
		{
			name: "nfc_special_token",
			tokenizer: `{
		"model":{"type":"BPE","vocab":{"Ã":0,"©":1,"Ã©":2,"[é]":3},"merges":["Ã ©"]},
		"normalizer":{"type":"NFC"},
		"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
		"added_tokens":[{"id":3,"content":"[é]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}]
	}`,
			cases: []normalizationCase{
				{name: "composed text", input: "é[e\u0301]", want: []int32{2, 3}},
				{name: "decomposed text", input: "e\u0301[e\u0301]", want: []int32{2, 3}},
			},
		},
		{
			name: "added_nfc_false",
			tokenizer: `{
  "version": "1.0", "truncation": null, "padding": null,
  "added_tokens": [
    {"id": 3, "content": "éé", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}
  ],
  "normalizer": {"type": "NFC"},
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
  "post_processor": null, "decoder": null,
  "model": {
    "type": "BPE", "dropout": null, "unk_token": null,
    "continuing_subword_prefix": null, "end_of_word_suffix": null,
    "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
    "vocab": {"Ã": 0, "©": 1, "Ã©": 2, "éé": 3, "a": 4},
    "merges": [["Ã", "©"]]
  }
}`,
			cases: []normalizationCase{
				{name: "composed", input: "éé", want: []int32{3}},
				{name: "decomposed", input: "e\u0301e\u0301", want: []int32{2, 2}},
				{name: "prefix", input: "aéé", want: []int32{4, 2, 2}},
				{name: "suffix", input: "ééa", want: []int32{2, 2, 4}},
				{name: "interior", input: "aééa", want: []int32{4, 2, 2, 4}},
				{name: "repeated", input: "éééé", want: []int32{2, 2, 2, 2}},
				{name: "literal interior", input: "aééa", want: []int32{4, 3, 4}},
				{name: "mixed literal and normalized", input: "ééaéé", want: []int32{3, 4, 2, 2}},
			},
		},
		{
			name: "added_nfc_true",
			tokenizer: `{
  "version": "1.0", "truncation": null, "padding": null,
  "added_tokens": [
    {"id": 3, "content": "éé", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": false}
  ],
  "normalizer": {"type": "NFC"},
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
  "post_processor": null, "decoder": null,
  "model": {
    "type": "BPE", "dropout": null, "unk_token": null,
    "continuing_subword_prefix": null, "end_of_word_suffix": null,
    "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
    "vocab": {"Ã": 0, "©": 1, "Ã©": 2, "éé": 3, "a": 4},
    "merges": [["Ã", "©"]]
  }
}`,
			cases: []normalizationCase{
				{name: "composed", input: "éé", want: []int32{3}},
				{name: "decomposed", input: "e\u0301e\u0301", want: []int32{3}},
				{name: "prefix", input: "aéé", want: []int32{4, 3}},
				{name: "suffix", input: "ééa", want: []int32{3, 4}},
				{name: "interior", input: "aééa", want: []int32{4, 3, 4}},
				{name: "repeated", input: "éééé", want: []int32{3, 3}},
				{name: "literal interior", input: "aééa", want: []int32{4, 3, 4}},
				{name: "mixed literal and normalized", input: "ééaéé", want: []int32{3, 4, 3}},
			},
		},
		{
			name: "normalized_spelling_and_raw_precedence",
			tokenizer: `{
  "normalizer": {"type": "NFC"},
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
  "model": {"type": "BPE", "vocab": {"a": 0, "Ã": 1, "©": 2, "Ã©": 3, "éé": 4, "é": 5, "e": 6}, "merges": [["Ã", "©"]]},
  "added_tokens": [
    {"id": 4, "content": "éé", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": false},
    {"id": 5, "content": "é", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 6, "content": "e", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": false}
  ]
}`,
			cases: []normalizationCase{
				{name: "normalized spelling", input: "ae\u0301e\u0301a", want: []int32{0, 4, 0}},
				{name: "raw tokens win", input: "éé", want: []int32{5, 5}},
				{name: "raw token separates normalized matches", input: "e\u0301ée\u0301", want: []int32{3, 5, 3}},
				{name: "normalize before matching", input: "e\u0301", want: []int32{3}},
				{name: "unchanged normalized token", input: "aea", want: []int32{0, 6, 0}},
			},
		},
		{
			name: "normalized_collision",
			tokenizer: `{
  "normalizer": {"type": "NFC"},
  "model": {"type": "BPE", "vocab": {"é": 0, "é": 1}, "merges": []},
  "added_tokens": [
    {"id": 0, "content": "é", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": false},
    {"id": 1, "content": "é", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": true}
  ]
}`,
			cases: []normalizationCase{
				{name: "composed", input: "é", want: []int32{1}},
				{name: "decomposed", input: "e\u0301", want: []int32{1}},
			},
		},
		{
			name: "added_metaspace_replace_false",
			tokenizer: `{
  "version": "1.0", "truncation": null, "padding": null,
  "added_tokens": [
    {"id": 1, "content": "▁▁", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
    {"id": 3, "content": "▁a", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}
  ],
  "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
  "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
  "post_processor": null, "decoder": null,
  "model": {
    "type": "BPE", "dropout": null, "unk_token": null,
    "continuing_subword_prefix": null, "end_of_word_suffix": null,
    "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
    "vocab": {"▁": 0, "▁▁": 1, "a": 2, "▁a": 3}, "merges": []
  }
}`,
			cases: []normalizationCase{
				{name: "spaces", input: "  ", want: []int32{0, 0}},
				{name: "space word", input: " a", want: []int32{0, 2}},
				{name: "prefix only", input: "a", want: []int32{0, 2}},
				{name: "literal token", input: "▁a", want: []int32{3}},
				{name: "mixed", input: mixed, want: []int32{0, 0, 1, 0, 0}},
				{name: "large mixed", input: strings.Repeat(mixed, repeats), want: slices.Repeat([]int32{0, 0, 1, 0, 0}, repeats)},
			},
		},
		{
			name: "added_metaspace_replace_true",
			tokenizer: `{
  "version": "1.0", "truncation": null, "padding": null,
  "added_tokens": [
    {"id": 1, "content": "▁▁", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": false},
    {"id": 3, "content": "▁a", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": false}
  ],
  "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
  "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
  "post_processor": null, "decoder": null,
  "model": {
    "type": "BPE", "dropout": null, "unk_token": null,
    "continuing_subword_prefix": null, "end_of_word_suffix": null,
    "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
    "vocab": {"▁": 0, "▁▁": 1, "a": 2, "▁a": 3}, "merges": []
  }
}`,
			cases: []normalizationCase{
				{name: "spaces", input: "  ", want: []int32{1}},
				{name: "space word", input: " a", want: []int32{3}},
				{name: "prefix only", input: "a", want: []int32{0, 2}},
				{name: "literal token", input: "▁a", want: []int32{3}},
				{name: "interior", input: "a  a", want: []int32{0, 2, 1, 0, 2}},
				{name: "adjacent", input: " a a", want: []int32{3, 3}},
				{name: "mixed", input: mixed, want: []int32{1, 1, 1}},
				{name: "large mixed", input: strings.Repeat(mixed, repeats), want: slices.Repeat([]int32{1, 1, 1}, repeats)},
			},
		},
		{
			name: "added_metaspace_first",
			tokenizer: `{
  "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
  "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": true},
  "model": {"type": "BPE", "vocab": {"▁": 0, "▁▁": 1, "a": 2}, "merges": []},
  "added_tokens": [{"id": 1, "content": "▁▁", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": false}]
}`,
			cases: []normalizationCase{
				{name: "interior", input: "a  a", want: []int32{0, 2, 1, 2}},
				{name: "initial", input: "  a", want: []int32{1, 2}},
				{name: "suffix", input: "a  ", want: []int32{0, 2, 1}},
			},
		},
		{
			name: "added_metaspace_prefix_false",
			tokenizer: `{
  "version": "1.0", "truncation": null, "padding": null,
  "added_tokens": [
    {"id": 1, "content": "▁▁", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
    {"id": 3, "content": "▁a", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
  "post_processor": null, "decoder": null,
  "model": {
    "type": "BPE", "dropout": null, "unk_token": null,
    "continuing_subword_prefix": null, "end_of_word_suffix": null,
    "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
    "vocab": {"▁": 0, "▁▁": 1, "a": 2, "▁a": 3}, "merges": []
  }
}`,
			cases: []normalizationCase{
				{name: "spaces", input: "  ", want: []int32{0, 0}},
				{name: "space word", input: " a", want: []int32{0, 2}},
				{name: "prefix only", input: "a", want: []int32{0, 2}},
				{name: "literal token", input: "▁a", want: []int32{3}},
			},
		},
		{
			name: "added_metaspace_prefix_true",
			tokenizer: `{
  "version": "1.0", "truncation": null, "padding": null,
  "added_tokens": [
    {"id": 1, "content": "▁▁", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": false},
    {"id": 3, "content": "▁a", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
  "post_processor": null, "decoder": null,
  "model": {
    "type": "BPE", "dropout": null, "unk_token": null,
    "continuing_subword_prefix": null, "end_of_word_suffix": null,
    "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
    "vocab": {"▁": 0, "▁▁": 1, "a": 2, "▁a": 3}, "merges": []
  }
}`,
			cases: []normalizationCase{
				{name: "spaces", input: "  ", want: []int32{0, 0}},
				{name: "space word", input: " a", want: []int32{0, 2}},
				{name: "prefix only", input: "a", want: []int32{0, 2}},
				{name: "literal token", input: "▁a", want: []int32{3}},
			},
		},
	} {
		t.Run(fixture.name, func(t *testing.T) {
			tok, err := LoadFromBytes([]byte(fixture.tokenizer))
			if err != nil {
				t.Fatal(err)
			}
			for _, tc := range fixture.cases {
				t.Run(tc.name, func(t *testing.T) {
					if got := tok.Encode(tc.input, false); !slices.Equal(got, tc.want) {
						t.Errorf("Encode(%q) = %v, want %v", tc.input, got, tc.want)
					}
				})
			}
		})
	}
}

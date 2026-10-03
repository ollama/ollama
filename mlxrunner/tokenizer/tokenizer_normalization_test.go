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
		"added_tokens":[{"id":3,"content":"[é]","special":true}]
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
    "vocab": {"Ã": 0, "©": 1, "Ã©": 2, "éé": 3},
    "merges": [["Ã", "©"]]
  }
}`,
			cases: []normalizationCase{
				{name: "composed", input: "éé", want: []int32{3}},
				{name: "decomposed", input: "e\u0301e\u0301", want: []int32{2, 2}},
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
    "vocab": {"Ã": 0, "©": 1, "Ã©": 2, "éé": 3},
    "merges": [["Ã", "©"]]
  }
}`,
			cases: []normalizationCase{
				{name: "composed", input: "éé", want: []int32{3}},
				{name: "decomposed", input: "e\u0301e\u0301", want: []int32{3}},
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

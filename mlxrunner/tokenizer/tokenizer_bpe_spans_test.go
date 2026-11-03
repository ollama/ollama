package tokenizer

import (
	"fmt"
	"slices"
	"strings"
	"testing"
)

func TestBPEMergeSpans(t *testing.T) {
	tok := &Tokenizer{vocab: &Vocabulary{
		Reverse: map[string]int32{"a": 1, "b": 2, "c": 3, "ab": 4, "bc": 5, "abc": 6, "é": 7, "�": 8, "�a": 9, "aa": 10},
		Merges:  map[string]int{"b c": 0, "a b": 1, "ab c": 2, "� a": 3, "a a": 4},
	}}
	for _, tc := range []struct {
		name, input string
		want        []int32
	}{
		{"competing merges", "abc", []int32{1, 5}},
		{"equal rank merges leftmost", "aaa", []int32{10, 1}},
		{"invalid UTF8", "\xffa", []int32{9}},
		{"replacement rune", "�a", []int32{9}},
		{"adjacent invalid bytes", "\xff\xfea", []int32{8, 9}},
		{"truncated UTF8", "\xe2\x82a", []int32{8, 9}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tok.encodeBPEMerge(tc.input, nil); !slices.Equal(got, tc.want) {
				t.Fatalf("encodeBPEMerge(%q) = %v, want %v", tc.input, got, tc.want)
			}
		})
	}
	// Overlapping candidates exercise heap growth, rank ordering and stale pairs.
	for _, n := range []int{31, 32, 33, 255, 256, 257} {
		t.Run(fmt.Sprintf("equal rank/%d", n), func(t *testing.T) {
			input := strings.Repeat("a", n)
			want := slices.Repeat([]int32{10}, n/2)
			if n%2 != 0 {
				want = append(want, 1)
			}
			if got := tok.encodeBPEMerge(input, nil); !slices.Equal(got, want) {
				t.Fatalf("encodeBPEMerge(%q) = %v, want %v", input, got, want)
			}
		})
		t.Run(fmt.Sprintf("competing ranks/%d", n), func(t *testing.T) {
			input := strings.Repeat("abc", n)
			want := slices.Repeat([]int32{1, 5}, n)
			if got := tok.encodeBPEMerge(input, nil); !slices.Equal(got, want) {
				t.Fatalf("encodeBPEMerge(%q) = %v, want %v", input, got, want)
			}
		})
	}
	for _, n := range []int{29, 30, 31, 255, 256, 257} {
		input := strings.Repeat("é", n) + "abc"
		want := append(slices.Repeat([]int32{7}, n), 1, 5)
		if got := tok.encodeBPEMerge(input, nil); !slices.Equal(got, want) {
			t.Fatalf("encodeBPEMerge with %d multibyte runes = %v, want %v", n, got, want)
		}
	}
}

func TestByteFallback(t *testing.T) {
	vocab := &Vocabulary{Reverse: map[string]int32{
		"<0xC3>": 0, "<0xA9>": 1, "<0xF0>": 2, "<0x9F>": 3,
		"<0x98>": 4, "<0x80>": 5, "<0x61>": 6,
	}}
	for _, tc := range []struct {
		name  string
		typ   TokenizerType
		input string
		want  []int32
	}{
		{"BPE empty", TokenizerBPE, "", nil},
		{"BPE ASCII", TokenizerBPE, "a", []int32{6}},
		{"BPE missing byte", TokenizerBPE, "x", nil},
		{"SentencePiece empty", TokenizerSentencePiece, "", nil},
		{"SentencePiece two byte rune", TokenizerSentencePiece, "é", []int32{0, 1}},
		{"SentencePiece four byte rune", TokenizerSentencePiece, "😀", []int32{2, 3, 4, 5}},
		{"SentencePiece missing byte", TokenizerSentencePiece, "x", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tok := &Tokenizer{typ: tc.typ, vocab: vocab}
			initByteTokens(tok)
			if got := tok.appendByteFallback(nil, tc.input); !slices.Equal(got, tc.want) {
				t.Errorf("appendByteFallback(%q) = %v, want %v", tc.input, got, tc.want)
			}
		})
	}
}

func TestByteLevelMapping(t *testing.T) {
	for b, r := range byteToRune {
		if got, ok := decodeByteLevelRune(r); !ok || got != byte(b) {
			t.Errorf("decodeByteLevelRune(%U) = %d, %v, want %d, true", r, got, ok, b)
		}
	}
	if _, ok := decodeByteLevelRune('😀'); ok {
		t.Error("decodeByteLevelRune accepted a rune outside the byte alphabet")
	}
}

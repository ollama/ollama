package tokenizer

import (
	"slices"
	"testing"
)

// makeUnigramVocab builds a SentencePiece from (piece, score) pairs.
// All tokens are TOKEN_TYPE_NORMAL unless specified via the byteTokens map.
func makeUnigramVocab(pieces []struct {
	value string
	score float32
	typ   int32
}) SentencePiece {
	var v Vocabulary
	for _, p := range pieces {
		v.Values = append(v.Values, p.value)
		v.Scores = append(v.Scores, p.score)
		if p.typ == 0 {
			v.Types = append(v.Types, TOKEN_TYPE_NORMAL)
		} else {
			v.Types = append(v.Types, p.typ)
		}
	}
	return NewSentencePiece(&v)
}

// TestSentencePieceViterbiGlobalOptimum verifies that Viterbi finds the globally
// optimal segmentation even when a greedy left-to-right approach would not.
//
// Vocab scores: "a"=-1, "b"=-1, "c"=-1, "ab"=-0.5, "bc"=-0.3, "abc"=-3.0
//
// For text "abc":
//   - Greedy (picks best-scoring prefix at each position) → [ab, c] = -1.5
//   - Viterbi (global DP) → [a, bc] = -1.3   ← optimal
func TestSentencePieceViterbiGlobalOptimum(t *testing.T) {
	spm := makeUnigramVocab([]struct {
		value string
		score float32
		typ   int32
	}{
		{"a", -1.0, 0},
		{"b", -1.0, 0},
		{"c", -1.0, 0},
		{"ab", -0.5, 0},  // greedy would pick this over "a" alone
		{"bc", -0.3, 0},  // but "bc" makes the global optimum [a, bc]
		{"abc", -3.0, 0}, // whole word is least optimal
	})

	ids, err := spm.Encode("abc", false)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	// id of "a"=0, "bc"=4 — Viterbi must choose [a, bc] = -1.0 + -0.3 = -1.3
	// over [ab, c] = -0.5 + -1.0 = -1.5
	want := []int32{0, 4}
	if !slices.Equal(ids, want) {
		pieces := make([]string, len(ids))
		for i, id := range ids {
			pieces[i] = spm.vocab.Values[id]
		}
		t.Errorf("got %v (%v), want %v ([a, bc] — global optimum)", ids, pieces, want)
	}
}

// TestSentencePieceViterbiPrefersSingleToken verifies that when one long token
// has a better score than any segmentation into pieces, Viterbi picks it.
func TestSentencePieceViterbiPrefersSingleToken(t *testing.T) {
	spm := makeUnigramVocab([]struct {
		value string
		score float32
		typ   int32
	}{
		{"x", -5.0, 0},
		{"y", -5.0, 0},
		{"z", -5.0, 0},
		{"xy", -3.0, 0},
		{"yz", -3.0, 0},
		{"xyz", -1.0, 0}, // single token is best
	})

	ids, err := spm.Encode("xyz", false)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	// [xyz] = -1.0 > [xy, z] = -8.0 > [x, yz] = -8.0 > [x, y, z] = -15.0
	want := []int32{5} // id of "xyz"
	if !slices.Equal(ids, want) {
		pieces := make([]string, len(ids))
		for i, id := range ids {
			pieces[i] = spm.vocab.Values[id]
		}
		t.Errorf("got %v (%v), want %v ([xyz])", ids, pieces, want)
	}
}

// TestSentencePieceViterbiThreeWaySplit verifies a case where neither a greedy
// left-to-right nor right-to-left approach finds the optimum.
//
// Vocab: "a"=-1, "b"=-2, "c"=-2, "d"=-1, "ab"=-0.8, "bc"=-0.5, "cd"=-5.0
//
// "abcd" optimal is [a, bc, d] = -1.0 + -0.5 + -1.0 = -2.5
// Greedy (picks "ab" first at pos 0, since -0.8 > -1.0) → [ab, c, d] = -0.8 + -2.0 + -1.0 = -3.8
func TestSentencePieceViterbiThreeWaySplit(t *testing.T) {
	spm := makeUnigramVocab([]struct {
		value string
		score float32
		typ   int32
	}{
		{"a", -1.0, 0},   // 0
		{"b", -2.0, 0},   // 1
		{"c", -2.0, 0},   // 2
		{"d", -1.0, 0},   // 3
		{"ab", -0.8, 0},  // 4: greedy picks this first (better than "a")
		{"bc", -0.5, 0},  // 5: Viterbi finds [a, bc, d] via bc
		{"cd", -5.0, 0},  // 6: poor score, never chosen
	})

	ids, err := spm.Encode("abcd", false)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	// Scores:
	// [a, bc, d] = -1.0 + -0.5 + -1.0 = -2.5  ← optimal
	// [ab, c, d] = -0.8 + -2.0 + -1.0 = -3.8
	// [a, b, cd] = -1.0 + -2.0 + -5.0 = -8.0
	// [a, b, c, d] = -7.0
	want := []int32{0, 5, 3} // a=0, bc=5, d=3
	if !slices.Equal(ids, want) {
		pieces := make([]string, len(ids))
		for i, id := range ids {
			pieces[i] = spm.vocab.Values[id]
		}
		t.Errorf("got %v (%v), want %v ([a, bc, d])", ids, pieces, want)
	}
}

// TestSentencePieceAddSpacePrefix verifies that setting AddSpacePrefix=true
// prepends a ▁ to the first word, matching SentencePiece add_dummy_prefix semantics.
func TestSentencePieceAddSpacePrefix(t *testing.T) {
	pieces := []struct {
		value string
		score float32
		typ   int32
	}{
		{"hello", -1.0, 0},    // 0: no leading ▁
		{"▁hello", -1.0, 0},   // 1: with leading ▁
		{"▁world", -1.0, 0},   // 2
		{"world", -1.0, 0},    // 3: no leading ▁
		{"h", -10.0, 0},       // 4: fallback chars
		{"e", -10.0, 0},       // 5
		{"l", -10.0, 0},       // 6
		{"o", -10.0, 0},       // 7
		{"w", -10.0, 0},       // 8
		{"r", -10.0, 0},       // 9
		{"d", -10.0, 0},       // 10
		{"▁", -100.0, 0},      // 11
	}

	withoutPrefix := makeUnigramVocab(pieces)
	withoutPrefix.vocab.AddSpacePrefix = false

	withPrefix := makeUnigramVocab(pieces)
	withPrefix.vocab.AddSpacePrefix = true

	t.Run("without_add_space_prefix", func(t *testing.T) {
		// "hello world" → replace space with ▁ → "hello▁world"
		// tokenizes as: "hello" + "▁world"
		ids, err := withoutPrefix.Encode("hello world", false)
		if err != nil {
			t.Fatal(err)
		}
		want := []int32{0, 2} // "hello", "▁world"
		if !slices.Equal(ids, want) {
			pieces := make([]string, len(ids))
			for i, id := range ids {
				pieces[i] = withoutPrefix.vocab.Values[id]
			}
			t.Errorf("got %v (%v), want %v ([hello, ▁world])", ids, pieces, want)
		}
	})

	t.Run("with_add_space_prefix", func(t *testing.T) {
		// add_space_prefix prepends " " → " hello world" → "▁hello▁world"
		// tokenizes as: "▁hello" + "▁world"
		ids, err := withPrefix.Encode("hello world", false)
		if err != nil {
			t.Fatal(err)
		}
		want := []int32{1, 2} // "▁hello", "▁world"
		if !slices.Equal(ids, want) {
			pieces := make([]string, len(ids))
			for i, id := range ids {
				pieces[i] = withPrefix.vocab.Values[id]
			}
			t.Errorf("got %v (%v), want %v ([▁hello, ▁world])", ids, pieces, want)
		}
	})

	t.Run("first_word_gets_prefix", func(t *testing.T) {
		// Single word with add_space_prefix: "hello" → " hello" → "▁hello"
		ids, err := withPrefix.Encode("hello", false)
		if err != nil {
			t.Fatal(err)
		}
		want := []int32{1} // "▁hello"
		if !slices.Equal(ids, want) {
			pieces := make([]string, len(ids))
			for i, id := range ids {
				pieces[i] = withPrefix.vocab.Values[id]
			}
			t.Errorf("got %v (%v), want %v ([▁hello])", ids, pieces, want)
		}
	})

	t.Run("empty_string", func(t *testing.T) {
		ids, err := withPrefix.Encode("", false)
		if err != nil {
			t.Fatal(err)
		}
		// " " → "▁" → should produce the ▁ token or nothing if only space
		// After replacing " " with "▁", we get "▁" which is a token
		// (or if add_space_prefix=true with empty, " " → "▁", encoded as id=11)
		if len(ids) > 1 {
			t.Errorf("empty string with add_space_prefix got %d tokens, want ≤1", len(ids))
		}
	})
}

// TestSentencePieceByteTokenFallback verifies that characters with no vocabulary
// entry fall back to byte-level tokens.
func TestSentencePieceByteTokenFallback(t *testing.T) {
	spm := makeUnigramVocab([]struct {
		value string
		score float32
		typ   int32
	}{
		{"hello", -1.0, TOKEN_TYPE_NORMAL},  // 0
		{"<0xC3>", -1.0, TOKEN_TYPE_BYTE},   // 1: first byte of é in UTF-8
		{"<0xA9>", -1.0, TOKEN_TYPE_BYTE},   // 2: second byte of é
		{"<0x21>", -1.0, TOKEN_TYPE_BYTE},   // 3: '!'
		{"world", -1.0, TOKEN_TYPE_NORMAL},  // 4: padding to reach min trace size
	})

	t.Run("utf8_byte_fallback", func(t *testing.T) {
		// "é" = 0xC3 0xA9 in UTF-8; no "é" token → fall back to bytes
		ids, err := spm.Encode("é", false)
		if err != nil {
			t.Fatal(err)
		}
		want := []int32{1, 2} // <0xC3>, <0xA9>
		if !slices.Equal(ids, want) {
			t.Errorf("got %v, want %v (<0xC3>, <0xA9>)", ids, want)
		}
	})

	t.Run("mixed_known_and_byte_fallback", func(t *testing.T) {
		// "hello!" — "hello" is in vocab, "!" falls back to <0x21>
		ids, err := spm.Encode("hello!", false)
		if err != nil {
			t.Fatal(err)
		}
		want := []int32{0, 3} // hello, <0x21>
		if !slices.Equal(ids, want) {
			t.Errorf("got %v, want %v (hello, <0x21>)", ids, want)
		}
	})

	t.Run("multi_byte_utf8_byte_fallback", func(t *testing.T) {
		// "é!" — both use byte fallback for é, then <0x21> for !
		ids, err := spm.Encode("é!", false)
		if err != nil {
			t.Fatal(err)
		}
		want := []int32{1, 2, 3} // <0xC3>, <0xA9>, <0x21>
		if !slices.Equal(ids, want) {
			t.Errorf("got %v, want %v (<0xC3>, <0xA9>, <0x21>)", ids, want)
		}
	})
}

// TestSentencePieceEdgeCases covers boundary conditions.
func TestSentencePieceEdgeCases(t *testing.T) {
	spm := makeUnigramVocab([]struct {
		value string
		score float32
		typ   int32
	}{
		{"a", -1.0, 0},
		{"b", -1.0, 0},
		{"ab", -0.5, 0},
		{"▁a", -1.0, 0},
		{"▁b", -1.0, 0},
		{"<0x61>", -5.0, TOKEN_TYPE_BYTE}, // byte for 'a'
	})

	t.Run("empty_string", func(t *testing.T) {
		ids, err := spm.Encode("", false)
		if err != nil {
			t.Fatal(err)
		}
		if len(ids) != 0 {
			t.Errorf("got %v, want empty", ids)
		}
	})

	t.Run("single_char", func(t *testing.T) {
		ids, err := spm.Encode("a", false)
		if err != nil {
			t.Fatal(err)
		}
		if len(ids) == 0 {
			t.Error("got empty, want non-empty")
		}
	})

	t.Run("single_char_in_vocab", func(t *testing.T) {
		ids, err := spm.Encode("a", false)
		if err != nil {
			t.Fatal(err)
		}
		want := []int32{0} // "a"
		if !slices.Equal(ids, want) {
			t.Errorf("got %v, want %v", ids, want)
		}
	})

	t.Run("exact_vocab_match_whole_string", func(t *testing.T) {
		ids, err := spm.Encode("ab", false)
		if err != nil {
			t.Fatal(err)
		}
		// "ab" is in vocab as single token, score -0.5 > "a"(-1.0) + "b"(-1.0) = -2.0
		want := []int32{2}
		if !slices.Equal(ids, want) {
			t.Errorf("got %v, want %v ([ab])", ids, want)
		}
	})

	t.Run("space_becomes_spm_whitespace", func(t *testing.T) {
		// " a" → "▁a"
		ids, err := spm.Encode(" a", false)
		if err != nil {
			t.Fatal(err)
		}
		want := []int32{3} // "▁a"
		if !slices.Equal(ids, want) {
			t.Errorf("got %v, want %v ([▁a])", ids, want)
		}
	})

	t.Run("space_word_space_word", func(t *testing.T) {
		// " a b" → "▁a▁b"
		ids, err := spm.Encode(" a b", false)
		if err != nil {
			t.Fatal(err)
		}
		want := []int32{3, 4} // "▁a", "▁b"
		if !slices.Equal(ids, want) {
			t.Errorf("got %v, want %v ([▁a, ▁b])", ids, want)
		}
	})
}

// TestSentencePieceSpecialTokenHandling verifies that control/special tokens
// are split out before Viterbi tokenization and passed through as-is.
func TestSentencePieceSpecialTokenHandling(t *testing.T) {
	v := &Vocabulary{
		Values: []string{
			"hello",     // 0: NORMAL
			"▁world",    // 1: NORMAL
			"<s>",       // 2: CONTROL (BOS)
			"</s>",      // 3: CONTROL (EOS)
			"h",         // 4: NORMAL fallback
			"e",         // 5: NORMAL fallback
			"l",         // 6: NORMAL fallback
			"o",         // 7: NORMAL fallback
		},
		Scores: []float32{-1.0, -1.0, 0.0, 0.0, -5.0, -5.0, -5.0, -5.0},
		Types: []int32{
			TOKEN_TYPE_NORMAL,
			TOKEN_TYPE_NORMAL,
			TOKEN_TYPE_CONTROL,
			TOKEN_TYPE_CONTROL,
			TOKEN_TYPE_NORMAL,
			TOKEN_TYPE_NORMAL,
			TOKEN_TYPE_NORMAL,
			TOKEN_TYPE_NORMAL,
		},
		BOS:    []int32{2},
		EOS:    []int32{3},
		AddBOS: true,
		AddEOS: true,
	}

	spm := NewSentencePiece(v)

	t.Run("addSpecial_wraps_with_bos_eos", func(t *testing.T) {
		ids, err := spm.Encode("hello", true)
		if err != nil {
			t.Fatal(err)
		}
		// [<s>, hello, </s>]
		want := []int32{2, 0, 3}
		if !slices.Equal(ids, want) {
			t.Errorf("got %v, want %v", ids, want)
		}
	})

	t.Run("no_addSpecial", func(t *testing.T) {
		ids, err := spm.Encode("hello", false)
		if err != nil {
			t.Fatal(err)
		}
		want := []int32{0}
		if !slices.Equal(ids, want) {
			t.Errorf("got %v, want %v", ids, want)
		}
	})

	t.Run("control_token_in_text_is_passthrough", func(t *testing.T) {
		// "<s>" in the input text should be treated as a special token
		ids, err := spm.Encode("<s>hello</s>", false)
		if err != nil {
			t.Fatal(err)
		}
		// ["<s>", "hello", "</s>"]
		want := []int32{2, 0, 3}
		if !slices.Equal(ids, want) {
			t.Errorf("got %v, want %v", ids, want)
		}
	})
}

// TestSentencePieceRoundtrip verifies encode→decode round-trip for the Unigram model.
// Note: with add_space_prefix=true, the leading ▁ is part of the encoded token stream
// and decodes back to a leading space — round-trip input must include the leading space.
func TestSentencePieceRoundtrip(t *testing.T) {
	spm := makeUnigramVocab([]struct {
		value string
		score float32
		typ   int32
	}{
		{"▁hello", -1.0, 0},
		{"▁world", -1.0, 0},
		{"▁", -5.0, 0},
		{"h", -10.0, 0},
		{"e", -10.0, 0},
		{"l", -10.0, 0},
		{"o", -10.0, 0},
		{"w", -10.0, 0},
		{"r", -10.0, 0},
		{"d", -10.0, 0},
		{"▁t", -2.0, 0},
		{"es", -2.0, 0},
		{"t", -10.0, 0},
	})
	spm.vocab.AddSpacePrefix = false

	// These strings already have leading ▁-equivalents embedded via space→▁ replacement.
	// encode(" hello world") → "▁hello▁world" → [▁hello, ▁world] → decode → " hello world"
	cases := []string{
		" hello",
		" hello world",
		" test",
	}
	for _, want := range cases {
		ids, err := spm.Encode(want, false)
		if err != nil {
			t.Fatalf("Encode(%q): %v", want, err)
		}
		got, err := spm.Decode(ids)
		if err != nil {
			t.Fatalf("Decode: %v", err)
		}
		if got != want {
			t.Errorf("roundtrip(%q): got %q", want, got)
		}
	}
}

// TestSentencePieceMaxTokenLen verifies that Viterbi doesn't look beyond maxTokenLen.
func TestSentencePieceMaxTokenLen(t *testing.T) {
	// Only "abc" and individual chars — no token longer than 3
	spm := makeUnigramVocab([]struct {
		value string
		score float32
		typ   int32
	}{
		{"a", -1.0, 0},
		{"b", -1.0, 0},
		{"c", -1.0, 0},
		{"d", -1.0, 0},
		{"abc", -0.1, 0},
		{"abcd", -0.05, 0}, // longer than maxTokenLen if we limit it — but maxTokenLen is computed from vocab
	})

	// "abcd" should be tokenized as [abcd] since it's in vocab and best overall
	ids, err := spm.Encode("abcd", false)
	if err != nil {
		t.Fatal(err)
	}
	// [abcd] = -0.05 is best
	want := []int32{5} // "abcd"
	if !slices.Equal(ids, want) {
		pieces := make([]string, len(ids))
		for i, id := range ids {
			pieces[i] = spm.vocab.Values[id]
		}
		t.Errorf("got %v (%v), want [abcd]", ids, pieces)
	}
}

// TestSentencePieceUnicodeRunes verifies correct rune-based handling for
// multi-byte Unicode characters.
func TestSentencePieceUnicodeRunes(t *testing.T) {
	spm := makeUnigramVocab([]struct {
		value string
		score float32
		typ   int32
	}{
		{"▁こんにちは", -1.0, 0}, // Japanese greeting as one token
		{"▁世界", -1.0, 0},       // "world" in Chinese/Japanese
		{"こ", -5.0, 0},
		{"ん", -5.0, 0},
		{"に", -5.0, 0},
		{"ち", -5.0, 0},
		{"は", -5.0, 0},
		{"世", -5.0, 0},
		{"界", -5.0, 0},
	})
	spm.vocab.AddSpacePrefix = true

	t.Run("japanese_single_token", func(t *testing.T) {
		ids, err := spm.Encode("こんにちは", false)
		if err != nil {
			t.Fatal(err)
		}
		// AddSpacePrefix=true → " こんにちは" → "▁こんにちは" → id 0
		want := []int32{0}
		if !slices.Equal(ids, want) {
			pieces := make([]string, len(ids))
			for i, id := range ids {
				pieces[i] = spm.vocab.Values[id]
			}
			t.Errorf("got %v (%v), want [▁こんにちは]", ids, pieces)
		}
	})

	t.Run("roundtrip_japanese", func(t *testing.T) {
		// With add_space_prefix=true, encode prepends " " so the decoded output has a
		// leading space. Roundtrip from " こんにちは 世界" → encode → decode → same string.
		want := " こんにちは 世界"
		ids, err := spm.Encode("こんにちは 世界", false)
		if err != nil {
			t.Fatal(err)
		}
		got, err := spm.Decode(ids)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("roundtrip: got %q, want %q", got, want)
		}
	})
}

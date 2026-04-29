package tokenizer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func llama(t testing.TB) BytePairEncoding {
	t.Helper()

	f, err := os.Open(filepath.FromSlash("testdata/llama3.2/encoder.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	vocab := make(map[string]int32)
	if err := json.NewDecoder(f).Decode(&vocab); err != nil {
		t.Fatal(err)
	}

	types := make([]int32, len(vocab))
	tokens := make([]string, len(vocab))
	for token, id := range vocab {
		tokens[id] = token
		types[id] = 1
	}

	for _, token := range []string{"<|begin_of_text|>", "<|end_of_text|>"} {
		if _, ok := vocab[token]; !ok {
			tokens = append(tokens, token) //nolint:makezero
			types = append(types, 3)       //nolint:makezero
			vocab[token] = int32(len(vocab))
		}
	}

	f, err = os.Open(filepath.FromSlash("testdata/llama3.2/vocab.bpe"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	merges := make([]string, 0, 50000)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		if !strings.HasPrefix(scanner.Text(), "#") {
			merges = append(merges, scanner.Text())
		}
	}

	return NewBytePairEncoding(
		&Vocabulary{
			Values: tokens,
			Types:  types,
			Merges: merges,
		},
		"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
	)
}

func TestLlama(t *testing.T) {
	tokenizer := llama(t)

	t.Run("simple", func(t *testing.T) {
		t.Parallel()

		ids, err := tokenizer.Encode("hello world", true)
		if err != nil {
			t.Error(err)
		}

		if diff := cmp.Diff([]int32{15339, 1917}, ids); diff != "" {
			t.Errorf("no match (-theirs +ours):\n%s", diff)
		}

		s, err := tokenizer.Decode([]int32{15339, 1917})
		if err != nil {
			t.Fatal(err)
		}

		if s != "hello world" {
			t.Errorf("got %q, want hello world", s)
		}

		ids, err = tokenizer.Encode("hello <|end_of_text|>", true)
		if err != nil {
			t.Error(err)
		}

		if diff := cmp.Diff([]int32{15339, 220, 128001}, ids); diff != "" {
			t.Errorf("no match (-theirs +ours):\n%s", diff)
		}
	})

	t.Run("simple repeated", func(t *testing.T) {
		t.Parallel()

		cases := map[string][]int32{
			strings.Repeat("0", 1):  {15},
			strings.Repeat("0", 2):  {410},
			strings.Repeat("0", 3):  {931},
			strings.Repeat("0", 4):  {931, 15},
			strings.Repeat("0", 5):  {931, 410},
			strings.Repeat("0", 6):  {931, 931},
			strings.Repeat("0", 7):  {931, 931, 15},
			strings.Repeat("0", 8):  {931, 931, 410},
			strings.Repeat("0", 9):  {931, 931, 931},
			strings.Repeat("0", 10): {931, 931, 931, 15},
			strings.Repeat("0", 11): {931, 931, 931, 410},
			strings.Repeat("0", 12): {931, 931, 931, 931},
			strings.Repeat("0", 13): {931, 931, 931, 931, 15},
			strings.Repeat("0", 14): {931, 931, 931, 931, 410},
			strings.Repeat("0", 15): {931, 931, 931, 931, 931},
			strings.Repeat("0", 16): {931, 931, 931, 931, 931, 15},
			strings.Repeat("0", 17): {931, 931, 931, 931, 931, 410},
		}

		for s, want := range cases {
			ids, err := tokenizer.Encode(s, true)
			if err != nil {
				t.Error(err)
			}

			if diff := cmp.Diff(want, ids); diff != "" {
				t.Errorf("%q no match (-theirs +ours):\n%s", s, diff)
			}
		}
	})

	t.Run("basic roundtrip", func(t *testing.T) {
		t.Parallel()

		cases := []string{
			"hello",
			"hello ",
			"hello  ",
			" hello",
			" hello ",
			" hello  ",
			"hello world",
			"请考试我的软件！12345",
		}

		for _, want := range cases {
			ids, err := tokenizer.Encode(want, true)
			if err != nil {
				t.Error(err)
			}

			if got, err := tokenizer.Decode(ids); err != nil {
				t.Fatal(err)
			} else if got != want {
				t.Errorf("got %q, want %q", got, want)
			}
		}
	})

	t.Run("special", func(t *testing.T) {
		t.Parallel()

		cases := map[string][]int32{
			"<|begin_of_text|>A B!":                                               {128000, 32, 426, 0},
			"<|begin_of_text|>A<|end_of_text|>B!":                                 {128000, 32, 128001, 33, 0},
			"<|begin_of_text|>A<|end_of_text|>B<|begin_of_text|>!":                {128000, 32, 128001, 33, 128000, 0},
			"<|begin_of_text|>A<|end_of_text|>B<|begin_of_text|>!<|end_of_text|>": {128000, 32, 128001, 33, 128000, 0, 128001},
		}

		for s, want := range cases {
			ids, err := tokenizer.Encode(s, true)
			if err != nil {
				t.Fatal(err)
			}

			if diff := cmp.Diff(want, ids); diff != "" {
				t.Errorf("no match (-theirs +ours):\n%s", diff)
			}
		}
	})

	t.Run("split", func(t *testing.T) {
		t.Parallel()

		cases := map[string][]string{
			"Hello World!":                   {"Hello", " World", "!"},
			"I'm don't won't":                {"I", "'m", " don", "'t", " won", "'t"},
			"In 2024 there are 366 days":     {"In", " ", "202", "4", " there", " are", " ", "366", " days"},
			"Hello!! ...world":               {"Hello", "!!", " ...", "world"},
			"Hello    World":                 {"Hello", "   ", " World"},
			"Hello\nWorld":                   {"Hello", "\n", "World"},
			"Hello, WORLD!! How's it going?": {"Hello", ",", " WORLD", "!!", " How", "'s", " it", " going", "?"},
		}

		for s, want := range cases {
			got := slices.Collect(tokenizer.split(s))
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("no match (-theirs +ours):\n%s", diff)
			}
		}
	})

	t.Run("roundtriping 0x00-0xFF", func(t *testing.T) {
		t.Parallel()

		for b := 0x00; b <= 0xFF; b++ {
			input := string(rune(b))
			ids, err := tokenizer.Encode(input, false)
			if err != nil {
				t.Errorf("failed to encode rune 0x%02X: %v", b, err)
				continue
			}

			decoded, err := tokenizer.Decode(ids)
			if err != nil {
				t.Errorf("failed to decode rune 0x%02X: %v", b, err)
				continue
			}

			if b == 0x00 {
				if len(decoded) != 0 {
					t.Errorf("Decode(Encode(0x00)) should be empty, got %v", ids)
				}
				continue
			}

			if decoded != input {
				t.Errorf("rune 0x%02X failed roundtrip: got %q, want %q", b, decoded, input)
			}
		}
	})
}

// spmBPE builds a SentencePiece-style BPE tokenizer for testing.
//
// Models that use SentencePiece BPE differ from GPT-2 BPE in how they
// handle spaces: the vocabulary stores ▁ (U+2581) instead of GPT-2's
// shifted-byte encoding (0x0100–0x0143). Without WithSentencePieceNormalizer,
// spaces are mapped through the GPT-2 byte table which produces wrong token
// IDs for any vocabulary that uses ▁-prefixed tokens. The decode path has
// the inverse problem: high codepoints like CJK characters and ▁ itself
// would be mangled by the GPT-2 reverse mapping instead of being passed
// through (or converted to spaces in the ▁ case).
func spmBPE(t testing.TB) BytePairEncoding {
	t.Helper()

	tokens := []string{
		// Control tokens (low IDs, as in real SentencePiece vocabs)
		"<pad>",    // 0
		"<eos>",    // 1
		"<bos>",    // 2
		"<|start>", // 3 - asymmetric open/close special tokens
		"<end|>",   // 4
		"<|q>",     // 5 - short special token (like <|"|>)

		// ▁-prefixed word tokens (the core of what SPM BPE changes)
		"▁hello", // 6
		"▁world", // 7
		"hello",  // 8
		"▁Run",   // 9
		"▁a",     // 10

		// Punctuation and structure
		",", // 11
		"!", // 12
		":", // 13
		"{", // 14
		"}", // 15

		// Whitespace separator
		"▁", // 16

		// Subword tokens used in tool-declaration-like patterns
		"description", // 17
		"▁command",    // 18
		"declaration", // 19

		// Unicode token for decode passthrough testing (must be > U+0143
		// to exercise the SPM decode path rather than GPT-2 byte reversal)
		"▁中文", // 20

		// Unicode tokens with codepoints in the GPT-2 byte range (0x0100-0x0142).
		// Without the SPM decode path, these get mangled by GPT-2 byte reversal.
		"ą", // 21 (U+0105) — would become 0x05 via GPT-2 reversal
		"ę", // 22 (U+0119) — would become 0x19
		"ć", // 23 (U+0107) — would become 0x07
		"ł", // 24 (U+0142) — would become 0xA0

		// Byte fallback tokens (SentencePiece BYTE type)
		"<0x00>", // 25
		"<0x01>", // 26
	}

	// Add all 256 byte tokens starting at index 27
	for b := 2; b < 256; b++ {
		tokens = append(tokens, fmt.Sprintf("<0x%02X>", b))
	}

	types := make([]int32, len(tokens))
	for i := range types {
		types[i] = TOKEN_TYPE_NORMAL
	}
	types[0] = TOKEN_TYPE_CONTROL      // <pad>
	types[1] = TOKEN_TYPE_CONTROL      // <eos>
	types[2] = TOKEN_TYPE_CONTROL      // <bos>
	types[3] = TOKEN_TYPE_USER_DEFINED // <|start>
	types[4] = TOKEN_TYPE_USER_DEFINED // <end|>
	types[5] = TOKEN_TYPE_USER_DEFINED // <|q>
	for i := 21; i < len(types); i++ {
		types[i] = TOKEN_TYPE_BYTE
	}

	return NewBytePairEncodingWithOptions(
		&Vocabulary{
			Values: tokens,
			Types:  types,
			BOS:    []int32{2},
			EOS:    []int32{1},
			AddBOS: false,
		},
		// Empty pretokenizer list: falls back to the default pattern.
		// Real SentencePiece BPE models are configured this way.
		[]string{},
		WithSentencePieceNormalizer(),
	)
}

func TestSentencePieceBPE(t *testing.T) {
	tok := spmBPE(t)

	// Test 1: Space-to-▁ normalization and roundtrip.
	//
	// SentencePiece BPE has no pretokenizer — the BPE merges handle word
	// boundaries via ▁ markers. With no merges in the test vocab, multi-char
	// tokens won't be found, but the roundtrip must still be lossless.
	t.Run("spm space normalization roundtrip", func(t *testing.T) {
		t.Parallel()

		for _, input := range []string{
			"hello",
			" hello",
			"hello, world!",
			"  leading spaces",
			"multiple   spaces",
		} {
			ids, err := tok.Encode(input, false)
			if err != nil {
				t.Fatalf("Encode(%q): %v", input, err)
			}
			if len(ids) == 0 {
				t.Fatalf("Encode(%q) returned empty IDs", input)
			}

			got, err := tok.Decode(ids)
			if err != nil {
				t.Fatalf("Decode(%v): %v", ids, err)
			}
			if got != input {
				t.Errorf("roundtrip %q: Decode(Encode) = %q", input, got)
			}
		}
	})

	// Test 2: Special tokens interleaved with SPM-normalized text.
	//
	// This mimics tool declaration patterns like:
	//   <|tool>declaration:bash{description:<|"|>Run a command<|"|>}<tool|>
	// where special tokens (<|tool>, <|"|>, <tool|>) must be extracted
	// first, then the remaining text fragments go through SPM normalization.
	t.Run("special tokens with spm text fragments", func(t *testing.T) {
		t.Parallel()

		input := "<|start>declaration:description:<|q> Run a command<|q>}<end|>"
		ids, err := tok.Encode(input, false)
		if err != nil {
			t.Fatal(err)
		}

		// Special tokens should be extracted as single IDs at the right positions.
		// The text between them is SPM-normalized and BPE-encoded (specific IDs
		// depend on merges, so we verify the special token positions + roundtrip).
		specialPositions := map[int32]bool{3: true, 4: true, 5: true} // <|start>, <end|>, <|q>
		foundSpecials := 0
		for _, id := range ids {
			if specialPositions[id] {
				foundSpecials++
			}
		}
		if foundSpecials != 4 { // <|start>, <|q>, <|q>, <end|>
			t.Errorf("expected 4 special tokens, found %d in %v", foundSpecials, ids)
		}

		// First token must be <|start>(3), last must be <end|>(4)
		if ids[0] != 3 {
			t.Errorf("first token = %d, want 3 (<|start>)", ids[0])
		}
		if ids[len(ids)-1] != 4 {
			t.Errorf("last token = %d, want 4 (<end|>)", ids[len(ids)-1])
		}
	})

	// Test 3: Byte fallback for characters not in the vocabulary.
	//
	// SentencePiece vocabs include <0xHH> byte tokens for every byte value.
	// When a character (e.g. "ą" = U+0105 = C4 85) isn't in the vocab as a
	// direct token, the encoder must fall back to its UTF-8 bytes:
	// <0xC4> <0x85>. Without this fallback, the character is silently dropped.
	// See: https://github.com/ollama/ollama/issues/15229
	t.Run("byte fallback for unknown chars", func(t *testing.T) {
		t.Parallel()

		// "ą" is not in the vocab — should fall back to byte tokens
		ids, err := tok.Encode("ą", false)
		if err != nil {
			t.Fatalf("Encode(ą): %v", err)
		}
		if len(ids) == 0 {
			t.Fatal("Encode(ą) returned empty IDs — character was silently dropped")
		}

		got, err := tok.Decode(ids)
		if err != nil {
			t.Fatalf("Decode: %v", err)
		}
		if got != "ą" {
			t.Errorf("roundtrip = %q, want %q", got, "ą")
		}
	})

	// Test 4: Byte fallback preserves known tokens around unknown chars.
	t.Run("byte fallback mixed with known tokens", func(t *testing.T) {
		t.Parallel()

		// "hello" is in vocab, "é" is not
		ids, err := tok.Encode("helloé", false)
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}

		got, err := tok.Decode(ids)
		if err != nil {
			t.Fatalf("Decode: %v", err)
		}
		if got != "helloé" {
			t.Errorf("roundtrip = %q, want %q", got, "helloé")
		}
	})

	// Test 5: Decode doesn't mangle Unicode in the GPT-2 byte range.
	//
	// Characters like ą (U+0105), ę (U+0119), ć (U+0107), ł (U+0142) have
	// codepoints in the 0x0100-0x0142 range that GPT-2 byte reversal would
	// remap to control characters. SentencePiece decode must pass them through.
	t.Run("decode unicode in gpt2 byte range", func(t *testing.T) {
		t.Parallel()

		// Token IDs 21-24 are ą, ę, ć, ł
		ids := []int32{21, 22, 23, 24}
		got, err := tok.Decode(ids)
		if err != nil {
			t.Fatalf("Decode: %v", err)
		}
		if got != "ąęćł" {
			t.Errorf("Decode = %q, want %q", got, "ąęćł")
		}
	})

	// Test 6: Decode handles non-GPT2 Unicode correctly.
	//
	// GPT-2 BPE decode reverses the byte→codepoint shift for runes in
	// 0x0100–0x0143. But SentencePiece vocabs store real Unicode (CJK,
	// accented chars, etc.) which have codepoints well above 0x0143.
	// Without the > 0x0143 passthrough in Decode, these would be mangled
	// by the GPT-2 reverse mapping (e.g., written as raw bytes instead
	// of the original characters).
	t.Run("decode non-gpt2 unicode passthrough", func(t *testing.T) {
		t.Parallel()

		cases := map[string][]int32{
			" 中文": {20}, // ▁→space, then CJK passes through as-is
		}

		for want, ids := range cases {
			got, err := tok.Decode(ids)
			if err != nil {
				t.Fatalf("Decode(%v): %v", ids, err)
			}
			if got != want {
				t.Errorf("Decode(%v) = %q, want %q", ids, got, want)
			}
		}
	})
}

func BenchmarkBytePairEncoding(b *testing.B) {
	tokenizer := llama(b)
	bts, err := os.ReadFile(filepath.Join("testdata", "war-and-peace.txt"))
	if err != nil {
		b.Fatal(err)
	}

	for i := range 8 {
		n := min(int(math.Pow10(i)), len(bts))
		bts := bts[:n]
		b.Run("encode"+strconv.Itoa(n), func(b *testing.B) {
			b.ResetTimer()
			for b.Loop() {
				_, err := tokenizer.Encode(string(bts), true)
				if err != nil {
					b.Fatal(err)
				}
			}
		})

		b.Run("decode"+strconv.Itoa(n), func(b *testing.B) {
			ids, err := tokenizer.Encode(string(bts), true)
			if err != nil {
				b.Fatal(err)
			}

			b.ResetTimer()
			for b.Loop() {
				_, err := tokenizer.Decode(ids)
				if err != nil {
					b.Fatal(err)
				}
			}
		})

		b.Run("split"+strconv.Itoa(n), func(b *testing.B) {
			b.ResetTimer()
			for b.Loop() {
				slices.Collect(tokenizer.split(string(bts)))
			}
		})
	}
}

func TestBytePairEncodingSplitMultipleRegexpsPreservesOffsets(t *testing.T) {
	t.Parallel()

	bpe := NewBytePairEncoding(
		nil,
		`(?:\r?\n)+(?!\r?\n)`,
		`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
	)

	input := "One line\nTwo lines\n\nThree"
	got := slices.Collect(bpe.split(input))
	want := []string{"One", " line", "\n", "Two", " lines", "\n\n", "Three"}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("split mismatch (-want +got):\n%s", diff)
	}
}

func TestBytePairEncodingSplitRefactPreservesOffsets(t *testing.T) {
	t.Parallel()

	bpe := NewBytePairEncoding(
		nil,
		`\p{N}`,
		`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`,
	)

	input := "One line\nTwo lines\n\nThree"
	got := slices.Collect(bpe.split(input))
	want := []string{"One", " line", "\n", "Two", " lines", "\n", "\n", "Three"}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("split mismatch (-want +got):\n%s", diff)
	}
}

func TestBytePairEncodingSplitDeepSeekV3PreservesOffsets(t *testing.T) {
	t.Parallel()

	bpe := NewBytePairEncoding(
		nil,
		"\\p{N}{1,3}",
		`[一-龥぀-ゟ゠-ヿ]+`,
		"[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
	)

	input := "One line\nTwo lines\n\nThree"
	got := slices.Collect(bpe.split(input))
	want := []string{"One", " line", "\n", "Two", " lines", "\n\n", "Three"}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("split mismatch (-want +got):\n%s", diff)
	}
}

func TestSplit(t *testing.T) {
	cases := []struct {
		name string
		patterns,
		want []string
	}{
		{
			name: "default",
			want: []string{"Hello", ",", " WORLD", "!!", " How", "'s", " it", " going", "?", " 123", " 一二三"},
		},
		{
			name: "unicode",
			patterns: []string{
				"\\p{N}{1,3}",
				`[一-龥぀-ゟ゠-ヿ]+`,
				"[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
			},
			want: []string{"Hello", ",", " WORLD", "!!", " How", "'s", " it", " going", "?", " ", "123", " ", "一二三"},
		},
		{
			name: "individual digits",
			patterns: []string{
				"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
			},
			want: []string{"Hello", ",", " WORLD", "!!", " How", "'s", " it", " going", "?", " ", "1", "2", "3", " 一二三"},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			tokenizer := NewBytePairEncoding(nil, tt.patterns...)
			if diff := cmp.Diff(tt.want, slices.Collect(tokenizer.split("Hello, WORLD!! How's it going? 123 一二三"))); diff != "" {
				t.Errorf("no match (-theirs +ours):\n%s", diff)
			}
		})
	}
}

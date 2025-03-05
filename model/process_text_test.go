package model

import (
	"bufio"
	"encoding/json"
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

	f, err := os.Open(filepath.Join("testdata", "llama3.2", "encoder.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	vocab := make(map[string]int32)
	if err := json.NewDecoder(f).Decode(&vocab); err != nil {
		t.Fatal(err)
	}

	types := make([]uint32, len(vocab))
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

	f, err = os.Open(filepath.Join("testdata", "llama3.2", "vocab.bpe"))
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
		`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		&Vocabulary{
			Values: tokens,
			Types:  types,
			Merges: merges,
		},
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
			for range b.N {
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
			for range b.N {
				_, err := tokenizer.Decode(ids)
				if err != nil {
					b.Fatal(err)
				}
			}
		})

		b.Run("split"+strconv.Itoa(n), func(b *testing.B) {
			b.ResetTimer()
			for range b.N {
				slices.Collect(tokenizer.split(string(bts)))
			}
		})
	}
}

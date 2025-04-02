package model

import (
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/ollama/ollama/convert/sentencepiece"
)

func loadSentencePieceVocab(t *testing.T) SentencePieceModel {
	t.Helper()

	bts, err := os.ReadFile(filepath.Join("testdata", "gemma2", "tokenizer.model"))
	if err != nil {
		t.Fatal(err)
	}

	var spm sentencepiece.ModelProto
	if err := proto.Unmarshal(bts, &spm); err != nil {
		t.Fatal(err)
	}

	var v Vocabulary

	for _, piece := range spm.GetPieces() {
		v.Values = append(v.Values, piece.GetPiece())
		v.Scores = append(v.Scores, piece.GetScore())
		switch t := piece.GetType(); t {
		case sentencepiece.ModelProto_SentencePiece_UNKNOWN,
			sentencepiece.ModelProto_SentencePiece_CONTROL,
			sentencepiece.ModelProto_SentencePiece_UNUSED,
			sentencepiece.ModelProto_SentencePiece_BYTE:
			v.Types = append(v.Types, uint32(t))
		default:
			tt := uint32(sentencepiece.ModelProto_SentencePiece_NORMAL)
			// todo parse the special tokens file
			//   - this will roundtrip correctly but the <start_of_turn> and
			//     <end_of_turn> tokens aren't processed
			v.Types = append(v.Types, tt)
		}
	}

	return NewSentencePieceModel(&v)
}

func TestSentencePieceEncode(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelDebug}))
	slog.SetDefault(logger)

	tokenizer := loadSentencePieceVocab(t)

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
			"你好",
			"Hello 你好 world!",
			"Special characters: !@#$%^&*()_+-=[]{}|;':\",./<>?",
			"Multilingual: 你好 こんにちは Привет Hola مرحبا",
			"Numbers and symbols: 123456789 +- */",
			"Special tokens: <bos> text <eos>",
			"Code snippets: func main() { fmt.Println(\"Hello World\") }",
			"Long text: " + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " +
				"Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " +
				"Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
		}

		for _, want := range cases {
			ids, err := tokenizer.Encode(want, true)
			if err != nil {
				t.Fatal(err)
			}

			if got, err := tokenizer.Decode(ids); err != nil {
				t.Fatal(err)
			} else if got != want {
				t.Errorf("got %q, want %q [%#v]", got, want, ids)
			}
		}
	})

	t.Run("special tokens", func(t *testing.T) {
		type candidate struct {
			token string
			ids   []int32
		}

		cases := []candidate{
			{"<bos>", []int32{2}},
			{"<eos>", []int32{1}},
		}

		for _, want := range cases {
			ids, err := tokenizer.Encode(want.token, true)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(ids, want.ids) {
				t.Errorf("got %#v, want %#v", ids, want.ids)
			}
		}
	})
}

func TestSentencePieceEncodeAndPrintTokens(t *testing.T) {
	tokenizer := loadSentencePieceVocab(t)
	tt := []struct {
		name  string
		input string
		want  []string
	}{
		{
			name:  "basic",
			input: `How are you doing? how's today's weather`,
			want:  []string{"How", " are", " you", " doing", "?", " how", "'", "s", " today", "'", "s", " weather"},
		},
		{
			name: "whitespace",
			input: `Below is a list of items:
    * **Item 1**
    * **Item 2**
    * **Item 3**`,
			want: []string{"Below", " is", " a", " list", " of", " items", ":", "\n", "    ", "*", " **", "Item", " ", "1", "**", "\n", "    ", "*", " **", "Item", " ", "2", "**", "\n", "    ", "*", " **", "Item", " ", "3", "**"},
		},
	}

	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			ids, err := tokenizer.Encode(tc.input, true)
			if err != nil {
				t.Fatal(err)
			}

			for i, id := range ids {
				token, err := tokenizer.Decode([]int32{id})
				if err != nil {
					t.Errorf("Failed to decode token ID %d: %v", id, err)
				}
				if token != tc.want[i] {
					t.Errorf("got %q, want %q at position %d", token, tc.want[i], i)
				}
			}
		})
	}
}

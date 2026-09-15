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

func loadSentencePieceVocab(t *testing.T) SentencePiece {
	t.Helper()

	bts, err := os.ReadFile(filepath.Join("..", "..", "model", "testdata", "gemma2", "tokenizer.model"))
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
			v.Types = append(v.Types, int32(t))
		default:
			tt := int32(sentencepiece.ModelProto_SentencePiece_NORMAL)
			// todo parse the special tokens file
			//   - this will roundtrip correctly but the <start_of_turn> and
			//     <end_of_turn> tokens aren't processed
			v.Types = append(v.Types, tt)
		}
	}

	return NewSentencePiece(&v)
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

func TestSentencePieceDecodeByteTokens(t *testing.T) {
	vocab := &Vocabulary{
		Values: []string{
			"normal",
			"<0xEA>",
			"<0x41>",
			"<0xC3>",
			"<0xA3>",
		},
		Types: []int32{
			TOKEN_TYPE_NORMAL,
			TOKEN_TYPE_BYTE,
			TOKEN_TYPE_BYTE,
			TOKEN_TYPE_BYTE,
			TOKEN_TYPE_BYTE,
		},
		Scores: []float32{0, 0, 0, 0, 0},
	}

	spm := NewSentencePiece(vocab)

	tests := []struct {
		name     string
		ids      []int32
		expected string
	}{
		{
			name:     "single byte token",
			ids:      []int32{1},
			expected: "\xea",
		},
		{
			name:     "ASCII byte token",
			ids:      []int32{2},
			expected: "A",
		},
		{
			name:     "multiple byte tokens forming UTF-8 character",
			ids:      []int32{3, 4},
			expected: "ã",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := spm.Decode(tt.ids)
			if err != nil {
				t.Errorf("failed to decode token IDs %v: %v", tt.ids, err)
			}
			if result != tt.expected {
				t.Errorf("got %q, want %q", result, tt.expected)
			}
		})
	}
}

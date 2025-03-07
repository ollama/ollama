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

	preTokenizer := `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

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

	return NewSentencePieceModel(preTokenizer, &v)
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

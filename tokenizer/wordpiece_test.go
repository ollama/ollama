package tokenizer

import (
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestWordPiece(t *testing.T) {
	wpm := NewWordPiece(
		&Vocabulary{
			Values: []string{"[UNK]", "[CLS]", "[SEP]", "▁hello", "▁world", "s", "▁!", "▁@", "▁#"},
			AddBOS: true,
			AddEOS: true,
			BOS:    []int32{1},
			EOS:    []int32{2},
		},
		true, // lowercase
	)

	ids, err := wpm.Encode("Hello world!", true)
	if err != nil {
		t.Fatal(err)
	}

	if diff := cmp.Diff([]int32{1, 3, 4, 6, 2}, ids); diff != "" {
		t.Errorf("unexpected ids (-want +got):\n%s", diff)
	}

	words, err := wpm.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}

	if diff := cmp.Diff("[CLS] hello world! [SEP]", words); diff != "" {
		t.Errorf("unexpected words (-want +got):\n%s", diff)
	}
}

func TestWordPieceEncodeReturnsErrorWhenUnkMissing(t *testing.T) {
	// Regression test for issue #15174: WordPiece used to silently emit
	// -1 when a word could not be tokenized and [UNK] was absent from the
	// vocab. That -1 then crashed the embedding forward pass with a
	// GGML_ASSERT in ggml_get_rows.
	wpm := NewWordPiece(
		&Vocabulary{
			Values: []string{"[CLS]", "[SEP]", "▁hello"},
			BOS:    []int32{0},
			EOS:    []int32{1},
		},
		true,
	)

	if _, err := wpm.Encode("hello world!", false); err == nil {
		t.Error("expected error when word is OOV and [UNK] is missing, got nil")
	}
}

func TestWordPieceWords(t *testing.T) {
	var wpm WordPiece

	basic := slices.Collect(wpm.words("Hey friend!     How are you?!?"))
	if diff := cmp.Diff([]string{"Hey", "friend", "!", "How", "are", "you", "?", "!", "?"}, basic); diff != "" {
		t.Errorf("unexpected words (-want +got):\n%s", diff)
	}

	chinese := slices.Collect(wpm.words("野口里佳 Noguchi Rika"))
	if diff := cmp.Diff([]string{"野", "口", "里", "佳", "Noguchi", "Rika"}, chinese); diff != "" {
		t.Errorf("unexpected words (-want +got):\n%s", diff)
	}
}

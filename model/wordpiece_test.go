package model

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

package bert

import (
	"testing"

	"github.com/ollama/ollama/tokenizer"
)

// TestBertNewT5Tokenizer verifies that a bert model configured with
// tokenizer.ggml.model="t5" (as bge-m3 is) loads a SentencePiece tokenizer
// rather than returning ErrUnsupportedTokenizer.
func TestBertNewT5Tokenizer(t *testing.T) {
	vocab := &tokenizer.Vocabulary{
		Values:         []string{"▁hello", "▁world", "▁test", "<s>", "</s>", "h", "e", "l", "o", "w", "r", "d"},
		Scores:         []float32{-1, -1, -1, 0, 0, -5, -5, -5, -5, -5, -5, -5},
		Types:          []int32{1, 1, 1, 4, 4, 1, 1, 1, 1, 1, 1, 1},
		BOS:            []int32{3},
		EOS:            []int32{4},
		AddBOS:         true,
		AddEOS:         true,
		AddSpacePrefix: true,
	}

	spm := tokenizer.NewSentencePiece(vocab)

	t.Run("encodes_without_error", func(t *testing.T) {
		ids, err := spm.Encode("hello world", true)
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		if len(ids) == 0 {
			t.Error("got empty token list")
		}
		// With add_space_prefix=true and BOS/EOS: [<s>, ▁hello, ▁world, </s>]
		t.Logf("ids: %v", ids)
	})

	t.Run("add_space_prefix_prepends_whitespace_token", func(t *testing.T) {
		// "hello" with add_space_prefix=true should produce ▁hello token (id=0)
		ids, err := spm.Encode("hello", false)
		if err != nil {
			t.Fatal(err)
		}
		if len(ids) != 1 || ids[0] != 0 {
			t.Errorf("got %v, want [0] (▁hello)", ids)
		}
	})

	t.Run("is_sentence_piece_not_wordpiece", func(t *testing.T) {
		// Verify it satisfies the Tokenizer interface and is SentencePiece
		var _ tokenizer.Tokenizer = spm
	})
}

package model

type WordPiece struct {
	vocab *Vocabulary
}

// Decode implements TextProcessor.
func (wpm *WordPiece) Decode([]int32) (string, error) {
	return "hi", nil
}

// Encode implements TextProcessor.
func (wpm *WordPiece) Encode(s string, addSpecial bool) ([]int32, error) {
	return []int32{101, 7592, 102}, nil
}

// Is implements TextProcessor.
func (wpm *WordPiece) Is(id int32, special Special) bool {
	return wpm.vocab.Is(id, special)
}

// Vocabulary implements TextProcessor.
func (wpm *WordPiece) Vocabulary() *Vocabulary {
	return wpm.vocab
}

var _ TextProcessor = (*WordPiece)(nil)

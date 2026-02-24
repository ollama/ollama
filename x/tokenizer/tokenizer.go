//go:build mlx

// tokenizer.go - BPE and SentencePiece tokenizer for HuggingFace models
//
// Based on standard BPE algorithm (Sennrich et al. 2015) with:
// - GPT-2 byte-level encoding (OpenAI tiktoken)
// - HuggingFace tokenizer.json pretokenizer patterns
// - SentencePiece ▁-style space handling

package tokenizer

import "regexp"

// TokenizerType identifies the tokenization algorithm
type TokenizerType int

const (
	TokenizerBPE           TokenizerType = iota // GPT-2 style byte-level BPE
	TokenizerSentencePiece                      // SentencePiece with ▁ for spaces
)

// Vocabulary holds the tokenizer vocabulary and merges
type Vocabulary struct {
	Values  []string
	Reverse map[string]int32
	Merges  map[string]int

	BOS    int32
	EOS    []int32 // Multiple EOS tokens supported (e.g., Gemma has <eos> and <end_of_turn>)
	PAD    int32   // Padding token (often <|endoftext|> or <pad>)
	AddBOS bool
	AddEOS bool

	// Precomputed byte token IDs for <0xNN> fallback (256 entries, -1 if not found)
	byteTokens [256]int32
}

// Tokenizer handles BPE and SentencePiece tokenization
type Tokenizer struct {
	vocab               *Vocabulary
	pretokenizer        *regexp.Regexp
	specialTokens       map[string]int32 // Special tokens for direct lookup
	sortedSpecialTokens []string         // Special tokens sorted by length, longest first
	typ                 TokenizerType    // Algorithm type
}

// Precomputed GPT-2 byte-level encoding table
// Maps byte values to their encoded rune equivalents
var byteToRune [256]rune

func init() {
	for b := 0; b < 256; b++ {
		r := rune(b)
		switch {
		case r == 0x00ad:
			r = 0x0143
		case r <= 0x0020:
			r = r + 0x0100
		case r >= 0x007f && r <= 0x00a0:
			r = r + 0x00a2
		}
		byteToRune[b] = r
	}
}

// VocabSize returns the vocabulary size
func (t *Tokenizer) VocabSize() int {
	return len(t.vocab.Values)
}

// BOS returns the beginning of sequence token ID
func (t *Tokenizer) BOS() int32 {
	return t.vocab.BOS
}

// EOS returns the first end of sequence token ID (for backwards compatibility)
func (t *Tokenizer) EOS() int32 {
	if len(t.vocab.EOS) > 0 {
		return t.vocab.EOS[0]
	}
	return -1
}

// EOSTokens returns all end of sequence token IDs
func (t *Tokenizer) EOSTokens() []int32 {
	return t.vocab.EOS
}

// PAD returns the padding token ID, or -1 if not set
func (t *Tokenizer) PAD() int32 {
	return t.vocab.PAD
}

// IsEOS returns true if the token ID is an end of sequence token
func (t *Tokenizer) IsEOS(id int32) bool {
	for _, eos := range t.vocab.EOS {
		if id == eos {
			return true
		}
	}
	return false
}

// GetSpecialToken returns the token ID for a special token string
func (t *Tokenizer) GetSpecialToken(name string) (int32, bool) {
	id, ok := t.specialTokens[name]
	return id, ok
}

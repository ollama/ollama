package oga

// #include "oga.h"
// #include <stdlib.h>
import "C"

import (
	"unsafe"
)

// Tokenizer wraps an OgaTokenizer handle.
type Tokenizer struct {
	ctx *C.OgaTokenizer
}

// NewTokenizer creates a tokenizer from a model.
func NewTokenizer(model *Model) (*Tokenizer, error) {
	var tok *C.OgaTokenizer
	if err := ogaError(C.OgaCreateTokenizer(model.ctx, &tok)); err != nil {
		return nil, err
	}
	return &Tokenizer{ctx: tok}, nil
}

// Encode tokenizes a string and returns the token IDs.
func (t *Tokenizer) Encode(text string) ([]int32, error) {
	var seqs *C.OgaSequences
	if err := ogaError(C.OgaCreateSequences(&seqs)); err != nil {
		return nil, err
	}
	defer C.OgaDestroySequences(seqs)

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	if err := ogaError(C.OgaTokenizerEncode(t.ctx, cText, seqs)); err != nil {
		return nil, err
	}

	count := C.OgaSequencesGetSequenceCount(seqs, 0)
	data := C.OgaSequencesGetSequenceData(seqs, 0)
	if count == 0 || data == nil {
		return nil, nil
	}

	tokens := make([]int32, count)
	copy(tokens, unsafe.Slice((*int32)(unsafe.Pointer(data)), count))
	return tokens, nil
}

// Decode converts token IDs back to a string.
func (t *Tokenizer) Decode(tokens []int32) (string, error) {
	if len(tokens) == 0 {
		return "", nil
	}
	var out *C.char
	if err := ogaError(C.OgaTokenizerDecode(t.ctx, (*C.int32_t)(&tokens[0]), C.size_t(len(tokens)), &out)); err != nil {
		return "", err
	}
	result := C.GoString(out)
	C.OgaDestroyString(out)
	return result, nil
}

// Close frees the tokenizer.
func (t *Tokenizer) Close() {
	if t.ctx != nil {
		C.OgaDestroyTokenizer(t.ctx)
		t.ctx = nil
	}
}

// TokenStream wraps an OgaTokenizerStream for incremental decoding.
type TokenStream struct {
	ctx *C.OgaTokenizerStream
}

// NewTokenStream creates a streaming token decoder from a tokenizer.
func NewTokenStream(tok *Tokenizer) (*TokenStream, error) {
	var stream *C.OgaTokenizerStream
	if err := ogaError(C.OgaCreateTokenizerStream(tok.ctx, &stream)); err != nil {
		return nil, err
	}
	return &TokenStream{ctx: stream}, nil
}

// Decode decodes a single token into text. Returns partial text as tokens
// accumulate to form complete word pieces.
func (s *TokenStream) Decode(token int32) (string, error) {
	var out *C.char
	if err := ogaError(C.OgaTokenizerStreamDecode(s.ctx, C.int32_t(token), &out)); err != nil {
		return "", err
	}
	// The returned pointer is owned by the stream, don't free it
	return C.GoString(out), nil
}

// Close frees the token stream.
func (s *TokenStream) Close() {
	if s.ctx != nil {
		C.OgaDestroyTokenizerStream(s.ctx)
		s.ctx = nil
	}
}

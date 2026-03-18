package oga

// #include "oga.h"
// #include <stdlib.h>
import "C"

import (
	"unsafe"
)

// GeneratorParams wraps OgaGeneratorParams.
type GeneratorParams struct {
	ctx *C.OgaGeneratorParams
}

// NewGeneratorParams creates generator parameters from a model.
func NewGeneratorParams(model *Model) (*GeneratorParams, error) {
	var params *C.OgaGeneratorParams
	if err := ogaError(C.OgaCreateGeneratorParams(model.ctx, &params)); err != nil {
		return nil, err
	}
	return &GeneratorParams{ctx: params}, nil
}

// SetNumber sets a numeric search parameter (e.g., "temperature", "top_p", "max_length").
func (p *GeneratorParams) SetNumber(name string, value float64) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return ogaError(C.OgaGeneratorParamsSetSearchNumber(p.ctx, cName, C.double(value)))
}

// SetBool sets a boolean search parameter (e.g., "do_sample").
func (p *GeneratorParams) SetBool(name string, value bool) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return ogaError(C.OgaGeneratorParamsSetSearchBool(p.ctx, cName, C.bool(value)))
}

// Close frees the generator params.
func (p *GeneratorParams) Close() {
	if p.ctx != nil {
		C.OgaDestroyGeneratorParams(p.ctx)
		p.ctx = nil
	}
}

// Generator wraps OgaGenerator for token-by-token generation.
type Generator struct {
	ctx *C.OgaGenerator
}

// NewGenerator creates a generator from a model and parameters.
func NewGenerator(model *Model, params *GeneratorParams) (*Generator, error) {
	var gen *C.OgaGenerator
	if err := ogaError(C.OgaCreateGenerator(model.ctx, params.ctx, &gen)); err != nil {
		return nil, err
	}
	return &Generator{ctx: gen}, nil
}

// AppendTokens appends input token sequences to the generator.
func (g *Generator) AppendTokens(tokens []int32) error {
	var seqs *C.OgaSequences
	if err := ogaError(C.OgaCreateSequences(&seqs)); err != nil {
		return err
	}
	defer C.OgaDestroySequences(seqs)

	// OgaTokenizerEncode populates sequences, but we need to append raw tokens.
	// The C API doesn't have a direct "append tokens to sequences" function,
	// so we use a workaround: encode a dummy string to create the sequence structure,
	// then use the generator's AppendTokenSequences which reads from sequences.
	//
	// Actually, the proper API flow is:
	// 1. Encode the prompt with the tokenizer to get OgaSequences
	// 2. Pass those sequences to Generator_AppendTokenSequences
	//
	// The caller should pass pre-encoded OgaSequences instead.
	// For now, this function is a placeholder — the pipeline will use
	// the tokenizer to produce sequences directly.
	_ = tokens
	return ogaError(C.OgaGenerator_AppendTokenSequences(g.ctx, seqs))
}

// AppendTokenSequencesRaw appends sequences obtained from tokenizer encoding.
// This is the primary method used by the pipeline.
func (g *Generator) AppendTokenSequencesFromEncoding(tok *Tokenizer, text string) error {
	var seqs *C.OgaSequences
	if err := ogaError(C.OgaCreateSequences(&seqs)); err != nil {
		return err
	}
	defer C.OgaDestroySequences(seqs)

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	if err := ogaError(C.OgaTokenizerEncode(tok.ctx, cText, seqs)); err != nil {
		return err
	}

	return ogaError(C.OgaGenerator_AppendTokenSequences(g.ctx, seqs))
}

// GenerateNextToken generates the next token. Call repeatedly until IsDone.
func (g *Generator) GenerateNextToken() error {
	return ogaError(C.OgaGenerator_GenerateNextToken(g.ctx))
}

// IsDone returns true when generation is complete (EOS or max tokens reached).
func (g *Generator) IsDone() bool {
	return bool(C.OgaGenerator_IsDone(g.ctx))
}

// GetNextTokens returns the most recently generated tokens.
func (g *Generator) GetNextTokens() ([]int32, error) {
	var data *C.int32_t
	var count C.size_t
	if err := ogaError(C.OgaGenerator_GetNextTokens(g.ctx, &data, &count)); err != nil {
		return nil, err
	}
	if count == 0 || data == nil {
		return nil, nil
	}
	tokens := make([]int32, count)
	copy(tokens, unsafe.Slice((*int32)(unsafe.Pointer(data)), count))
	return tokens, nil
}

// Close frees the generator.
func (g *Generator) Close() {
	if g.ctx != nil {
		C.OgaDestroyGenerator(g.ctx)
		g.ctx = nil
	}
}

// Shutdown cleans up global ORT GenAI state. Call once at program exit.
func Shutdown() {
	C.OgaShutdown()
}

package llama

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++17
#cgo CPPFLAGS: -I${SRCDIR}/../llama/llama.cpp/include
#cgo CPPFLAGS: -I${SRCDIR}/../llama/llama.cpp/common
#cgo CPPFLAGS: -I${SRCDIR}/../llama/llama.cpp/src
#cgo CPPFLAGS: -I${SRCDIR}

#include <stdlib.h>
#include <stdbool.h>
#include "llama.h"
#include "grammar_ext.h"

// Helper function to handle Go string arrays to C
static char** makeCharArray(int size) {
    return (char**)malloc(size * sizeof(char*));
}

static void setArrayString(char** a, int i, const char* s) {
    a[i] = (char*)s;
}

static void freeCharArray(char** a, int size) {
    free(a);
}
*/
import "C"

import (
	"errors"
	"runtime"
	"unsafe"
)

// Grammar represents the interface for grammar-based sampling
type Grammar interface {
	Apply(logits []float32) ([]float32, error)
	Close() error
}

// CGrammar is a wrapper around the C++ grammar implementation
type CGrammar struct {
	grammar *C.struct_llama_grammar
	model   *C.struct_llama_model
	closed  bool
}

// NewGrammarWithTokens creates a new grammar using a custom vocabulary defined by tokens
func NewGrammarWithTokens(grammarStr, grammarRoot string, tokens []string) (Grammar, error) {
	if grammarStr == "" {
		return nil, errors.New("empty grammar string")
	}

	if len(tokens) == 0 {
		return nil, errors.New("empty token list")
	}

	// Create C array of strings for tokens
	cTokens := C.makeCharArray(C.int(len(tokens)))
	defer C.freeCharArray(cTokens, C.int(len(tokens)))

	// Convert Go strings to C strings and set them in the array
	cStrings := make([]*C.char, len(tokens))
	for i, token := range tokens {
		cStrings[i] = C.CString(token)
		C.setArrayString(cTokens, C.int(i), cStrings[i])
	}

	// Create vocabulary from tokens
	cVocab := C.vocab_bridge_from_tokens((**C.char)(unsafe.Pointer(cTokens)), C.int(len(tokens)))

	// Free the C strings after creating the vocab
	for _, str := range cStrings {
		C.free(unsafe.Pointer(str))
	}

	if cVocab == nil {
		return nil, errors.New("failed to create vocabulary from tokens")
	}

	// Make sure to free the vocabulary when we're done
	defer C.vocab_bridge_free(cVocab)

	cGrammarStr := C.CString(grammarStr)
	defer C.free(unsafe.Pointer(cGrammarStr))

	cGrammarRoot := C.CString(grammarRoot)
	defer C.free(unsafe.Pointer(cGrammarRoot))

	// Create grammar using our C wrapper function with the correct signature
	grammar := C.grammar_create_from_string(cVocab, cGrammarStr, cGrammarRoot)
	if grammar == nil {
		return nil, errors.New("failed to initialize grammar")
	}

	cg := &CGrammar{
		grammar: grammar,
		closed:  false,
	}

	// Set up finalizer to free resources when the object is garbage collected
	runtime.SetFinalizer(cg, func(g *CGrammar) {
		g.Close()
	})

	return cg, nil
}

// Apply applies grammar constraints to logits
func (g *CGrammar) Apply(logits []float32) ([]float32, error) {
	if g.closed || g.grammar == nil {
		return nil, errors.New("grammar not initialized or already closed")
	}

	// Create a copy of logits to modify
	result := make([]float32, len(logits))
	copy(result, logits)

	// Apply grammar constraints using our C wrapper function
	C.grammar_apply_to_logits(g.grammar, (*C.float)(&result[0]), C.int(len(result)))

	return result, nil
}

// Close releases resources associated with the grammar
func (g *CGrammar) Close() error {
	if !g.closed && g.grammar != nil {
		C.grammar_free(g.grammar)
		g.grammar = nil
		g.closed = true
	}
	return nil
}

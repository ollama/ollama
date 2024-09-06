package llm

// #cgo CFLAGS: -Illama.cpp -Illama.cpp/include -Illama.cpp/ggml/include
// #cgo darwin,arm64 LDFLAGS: ${SRCDIR}/build/darwin/arm64_static/src/libllama.a ${SRCDIR}/build/darwin/arm64_static/ggml/src/libggml.a -framework Accelerate -lstdc++
// #cgo darwin,amd64 LDFLAGS: ${SRCDIR}/build/darwin/x86_64_static/src/libllama.a ${SRCDIR}/build/darwin/x86_64_static/ggml/src/libggml.a -framework Accelerate -lstdc++
// #cgo windows,amd64 LDFLAGS: ${SRCDIR}/build/windows/amd64_static/src/libllama.a ${SRCDIR}/build/windows/amd64_static/ggml/src/libggml.a -static -lstdc++
// #cgo windows,arm64 LDFLAGS: ${SRCDIR}/build/windows/arm64_static/src/libllama.a ${SRCDIR}/build/windows/arm64_static/ggml/src/libggml.a -static -lstdc++
// #cgo linux,amd64 LDFLAGS: ${SRCDIR}/build/linux/x86_64_static/src/libllama.a ${SRCDIR}/build/linux/x86_64_static/ggml/src/libggml.a -lstdc++
// #cgo linux,arm64 LDFLAGS: ${SRCDIR}/build/linux/arm64_static/src/libllama.a ${SRCDIR}/build/linux/arm64_static/ggml/src/libggml.a -lstdc++
// #include <stdlib.h>
// #include "llama.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// SystemInfo is an unused example of calling llama.cpp functions using CGo
func SystemInfo() string {
	return C.GoString(C.llama_print_system_info())
}

func Quantize(infile, outfile string, ftype fileType) error {
	cinfile := C.CString(infile)
	defer C.free(unsafe.Pointer(cinfile))

	coutfile := C.CString(outfile)
	defer C.free(unsafe.Pointer(coutfile))

	params := C.llama_model_quantize_default_params()
	params.nthread = -1
	params.ftype = ftype.Value()

	if rc := C.llama_model_quantize(cinfile, coutfile, &params); rc != 0 {
		return fmt.Errorf("llama_model_quantize: %d", rc)
	}

	return nil
}

type loadedModel struct {
	model *C.struct_llama_model
}

func loadModel(modelfile string, vocabOnly bool) (*loadedModel, error) {
	// TODO figure out how to quiet down the logging so we don't have 2 copies of the model metadata showing up
	params := C.llama_model_default_params()
	params.vocab_only = C.bool(vocabOnly)

	cmodelfile := C.CString(modelfile)
	defer C.free(unsafe.Pointer(cmodelfile))

	model := C.llama_load_model_from_file(cmodelfile, params)
	if model == nil {
		return nil, fmt.Errorf("failed to load model %s", modelfile)
	}
	return &loadedModel{model}, nil
}

func freeModel(model *loadedModel) {
	C.llama_free_model(model.model)
}

func tokenize(model *loadedModel, content string) ([]int, error) {
	ccontent := C.CString(content)
	defer C.free(unsafe.Pointer(ccontent))

	tokenCount := len(content) + 2
	tokens := make([]C.int32_t, tokenCount)

	tokenCount = int(C.llama_tokenize(model.model, ccontent, C.int32_t(len(content)),
		&tokens[0], C.int32_t(tokenCount), true, true))
	if tokenCount < 0 {
		tokenCount = -tokenCount
		tokens = make([]C.int32_t, tokenCount)
		tokenCount = int(C.llama_tokenize(model.model, ccontent, C.int32_t(len(content)), &tokens[0],
			C.int32_t(tokenCount), true, true))

		if tokenCount < 0 {
			return nil, fmt.Errorf("failed to tokenize: %d", tokenCount)
		}
	} else if tokenCount == 0 {
		return nil, nil
	}
	ret := make([]int, tokenCount)
	for i := range tokenCount {
		ret[i] = int(tokens[i])
	}

	return ret, nil
}

func detokenize(model *loadedModel, tokens []int) string {
	var resp string
	for _, token := range tokens {
		buf := make([]C.char, 8)
		nTokens := C.llama_token_to_piece(model.model, C.int(token), &buf[0], 8, 0, true)
		if nTokens < 0 {
			buf = make([]C.char, -nTokens)
			nTokens = C.llama_token_to_piece(model.model, C.int(token), &buf[0], -nTokens, 0, true)
		}
		tokString := C.GoStringN(&buf[0], nTokens)
		resp += tokString
	}

	return resp
}

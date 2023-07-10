package llama

// #cgo LDFLAGS: -Lbuild -lllama -lm -lggml_static -lstdc++
// #cgo CXXFLAGS: -std=c++11
// #cgo darwin LDFLAGS: -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
// #include "build/llama.h"
// #include <stdlib.h>
import "C"

import (
	"errors"
	"unsafe"
)

type LLama struct {
	ctx         *C.struct_llama_context
}

func New(modelpath string, mo ModelOptions) (*LLama, error) {
	lparams := C.llama_context_default_params()
	lparams.embedding = C.bool(mo.Embeddings)
	lparams.f16_kv = C.bool(mo.F16Memory)
	lparams.low_vram = C.bool(mo.LowVRAM)
	lparams.n_batch = C.int(mo.NBatch)
	lparams.n_gpu_layers = C.int(mo.NGPULayers)
	lparams.seed = C.uint(mo.Seed)
	lparams.use_mlock = C.bool(mo.MLock)
	lparams.use_mmap = C.bool(mo.MMap)
	lparams.vocab_only = C.bool(mo.VocabOnly)

	C.llama_init_backend(C.bool(mo.NUMA))

	cpath := C.CString(modelpath)
	defer C.free(unsafe.Pointer(cpath))

	ctx := C.llama_init_from_file(cpath, lparams)
	if ctx == nil {
		return nil, errors.New("init failed")
	}

	return &LLama{ctx: ctx}, nil
}

func (l *LLama) Free() {
	C.llama_free(l.ctx)
}

func sampleTokenGreedy(candidates []C.llama_token_data) C.int {
	max := candidates[0]
	for _, c := range candidates {
		if c.logit > max.logit {
			max = c
		}
	}

	return max.id
}

func (l *LLama) Predict(prompt string, po PredictOptions, ch chan string) error {
	toks := make([]C.llama_token, len(prompt))
	toksPtr := (*C.llama_token)(unsafe.Pointer(&toks[0]))

	cprompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cprompt))

	numTokens := C.llama_tokenize(l.ctx, cprompt, toksPtr, C.int(len(toks)), C.bool(false))
	if numTokens < 0 {
		return errors.New("llama_tokenize: invalid size")
	}

	toks = toks[:numTokens]
	toksPtr = (*C.llama_token)(unsafe.Pointer(&toks[0]))

	maxContextSize := C.llama_n_ctx(l.ctx)

	for {
		count := C.llama_get_kv_cache_token_count(l.ctx)

		if count >= maxContextSize {
			break
		}

		res := C.llama_eval(l.ctx, toksPtr, C.int(len(toks)), C.llama_get_kv_cache_token_count(l.ctx), C.int(po.Threads))
		if res != 0 {
			return errors.New("eval failed")
		}

		var newTokenId C.llama_token = 0
		logits := C.llama_get_logits(l.ctx)
		nVocab := C.llama_n_vocab(l.ctx)

		candidates := make([]C.llama_token_data, nVocab)
		for i := range candidates {
			candidates[i] = C.llama_token_data{
				id: C.llama_token(i),
				logit: *(*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(logits)) + uintptr(i)*unsafe.Sizeof(*logits))),
				p: C.float(0.0),
			}
		}

		newTokenId = sampleTokenGreedy(candidates)

		if newTokenId == C.llama_token_eos() {
			break;
		}

		ch <- C.GoString(C.llama_token_to_str(l.ctx, newTokenId))

		toks = []C.llama_token{newTokenId}
		toksPtr = (*C.llama_token)(unsafe.Pointer(&toks[0]))
	}

	return nil
}

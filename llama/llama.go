// MIT License

// Copyright (c) 2023 go-skynet authors

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//go:generate cmake -S . -B build
//go:generate cmake --build build
package llama

// #cgo LDFLAGS: -Lbuild -lbinding -lllama -lggml_static -lstdc++
// #cgo darwin LDFLAGS: -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
// #cgo darwin CXXFLAGS: -std=c++11
// #include "binding/binding.h"
import "C"
import (
	"fmt"
	"os"
	"strings"
	"sync"
	"unsafe"
)

type LLama struct {
	state       unsafe.Pointer
	embeddings  bool
	contextSize int
}

func New(model string, opts ...ModelOption) (*LLama, error) {
	mo := NewModelOptions(opts...)
	modelPath := C.CString(model)
	result := C.load_model(modelPath, C.int(mo.ContextSize), C.int(mo.Seed), C.bool(mo.F16Memory), C.bool(mo.MLock), C.bool(mo.Embeddings), C.bool(mo.MMap), C.bool(mo.LowVRAM), C.bool(mo.VocabOnly), C.int(mo.NGPULayers), C.int(mo.NBatch), C.CString(mo.MainGPU), C.CString(mo.TensorSplit), C.bool(mo.NUMA))
	if result == nil {
		return nil, fmt.Errorf("failed loading model")
	}

	ll := &LLama{state: result, contextSize: mo.ContextSize, embeddings: mo.Embeddings}

	return ll, nil
}

func (l *LLama) Free() {
	C.llama_binding_free_model(l.state)
}

func (l *LLama) LoadState(state string) error {
	d := C.CString(state)
	w := C.CString("rb")

	result := C.load_state(l.state, d, w)
	if result != 0 {
		return fmt.Errorf("error while loading state")
	}

	return nil
}

func (l *LLama) SaveState(dst string) error {
	d := C.CString(dst)
	w := C.CString("wb")

	C.save_state(l.state, d, w)

	_, err := os.Stat(dst)
	return err
}

// Token Embeddings
func (l *LLama) TokenEmbeddings(tokens []int, opts ...PredictOption) ([]float32, error) {
	if !l.embeddings {
		return []float32{}, fmt.Errorf("model loaded without embeddings")
	}

	po := NewPredictOptions(opts...)

	outSize := po.Tokens
	if po.Tokens == 0 {
		outSize = 9999999
	}

	floats := make([]float32, outSize)

	myArray := (*C.int)(C.malloc(C.size_t(len(tokens)) * C.sizeof_int))

	// Copy the values from the Go slice to the C array
	for i, v := range tokens {
		(*[1<<31 - 1]int32)(unsafe.Pointer(myArray))[i] = int32(v)
	}

	params := C.llama_allocate_params(C.CString(""), C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), nil, C.int(0),
		C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL), C.CString(po.LogitBias),
		C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		C.CString(po.MainGPU), C.CString(po.TensorSplit),
		C.bool(po.PromptCacheRO),
	)
	ret := C.get_token_embeddings(params, l.state, myArray, C.int(len(tokens)), (*C.float)(&floats[0]))
	if ret != 0 {
		return floats, fmt.Errorf("embedding inference failed")
	}
	return floats, nil
}

// Embeddings
func (l *LLama) Embeddings(text string, opts ...PredictOption) ([]float32, error) {
	if !l.embeddings {
		return []float32{}, fmt.Errorf("model loaded without embeddings")
	}

	po := NewPredictOptions(opts...)

	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}
	floats := make([]float32, po.Tokens)
	reverseCount := len(po.StopPrompts)
	reversePrompt := make([]*C.char, reverseCount)
	var pass **C.char
	for i, s := range po.StopPrompts {
		cs := C.CString(s)
		reversePrompt[i] = cs
		pass = &reversePrompt[0]
	}

	params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), pass, C.int(reverseCount),
		C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL), C.CString(po.LogitBias),
		C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		C.CString(po.MainGPU), C.CString(po.TensorSplit),
		C.bool(po.PromptCacheRO),
	)

	ret := C.get_embeddings(params, l.state, (*C.float)(&floats[0]))
	if ret != 0 {
		return floats, fmt.Errorf("embedding inference failed")
	}

	return floats, nil
}

func (l *LLama) Eval(text string, opts ...PredictOption) error {
	po := NewPredictOptions(opts...)

	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}

	reverseCount := len(po.StopPrompts)
	reversePrompt := make([]*C.char, reverseCount)
	var pass **C.char
	for i, s := range po.StopPrompts {
		cs := C.CString(s)
		reversePrompt[i] = cs
		pass = &reversePrompt[0]
	}

	params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), pass, C.int(reverseCount),
		C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL), C.CString(po.LogitBias),
		C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		C.CString(po.MainGPU), C.CString(po.TensorSplit),
		C.bool(po.PromptCacheRO),
	)
	ret := C.eval(params, l.state, input)
	if ret != 0 {
		return fmt.Errorf("inference failed")
	}

	C.llama_free_params(params)

	return nil
}

func (l *LLama) Predict(text string, opts ...PredictOption) (string, error) {
	po := NewPredictOptions(opts...)

	if po.TokenCallback != nil {
		setCallback(l.state, po.TokenCallback)
	}

	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}
	out := make([]byte, po.Tokens)

	reverseCount := len(po.StopPrompts)
	reversePrompt := make([]*C.char, reverseCount)
	var pass **C.char
	for i, s := range po.StopPrompts {
		cs := C.CString(s)
		reversePrompt[i] = cs
		pass = &reversePrompt[0]
	}

	params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), pass, C.int(reverseCount),
		C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL), C.CString(po.LogitBias),
		C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		C.CString(po.MainGPU), C.CString(po.TensorSplit),
		C.bool(po.PromptCacheRO),
	)
	ret := C.llama_predict(params, l.state, (*C.char)(unsafe.Pointer(&out[0])), C.bool(po.DebugMode))
	if ret != 0 {
		return "", fmt.Errorf("inference failed")
	}
	res := C.GoString((*C.char)(unsafe.Pointer(&out[0])))

	res = strings.TrimPrefix(res, " ")
	res = strings.TrimPrefix(res, text)
	res = strings.TrimPrefix(res, "\n")

	for _, s := range po.StopPrompts {
		res = strings.TrimRight(res, s)
	}

	C.llama_free_params(params)

	if po.TokenCallback != nil {
		setCallback(l.state, nil)
	}

	return res, nil
}

// CGo only allows us to use static calls from C to Go, we can't just dynamically pass in func's.
// This is the next best thing, we register the callbacks in this map and call tokenCallback from
// the C code. We also attach a finalizer to LLama, so it will unregister the callback when the
// garbage collection frees it.

// SetTokenCallback registers a callback for the individual tokens created when running Predict. It
// will be called once for each token. The callback shall return true as long as the model should
// continue predicting the next token. When the callback returns false the predictor will return.
// The tokens are just converted into Go strings, they are not trimmed or otherwise changed. Also
// the tokens may not be valid UTF-8.
// Pass in nil to remove a callback.
//
// It is save to call this method while a prediction is running.
func (l *LLama) SetTokenCallback(callback func(token string) bool) {
	setCallback(l.state, callback)
}

var (
	m         sync.Mutex
	callbacks = map[uintptr]func(string) bool{}
)

//export tokenCallback
func tokenCallback(statePtr unsafe.Pointer, token *C.char) bool {
	m.Lock()
	defer m.Unlock()

	if callback, ok := callbacks[uintptr(statePtr)]; ok {
		return callback(C.GoString(token))
	}

	return true
}

// setCallback can be used to register a token callback for LLama. Pass in a nil callback to
// remove the callback.
func setCallback(statePtr unsafe.Pointer, callback func(string) bool) {
	m.Lock()
	defer m.Unlock()

	if callback == nil {
		delete(callbacks, uintptr(statePtr))
	} else {
		callbacks[uintptr(statePtr)] = callback
	}
}

package constraint

// #cgo linux LDFLAGS: -ldl
// #include "dynamic.h"
// #include <stdlib.h>
import "C"

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"
)

type Kind int

const (
	JSON Kind = iota
	JSONSchema
)

// The native library is loaded once and never unloaded.
var library struct {
	sync.Mutex
	handle C.ollama_constraint_dynamic_handle
	path   string
}

func Load(dir string) error {
	library.Lock()
	defer library.Unlock()
	if library.path != "" {
		return nil
	}

	var name string
	switch runtime.GOOS {
	case "darwin":
		name = "libollama_constraints.dylib"
	case "linux":
		name = "libollama_constraints.so"
	case "windows":
		name = "ollama_constraints.dll"
	default:
		return fmt.Errorf("constraint library is not supported on %s", runtime.GOOS)
	}
	path, err := filepath.Abs(filepath.Join(dir, name))
	if err != nil {
		return fmt.Errorf("resolve constraint library path: %w", err)
	}
	if _, err := os.Stat(path); err != nil {
		return fmt.Errorf("constraint library not found at %s: %w", path, err)
	}
	cPath := C.CString(path)
	result := C.ollama_constraint_dynamic_load(&library.handle, cPath)
	C.free(unsafe.Pointer(cPath))
	if result != 0 {
		return fmt.Errorf("load constraint library %s: %s", path, C.GoString(C.ollama_constraint_dynamic_error()))
	}
	library.path = path
	return nil
}

func LoadedLibraryPath() string {
	library.Lock()
	defer library.Unlock()
	return library.path
}

type Model struct {
	ctxMu     sync.Mutex
	ctx       *C.ollama_constraint_model
	vocabSize int
}

func NewModel(pieces []string, vocabSize int, stopIDs []int32) (*Model, error) {
	library.Lock()
	loaded := library.path != ""
	library.Unlock()
	if !loaded {
		return nil, errors.New("constraint library is not loaded")
	}
	if vocabSize <= 0 || len(pieces) > vocabSize {
		return nil, fmt.Errorf("invalid vocabulary size %d for %d token pieces", vocabSize, len(pieces))
	}
	if len(stopIDs) == 0 {
		return nil, errors.New("tokenizer has no stop tokens")
	}

	var data []byte
	offsets := make([]C.uint64_t, len(pieces))
	for i, piece := range pieces {
		data = append(data, piece...)
		offsets[i] = C.uint64_t(len(data))
	}
	cStops := make([]C.int32_t, len(stopIDs))
	for i, id := range stopIDs {
		cStops[i] = C.int32_t(id)
	}

	var dataPtr *C.char
	if len(data) > 0 {
		dataPtr = (*C.char)(unsafe.Pointer(&data[0]))
	}
	var offsetPtr *C.uint64_t
	if len(offsets) > 0 {
		offsetPtr = &offsets[0]
	}
	var ctx *C.ollama_constraint_model
	var cError *C.char
	if C.ollama_constraint_dynamic_model_new(
		dataPtr, C.size_t(len(data)), offsetPtr, C.size_t(len(offsets)), C.int32_t(vocabSize),
		&cStops[0], C.size_t(len(cStops)), &ctx, &cError,
	) != 0 {
		return nil, nativeError("create constraint model", cError)
	}
	return &Model{ctx: ctx, vocabSize: vocabSize}, nil
}

func (m *Model) Close() {
	if m == nil {
		return
	}
	m.ctxMu.Lock()
	defer m.ctxMu.Unlock()
	if m.ctx != nil {
		C.ollama_constraint_dynamic_model_free(m.ctx)
		m.ctx = nil
	}
}

func (m *Model) Compile(kind Kind, source string) (*Matcher, error) {
	if m == nil {
		return nil, errors.New("constraint model is unavailable")
	}
	m.ctxMu.Lock()
	defer m.ctxMu.Unlock()
	if m.ctx == nil {
		return nil, errors.New("constraint model is closed")
	}

	var sourcePtr *C.char
	if len(source) > 0 {
		sourcePtr = (*C.char)(unsafe.Pointer(unsafe.StringData(source)))
	}
	var ctx *C.ollama_constraint_matcher
	var cError *C.char
	if C.ollama_constraint_dynamic_matcher_new(
		m.ctx, C.ollama_constraint_kind(kind), sourcePtr, C.size_t(len(source)), &ctx, &cError,
	) != 0 {
		return nil, nativeError("compile constraint", cError)
	}
	return &Matcher{ctx: ctx, vocabSize: m.vocabSize}, nil
}

type Matcher struct {
	ctxMu     sync.Mutex
	ctx       *C.ollama_constraint_matcher
	vocabSize int
}

func (m *Matcher) VocabSize() int {
	if m == nil {
		return 0
	}
	return m.vocabSize
}

func (m *Matcher) Fill() ([]int32, bool, error) {
	if m == nil {
		return nil, false, errors.New("constraint matcher is closed")
	}
	m.ctxMu.Lock()
	defer m.ctxMu.Unlock()
	if m.ctx == nil {
		return nil, false, errors.New("constraint matcher is closed")
	}
	mask := make([]int32, (m.vocabSize+31)/32)
	var needsApply C.int
	var cError *C.char
	if C.ollama_constraint_dynamic_matcher_fill(m.ctx, (*C.int32_t)(unsafe.Pointer(&mask[0])), C.size_t(len(mask)), &needsApply, &cError) != 0 {
		return nil, false, nativeError("build token mask", cError)
	}
	return mask, needsApply != 0, nil
}

func (m *Matcher) Accept(tokenID int32) error {
	if m == nil {
		return errors.New("constraint matcher is closed")
	}
	m.ctxMu.Lock()
	defer m.ctxMu.Unlock()
	if m.ctx == nil {
		return errors.New("constraint matcher is closed")
	}
	var accepted C.int
	var cError *C.char
	if C.ollama_constraint_dynamic_matcher_accept(m.ctx, C.int32_t(tokenID), &accepted, &cError) != 0 {
		return nativeError("accept token", cError)
	}
	if accepted == 0 {
		return fmt.Errorf("constraint rejected sampled token %d", tokenID)
	}
	return nil
}

func (m *Matcher) Close() {
	if m == nil {
		return
	}
	m.ctxMu.Lock()
	defer m.ctxMu.Unlock()
	if m.ctx != nil {
		C.ollama_constraint_dynamic_matcher_free(m.ctx)
		m.ctx = nil
	}
}

func nativeError(op string, cMessage *C.char) error {
	message := "unknown error"
	if cMessage != nil {
		defer C.free(unsafe.Pointer(cMessage))
		message = C.GoString(cMessage)
	}
	if message == "" {
		message = "unknown error"
	}
	return fmt.Errorf("%s: %s", op, message)
}

package xgrammar

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
	"slices"
	"sync"
	"unsafe"
)

type Kind int

const (
	JSONSchema Kind = C.OLLAMA_XGRAMMAR_JSON_SCHEMA
)

// The native library is loaded once per process and never unloaded.
var (
	loadOnce sync.Once
	loadDir  string
	loadPath string
	loadErr  error
)

func loadLibrary(dir string) (string, error) {
	loadOnce.Do(func() {
		loadDir = dir
		loadPath, loadErr = openLibrary(dir)
	})
	if loadErr != nil {
		return "", loadErr
	}
	if dir != loadDir {
		return "", fmt.Errorf("xgrammar library already loaded from %s", loadDir)
	}
	return loadPath, nil
}

func openLibrary(dir string) (string, error) {
	var name string
	switch runtime.GOOS {
	case "darwin":
		name = "libollama_xgrammar.dylib"
	case "linux":
		name = "libollama_xgrammar.so"
	case "windows":
		name = "ollama_xgrammar.dll"
	default:
		return "", fmt.Errorf("xgrammar library is not supported on %s", runtime.GOOS)
	}
	path, err := filepath.Abs(filepath.Join(dir, name))
	if err != nil {
		return "", fmt.Errorf("resolve xgrammar library path: %w", err)
	}
	if _, err := os.Stat(path); err != nil {
		return "", fmt.Errorf("xgrammar library not found at %s: %w", path, err)
	}
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	var handle C.ollama_xgrammar_dynamic_handle
	if C.ollama_xgrammar_dynamic_load(&handle, cPath) != 0 {
		return "", fmt.Errorf("load xgrammar library %s: %s", path, C.GoString(C.ollama_xgrammar_dynamic_error()))
	}
	return path, nil
}

// Compiler compiles grammars for one vocabulary.
type Compiler struct {
	ctx       *C.ollama_xgrammar_compiler
	path      string
	vocabSize int
	stops     []int32
}

// New loads the native library from dir — once per process, never unloaded —
// and binds a grammar compiler to the vocabulary. cacheBytes bounds the
// engine's compiled-grammar cache; <= 0 disables it.
func New(dir string, pieces []string, vocabSize int, stopIDs []int32, threads int, cacheBytes int64) (*Compiler, error) {
	path, err := loadLibrary(dir)
	if err != nil {
		return nil, err
	}
	if vocabSize <= 0 || len(pieces) > vocabSize {
		return nil, fmt.Errorf("invalid vocabulary size %d for %d token pieces", vocabSize, len(pieces))
	}
	if len(stopIDs) == 0 {
		return nil, errors.New("tokenizer has no stop tokens")
	}
	for _, id := range stopIDs {
		if id < 0 || int(id) >= vocabSize {
			return nil, fmt.Errorf("stop token %d is outside the vocabulary", id)
		}
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
	var ctx *C.ollama_xgrammar_compiler
	var cError *C.char
	if C.ollama_xgrammar_dynamic_compiler_new(
		dataPtr, C.size_t(len(data)), offsetPtr, C.size_t(len(offsets)), C.int32_t(vocabSize),
		&cStops[0], C.size_t(len(cStops)), C.int32_t(threads), C.int64_t(cacheBytes), &ctx, &cError,
	) != 0 {
		return nil, nativeError("create grammar compiler", cError)
	}
	return &Compiler{ctx: ctx, path: path, vocabSize: vocabSize, stops: slices.Clone(stopIDs)}, nil
}

// Path returns the loaded native library's location.
func (c *Compiler) Path() string {
	return c.path
}

// Version returns the pinned xgrammar release the library was built from.
func (c *Compiler) Version() string {
	return C.GoString(C.ollama_xgrammar_dynamic_version())
}

func (c *Compiler) Compile(kind Kind, source string) (*Matcher, error) {
	if c == nil {
		return nil, errors.New("grammar compiler is unavailable")
	}
	if c.ctx == nil {
		return nil, errors.New("grammar compiler is closed")
	}

	var sourcePtr *C.char
	if len(source) > 0 {
		sourcePtr = (*C.char)(unsafe.Pointer(unsafe.StringData(source)))
	}
	var ctx *C.ollama_xgrammar_matcher
	var cError *C.char
	if C.ollama_xgrammar_dynamic_matcher_new(
		c.ctx, C.ollama_xgrammar_kind(kind), sourcePtr, C.size_t(len(source)), &ctx, &cError,
	) != 0 {
		return nil, nativeError("compile grammar", cError)
	}
	return &Matcher{ctx: ctx, vocabSize: c.vocabSize, stops: c.stops}, nil
}

func (c *Compiler) Close() {
	if c == nil {
		return
	}
	if c.ctx != nil {
		C.ollama_xgrammar_dynamic_compiler_free(c.ctx)
		c.ctx = nil
	}
}

type Matcher struct {
	ctx       *C.ollama_xgrammar_matcher
	vocabSize int
	stops     []int32
	// terminated: a stop token was accepted, ending the grammar; the matcher
	// no longer constrains sampling.
	terminated bool
}

// Terminated reports whether an accepted stop token ended the grammar; a
// terminated matcher no longer constrains sampling.
func (m *Matcher) Terminated() bool {
	if m == nil {
		return true
	}
	return m.terminated
}

// Fill writes the packed allowed-token bitmask for the next position into
// row — bit id%32 of word id/32 is set when token id is allowed — and
// reports whether it constrains sampling. A false return means every token
// is allowed or the grammar has terminated; row contents are meaningful
// only on true. row must hold ceil(vocabSize/32) words and may be one row
// of a larger batch buffer.
func (m *Matcher) Fill(row []int32) (bool, error) {
	if m == nil {
		return false, errors.New("grammar matcher is closed")
	}
	if m.ctx == nil {
		return false, errors.New("grammar matcher is closed")
	}
	if want := (m.vocabSize + 31) / 32; len(row) != want {
		return false, fmt.Errorf("mask row holds %d words; the vocabulary needs %d", len(row), want)
	}
	if m.terminated {
		return false, nil
	}

	var needsApply C.int
	var cError *C.char
	if C.ollama_xgrammar_dynamic_matcher_fill(m.ctx, (*C.int32_t)(unsafe.Pointer(&row[0])), C.size_t(len(row)), &needsApply, &cError) != 0 {
		return false, nativeError("build token mask", cError)
	}
	return needsApply != 0, nil
}

func (m *Matcher) Accept(tokenID int32) error {
	if m == nil {
		return errors.New("grammar matcher is closed")
	}
	if m.ctx == nil {
		return errors.New("grammar matcher is closed")
	}
	var accepted C.int
	var cError *C.char
	if C.ollama_xgrammar_dynamic_matcher_accept(m.ctx, C.int32_t(tokenID), &accepted, &cError) != 0 {
		return nativeError("accept token", cError)
	}
	if accepted == 0 {
		return fmt.Errorf("grammar rejected sampled token %d", tokenID)
	}
	if slices.Contains(m.stops, tokenID) {
		m.terminated = true
	}
	return nil
}

func (m *Matcher) Close() {
	if m == nil {
		return
	}
	if m.ctx != nil {
		C.ollama_xgrammar_dynamic_matcher_free(m.ctx)
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

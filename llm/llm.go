package llm

// #cgo CFLAGS: -Illama.cpp
// #cgo darwin,arm64 LDFLAGS: ${SRCDIR}/build/darwin/arm64_static/libllama.a -lstdc++
// #cgo darwin,amd64 LDFLAGS: ${SRCDIR}/build/darwin/x86_64_static/libllama.a -lstdc++
// #cgo windows,amd64 LDFLAGS: ${SRCDIR}/build/windows/amd64_static/libllama.a -static -lstdc++
// #cgo windows,arm64 LDFLAGS: ${SRCDIR}/build/windows/arm64_static/libllama.a -static -lstdc++
// #cgo linux,amd64 LDFLAGS: ${SRCDIR}/build/linux/x86_64_static/libllama.a -lstdc++
// #cgo linux,arm64 LDFLAGS: ${SRCDIR}/build/linux/arm64_static/libllama.a -lstdc++
// #include <stdlib.h>
// #include "llama.h"
import "C"
import (
	"fmt"
	"strings"
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

type llamaModel struct {
	m *C.struct_llama_model
}

func newLlamaModel(p string) *llamaModel {
	cs := C.CString(p)
	defer C.free(unsafe.Pointer(cs))

	params := C.llama_model_default_params()
	params.vocab_only = true

	return &llamaModel{
		C.llama_load_model_from_file(cs, params),
	}
}

func (llm *llamaModel) Close() {
	C.llama_free_model(llm.m)
}

func (llm *llamaModel) Tokenize(s string) []int {
	cs := C.CString(s)
	defer C.free(unsafe.Pointer(cs))

	ltokens := make([]C.llama_token, len(s)+2)
	n := C.llama_tokenize(
		llm.m,
		cs,
		C.int32_t(len(s)),
		&ltokens[0],
		C.int32_t(len(ltokens)),
		false,
		true,
	)

	if n < 0 {
		return nil
	}

	tokens := make([]int, n)
	for i := 0; i < int(n); i++ {
		tokens[i] = int(ltokens[i])
	}

	return tokens
}

func (llm *llamaModel) Detokenize(i32s []int) string {
	var sb strings.Builder
	for _, i32 := range i32s {
		c := make([]byte, 512)
		if n := C.llama_token_to_piece(llm.m, C.llama_token(i32), (*C.char)(unsafe.Pointer(&c[0])), C.int(len(c)), false); n > 0 {
			sb.WriteString(unsafe.String(&c[0], n))
		}
	}

	return sb.String()
}

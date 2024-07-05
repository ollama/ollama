package llm

// #cgo CFLAGS: -Illama.cpp/include -Illama.cpp/ggml/include
// #cgo darwin,arm64 LDFLAGS: ${SRCDIR}/build/darwin/arm64_static/src/libllama.a ${SRCDIR}/build/darwin/arm64_static/ggml/src/libggml.a -lstdc++ -framework Accelerate -framework Metal
// #cgo darwin,amd64 LDFLAGS: ${SRCDIR}/build/darwin/x86_64_static/src/libllama.a ${SRCDIR}/build/darwin/x86_64_static/ggml/src/libggml.a -lstdc++ -framework Accelerate -framework Metal
// #cgo windows,amd64 LDFLAGS: ${SRCDIR}/build/windows/amd64_static/src/libllama.a ${SRCDIR}/build/windows/amd64_static/ggml/src/libggml.a  -static -lstdc++
// #cgo windows,arm64 LDFLAGS: ${SRCDIR}/build/windows/arm64_static/src/libllama.a ${SRCDIR}/build/windows/arm64_static/ggml/src/libggml.a -static -lstdc++
// #cgo linux,amd64 LDFLAGS: ${SRCDIR}/build/linux/x86_64_static/src/libllama.a ${SRCDIR}/build/linux/x86_64_static/ggml/src/libggml.a -lstdc++
// #cgo linux,arm64 LDFLAGS: ${SRCDIR}/build/linux/arm64_static/src/libllama.a ${SRCDIR}/build/linux/arm64_static/ggml/src/libggml. -lstdc++
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

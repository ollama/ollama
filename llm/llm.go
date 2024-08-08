package llm

// #cgo CFLAGS: -Illama.cpp -Illama.cpp/include -Illama.cpp/ggml/include
// #cgo LDFLAGS: -lllama -lggml -lstdc++ -lpthread
// #cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/build/darwin/arm64_static -L${SRCDIR}/build/darwin/arm64_static/src -L${SRCDIR}/build/darwin/arm64_static/ggml/src -framework Accelerate -framework Metal
// #cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/build/darwin/x86_64_static -L${SRCDIR}/build/darwin/x86_64_static/src -L${SRCDIR}/build/darwin/x86_64_static/ggml/src
// #cgo windows,amd64 LDFLAGS: -static-libstdc++ -static-libgcc -static -L${SRCDIR}/build/windows/amd64_static -L${SRCDIR}/build/windows/amd64_static/src -L${SRCDIR}/build/windows/amd64_static/ggml/src
// #cgo windows,arm64 LDFLAGS: -static-libstdc++ -static-libgcc -static -L${SRCDIR}/build/windows/arm64_static -L${SRCDIR}/build/windows/arm64_static/src -L${SRCDIR}/build/windows/arm64_static/ggml/src
// #cgo linux,amd64 LDFLAGS: -L${SRCDIR}/build/linux/x86_64_static -L${SRCDIR}/build/linux/x86_64_static/src -L${SRCDIR}/build/linux/x86_64_static/ggml/src
// #cgo linux,arm64 LDFLAGS: -L${SRCDIR}/build/linux/arm64_static -L${SRCDIR}/build/linux/arm64_static/src -L${SRCDIR}/build/linux/arm64_static/ggml/src
// #cgo linux,riscv64 LDFLAGS: -L${SRCDIR}/build/linux/riscv64_static -L${SRCDIR}/build/linux/riscv64_static/src -L${SRCDIR}/build/linux/riscv64_static/ggml/src
// #include <stdlib.h>
// #include "llama.h"
import "C"

import (
	"errors"
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
		return errors.New("failed to quantize model. This model architecture may not be supported, or you may need to upgrade Ollama to the latest version")
	}

	return nil
}

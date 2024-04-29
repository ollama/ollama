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
	"unsafe"
)

// SystemInfo is an unused example of calling llama.cpp functions using CGo
func SystemInfo() string {
	return C.GoString(C.llama_print_system_info())
}

func Quantize(infile, outfile, filetype string) error {
	cinfile := C.CString(infile)
	defer C.free(unsafe.Pointer(cinfile))

	coutfile := C.CString(outfile)
	defer C.free(unsafe.Pointer(coutfile))

	params := C.llama_model_quantize_default_params()
	params.nthread = -1

	switch filetype {
	case "F32":
		params.ftype = fileTypeF32
	case "F16":
		params.ftype = fileTypeF16
	case "Q4_0":
		params.ftype = fileTypeQ4_0
	case "Q4_1":
		params.ftype = fileTypeQ4_1
	case "Q4_1_F16":
		params.ftype = fileTypeQ4_1_F16
	case "Q8_0":
		params.ftype = fileTypeQ8_0
	case "Q5_0":
		params.ftype = fileTypeQ5_0
	case "Q5_1":
		params.ftype = fileTypeQ5_1
	case "Q2_K":
		params.ftype = fileTypeQ2_K
	case "Q3_K_S":
		params.ftype = fileTypeQ3_K_S
	case "Q3_K_M":
		params.ftype = fileTypeQ3_K_M
	case "Q3_K_L":
		params.ftype = fileTypeQ3_K_L
	case "Q4_K_S":
		params.ftype = fileTypeQ4_K_S
	case "Q4_K_M":
		params.ftype = fileTypeQ4_K_M
	case "Q5_K_S":
		params.ftype = fileTypeQ5_K_S
	case "Q5_K_M":
		params.ftype = fileTypeQ5_K_M
	case "Q6_K":
		params.ftype = fileTypeQ6_K
	case "IQ2_XXS":
		params.ftype = fileTypeIQ2_XXS
	case "IQ2_XS":
		params.ftype = fileTypeIQ2_XS
	case "Q2_K_S":
		params.ftype = fileTypeQ2_K_S
	case "Q3_K_XS":
		params.ftype = fileTypeQ3_K_XS
	case "IQ3_XXS":
		params.ftype = fileTypeIQ3_XXS
	default:
		return fmt.Errorf("unknown filetype: %s", filetype)
	}

	if retval := C.llama_model_quantize(cinfile, coutfile, &params); retval != 0 {
		return fmt.Errorf("llama_model_quantize: %d", retval)
	}

	return nil
}

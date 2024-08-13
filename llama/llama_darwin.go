package llama

// extern const char *ggml_metallib_start;
// extern const char *ggml_metallib_end;
import "C"

import (
	_ "embed"
	"strings"
	"unsafe"
)

//go:embed ggml-common.h
var ggmlCommon string

//go:embed ggml-metal.metal
var ggmlMetal string

func init() {
	metal := strings.ReplaceAll(ggmlMetal, `#include "ggml-common.h"`, ggmlCommon)
	cMetal := C.CString(metal)
	C.ggml_metallib_start = cMetal
	C.ggml_metallib_end = (*C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(cMetal)) + uintptr(len(metal))))
}

package ggml

// #include "ggml-metal.h"
import "C"

func newBackend() *C.struct_ggml_backend {
	return C.ggml_backend_metal_init()
}

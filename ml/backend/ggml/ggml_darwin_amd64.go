package ggml

// #include "ggml-backend.h"
import "C"

func newBackend() *C.struct_ggml_backend {
	return newCPUBackend()
}

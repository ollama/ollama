package ggml

// #include "ggml-backend.h"
import "C"

func newCPUBackend() *C.struct_ggml_backend {
	return C.ggml_backend_cpu_init()
}

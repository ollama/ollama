package ggml

// #cgo CPPFLAGS: -D_GNU_SOURCE
// #cgo LDFLAGS: -lm
// #include "ggml-backend.h"
import "C"

func newBackend() *C.struct_ggml_backend {
	return newCPUBackend()
}

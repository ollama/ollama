package ggml

// #include "ggml-metal.h"
import "C"
import _ "github.com/ollama/ollama/ml/backend/ggml/ggml/ggml-metal"

func newBackend() *C.struct_ggml_backend {
	return C.ggml_backend_metal_init()
}

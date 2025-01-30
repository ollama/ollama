package ggml

// #cgo CPPFLAGS: -DGGML_USE_METAL -DGGML_USE_BLAS
// #cgo LDFLAGS: -framework Foundation
import "C"

import (
	_ "github.com/ollama/ollama/ml/backend/ggml/ggml/src/ggml-blas"
	_ "github.com/ollama/ollama/ml/backend/ggml/ggml/src/ggml-metal"
)

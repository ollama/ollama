//go:build cuda

package ggml

// #cgo CPPFLAGS: -DGGML_USE_CUDA
// #cgo rocm CPPFLAGS: -DGGML_USE_HIP
import "C"
import _ "github.com/ollama/ollama/ml/backend/ggml/ggml/ggml-cuda"

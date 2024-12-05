//go:build cuda

package ggml

// #cgo CPPFLAGS: -DGGML_USE_CUDA
import "C"
import _ "github.com/ollama/ollama/ml/backend/ggml/ggml/ggml-cuda"

package ggml

// #cgo CPPFLAGS: -DGGML_USE_METAL
import "C"
import _ "github.com/ollama/ollama/ml/backend/ggml/ggml/ggml-metal"

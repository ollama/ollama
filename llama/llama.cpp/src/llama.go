package llama

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/../include
// #cgo CPPFLAGS: -I${SRCDIR}/../../../ml/backend/ggml/ggml/include
// #cgo windows CPPFLAGS: -D_WIN32_WINNT=0x0602
import "C"

import (
	_ "github.com/ollama/ollama/llama/llama.cpp/src/models"
	_ "github.com/ollama/ollama/ml/backend/ggml/ggml/src"
)

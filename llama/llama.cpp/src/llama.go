package llama

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/../include
// #cgo CPPFLAGS: -I${SRCDIR}/../../../ml/backend/ggml/ggml/include
import "C"
import _ "github.com/ollama/ollama/ml/backend/ggml/ggml/src"

package cpu

// #cgo CFLAGS: -Wno-implicit-function-declaration
// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -O3
// #cgo CPPFLAGS: -DGGML_USE_LLAMAFILE
// #cgo CPPFLAGS: -I${SRCDIR}/amx -I${SRCDIR}/llamafile -I${SRCDIR}/.. -I${SRCDIR}/../../include
// #cgo linux CPPFLAGS: -D_GNU_SOURCE
// #cgo arm64 CPPFLAGS: -DGGML_USE_AARCH64
// #cgo darwin,arm64 CPPFLAGS: -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
// #cgo darwin,arm64 LDFLAGS: -framework Accelerate
import "C"
import _ "github.com/ollama/ollama/ml/backend/ggml/ggml/src/ggml-cpu/llamafile"

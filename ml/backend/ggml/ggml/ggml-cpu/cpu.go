package cpu

// #cgo CXXFLAGS: -std=c++11
// #cgo CPPFLAGS: -I${SRCDIR}/.. -I${SRCDIR}/../include -I${SRCDIR}/amx
// #cgo CPPFLAGS: -D_GNU_SOURCE
// #cgo amd64,avx CPPFLAGS: -mavx
// #cgo amd64,avx2 CPPFLAGS: -mavx2 -mfma -mf16c
// #cgo arm64 CPPFLAGS: -D__aarch64__ -D__ARM_NEON -D__ARM_FEATURE_FMA
import "C"
import (
	_ "github.com/ollama/ollama/ml/backend/ggml/ggml/ggml-cpu/amx"
	_ "github.com/ollama/ollama/ml/backend/ggml/ggml/ggml-cpu/llamafile"
)

package ggml

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR} -I${SRCDIR}/include -I${SRCDIR}/ggml-cpu
// #cgo CPPFLAGS: -DNDEBUG -DGGML_USE_CPU
// #cgo darwin LDFLAGS: -framework Foundation
// #cgo amd64,avx CPPFLAGS: -mavx
// #cgo amd64,avx2 CPPFLAGS: -mavx2 -mfma
// #cgo amd64,f16c CPPFLAGS: -mf16c
// #cgo arm64 CPPFLAGS: -D__aarch64__ -D__ARM_NEON -D__ARM_FEATURE_FMA
import "C"
import _ "github.com/ollama/ollama/ml/backend/ggml/ggml/ggml-cpu"

package cpu

// #cgo CFLAGS: -O3 -Wno-implicit-function-declaration
// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/amx -I${SRCDIR}/llamafile -I${SRCDIR}/.. -I${SRCDIR}/../../include
// #cgo CPPFLAGS: -DNDEBUG -DGGML_USE_LLAMAFILE
// #cgo linux CPPFLAGS: -D_GNU_SOURCE
// #cgo darwin,arm64 CPPFLAGS: -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
// #cgo darwin,arm64 LDFLAGS: -framework Accelerate
//
// #cgo amd64 CFLAGS: -D__amd64__
// #cgo amd64 CPPFLAGS: -D__amd64__
//
// #cgo arm64 CFLAGS: -D__arm64__
// #cgo arm64 CPPFLAGS: -D__arm64__
//
// #cgo loong64 CFLAGS: -D__loong64__
// #cgo loong64 CPPFLAGS: -D__loong64__
//
// #cgo ppc64 CFLAGS: -D__ppc64__
// #cgo ppc64 CPPFLAGS: -D__ppc64__
//
// #cgo riscv64 CFLAGS: -D__riscv64__
// #cgo riscv64 CPPFLAGS: -D__riscv64__
//
// #cgo s390x CFLAGS: -D__s390x__
// #cgo s390x CPPFLAGS: -D__s390x__
//
// #cgo wasm CFLAGS: -D__wasm__
// #cgo wasm CPPFLAGS: -D__wasm__
import "C"
import _ "github.com/ollama/ollama/ml/backend/ggml/ggml/src/ggml-cpu/llamafile"

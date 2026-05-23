package mtmd

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/../../include -I${SRCDIR}/../../common -I${SRCDIR}/../../vendor
// #cgo CPPFLAGS: -I${SRCDIR}/../../../../ml/backend/ggml/ggml/include
// #cgo CPPFLAGS: -DGGML_TURBOQUANT
import "C"

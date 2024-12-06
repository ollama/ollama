package ggml

// #cgo CFLAGS: -std=c17
// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/include -I${SRCDIR}/ggml-cpu
// #cgo CPPFLAGS: -DNDEBUG
import "C"

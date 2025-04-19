package sycl

// #cgo CXXFLAGS: -std=c++11
// #cgo CPPFLAGS: -DGGML_USE_SYCL
// #cgo CPPFLAGS: -I${SRCDIR}/.. -I${SRCDIR}/../../include
import "C"

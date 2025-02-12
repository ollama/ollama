//go:build darwin && arm64

package blas

// #cgo CXXFLAGS: -std=c++11
// #cgo CPPFLAGS: -DGGML_USE_BLAS
// #cgo CPPFLAGS: -I${SRCDIR}/.. -I${SRCDIR}/../../include
// #cgo darwin,arm64 CPPFLAGS: -DGGML_BLAS_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
// #cgo darwin,arm64 LDFLAGS: -framework Accelerate
import "C"

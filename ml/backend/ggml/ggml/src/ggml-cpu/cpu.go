package cpu

// #cgo CXXFLAGS: -std=c++11
// #cgo CPPFLAGS: -I${SRCDIR}/amx -I${SRCDIR}/.. -I${SRCDIR}/../../include
// #cgo linux CPPFLAGS: -D_GNU_SOURCE
// #cgo darwin,arm64 CPPFLAGS: -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
// #cgo darwin,arm64 LDFLAGS: -framework Accelerate
import "C"

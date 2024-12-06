package ggml

// #cgo CPPFLAGS: -Wno-incompatible-pointer-types-discards-qualifiers
// #cgo LDFLAGS: -framework Foundation
// #cgo amd64,avx2 CPPFLAGS: -DGGML_USE_ACCELERATE -DACCELERATE_USE_LAPACK -DACCELERATE_LAPACK_ILP64
// #cgo amd64,avx2 LDFLAGS: -framework Accelerate
import "C"

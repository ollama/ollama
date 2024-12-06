package ggml

// #cgo CPPFLAGS: -D_GNU_SOURCE
// #cgo LDFLAGS: -lm
// #cgo arm64 CPPFLAGS: -D__aarch64__ -D__ARM_NEON -D__ARM_FEATURE_FMA
// #cgo arm64,sve CPPFLAGS: -march=armv8.6-a+sve
import "C"

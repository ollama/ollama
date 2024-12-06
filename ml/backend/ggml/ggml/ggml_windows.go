package ggml

// #cgo CPPFLAGS: -D_WIN32_WINNT=0x602
// #cgo LDFLAGS: -lmsvcrt -static -static-libgcc -static-libstdc++
// #cgo arm64 CPPFLAGS: -D__aarch64__ -D__ARM_NEON -D__ARM_FEATURE_FMA
import "C"

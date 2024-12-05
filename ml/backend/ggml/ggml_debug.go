//go:build debug

package ggml

// #cgo CPPFLAGS: -DOLLAMA_DEBUG
import "C"

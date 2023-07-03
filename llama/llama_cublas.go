//go:build cublas
// +build cublas

package llama

/*
#cgo LDFLAGS: -lcublas -lcudart -L/usr/local/cuda/lib64/
*/
import "C"

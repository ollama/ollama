//go:build openblas
// +build openblas

package llama

/*
#cgo LDFLAGS: -lopenblas
*/
import "C"

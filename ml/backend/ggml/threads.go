//go:build !debug

package ggml

func Threads(n int) int {
	return n
}

//go:build debug

package ggml

func Threads(_ int) int {
	return 1
}

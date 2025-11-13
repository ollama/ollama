package tools

import (
	"math/rand"
	"strings"
)

func ToolCallID() string {
	const letterBytes = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, 8)
	for i := range b {
		b[i] = letterBytes[rand.Intn(len(letterBytes))]
	}
	return "call_" + strings.ToLower(string(b))
}

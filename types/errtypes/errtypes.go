// Package errtypes contains custom error types
package errtypes

import (
	"fmt"
	"strings"
)

const UnknownOllamaKeyErrMsg = "unknown ollama key"

type UnknownOllamaKey struct {
	Key string
}

func (e *UnknownOllamaKey) Error() string {
	return fmt.Sprintf("unauthorized: %s  %q", UnknownOllamaKeyErrMsg, strings.TrimSpace(e.Key))
}

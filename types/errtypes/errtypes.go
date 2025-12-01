// Package errtypes contains custom error types
package errtypes

import (
	"fmt"
	"strings"
)

const (
	UnknownOllamaKeyErrMsg = "unknown ollama key"
	InvalidModelNameErrMsg = "invalid model name"
	ModelNotFoundErrMsg    = "model not found"
)

// TODO: This should have a structured response from the API
type UnknownOllamaKey struct {
	Key string
}

func (e *UnknownOllamaKey) Error() string {
	return fmt.Sprintf("unauthorized: %s %q", UnknownOllamaKeyErrMsg, strings.TrimSpace(e.Key))
}

type ModelLoadError struct {
	Model  string
	Reason string
}

func (e *ModelLoadError) Error() string {
	return fmt.Sprintf("failed to load model %q: %s", e.Model, e.Reason)
}

type ModelNotFoundError struct {
	Model string
}

func (e *ModelNotFoundError) Error() string {
	return fmt.Sprintf("model %q not found: %s", e.Model, ModelNotFoundErrMsg)
}

type InvalidModelConfigError struct {
	Config string
	Reason string
}

func (e *InvalidModelConfigError) Error() string {
	return fmt.Sprintf("invalid model config %q: %s", e.Config, e.Reason)
}

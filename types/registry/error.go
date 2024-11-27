package registry

import (
	"fmt"
	"slices"
	"strings"
)

const ErrCodeAnonymous = "ANONYMOUS_ACCESS_DENIED"

type Err struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// Errs represents the structure of error responses from the registry
// TODO (brucemacd): this struct should be imported from some shared package that is used between the registry and ollama
type Errs struct {
	Errors []Err `json:"errors"`
}

// Error implements the error interface for RegistryError
func (e Errs) Error() string {
	if len(e.Errors) == 0 {
		return "unknown registry error"
	}
	var msgs []string
	for _, err := range e.Errors {
		msgs = append(msgs, fmt.Sprintf("%s: %s", err.Code, err.Message))
	}
	return strings.Join(msgs, "; ")
}

func (e Errs) HasCode(code string) bool {
	return slices.ContainsFunc(e.Errors, func(err Err) bool {
		return err.Code == code
	})
}

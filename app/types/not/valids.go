//go:build windows || darwin

package not

import (
	"fmt"
)

type ValidError struct {
	name string
	msg  string
	args []any
}

// Valid returns a new validation error with the given name and message.
func Valid(name, message string, args ...any) error {
	return ValidError{name, message, args}
}

// Message returns the formatted message for the validation error.
func (e *ValidError) Message() string {
	return fmt.Sprintf(e.msg, e.args...)
}

// Error implements the error interface.
func (e ValidError) Error() string {
	return fmt.Sprintf("invalid %s: %s", e.name, e.Message())
}

func (e ValidError) Field() string {
	return e.name
}

// Valids is for building a list of validation errors.
type Valids []ValidError

// Addf adds a validation error to the list with a formatted message using fmt.Sprintf.
func (b *Valids) Add(name, message string, args ...any) {
	*b = append(*b, ValidError{name, message, args})
}

func (b Valids) Error() string {
	if len(b) == 0 {
		return ""
	}

	var result string
	for i, err := range b {
		if i > 0 {
			result += "; "
		}
		result += err.Error()
	}
	return result
}

//go:build windows || darwin

package not_test

import (
	"errors"
	"fmt"

	"github.com/ollama/ollama/app/types/not"
)

func ExampleValids() {
	// This example demonstrates how to use the Valids type to create
	// a list of validation errors.
	//
	// The Valids type is a slice of ValidError values. Each ValidError
	// value represents a validation error.
	//
	// The Valids type has an Error method that returns a single error
	// value that represents all of the validation errors in the list.
	//
	// The Valids type is useful for collecting multiple validation errors
	// and returning them as a single error value.

	validate := func() error {
		var b not.Valids
		b.Add("name", "must be a valid name")
		b.Add("email", "%q: must be a valid email address", "invalid.email")
		return b
	}

	err := validate()
	var nv not.Valids
	if errors.As(err, &nv) {
		for _, v := range nv {
			fmt.Println(v)
		}
	}

	// Output:
	// invalid name: must be a valid name
	// invalid email: "invalid.email": must be a valid email address
}

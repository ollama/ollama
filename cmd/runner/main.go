package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/ollama/ollama/runner"
)

func main() {
	if err := runner.Execute(os.Args[1:]); err != nil {
		// Prefer showing the root cause when errors are wrapped.
		fmt.Fprintf(os.Stderr, "error: %v\n", err)

		exitCode := 1
		// If a wrapped os.PathError exists, treat it as a usage/environment error
		// (commonly "file not found", permission denied, etc.).
		var pe *os.PathError
		if errors.As(err, &pe) {
			exitCode = 2
		}

		os.Exit(exitCode)
	}
}

package main

import (
	"context"
	"fmt"
	"os"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd"
)

func main() {
	checkErr(cmd.NewCLI().ExecuteContext(context.Background()))
}

// checkErr prints the error message and exits the program if the message is not nil.
func checkErr(msg any) {
	if msg == nil {
		return
	}

	if errorResponse, ok := msg.(api.ErrorResponse); ok {
		// This error contains some additional information that we want to print
		fmt.Fprintln(os.Stderr, "Error: ", errorResponse.Err)
		if errorResponse.Hint != "" {
			fmt.Fprintf(os.Stderr, "\n%s\n", errorResponse.Hint)
		}
		os.Exit(1)
	}

	fmt.Fprintln(os.Stderr, "Error: ", msg)
	os.Exit(1)
}

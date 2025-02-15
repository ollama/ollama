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
func checkErr(err any) {
	if err == nil {
		return
	}

	switch e := err.(type) {
	case api.ErrorResponse:
		fmt.Fprintln(os.Stderr, "Error: ", e.Err)
		if e.Hint != "" {
			fmt.Fprintf(os.Stderr, "\n%s\n", e.Hint)
		}
	default:
		fmt.Fprintln(os.Stderr, "Error: ", err)
	}
	os.Exit(1)
}

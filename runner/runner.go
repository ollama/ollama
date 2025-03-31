package runner

import (
	"fmt"

	"github.com/ollama/ollama/runner/llamarunner"
	"github.com/ollama/ollama/runner/ollamarunner"
)

func Execute(args []string) error {
	print("=== start===\n")
	for i, arg := range args {
		fmt.Printf("args[%d]: %s\n", i, arg)
	}
	print("=== start end ===\n\n\n")

	if args[0] == "runner" {
		args = args[1:]
	}

	var newRunner bool
	if args[0] == "--ollama-engine" {
		args = args[1:]
		newRunner = true
	}
	if newRunner {
		return ollamarunner.Execute(args)
	} else {
		return llamarunner.Execute(args)
	}
}

package runner

import (
	"github.com/ollama/ollama/runner/newrunner"
	"github.com/ollama/ollama/runner/oldrunner"
)

func Execute(args []string) error {
	if args[0] == "runner" {
		args = args[1:]
	}

	var newRunner bool
	if args[0] == "--new-runner" {
		args = args[1:]
		newRunner = true
	}

	if newRunner {
		return newrunner.Execute(args)
	} else {
		return oldrunner.Execute(args)
	}
}

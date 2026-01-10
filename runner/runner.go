package runner

import (
	"github.com/ollama/ollama/runner/llamarunner"
	"github.com/ollama/ollama/runner/ollamarunner"
	imagerunner "github.com/ollama/ollama/x/imagegen/runner"
)

func Execute(args []string) error {
	if args[0] == "runner" {
		args = args[1:]
	}

	var newRunner bool
	var imageRunner bool
	if len(args) > 0 && args[0] == "--ollama-engine" {
		args = args[1:]
		newRunner = true
	}
	if len(args) > 0 && args[0] == "--image-engine" {
		args = args[1:]
		imageRunner = true
	}

	if imageRunner {
		return imagerunner.Execute(args)
	} else if newRunner {
		return ollamarunner.Execute(args)
	} else {
		return llamarunner.Execute(args)
	}
}

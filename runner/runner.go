package runner

import (
	"github.com/ollama/ollama/runner/llamarunner"
	"github.com/ollama/ollama/runner/ollamarunner"
	imagerunner "github.com/ollama/ollama/x/imagegen/runner"
)

func Execute(args []string) error {
	if len(args) > 0 {
		switch args[0] {
		case "--ollama-engine":
			return ollamarunner.Execute(args[1:])
		case "--image-engine":
			return imagerunner.Execute(args[1:])
		}
	}
	return llamarunner.Execute(args)
}

package runner

import (
	"github.com/ollama/ollama/runner/llamarunner"
	"github.com/ollama/ollama/runner/ollamarunner"
	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/mlxrunner"
)

func Execute(args []string) error {
	if args[0] == "runner" {
		args = args[1:]
	}

	if len(args) > 0 {
		switch args[0] {
		case "--ollama-engine":
			return ollamarunner.Execute(args[1:])
		case "--imagegen-engine":
			return imagegen.Execute(args[1:])
		case "--mlx-engine":
			return mlxrunner.Execute(args[1:])
		}
	}
	return llamarunner.Execute(args)
}

package main

import (
	"context"
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/cmd"
	"github.com/ollama/ollama/llama/runner"
)

func main() {
	if len(os.Args) >= 2 {
		if os.Args[1] == "_runner" {
			os.Args = append([]string{os.Args[0]}, os.Args[2:]...)
			runner.RunnerMain()
			return
		}
	}
	cobra.CheckErr(cmd.NewCLI().ExecuteContext(context.Background()))
}

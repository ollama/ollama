package main

import (
	"github.com/ollama/ollama/llama/runner"
	"github.com/spf13/cobra"
)

func main() {
	rootCmd := &cobra.Command{
		RunE: func(cmd *cobra.Command, arg []string) error {
			runner.RunnerMain(cmd)
			return nil
		},
	}
	runner.AddRunnerFlags(rootCmd)
	rootCmd.Execute()
}

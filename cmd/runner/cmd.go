package main

import (
	"github.com/spf13/cobra"

	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/llama/runner"
)

func main() {
	rootCmd := &cobra.Command{}
	runnerCmd := &cobra.Command{
		Use:   "runner",
		Short: llama.PrintSystemInfo(),
		RunE: func(cmd *cobra.Command, arg []string) error {
			runner.RunnerMain(cmd)
			return nil
		},
	}
	runner.AddRunnerFlags(runnerCmd)
	rootCmd.AddCommand(runnerCmd)
	_ = rootCmd.Execute()
}

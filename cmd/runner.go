package cmd

import (
	"os"

	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/llama/runner"
	"github.com/spf13/cobra"
)

func NewRunnerCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:    "runner",
		Short:  llama.PrintSystemInfo(),
		Hidden: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runner.Execute(os.Args[1:])
		},
		FParseErrWhitelist: cobra.FParseErrWhitelist{UnknownFlags: true},
	}

	cmd.SetHelpFunc(func(cmd *cobra.Command, args []string) {
		_ = runner.Execute(args[1:])
	})

	return cmd
}

package main

import (
	"context"

	"github.com/spf13/cobra"
	"ollama.com/cmd"
)

func main() {
	cobra.CheckErr(cmd.NewCLI().ExecuteContext(context.Background()))
}

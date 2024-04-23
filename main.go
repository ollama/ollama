package main

import (
	"context"

	"github.com/spf13/cobra"
	"github.com/uppercaveman/ollama-server/cmd"
)

func main() {
	cobra.CheckErr(cmd.NewCLI().ExecuteContext(context.Background()))
}

package main

import (
	"context"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/cmd"
)

func main() {
	cmd.PayloadFS = libEmbed
	cobra.CheckErr(cmd.NewCLI().ExecuteContext(context.Background()))
}

package main

//go:generate go run build.go -g -s

import (
	"context"

	"github.com/ollama/ollama/cmd"
	"github.com/spf13/cobra"
)

func main() {
	cobra.CheckErr(cmd.NewCLI().ExecuteContext(context.Background()))
}

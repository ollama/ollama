package main

import (
	"context"
	"log"

	"github.com/jmorganca/ollama/cmd"
	"github.com/spf13/cobra"
)

func main() {
	err := cmd.LoadDotEnvFromOllamaFolder()
	if err != nil {
		log.Fatal(err)
	}
	cobra.CheckErr(cmd.NewCLI().ExecuteContext(context.Background()))
}

package main

import (
	"context"
	"os"
	"os/signal"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/cmd"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt)
	go func() {
		<-sigChan
		cancel()
	}()

	cobra.CheckErr(cmd.NewCLI().ExecuteContext(ctx))
}

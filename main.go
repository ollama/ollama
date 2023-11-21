package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/jmorganca/ollama/cmd"
	"github.com/spf13/cobra"
)

func main() {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT)

	go func() {
		<-sigChan
		fmt.Print("\033[?25h")

		os.Exit(0)
	}()

	cobra.CheckErr(cmd.NewCLI().ExecuteContext(context.Background()))
}

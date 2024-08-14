package main

import (
	"context"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/cmd"

	"net/http"
	_ "net/http/pprof"
)

func main() {
	go func() {
		http.ListenAndServe("localhost:6060", nil)
	}()

	cobra.CheckErr(cmd.NewCLI().ExecuteContext(context.Background()))
}

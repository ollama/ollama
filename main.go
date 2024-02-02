package main

import (
	"context"
	"errors"
	"net/http"
	"os"

	"github.com/jmorganca/ollama/cmd"
	"github.com/spf13/cobra"
)

func main() {
	err := cmd.NewCLI().ExecuteContext(context.Background())
	if errors.Is(err, http.ErrServerClosed) {
		os.Exit(0)
	}
	cobra.CheckErr(err)
}

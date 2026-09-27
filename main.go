package main

import (
	"context"
	"fmt"
	"os"

	"github.com/ollama/ollama/cmd"
	"github.com/ollama/ollama/i18n"
)

func main() {
	// Replaces cobra.CheckErr so the "Error: " prefix and library-generated
	// messages (cobra/pflag/parser) follow the active locale. In English the
	// output is byte-identical to cobra.CheckErr.
	if err := cmd.NewCLI().ExecuteContext(context.Background()); err != nil {
		// Mirror cobra.CheckErr: an empty message prints nothing.
		if msg := i18n.Err(err).Error(); msg != "" {
			fmt.Fprintln(os.Stderr, i18n.T("Error: ")+msg)
		}
		os.Exit(1)
	}
}

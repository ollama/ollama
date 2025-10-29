package cmd

import (
	"fmt"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/client"
	"github.com/spf13/cobra"
)

func cmdRemove() *cobra.Command {
	return &cobra.Command{
		Use:     "remove [model]...",
		Aliases: []string{"rm"},
		Short:   "Remove one or more models from the local repository",
		Args:    cobra.MinimumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    removeHandler,
	}
}

func removeHandler(cmd *cobra.Command, args []string) error {
	c := client.New()
	for _, arg := range args {
		// TODO: stop model if it's running; skip if model is cloud
		if err := c.Delete(cmd.Context(), api.DeleteRequest{Model: arg}); err != nil {
			return err
		}
		fmt.Println("deleted", arg)
	}
	return nil
}

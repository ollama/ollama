package cmd

import (
	"fmt"
	"github.com/ollama/ollama/api"
	"github.com/spf13/cobra"
)

var deleteCmd = &cobra.Command{
	Use:     "rm MODEL [MODEL...]",
	Short:   "Remove a model",
	Args:    cobra.MinimumNArgs(1),
	PreRunE: checkServerHeartbeat,
	RunE:    DeleteHandler,
}

func init() {
	appendHostEnvDocs(deleteCmd)
}

func DeleteHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	for _, name := range args {
		req := api.DeleteRequest{Name: name}
		if err := client.Delete(cmd.Context(), &req); err != nil {
			return err
		}
		fmt.Printf("deleted '%s'\n", name)
	}
	return nil
}

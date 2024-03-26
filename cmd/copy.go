package cmd

import (
	"fmt"
	"github.com/jmorganca/ollama/api"
	"github.com/spf13/cobra"
)

var copyCmd = &cobra.Command{
	Use:     "cp SOURCE TARGET",
	Short:   "Copy a model",
	Args:    cobra.ExactArgs(2),
	PreRunE: checkServerHeartbeat,
	RunE:    CopyHandler,
}

func init() {
	appendHostEnvDocs(copyCmd)
	rootCmd.AddCommand(copyCmd)
}

func CopyHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	req := api.CopyRequest{Source: args[0], Destination: args[1]}
	if err := client.Copy(cmd.Context(), &req); err != nil {
		return err
	}
	fmt.Printf("copied '%s' to '%s'\n", args[0], args[1])
	return nil
}

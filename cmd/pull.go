package cmd

import (
	"os"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/client"
	"github.com/ollama/ollama/progress"
	"github.com/spf13/cobra"
)

func cmdPull() *cobra.Command {
	cmd := cobra.Command{
		Use:     "pull [model]",
		Short:   "Pull a model from a remote repository",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    pullHandler,
	}
	cmd.Flags().Bool("insecure", false, "Allow insecure server connections when pulling models")
	return &cmd
}

func pullHandler(cmd *cobra.Command, args []string) error {
	c := client.New()
	w, err := c.Pull(cmd.Context(), api.PullRequest{
		Name:     args[0],
		Insecure: must(cmd.Flags().GetBool("insecure")),
	})
	if err != nil {
		return err
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()
	return progressHandler(p, w)
}

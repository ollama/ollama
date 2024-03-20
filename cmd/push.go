package cmd

import (
	"fmt"
	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/progress"
	"github.com/spf13/cobra"
	"os"
)

var pushCmd = &cobra.Command{
	Use:     "push MODEL",
	Short:   "Push a model to a registry",
	Args:    cobra.ExactArgs(1),
	PreRunE: checkServerHeartbeat,
	RunE:    PushHandler,
}

func init() {
	pushCmd.Flags().Bool("insecure", false, "Use an insecure registry")
	appendHostEnvDocs(pushCmd)
	rootCmd.AddCommand(pushCmd)
}

func PushHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	insecure, err := cmd.Flags().GetBool("insecure")
	if err != nil {
		return err
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	bars := make(map[string]*progress.Bar)
	var status string
	var spinner *progress.Spinner

	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			if spinner != nil {
				spinner.Stop()
			}

			bar, ok := bars[resp.Digest]
			if !ok {
				bar = progress.NewBar(fmt.Sprintf("pushing %s...", resp.Digest[7:19]), resp.Total, resp.Completed)
				bars[resp.Digest] = bar
				p.Add(resp.Digest, bar)
			}

			bar.Set(resp.Completed)
		} else if status != resp.Status {
			if spinner != nil {
				spinner.Stop()
			}

			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}

		return nil
	}

	request := api.PushRequest{Name: args[0], Insecure: insecure}
	if err := client.Push(cmd.Context(), &request, fn); err != nil {
		return err
	}

	spinner.Stop()
	return nil
}

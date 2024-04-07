package cmd

import (
	"errors"
	"github.com/ollama/ollama/api"
	"github.com/spf13/cobra"
	"golang.org/x/term"
	"io"
	"net/http"
	"os"
	"slices"
	"strings"
)

var runCmd = &cobra.Command{
	Use:   "run MODEL [PROMPT]",
	Short: "Run a model",
	Args:  cobra.MinimumNArgs(1),
	RunE:  RunHandler,
}

func init() {
	runCmd.Flags().Bool("verbose", false, "Show timings for response")
	runCmd.Flags().Bool("insecure", false, "Use an insecure registry")
	runCmd.Flags().Bool("nowordwrap", false, "Don't wrap words to the next line automatically")
	runCmd.Flags().String("format", "", "Response format (e.g. json)")
	appendHostEnvDocs(runCmd)
}

func RunHandler(cmd *cobra.Command, args []string) error {
	if os.Getenv("OLLAMA_MODELS") != "" {
		return errors.New("OLLAMA_MODELS must only be set for 'ollama serve'")
	}

	if err := checkServerHeartbeat(cmd, args); err != nil {
		return err
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	name := args[0]

	// check if the model exists on the server
	show, err := client.Show(cmd.Context(), &api.ShowRequest{Name: name})
	var statusError api.StatusError
	switch {
	case errors.As(err, &statusError) && statusError.StatusCode == http.StatusNotFound:
		if err := PullHandler(cmd, []string{name}); err != nil {
			return err
		}

		show, err = client.Show(cmd.Context(), &api.ShowRequest{Name: name})
		if err != nil {
			return err
		}
	case err != nil:
		return err
	}

	interactive := true

	opts := runOptions{
		Model:       args[0],
		WordWrap:    os.Getenv("TERM") == "xterm-256color",
		Options:     map[string]interface{}{},
		MultiModal:  slices.Contains(show.Details.Families, "clip"),
		ParentModel: show.Details.ParentModel,
	}

	format, err := cmd.Flags().GetString("format")
	if err != nil {
		return err
	}
	opts.Format = format

	prompts := args[1:]
	// prepend stdin to the prompt if provided
	if !term.IsTerminal(int(os.Stdin.Fd())) {
		in, err := io.ReadAll(os.Stdin)
		if err != nil {
			return err
		}

		prompts = append([]string{string(in)}, prompts...)
		opts.WordWrap = false
		interactive = false
	}
	opts.Prompt = strings.Join(prompts, " ")
	if len(prompts) > 0 {
		interactive = false
	}

	nowrap, err := cmd.Flags().GetBool("nowordwrap")
	if err != nil {
		return err
	}
	opts.WordWrap = !nowrap

	if !interactive {
		return generate(cmd, opts)
	}

	return generateInteractive(cmd, opts)
}

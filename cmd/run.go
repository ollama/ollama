package cmd

import (
	"context"
	"errors"
	"fmt"
	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/progress"
	"github.com/spf13/cobra"
	"golang.org/x/term"
	"io"
	"net/http"
	"os"
	"os/signal"
	"slices"
	"strings"
	"syscall"
)

var runCmd = &cobra.Command{
	Use:     "run MODEL [PROMPT]",
	Short:   "Run a model",
	Args:    cobra.MinimumNArgs(1),
	PreRunE: checkServerHeartbeat,
	RunE:    RunHandler,
}

func init() {
	runCmd.Flags().Bool("verbose", false, "Show timings for response")
	runCmd.Flags().Bool("insecure", false, "Use an insecure registry")
	runCmd.Flags().Bool("nowordwrap", false, "Don't wrap words to the next line automatically")
	runCmd.Flags().String("format", "", "Response format (e.g. json)")
	appendHostEnvDocs(runCmd)
	rootCmd.AddCommand(runCmd)
}

func RunHandler(cmd *cobra.Command, args []string) error {
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

func generate(cmd *cobra.Command, opts runOptions) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	p := progress.NewProgress(os.Stderr)
	defer p.StopAndClear()

	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	var latest api.GenerateResponse

	generateContext, ok := cmd.Context().Value(generateContextKey("context")).([]int)
	if !ok {
		generateContext = []int{}
	}

	ctx, cancel := context.WithCancel(cmd.Context())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT)

	go func() {
		<-sigChan
		cancel()
	}()

	var state *displayResponseState = &displayResponseState{}

	fn := func(response api.GenerateResponse) error {
		p.StopAndClear()

		latest = response
		content := response.Response

		displayResponse(content, opts.WordWrap, state)

		return nil
	}

	if opts.MultiModal {
		opts.Prompt, opts.Images, err = extractFileData(opts.Prompt)
		if err != nil {
			return err
		}
	}

	request := api.GenerateRequest{
		Model:    opts.Model,
		Prompt:   opts.Prompt,
		Context:  generateContext,
		Images:   opts.Images,
		Format:   opts.Format,
		System:   opts.System,
		Template: opts.Template,
		Options:  opts.Options,
	}

	if err := client.Generate(ctx, &request, fn); err != nil {
		if errors.Is(err, context.Canceled) {
			return nil
		}
		return err
	}

	if opts.Prompt != "" {
		fmt.Println()
		fmt.Println()
	}

	if !latest.Done {
		return nil
	}

	verbose, err := cmd.Flags().GetBool("verbose")
	if err != nil {
		return err
	}

	if verbose {
		latest.Summary()
	}

	ctx = context.WithValue(cmd.Context(), generateContextKey("context"), latest.Context)
	cmd.SetContext(ctx)

	return nil
}

package cmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/progress"
)

// runDecision runs a decision model, which answers yes/no questions about
// text: the first argument (or stdin) is the text and the rest are questions.
func runDecision(cmd *cobra.Command, opts runOptions, args []string, interactive bool) error {
	if interactive {
		opts.Decision = true
		return generateInteractive(cmd, opts)
	}
	if len(args) < 2 {
		return fmt.Errorf("%s answers questions about text, e.g. ollama run %s \"My order never arrived\" \"Is this urgent?\"", opts.Model, opts.Model)
	}

	text := args[0]
	var images []api.ImageData
	if opts.MultiModal {
		var err error
		text, images, err = extractFileData(text)
		if err != nil {
			return err
		}
	}
	return decide(cmd, opts, text, images, args[1:]...)
}

// decide asks a decision model yes/no questions about text and prints the
// likelier answer to each with its probability.
func decide(cmd *cobra.Command, opts runOptions, text string, images []api.ImageData, questions ...string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}
	// Piped text and multiline messages end in a newline that isn't part of the text.
	state, err := json.Marshal(strings.TrimRight(text, "\r\n"))
	if err != nil {
		return err
	}
	req := &api.SystemOneRequest{
		Model:     opts.Model,
		State:     state,
		Images:    images,
		Questions: &api.SystemOneQuestions{},
		KeepAlive: opts.KeepAlive,
	}
	for _, question := range questions {
		instructions, _ := json.Marshal(question)
		req.Questions.Set(question, api.SystemOneQuestion{Type: "noul", Instructions: instructions})
	}

	ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt)
	defer stop()

	p := progress.NewProgress(os.Stderr)
	p.Add("", progress.NewSpinner(""))
	start := time.Now()
	resp, err := client.SystemOne(ctx, req)
	p.StopAndClear()
	if errors.Is(err, context.Canceled) {
		return nil
	} else if err != nil {
		return err
	}

	if opts.Format == "json" {
		enc := json.NewEncoder(os.Stdout)
		enc.SetEscapeHTML(false)
		return enc.Encode(resp)
	}

	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	for _, question := range questions {
		raw, ok := resp.Answers.Get(question)
		if !ok {
			return fmt.Errorf("no answer to %q", question)
		}
		var answer struct {
			Noul float64 `json:"noul"`
		}
		if err := json.Unmarshal(raw, &answer); err != nil {
			return fmt.Errorf("answer to %q: %w", question, err)
		}
		likelier, p := "yes", answer.Noul
		if p < 0.5 {
			likelier, p = "no", 1-p
		}
		if len(questions) > 1 {
			fmt.Fprintf(tw, "%s\t", question)
		}
		fmt.Fprintf(tw, "%s\t%.0f%%\n", likelier, p*100)
	}
	if err := tw.Flush(); err != nil {
		return err
	}

	if verbose, _ := cmd.Flags().GetBool("verbose"); verbose {
		metrics := api.Metrics{
			TotalDuration:         time.Since(start),
			PromptEvalCount:       resp.Usage.InputTokens,
			PromptEvalCachedCount: resp.PromptEvalCachedCount,
		}
		metrics.Summary()
	}
	return nil
}

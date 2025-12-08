package main

import (
	"cmp"
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
)

type flagOptions struct {
	models      *string
	epochs      *int
	maxTokens   *int
	temperature *float64
	seed        *int
	timeout     *int
	prompt      *string
	imageFile   *string
	keepAlive   *float64
	format      *string
	outputFile  *string
	debug       *bool
	verbose     *bool
}

type Metrics struct {
	Model    string
	Step     string
	Count    int
	Duration time.Duration
}

var once sync.Once

const DefaultPrompt = `Please write a descriptive story about a llama named Alonso who grows up to be President of the Land of Llamas. Include details about Alonso's childhood, adolescent years, and how he grew up to be a political mover and shaker. Write the story with a sense of whimsy.`

func OutputMetrics(w io.Writer, format string, metrics []Metrics, verbose bool) {
	switch format {
	case "benchstat":
		if verbose {
			printHeader := func() {
				fmt.Fprintf(w, "sysname: %s\n", runtime.GOOS)
				fmt.Fprintf(w, "machine: %s\n", runtime.GOARCH)
			}
			once.Do(printHeader)
		}
		for _, m := range metrics {
			if m.Step == "generate" || m.Step == "prefill" {
				if m.Count > 0 {
					nsPerToken := float64(m.Duration.Nanoseconds()) / float64(m.Count)
					tokensPerSec := float64(m.Count) / (float64(m.Duration.Nanoseconds()) + 1e-12) * 1e9

					fmt.Fprintf(w, "BenchmarkModel/name=%s/step=%s %d %.2f ns/token %.2f token/sec\n",
						m.Model, m.Step, m.Count, nsPerToken, tokensPerSec)
				} else {
					fmt.Fprintf(w, "BenchmarkModel/name=%s/step=%s %d 0 ns/token 0 token/sec\n",
						m.Model, m.Step, m.Count)
				}
			} else {
				var suffix string
				if m.Step == "load" {
					suffix = "/step=load"
				}
				fmt.Fprintf(w, "BenchmarkModel/name=%s%s 1 %d ns/request\n",
					m.Model, suffix, m.Duration.Nanoseconds())
			}
		}
	case "csv":
		printHeader := func() {
			headings := []string{"NAME", "STEP", "COUNT", "NS_PER_COUNT", "TOKEN_PER_SEC"}
			fmt.Fprintln(w, strings.Join(headings, ","))
		}
		once.Do(printHeader)

		for _, m := range metrics {
			if m.Step == "generate" || m.Step == "prefill" {
				var nsPerToken float64
				var tokensPerSec float64
				if m.Count > 0 {
					nsPerToken = float64(m.Duration.Nanoseconds()) / float64(m.Count)
					tokensPerSec = float64(m.Count) / (float64(m.Duration.Nanoseconds()) + 1e-12) * 1e9
				}
				fmt.Fprintf(w, "%s,%s,%d,%.2f,%.2f\n", m.Model, m.Step, m.Count, nsPerToken, tokensPerSec)
			} else {
				fmt.Fprintf(w, "%s,%s,1,%d,0\n", m.Model, m.Step, m.Duration.Nanoseconds())
			}
		}
	case "markdown":
		printHeader := func() {
			fmt.Fprintln(w, "| Model | Step | Count | Duration | nsPerToken | tokensPerSec |")
			fmt.Fprintln(w, "|-------|------|-------|----------|------------|--------------|")
		}
		once.Do(printHeader)

		for _, m := range metrics {
			var nsPerToken, tokensPerSec float64
			var nsPerTokenStr, tokensPerSecStr string

			if m.Step == "generate" || m.Step == "prefill" {
				nsPerToken = float64(m.Duration.Nanoseconds()) / float64(m.Count)
				tokensPerSec = float64(m.Count) / (float64(m.Duration.Nanoseconds()) + 1e-12) * 1e9
				nsPerTokenStr = fmt.Sprintf("%.2f", nsPerToken)
				tokensPerSecStr = fmt.Sprintf("%.2f", tokensPerSec)
			} else {
				nsPerTokenStr = "-"
				tokensPerSecStr = "-"
			}

			fmt.Fprintf(w, "| %s | %s | %d | %v | %s | %s |\n",
				m.Model, m.Step, m.Count, m.Duration, nsPerTokenStr, tokensPerSecStr)
		}
	default:
		fmt.Fprintf(os.Stderr, "Unknown output format '%s'\n", format)
	}
}

func BenchmarkChat(fOpt flagOptions) error {
	models := strings.Split(*fOpt.models, ",")

	// todo - add multi-image support
	var imgData api.ImageData
	var err error
	if *fOpt.imageFile != "" {
		imgData, err = readImage(*fOpt.imageFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ERROR: Couldn't read image '%s': %v\n", *fOpt.imageFile, err)
			return err
		}
	}

	if *fOpt.debug && imgData != nil {
		fmt.Fprintf(os.Stderr, "Read file '%s'\n", *fOpt.imageFile)
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: Couldn't create ollama client: %v\n", err)
		return err
	}

	var out io.Writer = os.Stdout
	if fOpt.outputFile != nil && *fOpt.outputFile != "" {
		f, err := os.OpenFile(*fOpt.outputFile, os.O_CREATE|os.O_WRONLY, 0o644)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ERROR: cannot open output file %s: %v\n", *fOpt.outputFile, err)
			return err
		}
		defer f.Close()
		out = f
	}

	for _, model := range models {
		for range *fOpt.epochs {
			options := make(map[string]interface{})
			if *fOpt.maxTokens > 0 {
				options["num_predict"] = *fOpt.maxTokens
			}
			options["temperature"] = *fOpt.temperature
			if fOpt.seed != nil && *fOpt.seed > 0 {
				options["seed"] = *fOpt.seed
			}

			var keepAliveDuration *api.Duration
			if *fOpt.keepAlive > 0 {
				duration := api.Duration{Duration: time.Duration(*fOpt.keepAlive * float64(time.Second))}
				keepAliveDuration = &duration
			}

			req := &api.ChatRequest{
				Model: model,
				Messages: []api.Message{
					{
						Role:    "user",
						Content: *fOpt.prompt,
					},
				},
				Options:   options,
				KeepAlive: keepAliveDuration,
			}

			if imgData != nil {
				req.Messages[0].Images = []api.ImageData{imgData}
			}

			var responseMetrics *api.Metrics

			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*fOpt.timeout)*time.Second)
			defer cancel()

			err = client.Chat(ctx, req, func(resp api.ChatResponse) error {
				if *fOpt.debug {
					fmt.Fprintf(os.Stderr, "%s", cmp.Or(resp.Message.Thinking, resp.Message.Content))
				}

				if resp.Done {
					responseMetrics = &resp.Metrics
				}
				return nil
			})

			if *fOpt.debug {
				fmt.Fprintln(os.Stderr)
			}

			if err != nil {
				if ctx.Err() == context.DeadlineExceeded {
					fmt.Fprintf(os.Stderr, "ERROR: Chat request timed out with model '%s' after %vs\n", model, 1)
					continue
				}
				fmt.Fprintf(os.Stderr, "ERROR: Couldn't chat with model '%s': %v\n", model, err)
				continue
			}

			if responseMetrics == nil {
				fmt.Fprintf(os.Stderr, "ERROR: No metrics received for model '%s'\n", model)
				continue
			}

			metrics := []Metrics{
				{
					Model:    model,
					Step:     "prefill",
					Count:    responseMetrics.PromptEvalCount,
					Duration: responseMetrics.PromptEvalDuration,
				},
				{
					Model:    model,
					Step:     "generate",
					Count:    responseMetrics.EvalCount,
					Duration: responseMetrics.EvalDuration,
				},
				{
					Model:    model,
					Step:     "load",
					Count:    1,
					Duration: responseMetrics.LoadDuration,
				},
				{
					Model:    model,
					Step:     "total",
					Count:    1,
					Duration: responseMetrics.TotalDuration,
				},
			}

			OutputMetrics(out, *fOpt.format, metrics, *fOpt.verbose)

			if *fOpt.keepAlive > 0 {
				time.Sleep(time.Duration(*fOpt.keepAlive*float64(time.Second)) + 200*time.Millisecond)
			}
		}
	}

	return nil
}

func readImage(filePath string) (api.ImageData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}

	return api.ImageData(data), nil
}

func main() {
	fOpt := flagOptions{
		models:      flag.String("model", "", "Model to benchmark"),
		epochs:      flag.Int("epochs", 6, "Number of epochs (iterations) per model"),
		maxTokens:   flag.Int("max-tokens", 200, "Maximum tokens for model response"),
		temperature: flag.Float64("temperature", 0, "Temperature parameter"),
		seed:        flag.Int("seed", 0, "Random seed"),
		timeout:     flag.Int("timeout", 60*5, "Timeout in seconds (default 300s)"),
		prompt:      flag.String("p", DefaultPrompt, "Prompt to use"),
		imageFile:   flag.String("image", "", "Filename for an image to include"),
		keepAlive:   flag.Float64("k", 0, "Keep alive duration in seconds"),
		format:      flag.String("format", "markdown", "Output format [benchstat|csv] (default benchstat)"),
		outputFile:  flag.String("output", "", "Output file for results (stdout if empty)"),
		verbose:     flag.Bool("v", false, "Show system information"),
		debug:       flag.Bool("debug", false, "Show debug information"),
	}

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [OPTIONS]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Description:\n")
		fmt.Fprintf(os.Stderr, "  Model benchmarking tool with configurable parameters\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  bench -model gpt-oss:20b -epochs 3 -temperature 0.7\n")
	}
	flag.Parse()

	if !slices.Contains([]string{"markdown", "benchstat", "csv"}, *fOpt.format) {
		fmt.Fprintf(os.Stderr, "ERROR: Unknown format '%s'\n", *fOpt.format)
		os.Exit(1)
	}

	if len(*fOpt.models) == 0 {
		fmt.Fprintf(os.Stderr, "ERROR: No model(s) specified to benchmark.\n")
		flag.Usage()
		return
	}

	BenchmarkChat(fOpt)
}

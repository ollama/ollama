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
	models       *string
	epochs       *int
	maxTokens    *int
	temperature  *float64
	seed         *int
	timeout      *int
	prompt       *string
	imageFile    *string
	keepAlive    *float64
	format       *string
	outputFile   *string
	debug        *bool
	verbose      *bool
	warmup       *int
	promptTokens *int
}

type Metrics struct {
	Model    string
	Step     string
	Count    int
	Duration time.Duration
}

type ModelInfo struct {
	Name              string
	ParameterSize     string
	QuantizationLevel string
	Family            string
	SizeBytes         int64
	VRAMBytes         int64
}

const DefaultPrompt = `Please write a descriptive story about a llama named Alonso who grows up to be President of the Land of Llamas. Include details about Alonso's childhood, adolescent years, and how he grew up to be a political mover and shaker. Write the story with a sense of whimsy.`

// Word list for generating prompts targeting a specific token count.
var promptWordList = []string{
	"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
	"a", "bright", "sunny", "day", "in", "the", "meadow", "where",
	"flowers", "bloom", "and", "birds", "sing", "their", "morning",
	"songs", "while", "gentle", "breeze", "carries", "sweet", "scent",
	"of", "pine", "trees", "across", "rolling", "hills", "toward",
	"distant", "mountains", "covered", "with", "fresh", "snow",
	"beneath", "clear", "blue", "sky", "children", "play", "near",
	"old", "stone", "bridge", "that", "crosses", "winding", "river",
}

func generatePromptForTokenCount(targetTokens int, epoch int) string {
	// ~1.3 tokens per word heuristic
	targetWords := int(float64(targetTokens) / 1.3)
	if targetWords < 1 {
		targetWords = 1
	}

	// Vary the starting offset by epoch to defeat KV cache prefix matching
	offset := epoch * 7 // stride by a prime to get good distribution
	n := len(promptWordList)
	words := make([]string, targetWords)
	for i := range words {
		words[i] = promptWordList[((i+offset)%n+n)%n]
	}
	return strings.Join(words, " ")
}

func buildGenerateRequest(model string, fOpt flagOptions, imgData api.ImageData, epoch int) *api.GenerateRequest {
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

	prompt := *fOpt.prompt
	if *fOpt.promptTokens > 0 {
		prompt = generatePromptForTokenCount(*fOpt.promptTokens, epoch)
	} else {
		// Vary the prompt per epoch to defeat KV cache prefix matching
		prompt = fmt.Sprintf("[%d] %s", epoch, prompt)
	}

	req := &api.GenerateRequest{
		Model:     model,
		Prompt:    prompt,
		Raw:       true,
		Options:   options,
		KeepAlive: keepAliveDuration,
	}

	if imgData != nil {
		req.Images = []api.ImageData{imgData}
	}

	return req
}

func fetchModelInfo(ctx context.Context, client *api.Client, model string) ModelInfo {
	info := ModelInfo{Name: model}
	resp, err := client.Show(ctx, &api.ShowRequest{Model: model})
	if err != nil {
		fmt.Fprintf(os.Stderr, "WARNING: Could not fetch model info for '%s': %v\n", model, err)
		return info
	}
	info.ParameterSize = resp.Details.ParameterSize
	info.QuantizationLevel = resp.Details.QuantizationLevel
	info.Family = resp.Details.Family
	return info
}

func fetchMemoryUsage(ctx context.Context, client *api.Client, model string) (size, vram int64) {
	resp, err := client.ListRunning(ctx)
	if err != nil {
		if debug := os.Getenv("OLLAMA_DEBUG"); debug != "" {
			fmt.Fprintf(os.Stderr, "WARNING: Could not fetch memory usage: %v\n", err)
		}
		return 0, 0
	}
	for _, m := range resp.Models {
		if m.Name == model || m.Model == model {
			return m.Size, m.SizeVRAM
		}
	}
	// Try prefix match (model names may include :latest or tags)
	for _, m := range resp.Models {
		if strings.HasPrefix(m.Name, model) || strings.HasPrefix(m.Model, model) {
			return m.Size, m.SizeVRAM
		}
	}
	return 0, 0
}

func outputFormatHeader(w io.Writer, format string, verbose bool) {
	switch format {
	case "benchstat":
		if verbose {
			fmt.Fprintf(w, "goos: %s\n", runtime.GOOS)
			fmt.Fprintf(w, "goarch: %s\n", runtime.GOARCH)
		}
	case "csv":
		headings := []string{"NAME", "STEP", "COUNT", "NS_PER_COUNT", "TOKEN_PER_SEC"}
		fmt.Fprintln(w, strings.Join(headings, ","))
	}
}

func outputModelInfo(w io.Writer, format string, info ModelInfo) {
	params := cmp.Or(info.ParameterSize, "unknown")
	quant := cmp.Or(info.QuantizationLevel, "unknown")
	family := cmp.Or(info.Family, "unknown")

	memStr := ""
	if info.SizeBytes > 0 {
		memStr = fmt.Sprintf(" | Size: %d | VRAM: %d", info.SizeBytes, info.VRAMBytes)
	}
	fmt.Fprintf(w, "# Model: %s | Params: %s | Quant: %s | Family: %s%s\n",
		info.Name, params, quant, family, memStr)
}

func OutputMetrics(w io.Writer, format string, metrics []Metrics, verbose bool) {
	switch format {
	case "benchstat":
		for _, m := range metrics {
			if m.Step == "generate" || m.Step == "prefill" {
				if m.Count > 0 {
					nsPerToken := float64(m.Duration.Nanoseconds()) / float64(m.Count)
					tokensPerSec := float64(m.Count) / (float64(m.Duration.Nanoseconds()) + 1e-12) * 1e9
					fmt.Fprintf(w, "BenchmarkModel/name=%s/step=%s 1 %.2f ns/token %.2f token/sec\n",
						m.Model, m.Step, nsPerToken, tokensPerSec)
				} else {
					fmt.Fprintf(w, "BenchmarkModel/name=%s/step=%s 1 0 ns/token 0 token/sec\n",
						m.Model, m.Step)
				}
			} else if m.Step == "ttft" {
				fmt.Fprintf(w, "BenchmarkModel/name=%s/step=ttft 1 %d ns/op\n",
					m.Model, m.Duration.Nanoseconds())
			} else {
				fmt.Fprintf(w, "BenchmarkModel/name=%s/step=%s 1 %d ns/op\n",
					m.Model, m.Step, m.Duration.Nanoseconds())
			}
		}
	case "csv":
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
	default:
		fmt.Fprintf(os.Stderr, "Unknown output format '%s'\n", format)
	}
}

func BenchmarkModel(fOpt flagOptions) error {
	models := strings.Split(*fOpt.models, ",")

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

	outputFormatHeader(out, *fOpt.format, *fOpt.verbose)

	// Log prompt-tokens info in debug mode
	if *fOpt.debug && *fOpt.promptTokens > 0 {
		prompt := generatePromptForTokenCount(*fOpt.promptTokens, 0)
		wordCount := len(strings.Fields(prompt))
		fmt.Fprintf(os.Stderr, "Generated prompt targeting ~%d tokens (%d words, varied per epoch)\n", *fOpt.promptTokens, wordCount)
	}

	for _, model := range models {
		// Fetch model info
		infoCtx, infoCancel := context.WithTimeout(context.Background(), 10*time.Second)
		info := fetchModelInfo(infoCtx, client, model)
		infoCancel()

		// Warmup phase (uses negative epoch numbers to avoid colliding with timed epochs)
		for i := range *fOpt.warmup {
			req := buildGenerateRequest(model, fOpt, imgData, -(i + 1))
			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*fOpt.timeout)*time.Second)

			err = client.Generate(ctx, req, func(resp api.GenerateResponse) error {
				return nil
			})
			cancel()

			if err != nil {
				fmt.Fprintf(os.Stderr, "WARNING: Warmup %d/%d for %s failed: %v\n", i+1, *fOpt.warmup, model, err)
			} else if *fOpt.debug {
				fmt.Fprintf(os.Stderr, "Warmup %d/%d for %s complete\n", i+1, *fOpt.warmup, model)
			}
		}

		// Fetch memory usage once after warmup (model is loaded and stable)
		memCtx, memCancel := context.WithTimeout(context.Background(), 5*time.Second)
		info.SizeBytes, info.VRAMBytes = fetchMemoryUsage(memCtx, client, model)
		memCancel()

		outputModelInfo(out, *fOpt.format, info)

		// Timed epoch loop
		for epoch := range *fOpt.epochs {
			req := buildGenerateRequest(model, fOpt, imgData, epoch)
			var responseMetrics *api.Metrics

			// TTFT tracking
			requestStart := time.Now()
			var ttft time.Duration
			var ttftOnce sync.Once

			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*fOpt.timeout)*time.Second)

			err = client.Generate(ctx, req, func(resp api.GenerateResponse) error {
				if *fOpt.debug {
					fmt.Fprintf(os.Stderr, "%s", cmp.Or(resp.Thinking, resp.Response))
				}

				// Capture TTFT on first content
				ttftOnce.Do(func() {
					if resp.Response != "" || resp.Thinking != "" {
						ttft = time.Since(requestStart)
					}
				})

				if resp.Done {
					responseMetrics = &resp.Metrics
				}
				return nil
			})
			cancel() // explicit cancel instead of defer

			if *fOpt.debug {
				fmt.Fprintln(os.Stderr)
			}

			if err != nil {
				if ctx.Err() == context.DeadlineExceeded {
					fmt.Fprintf(os.Stderr, "ERROR: Request timed out with model '%s' after %vs\n", model, *fOpt.timeout)
					continue
				}
				fmt.Fprintf(os.Stderr, "ERROR: Couldn't generate with model '%s': %v\n", model, err)
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
					Step:     "ttft",
					Count:    1,
					Duration: ttft,
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

			if *fOpt.debug && *fOpt.promptTokens > 0 {
				fmt.Fprintf(os.Stderr, "Generated prompt targeting ~%d tokens (actual: %d)\n",
					*fOpt.promptTokens, responseMetrics.PromptEvalCount)
			}

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
		models:       flag.String("model", "", "Model to benchmark"),
		epochs:       flag.Int("epochs", 6, "Number of epochs (iterations) per model"),
		maxTokens:    flag.Int("max-tokens", 200, "Maximum tokens for model response"),
		temperature:  flag.Float64("temperature", 0, "Temperature parameter"),
		seed:         flag.Int("seed", 0, "Random seed"),
		timeout:      flag.Int("timeout", 60*5, "Timeout in seconds (default 300s)"),
		prompt:       flag.String("p", DefaultPrompt, "Prompt to use"),
		imageFile:    flag.String("image", "", "Filename for an image to include"),
		keepAlive:    flag.Float64("k", 0, "Keep alive duration in seconds"),
		format:       flag.String("format", "benchstat", "Output format [benchstat|csv]"),
		outputFile:   flag.String("output", "", "Output file for results (stdout if empty)"),
		verbose:      flag.Bool("v", false, "Show system information"),
		debug:        flag.Bool("debug", false, "Show debug information"),
		warmup:       flag.Int("warmup", 1, "Number of warmup requests before timing"),
		promptTokens: flag.Int("prompt-tokens", 0, "Generate prompt targeting ~N tokens (0 = use -p prompt)"),
	}

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [OPTIONS]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Description:\n")
		fmt.Fprintf(os.Stderr, "  Model benchmarking tool with configurable parameters\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  bench -model gemma3,llama3 -epochs 6\n")
		fmt.Fprintf(os.Stderr, "  bench -model gemma3 -epochs 6 -prompt-tokens 512 -format csv\n")
	}
	flag.Parse()

	if !slices.Contains([]string{"benchstat", "csv"}, *fOpt.format) {
		fmt.Fprintf(os.Stderr, "ERROR: Unknown format '%s'\n", *fOpt.format)
		os.Exit(1)
	}

	if len(*fOpt.models) == 0 {
		fmt.Fprintf(os.Stderr, "ERROR: No model(s) specified to benchmark.\n")
		flag.Usage()
		return
	}

	BenchmarkModel(fOpt)
}

package main

import (
	"cmp"
	"context"
	"crypto/rand"
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
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
	numCtx       *int
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
	NumCtx            int64
}

const DefaultPrompt = `Please write a descriptive story about a llama named Alonso who grows up to be President of the Land of Llamas. Include details about Alonso's childhood, adolescent years, and how he grew up to be a political mover and shaker. Write the story with a sense of whimsy.`

// Generated prompts come from the MIT-licensed HumanEval problem set
// (openai/human-eval, see prompts/LICENSE). Real code avoids the repetition
// loops that synthetic filler triggers in speculative decoding drafts
// (MTP/dFlash), which inflate throughput and acceptance measurements.
//
//go:embed prompts/HumanEval.jsonl
var humanEvalJSONL []byte

type evalProblem struct {
	TaskID string `json:"task_id"`
	Prompt string `json:"prompt"`
}

var humanEvalProblems = sync.OnceValue(func() []evalProblem {
	var problems []evalProblem
	for line := range strings.Lines(string(humanEvalJSONL)) {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var p evalProblem
		if err := json.Unmarshal([]byte(line), &p); err == nil && p.Prompt != "" {
			problems = append(problems, p)
		}
	}
	return problems
})

// humanEvalWordBounds returns the smallest single problem's word count and the
// word count of the full problem set.
func humanEvalWordBounds() (small, total int) {
	small = math.MaxInt
	for _, p := range humanEvalProblems() {
		w := len(strings.Fields(strings.TrimSpace(p.Prompt)))
		small = min(small, w)
		total += w
	}
	return small, total
}

// tokensPerWordHeuristic seeds the prompt calibration loop; the exact ratio
// varies per model and is resolved against the live model's tokenizer.
const tokensPerWordHeuristic = 1.3

func estimatePromptWords(targetTokens int) int {
	words := int(float64(targetTokens) / tokensPerWordHeuristic)
	return max(words, 1)
}

// nonceHeader is the per-request cache-busting prefix. It reads like a
// vendored-file header rather than benchmark scaffolding, which reasoning
// models take as a cue to analyze the harness instead of writing code.
func nonceHeader(cacheBuster string) string {
	return "# -*- coding: utf-8 -*-\n# checksum: " + cacheBuster + "\n\n\n"
}

// generateCodePrompt packs whole problems (signature + docstring — the model
// completes the body), never truncated or repeated, up to wordCount words.
// cacheBuster defeats prefix caches; variation rotates the window for retries.
// Deterministic for a given (wordCount, variation).
func generateCodePrompt(wordCount, variation int, cacheBuster string) string {
	problems := humanEvalProblems()
	header := nonceHeader(cacheBuster)
	used := len(strings.Fields(header))

	var parts []string
	included := make([]bool, len(problems))
	for i := variation * 23 % len(problems); !included[i]; i = (i + 1) % len(problems) {
		body := strings.TrimSpace(problems[i].Prompt)
		if used+len(strings.Fields(body)) > wordCount {
			break
		}
		parts = append(parts, body)
		included[i] = true
		used += len(strings.Fields(body))
	}

	// Best-fit the remaining slack: the largest unused problem that fits.
	best, bestWords := -1, 0
	for j, p := range problems {
		if included[j] {
			continue
		}
		if w := len(strings.Fields(strings.TrimSpace(p.Prompt))); w > bestWords && used+w <= wordCount {
			best, bestWords = j, w
		}
	}
	if best >= 0 {
		parts = append(parts, strings.TrimSpace(problems[best].Prompt))
	}

	return header + strings.Join(parts, "\n\n\n")
}

func benchmarkOptions(fOpt flagOptions) map[string]any {
	options := make(map[string]interface{})
	if *fOpt.maxTokens > 0 {
		options["num_predict"] = *fOpt.maxTokens
	}
	options["temperature"] = *fOpt.temperature
	if fOpt.seed != nil && *fOpt.seed > 0 {
		options["seed"] = *fOpt.seed
	}
	if fOpt.numCtx != nil && *fOpt.numCtx > 0 {
		options["num_ctx"] = *fOpt.numCtx
	}
	return options
}

func benchmarkKeepAlive(fOpt flagOptions) *api.Duration {
	if *fOpt.keepAlive > 0 {
		return &api.Duration{Duration: time.Duration(*fOpt.keepAlive * float64(time.Second))}
	}
	return nil
}

// buildChatRequest builds a single-message benchmark request through the
// model's chat template. promptWords is the calibrated word count for
// generated prompts; ignored for -p prompts.
func buildChatRequest(model string, fOpt flagOptions, imgData api.ImageData, cacheBuster string, variation, promptWords int) *api.ChatRequest {
	var content string
	if *fOpt.promptTokens > 0 {
		content = generateCodePrompt(promptWords, variation, cacheBuster)
	} else {
		// A leading unique nonce defeats prefix cache reuse across runs.
		content = nonceHeader(cacheBuster) + *fOpt.prompt
	}

	msg := api.Message{Role: "user", Content: content}
	if imgData != nil {
		msg.Images = []api.ImageData{imgData}
	}

	return &api.ChatRequest{
		Model:     model,
		Messages:  []api.Message{msg},
		Options:   benchmarkOptions(fOpt),
		KeepAlive: benchmarkKeepAlive(fOpt),
	}
}

// promptTargetBounds is the acceptable rendered-token window: 1% under, 2% over.
func promptTargetBounds(targetTokens int) (int, int) {
	return targetTokens * 99 / 100, targetTokens * 102 / 100
}

// smallestCodePromptWords is the word budget for the smallest well-formed
// prompt: header plus the smallest problem.
func smallestCodePromptWords() int {
	small, _ := humanEvalWordBounds()
	const headerWords = 8 // "# -*- coding: utf-8 -*-" + "# checksum: <nonce>"
	return headerWords + small
}

// resolvePromptWords calibrates the prompt against the model's rendered token
// count (template included). Targets below the model-measured minimum are
// rejected (use -p); targets beyond the full problem set warn and use it.
// Errors when the server cannot count tokens.
func resolvePromptWords(ctx context.Context, client *api.Client, model string, fOpt flagOptions) (int, error) {
	targetTokens := *fOpt.promptTokens

	floorReq := buildChatRequest(model, fOpt, nil, rand.Text(), 0, smallestCodePromptWords())
	floorResp, err := client.ChatInputTokens(ctx, floorReq)
	if err != nil {
		return 0, fmt.Errorf("cannot count prompt tokens with model '%s': %w", model, err)
	}
	if targetTokens < floorResp.InputTokens {
		return 0, fmt.Errorf("prompt target %d tokens is below the minimum coding prompt size ~%d tokens for model %q; use -p for smaller prompts", targetTokens, floorResp.InputTokens, model)
	}

	_, total := humanEvalWordBounds()
	ceiling := total * 2
	if targetTokens > ceiling {
		fmt.Fprintf(os.Stderr, "WARNING: prompt target %d tokens exceeds the problem set (~%d tokens); the prompt will use the full set\n", targetTokens, ceiling)
	}

	minTokens, maxTokens := promptTargetBounds(targetTokens)
	words := estimatePromptWords(targetTokens)
	lastCount := 0
	for range 4 {
		req := buildChatRequest(model, fOpt, nil, rand.Text(), 0, words)
		resp, err := client.ChatInputTokens(ctx, req)
		if err != nil {
			return 0, fmt.Errorf("cannot count prompt tokens with model '%s': %w", model, err)
		}
		lastCount = resp.InputTokens
		if lastCount >= minTokens && lastCount <= maxTokens {
			return words, nil
		}
		words = words * targetTokens / max(lastCount, 1)
	}
	fmt.Fprintf(os.Stderr, "WARNING: prompt resolved to %d rendered tokens against target %d (tolerance -1%%/+2%%); results use the actual count\n", lastCount, targetTokens)
	return words, nil
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
	for _, m := range resp.Models {
		if strings.HasPrefix(m.Name, model) || strings.HasPrefix(m.Model, model) {
			return m.Size, m.SizeVRAM
		}
	}
	return 0, 0
}

func fetchContextLength(ctx context.Context, client *api.Client, model string) int64 {
	resp, err := client.ListRunning(ctx)
	if err != nil {
		return 0
	}
	for _, m := range resp.Models {
		if m.Name == model || m.Model == model || strings.HasPrefix(m.Name, model) || strings.HasPrefix(m.Model, model) {
			return int64(m.ContextLength)
		}
	}
	return 0
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
	ctxStr := ""
	if info.NumCtx > 0 {
		ctxStr = fmt.Sprintf(" | NumCtx: %d", info.NumCtx)
	}
	fmt.Fprintf(w, "# Model: %s | Params: %s | Quant: %s | Family: %s%s%s\n",
		info.Name, params, quant, family, memStr, ctxStr)
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
		fmt.Fprintf(os.Stderr, "Generated code prompt targeting ~%d tokens (unique per request)\n", *fOpt.promptTokens)
	}

	for _, model := range models {
		// Fetch model info
		infoCtx, infoCancel := context.WithTimeout(context.Background(), 10*time.Second)
		info := fetchModelInfo(infoCtx, client, model)
		infoCancel()

		// Resolve the generated prompt to the target token count against the
		// live model (count includes the chat template).
		promptWords := 0
		if *fOpt.promptTokens > 0 {
			calCtx, calCancel := context.WithTimeout(context.Background(), time.Duration(*fOpt.timeout)*time.Second)
			promptWords, err = resolvePromptWords(calCtx, client, model, fOpt)
			calCancel()
			if err != nil {
				fmt.Fprintf(os.Stderr, "ERROR: %v (server needs input token counting support)\n", err)
				return err
			}
		}

		// Warmup phase
		for i := range *fOpt.warmup {
			req := buildChatRequest(model, fOpt, imgData, rand.Text(), i, promptWords)
			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*fOpt.timeout)*time.Second)

			err = client.Chat(ctx, req, func(resp api.ChatResponse) error {
				return nil
			})
			cancel()

			if err != nil {
				fmt.Fprintf(os.Stderr, "WARNING: Warmup %d/%d for %s failed: %v\n", i+1, *fOpt.warmup, model, err)
			} else if *fOpt.debug {
				fmt.Fprintf(os.Stderr, "Warmup %d/%d for %s complete\n", i+1, *fOpt.warmup, model)
			}
		}

		// Fetch memory/context info once after warmup (model is loaded and stable)
		memCtx, memCancel := context.WithTimeout(context.Background(), 5*time.Second)
		info.SizeBytes, info.VRAMBytes = fetchMemoryUsage(memCtx, client, model)
		if fOpt.numCtx != nil && *fOpt.numCtx > 0 {
			info.NumCtx = int64(*fOpt.numCtx)
		} else {
			info.NumCtx = fetchContextLength(memCtx, client, model)
		}
		memCancel()

		outputModelInfo(out, *fOpt.format, info)

		// Timed epoch loop
		shortCount := 0
		for epoch := range *fOpt.epochs {
			var responseMetrics *api.Metrics
			var ttft time.Duration
			short := false

			// Retry loop: if the model hits a stop token before max-tokens,
			// retry with a different prompt (up to maxRetries times).
			const maxRetries = 3
			for attempt := range maxRetries + 1 {
				responseMetrics = nil
				ttft = 0
				var ttftOnce sync.Once

				req := buildChatRequest(model, fOpt, imgData, rand.Text(), attempt, promptWords)
				requestStart := time.Now()

				ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*fOpt.timeout)*time.Second)

				err = client.Chat(ctx, req, func(resp api.ChatResponse) error {
					if *fOpt.debug {
						fmt.Fprintf(os.Stderr, "%s", cmp.Or(resp.Message.Thinking, resp.Message.Content))
					}

					// Capture TTFT on first content
					ttftOnce.Do(func() {
						if resp.Message.Content != "" || resp.Message.Thinking != "" {
							ttft = time.Since(requestStart)
						}
					})

					if resp.Done {
						responseMetrics = &resp.Metrics
					}
					return nil
				})
				cancel()

				if *fOpt.debug {
					fmt.Fprintln(os.Stderr)
				}

				if err != nil {
					if ctx.Err() == context.DeadlineExceeded {
						fmt.Fprintf(os.Stderr, "ERROR: Request timed out with model '%s' after %vs\n", model, *fOpt.timeout)
					} else {
						fmt.Fprintf(os.Stderr, "ERROR: Couldn't generate with model '%s': %v\n", model, err)
					}
					break
				}

				if responseMetrics == nil {
					fmt.Fprintf(os.Stderr, "ERROR: No metrics received for model '%s'\n", model)
					break
				}

				// Check if the response was shorter than requested
				short = *fOpt.maxTokens > 0 && responseMetrics.EvalCount < *fOpt.maxTokens
				if !short || attempt == maxRetries {
					break
				}

				if *fOpt.debug {
					fmt.Fprintf(os.Stderr, "Short response (%d/%d tokens), retrying with different prompt (attempt %d/%d)\n",
						responseMetrics.EvalCount, *fOpt.maxTokens, attempt+1, maxRetries)
				}
			}

			if err != nil || responseMetrics == nil {
				continue
			}

			if short {
				shortCount++
				if *fOpt.debug {
					fmt.Fprintf(os.Stderr, "WARNING: Short response (%d/%d tokens) after %d retries for epoch %d\n",
						responseMetrics.EvalCount, *fOpt.maxTokens, maxRetries, epoch+1)
				}
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

		if shortCount > 0 {
			fmt.Fprintf(os.Stderr, "WARNING: %d/%d epochs for '%s' had short responses (<%d tokens). Generation metrics may be unreliable.\n",
				shortCount, *fOpt.epochs, model, *fOpt.maxTokens)
		}

		// Unload model before moving to the next one
		unloadModel(client, model, *fOpt.timeout)
	}

	return nil
}

func unloadModel(client *api.Client, model string, timeout int) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()

	zero := api.Duration{Duration: 0}
	req := &api.GenerateRequest{
		Model:     model,
		KeepAlive: &zero,
	}
	_ = client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		return nil
	})
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
		numCtx:       flag.Int("num-ctx", 0, "Context size (0 = server default)"),
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

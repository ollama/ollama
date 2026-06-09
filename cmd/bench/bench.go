package main

import (
	"cmp"
	"context"
	"errors"
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
	numCtx       *int

	// Target selection. When -runner is empty and -spawn is false, bench drives
	// the full ollama server via the public API (legacy default). When -runner
	// is set, bench probes it and auto-detects whether it is an MLX runner or a
	// llama-server.
	runner *string // host:port of a runner (MLX or llama-server) to drive directly

	// MLX-runner spawn options (only when -runner is empty).
	spawn     *bool
	ollamaBin *string

	// Profiling controls.
	mode      *string // prefill | decode | both
	ignoreEOS *bool   // disable stop tokens so generation runs exactly maxTokens
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

// Benchmark modes. prefill and decode produce single-phase workloads for clean
// profiler capture windows; both is the legacy mixed run.
const (
	modePrefill = "prefill"
	modeDecode  = "decode"
	modeBoth    = "both"
)

// completionParams is the backend-agnostic description of one completion call.
type completionParams struct {
	prompt      string
	numPredict  int // 0 = prefill-only
	temperature float64
	seed        int
	numCtx      int
	ignoreEOS   bool
	image       api.ImageData
	debug       bool
}

// completionResult carries the timing data every backend reports. Fields a
// backend cannot supply are left zero.
type completionResult struct {
	promptEvalCount    int
	promptEvalDuration time.Duration
	evalCount          int
	evalDuration       time.Duration
	ttft               time.Duration
	loadDuration       time.Duration
	totalDuration      time.Duration
}

// errNoMetrics signals that a completion finished without delivering a final
// (Done) metrics record, so its timings are unusable.
var errNoMetrics = errors.New("no metrics received")

// benchBackend abstracts the thing under test: the full ollama server, an MLX
// runner driven directly, or a llama-server driven directly.
type benchBackend interface {
	// Name identifies the backend for logging.
	Name() string
	// ModelInfo returns best-effort display metadata.
	ModelInfo(ctx context.Context, fOpt flagOptions) ModelInfo
	// Complete runs one completion and returns its timing metrics.
	Complete(ctx context.Context, p completionParams) (completionResult, error)
	// Cleanup tears the backend down (unload model, kill spawned subprocess).
	Cleanup(timeout int)
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

// tokensPerWord is the calibrated ratio of tokens to words for the current model.
// Initialized with a heuristic, then updated during warmup based on actual tokenization.
var tokensPerWord = 1.3

func generatePromptForTokenCount(targetTokens int, epoch int) string {
	targetWords := int(float64(targetTokens) / tokensPerWord)
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

// calibratePromptTokens adjusts tokensPerWord based on actual tokenization from a warmup run.
func calibratePromptTokens(targetTokens, actualTokens, wordCount int) {
	if actualTokens <= 0 || wordCount <= 0 {
		return
	}
	tokensPerWord = float64(actualTokens) / float64(wordCount)
	newWords := int(float64(targetTokens) / tokensPerWord)
	fmt.Fprintf(os.Stderr, "bench: calibrated %.2f tokens/word (target=%d, got=%d, words=%d → %d)\n",
		tokensPerWord, targetTokens, actualTokens, wordCount, newWords)
}

// buildParams derives the completion parameters for one epoch, shaping the
// prompt, num_predict, and ignore_eos according to the benchmark mode:
//
//   - prefill: vary the prompt per epoch (force a cache miss) and request
//     num_predict 0 so only the prompt is processed.
//   - decode: hold the prompt fixed across epochs so the runner's prefix cache
//     hits and the measured window is pure decode.
//   - both: legacy mixed run, prompt varied per epoch.
//
// numPredict convention: -1 = generate to the context limit, 0 = prefill-only,
// N>0 = exactly N tokens.
func buildParams(fOpt flagOptions, mode string, imgData api.ImageData, epoch int) completionParams {
	// decode mode keeps the prompt identical so the KV prefix cache hits;
	// prefill/both vary it per epoch to defeat the cache.
	promptEpoch := epoch
	if mode == modeDecode {
		promptEpoch = 0
	}

	var prompt string
	if *fOpt.promptTokens > 0 {
		prompt = generatePromptForTokenCount(*fOpt.promptTokens, promptEpoch)
	} else if mode == modeDecode {
		prompt = *fOpt.prompt
	} else {
		prompt = fmt.Sprintf("[%d] %s", epoch, *fOpt.prompt)
	}

	numPredict := -1
	if *fOpt.maxTokens > 0 {
		numPredict = *fOpt.maxTokens
	}
	if mode == modePrefill {
		numPredict = 0
	}

	seed := 0
	if fOpt.seed != nil {
		seed = *fOpt.seed
	}
	numCtx := 0
	if fOpt.numCtx != nil {
		numCtx = *fOpt.numCtx
	}

	return completionParams{
		prompt:      prompt,
		numPredict:  numPredict,
		temperature: *fOpt.temperature,
		seed:        seed,
		numCtx:      numCtx,
		ignoreEOS:   *fOpt.ignoreEOS && mode != modePrefill,
		image:       imgData,
		debug:       *fOpt.debug,
	}
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
	mode := *fOpt.mode
	if mode != modePrefill && mode != modeDecode && mode != modeBoth {
		return fmt.Errorf("unknown -mode %q (want prefill|decode|both)", mode)
	}

	var imgData api.ImageData
	var err error
	if *fOpt.imageFile != "" {
		imgData, err = readImage(*fOpt.imageFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ERROR: Couldn't read image '%s': %v\n", *fOpt.imageFile, err)
			return err
		}
		if *fOpt.debug {
			fmt.Fprintf(os.Stderr, "Read file '%s'\n", *fOpt.imageFile)
		}
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

	if *fOpt.debug && *fOpt.promptTokens > 0 {
		prompt := generatePromptForTokenCount(*fOpt.promptTokens, 0)
		fmt.Fprintf(os.Stderr, "Generated prompt targeting ~%d tokens (%d words)\n", *fOpt.promptTokens, len(strings.Fields(prompt)))
	}

	// Direct backends serve a single, already-loaded model; the -model value is
	// just a label. With -runner set, bench probes the endpoint and auto-detects
	// MLX runner vs llama-server. The serve path iterates the comma-separated
	// model list as before.
	switch {
	case *fOpt.runner != "" || *fOpt.spawn:
		backend, err := newDirectBackend(fOpt)
		if err != nil {
			return err
		}
		defer backend.Cleanup(*fOpt.timeout)
		runBenchmark(out, backend, fOpt, mode, imgData, *fOpt.models)
	default:
		client, err := api.ClientFromEnvironment()
		if err != nil {
			fmt.Fprintf(os.Stderr, "ERROR: Couldn't create ollama client: %v\n", err)
			return err
		}
		for _, model := range strings.Split(*fOpt.models, ",") {
			backend := &serveBackend{client: client, model: model}
			runBenchmark(out, backend, fOpt, mode, imgData, model)
			backend.Cleanup(*fOpt.timeout)
		}
	}

	return nil
}

// runBenchmark drives one backend: warmup (+ calibration and decode-cache
// priming), then the timed epoch loop, emitting metrics through OutputMetrics.
func runBenchmark(out io.Writer, backend benchBackend, fOpt flagOptions, mode string, imgData api.ImageData, model string) {
	timeout := time.Duration(*fOpt.timeout) * time.Second

	// Warmup. In decode mode the prompt is fixed, so warmup also primes the
	// runner's prefix cache for the timed epochs.
	for i := range *fOpt.warmup {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		res, err := backend.Complete(ctx, buildParams(fOpt, mode, imgData, -(i+1)))
		cancel()
		if err != nil {
			fmt.Fprintf(os.Stderr, "WARNING: Warmup %d/%d for %s failed: %v\n", i+1, *fOpt.warmup, model, err)
			continue
		}
		if *fOpt.debug {
			fmt.Fprintf(os.Stderr, "Warmup %d/%d for %s complete\n", i+1, *fOpt.warmup, model)
		}
		if i == *fOpt.warmup-1 && *fOpt.promptTokens > 0 && res.promptEvalCount > 0 {
			prompt := generatePromptForTokenCount(*fOpt.promptTokens, -(i + 1))
			calibratePromptTokens(*fOpt.promptTokens, res.promptEvalCount, len(strings.Fields(prompt)))
		}
	}

	// Decode mode needs a primed cache; if the user disabled warmup, prime once.
	if mode == modeDecode && *fOpt.warmup == 0 {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		if _, err := backend.Complete(ctx, buildParams(fOpt, mode, imgData, 0)); err != nil {
			fmt.Fprintf(os.Stderr, "WARNING: decode prime for %s failed: %v\n", model, err)
		}
		cancel()
	}

	info := backend.ModelInfo(context.Background(), fOpt)
	outputModelInfo(out, *fOpt.format, info)

	shortCount := 0
	for epoch := range *fOpt.epochs {
		var res completionResult
		var err error
		short := false

		// Retry only matters when stop tokens can truncate generation: not in
		// prefill mode and not when ignore_eos guarantees the full count.
		const maxRetries = 3
		for attempt := range maxRetries + 1 {
			p := buildParams(fOpt, mode, imgData, epoch+attempt*1000)
			ctx, cancel := context.WithTimeout(context.Background(), timeout)
			res, err = backend.Complete(ctx, p)
			timedOut := ctx.Err() == context.DeadlineExceeded
			cancel()

			if err != nil {
				switch {
				case errors.Is(err, errNoMetrics):
					fmt.Fprintf(os.Stderr, "ERROR: No metrics received for model '%s'\n", model)
				case timedOut:
					fmt.Fprintf(os.Stderr, "ERROR: Request timed out with model '%s' after %vs\n", model, *fOpt.timeout)
				default:
					fmt.Fprintf(os.Stderr, "ERROR: Couldn't generate with model '%s': %v\n", model, err)
				}
				break
			}

			canRetry := !p.ignoreEOS && p.numPredict > 0
			short = canRetry && res.evalCount < p.numPredict
			if !short || attempt == maxRetries {
				break
			}
			if *fOpt.debug {
				fmt.Fprintf(os.Stderr, "Short response (%d/%d tokens), retrying (attempt %d/%d)\n",
					res.evalCount, p.numPredict, attempt+1, maxRetries)
			}
		}

		if err != nil {
			continue
		}
		if short {
			shortCount++
		}

		metrics := []Metrics{
			{Model: model, Step: "prefill", Count: res.promptEvalCount, Duration: res.promptEvalDuration},
			{Model: model, Step: "generate", Count: res.evalCount, Duration: res.evalDuration},
			{Model: model, Step: "ttft", Count: 1, Duration: res.ttft},
			{Model: model, Step: "load", Count: 1, Duration: res.loadDuration},
			{Model: model, Step: "total", Count: 1, Duration: res.totalDuration},
		}
		OutputMetrics(out, *fOpt.format, metrics, *fOpt.verbose)

		if *fOpt.debug && *fOpt.promptTokens > 0 {
			fmt.Fprintf(os.Stderr, "Prompt targeting ~%d tokens (actual: %d)\n", *fOpt.promptTokens, res.promptEvalCount)
		}

		if *fOpt.keepAlive > 0 {
			time.Sleep(time.Duration(*fOpt.keepAlive*float64(time.Second)) + 200*time.Millisecond)
		}
	}

	if shortCount > 0 {
		fmt.Fprintf(os.Stderr, "WARNING: %d/%d epochs for '%s' had short responses (<%d tokens). Use -ignore-eos for exact counts.\n",
			shortCount, *fOpt.epochs, model, *fOpt.maxTokens)
	}
}

// serveBackend drives the full ollama server through the public API (the legacy
// default target).
type serveBackend struct {
	client *api.Client
	model  string
}

func (b *serveBackend) Name() string { return "ollama-serve" }

func (b *serveBackend) Complete(ctx context.Context, p completionParams) (completionResult, error) {
	options := map[string]any{"temperature": p.temperature}
	// num_predict convention: 0 = prefill-only, N>0 = exact; negative means
	// "unlimited", which we express by leaving the option unset.
	if p.numPredict >= 0 {
		options["num_predict"] = p.numPredict
	}
	if p.seed > 0 {
		options["seed"] = p.seed
	}
	if p.numCtx > 0 {
		options["num_ctx"] = p.numCtx
	}

	req := &api.GenerateRequest{
		Model:   b.model,
		Prompt:  p.prompt,
		Raw:     true,
		Options: options,
	}
	if p.image != nil {
		req.Images = []api.ImageData{p.image}
	}

	requestStart := time.Now()
	var res completionResult
	var ttftOnce sync.Once
	gotDone := false
	err := b.client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		if p.debug {
			fmt.Fprintf(os.Stderr, "%s", cmp.Or(resp.Thinking, resp.Response))
		}
		ttftOnce.Do(func() {
			if resp.Response != "" || resp.Thinking != "" {
				res.ttft = time.Since(requestStart)
			}
		})
		if resp.Done {
			gotDone = true
			res.promptEvalCount = resp.Metrics.PromptEvalCount
			res.promptEvalDuration = resp.Metrics.PromptEvalDuration
			res.evalCount = resp.Metrics.EvalCount
			res.evalDuration = resp.Metrics.EvalDuration
			res.loadDuration = resp.Metrics.LoadDuration
			res.totalDuration = resp.Metrics.TotalDuration
		}
		return nil
	})
	if p.debug {
		fmt.Fprintln(os.Stderr)
	}
	if err == nil && !gotDone {
		return res, errNoMetrics
	}
	return res, err
}

func (b *serveBackend) ModelInfo(ctx context.Context, fOpt flagOptions) ModelInfo {
	info := fetchModelInfo(ctx, b.client, b.model)
	info.SizeBytes, info.VRAMBytes = fetchMemoryUsage(ctx, b.client, b.model)
	if fOpt.numCtx != nil && *fOpt.numCtx > 0 {
		info.NumCtx = int64(*fOpt.numCtx)
	} else {
		info.NumCtx = fetchContextLength(ctx, b.client, b.model)
	}
	return info
}

func (b *serveBackend) Cleanup(timeout int) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()
	zero := api.Duration{Duration: 0}
	_ = b.client.Generate(ctx, &api.GenerateRequest{Model: b.model, KeepAlive: &zero}, func(api.GenerateResponse) error {
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

		runner:    flag.String("runner", "", "Drive a runner directly at host:port, bypassing ollama serve (auto-detects MLX runner vs llama-server)"),
		spawn:     flag.Bool("spawn", false, "Spawn the runner subprocess: MLX runner for an MLX model, or llama-server for a GGUF model/path"),
		ollamaBin: flag.String("ollama", "", "Path to the ollama binary for -spawn (default: PATH or this executable)"),
		mode:      flag.String("mode", "both", "Benchmark mode [prefill|decode|both]"),
		ignoreEOS: flag.Bool("ignore-eos", false, "Disable stop tokens so generation runs exactly -max-tokens (direct backends only)"),
	}

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [OPTIONS]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Description:\n")
		fmt.Fprintf(os.Stderr, "  Model benchmarking tool with configurable parameters\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  # Benchmark via ollama serve (default)\n")
		fmt.Fprintf(os.Stderr, "  bench -model gemma3,llama3 -epochs 6\n")
		fmt.Fprintf(os.Stderr, "  bench -model gemma3 -epochs 6 -prompt-tokens 512 -format csv\n\n")
		fmt.Fprintf(os.Stderr, "  # Profile an MLX runner directly. Start it under a profiler first, e.g.\n")
		fmt.Fprintf(os.Stderr, "  #   ollama runner --mlx-engine --model gemma3 --port 8081 --profile\n")
		fmt.Fprintf(os.Stderr, "  bench -model gemma3 -runner 127.0.0.1:8081 -mode prefill -prompt-tokens 2048\n")
		fmt.Fprintf(os.Stderr, "  bench -model gemma3 -runner 127.0.0.1:8081 -mode decode  -prompt-tokens 2048 -max-tokens 128 -ignore-eos\n\n")
		fmt.Fprintf(os.Stderr, "  # Spawn the runner for a quick (unprofiled) direct benchmark\n")
		fmt.Fprintf(os.Stderr, "  bench -model gemma3 -spawn -mode decode -ignore-eos               # MLX model -> MLX runner\n")
		fmt.Fprintf(os.Stderr, "  bench -model llama3.2:latest -spawn -mode decode -ignore-eos      # GGUF model -> llama-server\n\n")
		fmt.Fprintf(os.Stderr, "  # Compare against an already-running llama-server (same -runner flag; auto-detected)\n")
		fmt.Fprintf(os.Stderr, "  bench -model llama3.2 -runner 127.0.0.1:8091 -mode decode -ignore-eos\n")
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

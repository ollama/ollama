package main

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net/http"
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
	openaiURL    *string
	apiKey       *string
}

type Metrics struct {
	Model             string
	Step              string
	Count             int
	CachedPromptCount *int
	Duration          time.Duration
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

const (
	// tokensPerWordSeed seeds the calibration search only. HumanEval code runs
	// 2.0-2.5 tokens per word across tokenizers; the exact ratio varies per
	// model and is resolved against the live model's tokenizer.
	tokensPerWordSeed = 2.1
	// calibrationHeadroom places the first search probe under the target so a
	// prompt that satisfies the ceiling is always in hand.
	calibrationHeadroom = 0.9
	// maxCalibrationProbes bounds the sizing requests spent per model. Whole
	// problems cost 35-500 tokens each and word count predicts token count only
	// to within ~35% per problem, so the search samples rather than solves.
	maxCalibrationProbes = 8
	// padGoalDivisor sets how close to the target whole problems must land
	// before the search stops and lets pad letters cover the rest. Tightening it
	// buys less filler at the cost of more probes.
	padGoalDivisor = 256
	minPadGoal     = 4
	// nonceLetters is the base cache-busting nonce length, in pad letters.
	// 26^12 is far more than a run needs to keep every prefix distinct.
	nonceLetters            = 12
	maxShortResponseRetries = 3
	// problemStartStride spreads the calibrated problem set across HumanEval;
	// it is coprime with the number of embedded problems.
	problemStartStride = 23
)

// promptPlan is the resolved prompt shape for one model: a set of whole
// problems that measures at or under the target, plus the pad letters that
// close the remaining gap so every request lands on the target exactly.
type promptPlan struct {
	words int // word budget the problem set is packed to
	pad   int // pad letters beyond nonceLetters, one token each
}

// promptNonce returns n space-separated lowercase letters.
//
// Each " x" costs exactly one token on every tokenizer this was checked
// against, so the nonce contributes a fixed number of tokens no matter what it
// draws, and lengthening it moves the prompt size by exactly one token. A
// packed random string cannot do either job: its token count swings by ~5
// tokens per draw, which alone puts an exact prompt size out of reach.
func promptNonce(n int) string {
	letters := make([]byte, 0, 2*n)
	for i := range n {
		if i > 0 {
			letters = append(letters, ' ')
		}
		letters = append(letters, byte('a'+rand.IntN(26)))
	}
	return string(letters)
}

// nonceHeader is the per-request cache-busting prefix. It reads like a
// vendored-file header rather than benchmark scaffolding, which reasoning
// models take as a cue to analyze the harness instead of writing code.
func nonceHeader(cacheBuster string) string {
	return "# -*- coding: utf-8 -*-\n# checksum: " + cacheBuster + "\n\n\n"
}

func codePromptProblems(wordCount, variation int) []evalProblem {
	problems := humanEvalProblems()
	used := 0

	var selected []evalProblem
	included := make([]bool, len(problems))
	for i := 0; !included[i]; i = (i + problemStartStride) % len(problems) {
		problem := problems[i]
		words := len(strings.Fields(strings.TrimSpace(problem.Prompt)))
		if used+words > wordCount {
			break
		}
		selected = append(selected, problem)
		included[i] = true
		used += words
	}

	// Best-fit the remaining slack with complete problems until none fits, so
	// the packed set sits within the smallest problem of the word budget.
	for {
		best, bestWords := -1, 0
		for i, problem := range problems {
			if included[i] {
				continue
			}
			if words := len(strings.Fields(strings.TrimSpace(problem.Prompt))); words > bestWords && used+words <= wordCount {
				best, bestWords = i, words
			}
		}
		if best < 0 {
			break
		}
		selected = append(selected, problems[best])
		included[best] = true
		used += bestWords
	}
	if len(selected) == 0 {
		return nil
	}

	start := ((variation % len(selected)) + len(selected)) % len(selected)
	ordered := make([]evalProblem, 0, len(selected))
	ordered = append(ordered, selected[start:]...)
	ordered = append(ordered, selected[:start]...)
	return ordered
}

// codePromptBody packs whole problems (signature + docstring — the model
// completes the body), never truncated or repeated, up to wordCount words.
// variation rotates one fixed problem set so requests vary without changing the
// prompt material or its token count: problems are separated by blank lines, so
// the tokenizer treats each as its own run and the total is order-independent.
// Deterministic for a given (wordCount, variation).
func codePromptBody(wordCount, variation int) string {
	problems := codePromptProblems(wordCount, variation)
	parts := make([]string, len(problems))
	for i, problem := range problems {
		parts[i] = strings.TrimSpace(problem.Prompt)
	}
	return strings.Join(parts, "\n\n\n")
}

// generateCodePrompt renders a plan into a request-unique prompt. The pad
// letters ride in the header nonce, where they read as digest material and
// leave the coding request itself made only of whole problems.
func generateCodePrompt(plan promptPlan, variation int) string {
	return nonceHeader(promptNonce(nonceLetters+plan.pad)) + codePromptBody(plan.words, variation)
}

// benchmarkPromptVariation keeps primary epoch windows consecutive while
// assigning retries to disjoint windows after them.
func benchmarkPromptVariation(warmups, epochs, epoch, attempt int) int {
	return warmups + attempt*epochs + epoch
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

// Shared by both transports so a given variation is byte-identical on each.
func benchPromptContent(fOpt flagOptions, variation int, plan promptPlan) string {
	if *fOpt.promptTokens > 0 {
		return generateCodePrompt(plan, variation)
	}
	// A leading unique nonce defeats prefix cache reuse across runs.
	return nonceHeader(promptNonce(nonceLetters)) + *fOpt.prompt
}

// buildChatRequest builds a single-message benchmark request through the
// model's chat template. plan is the calibrated prompt shape for generated
// prompts; ignored for -p prompts.
func buildChatRequest(model string, fOpt flagOptions, imgData api.ImageData, variation int, plan promptPlan) *api.ChatRequest {
	msg := api.Message{Role: "user", Content: benchPromptContent(fOpt, variation, plan)}
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

// maxPadTokens caps the filler a plan may carry. Past this the prompt says more
// about the padding than about the coding request, so the run reports the size
// it could reach instead of padding out to the target.
func maxPadTokens(targetTokens int) int {
	return max(256, targetTokens/4)
}

// smallestCodePromptWords is the word budget for the smallest well-formed
// prompt: a single, complete problem.
func smallestCodePromptWords() int {
	small, _ := humanEvalWordBounds()
	return small
}

func fullCodePromptWords() int {
	_, total := humanEvalWordBounds()
	return total
}

func measurePromptTokens(ctx context.Context, client *api.Client, model string, fOpt flagOptions, imgData api.ImageData, plan promptPlan) (int, error) {
	maxTokens := 1
	fOpt.maxTokens = &maxTokens
	req := buildChatRequest(model, fOpt, imgData, 0, plan)

	var metrics *api.Metrics
	err := client.Chat(ctx, req, func(resp api.ChatResponse) error {
		if resp.Done {
			metrics = &resp.Metrics
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	if metrics == nil {
		return 0, errors.New("no metrics received")
	}
	return metrics.PromptEvalCount, nil
}

// calibratePrompt owns prompt sizing. The timed benchmark path only consumes
// the returned plan and never recalibrates between epochs.
//
// The target is a ceiling: the search only keeps a problem set that measures at
// or under it, then pads to the target exactly, so a run lands on the size the
// caller asked for. Whole problems get within a handful of tokens of the target;
// pad letters, which cost exactly one token each, cover the remainder.
//
// TODO: Replace this chat-based calibration with the token-count API when
// cmd/bench can depend on it. Keep the replacement behind this function so
// sizing remains outside warmups and timed requests.
//
// measurePlan is the only transport-specific step, so the OpenAI path
// calibrates without an Ollama client.
func calibratePrompt(measurePlan func(promptPlan) (int, error), model string, fOpt flagOptions) (promptPlan, error) {
	targetTokens := *fOpt.promptTokens
	maxWords := fullCodePromptWords()

	measured := make(map[int]int)
	measure := func(words int) (int, error) {
		if tokens, ok := measured[words]; ok {
			return tokens, nil
		}
		tokens, err := measurePlan(promptPlan{words: words})
		if err != nil {
			return 0, fmt.Errorf("cannot measure prompt tokens with model '%s': %w", model, err)
		}
		measured[words] = tokens
		return tokens, nil
	}

	// The smallest well-formed prompt doubles as the floor check and as a
	// feasible anchor, so the search always has something under the ceiling.
	plan := promptPlan{words: smallestCodePromptWords()}
	bestTokens, err := measure(plan.words)
	if err != nil {
		return promptPlan{}, err
	}
	if targetTokens < bestTokens {
		return promptPlan{}, fmt.Errorf("prompt target %d tokens is below the minimum coding prompt size ~%d tokens for model %q; use -p for smaller prompts", targetTokens, bestTokens, model)
	}

	padGoal := max(minPadGoal, targetTokens/padGoalDivisor)
	aim := targetTokens - max(1, padGoal/3)
	words := min(int(calibrationHeadroom*float64(targetTokens)/tokensPerWordSeed), maxWords)
	prevWords, prevTokens := 0, 0
	for len(measured) < maxCalibrationProbes && targetTokens-bestTokens > padGoal {
		if _, seen := measured[words]; seen {
			break // the search has stopped moving; take the best set so far
		}
		tokens, err := measure(words)
		if err != nil {
			return promptPlan{}, err
		}
		if tokens <= targetTokens && tokens > bestTokens {
			plan.words, bestTokens = words, tokens
		}

		// Word count predicts token count only loosely per problem, so step
		// with the secant slope between the last two probes where possible and
		// fall back to the running average.
		slope := float64(tokens) / float64(words)
		if prevWords != 0 && words != prevWords && tokens != prevTokens {
			slope = float64(tokens-prevTokens) / float64(words-prevWords)
		}
		prevWords, prevTokens = words, tokens
		next := words + int(math.Round(float64(aim-tokens)/max(slope, 0.5)))
		step := 1
		if tokens > aim {
			step = -1
		}
		for next >= 1 && next <= maxWords {
			if _, seen := measured[next]; !seen {
				break
			}
			next += step
		}
		words = min(max(next, 1), maxWords)
	}

	plan.pad = targetTokens - bestTokens
	if plan.pad > maxPadTokens(targetTokens) {
		if plan.words >= maxWords {
			fmt.Fprintf(os.Stderr, "WARNING: prompt target %d tokens exceeds the problem set (~%d tokens); the prompt will use the full set\n", targetTokens, bestTokens)
		} else {
			fmt.Fprintf(os.Stderr, "WARNING: could not size a %d-token prompt for model %q within %d probes; falling back to %d tokens\n", targetTokens, model, maxCalibrationProbes, bestTokens)
		}
		plan.pad = 0
		return plan, nil
	}

	// Confirm the padded prompt, and correct once if the model's tokenizer
	// prices pad letters at anything other than one token each.
	for range 2 {
		actual, err := measurePlan(plan)
		if err != nil {
			return promptPlan{}, fmt.Errorf("cannot measure prompt tokens with model '%s': %w", model, err)
		}
		if actual == targetTokens {
			return plan, nil
		}
		if corrected := plan.pad + targetTokens - actual; corrected >= 0 {
			plan.pad = corrected
			continue
		}
		break
	}

	// Padding cannot reach the target on this tokenizer. Drop it and keep the
	// ceiling: the unpadded set is known to measure at or under the target.
	fmt.Fprintf(os.Stderr, "WARNING: could not pin the prompt to %d tokens for model %q; falling back to %d tokens\n", targetTokens, model, bestTokens)
	plan.pad = 0
	return plan, nil
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
		headings := []string{"NAME", "STEP", "COUNT", "NS_PER_COUNT", "TOKEN_PER_SEC", "CACHED_PROMPT_COUNT"}
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

// Rate steps are tokens/sec; variants suffix the base name (prefill_server).
func isRateStep(step string) bool {
	return strings.HasPrefix(step, "prefill") || strings.HasPrefix(step, "generate")
}

func OutputMetrics(w io.Writer, format string, metrics []Metrics, verbose bool) {
	switch format {
	case "benchstat":
		for _, m := range metrics {
			if isRateStep(m.Step) {
				var promptCounts string
				if strings.HasPrefix(m.Step, "prefill") {
					promptCounts = fmt.Sprintf(" %d processed-prompt-token", m.Count)
					if m.CachedPromptCount != nil {
						promptCounts += fmt.Sprintf(" %d cached-prompt-token", *m.CachedPromptCount)
					}
				}
				if m.Count > 0 {
					nsPerToken := float64(m.Duration.Nanoseconds()) / float64(m.Count)
					tokensPerSec := float64(m.Count) / (float64(m.Duration.Nanoseconds()) + 1e-12) * 1e9
					fmt.Fprintf(w, "BenchmarkModel/name=%s/step=%s 1 %.2f ns/token %.2f token/sec%s\n",
						m.Model, m.Step, nsPerToken, tokensPerSec, promptCounts)
				} else {
					fmt.Fprintf(w, "BenchmarkModel/name=%s/step=%s 1 0 ns/token 0 token/sec%s\n",
						m.Model, m.Step, promptCounts)
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
			cachedPromptCount := ""
			if m.CachedPromptCount != nil {
				cachedPromptCount = fmt.Sprint(*m.CachedPromptCount)
			}
			if isRateStep(m.Step) {
				var nsPerToken float64
				var tokensPerSec float64
				if m.Count > 0 {
					nsPerToken = float64(m.Duration.Nanoseconds()) / float64(m.Count)
					tokensPerSec = float64(m.Count) / (float64(m.Duration.Nanoseconds()) + 1e-12) * 1e9
				}
				fmt.Fprintf(w, "%s,%s,%d,%.2f,%.2f,%s\n", m.Model, m.Step, m.Count, nsPerToken, tokensPerSec, cachedPromptCount)
			} else {
				fmt.Fprintf(w, "%s,%s,1,%d,0,%s\n", m.Model, m.Step, m.Duration.Nanoseconds(), cachedPromptCount)
			}
		}
	default:
		fmt.Fprintf(os.Stderr, "Unknown output format '%s'\n", format)
	}
}

func BenchmarkModel(fOpt flagOptions) error {
	models := strings.Split(*fOpt.models, ",")
	useOpenAI := fOpt.openaiURL != nil && *fOpt.openaiURL != ""

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
		fmt.Fprintf(os.Stderr, "Generated code prompt of exactly %d tokens (unique per request)\n", *fOpt.promptTokens)
	}

	if useOpenAI {
		return benchmarkOpenAI(fOpt, models, out)
	}

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

	for _, model := range models {
		// Fetch model info
		infoCtx, infoCancel := context.WithTimeout(context.Background(), 10*time.Second)
		info := fetchModelInfo(infoCtx, client, model)
		infoCancel()

		// Resolve the generated prompt to the target token count against the
		// live model (count includes the chat template).
		var plan promptPlan
		if *fOpt.promptTokens > 0 {
			calCtx, calCancel := context.WithTimeout(context.Background(), time.Duration(*fOpt.timeout)*time.Second)
			plan, err = calibratePrompt(func(p promptPlan) (int, error) {
				return measurePromptTokens(calCtx, client, model, fOpt, imgData, p)
			}, model, fOpt)
			calCancel()
			if err != nil {
				fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
				return err
			}
			if *fOpt.debug {
				fmt.Fprintf(os.Stderr, "Prompt resolved to %d tokens for %s: %d problems plus %d pad tokens\n",
					*fOpt.promptTokens, model, len(codePromptProblems(plan.words, 0)), plan.pad)
			}
		}

		// Warmup phase
		for i := range *fOpt.warmup {
			req := buildChatRequest(model, fOpt, imgData, i, plan)
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
		offTargetCount, offTargetExample := 0, 0
		for epoch := range *fOpt.epochs {
			var responseMetrics *api.Metrics
			var ttft time.Duration
			short := false

			// Retry loop: if the model hits a stop token before max-tokens,
			// retry with a different HumanEval window.
			for attempt := range maxShortResponseRetries + 1 {
				responseMetrics = nil
				ttft = 0
				var ttftOnce sync.Once

				variation := benchmarkPromptVariation(*fOpt.warmup, *fOpt.epochs, epoch, attempt)
				req := buildChatRequest(model, fOpt, imgData, variation, plan)
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
				if !short || attempt == maxShortResponseRetries {
					break
				}

				if *fOpt.debug {
					fmt.Fprintf(os.Stderr, "Short response (%d/%d tokens), retrying with different prompt (attempt %d/%d)\n",
						responseMetrics.EvalCount, *fOpt.maxTokens, attempt+1, maxShortResponseRetries)
				}
			}

			if err != nil || responseMetrics == nil {
				continue
			}

			if short {
				shortCount++
				if *fOpt.debug {
					fmt.Fprintf(os.Stderr, "WARNING: Short response (%d/%d tokens) after %d retries for epoch %d\n",
						responseMetrics.EvalCount, *fOpt.maxTokens, maxShortResponseRetries, epoch+1)
				}
			}

			cachedPromptCount := 0
			if responseMetrics.PromptEvalCachedCount != nil {
				cachedPromptCount = *responseMetrics.PromptEvalCachedCount
			}
			metrics := []Metrics{
				{
					Model:             model,
					Step:              "prefill",
					Count:             max(0, responseMetrics.PromptEvalCount-cachedPromptCount),
					CachedPromptCount: responseMetrics.PromptEvalCachedCount,
					Duration:          responseMetrics.PromptEvalDuration,
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
				if responseMetrics.PromptEvalCachedCount == nil {
					fmt.Fprintf(os.Stderr, "Generated prompt targeting ~%d tokens (actual: %d, cached: unavailable)\n",
						*fOpt.promptTokens, responseMetrics.PromptEvalCount)
				} else {
					fmt.Fprintf(os.Stderr, "Generated prompt targeting ~%d tokens (actual: %d, cached: %d)\n",
						*fOpt.promptTokens, responseMetrics.PromptEvalCount, cachedPromptCount)
				}
			}

			// The plan is calibrated once, so hold every timed request to it
			// rather than trusting that it held.
			if *fOpt.promptTokens > 0 && responseMetrics.PromptEvalCount != *fOpt.promptTokens {
				offTargetCount++
				offTargetExample = responseMetrics.PromptEvalCount
			}

			if *fOpt.keepAlive > 0 {
				time.Sleep(time.Duration(*fOpt.keepAlive*float64(time.Second)) + 200*time.Millisecond)
			}
		}

		if shortCount > 0 {
			fmt.Fprintf(os.Stderr, "WARNING: %d/%d epochs for '%s' had short responses (<%d tokens). Generation metrics may be unreliable.\n",
				shortCount, *fOpt.epochs, model, *fOpt.maxTokens)
		}

		if offTargetCount > 0 {
			fmt.Fprintf(os.Stderr, "WARNING: %d/%d epochs for '%s' ran at a prompt size other than the requested %d tokens (e.g. %d). Prefill comparisons across prompt sizes are not valid.\n",
				offTargetCount, *fOpt.epochs, model, *fOpt.promptTokens, offTargetExample)
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

// openaiMetrics holds performance data extracted from an OpenAI-compatible streaming response.
type openaiMetrics struct {
	PromptTokens     int
	CompletionTokens int
	TTFT             time.Duration
	TotalDuration    time.Duration

	// Server-reported timings (when available). Zero if not provided.
	ServerPromptMS    float64
	ServerPredictedMS float64
	HasTimings        bool

	// Server-side counts, which differ from the usage totals: a cache hit
	// shrinks the prompt count, and Splash's decode count omits the
	// speculative block emitted before the first token.
	ServerPromptTokens    int
	ServerPredictedTokens int
	CachedTokens          int
	HasCachedTokens       bool
}

func openaiGenerate(ctx context.Context, baseURL, apiKey, model string, fOpt flagOptions, variation int, plan promptPlan, keepAlive *float64) (*openaiMetrics, error) {
	body := map[string]any{
		"model":      model,
		"stream":     true,
		"max_tokens": *fOpt.maxTokens,
		"messages": []map[string]string{
			{"role": "user", "content": benchPromptContent(fOpt, variation, plan)},
		},
	}
	// Always sent. Omitting it lets each server pick its own default (Splash
	// 1.0, Ollama 0.8), so a requested temperature of 0 would not be greedy.
	body["temperature"] = *fOpt.temperature
	if fOpt.seed != nil && *fOpt.seed > 0 {
		body["seed"] = *fOpt.seed
	}
	// Ollama extension so runs unload between models; others ignore it.
	if keepAlive != nil {
		body["keep_alive"] = *keepAlive
	}
	body["stream_options"] = map[string]any{"include_usage": true}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	url := strings.TrimRight(baseURL, "/") + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	start := time.Now()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	metrics := &openaiMetrics{}
	var ttftOnce sync.Once
	chunkTokens := 0

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
					// vLLM spells it "reasoning", Splash "reasoning_content".
					Reasoning        string `json:"reasoning"`
					ReasoningContent string `json:"reasoning_content"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				// Stock OpenAI; Splash and vLLM both report it.
				PromptTokensDetails *struct {
					CachedTokens int `json:"cached_tokens"`
				} `json:"prompt_tokens_details"`
			} `json:"usage"`
			// Splash. start_to_first_token excludes queueing; prefill.tokens
			// excludes what the cache served.
			Metrics *struct {
				Prefill *struct {
					Tokens int `json:"tokens"`
				} `json:"prefill"`
				Decode *struct {
					Tokens int `json:"tokens"`
				} `json:"decode"`
				RequestLatency *struct {
					StartToFirstTokenMS float64 `json:"start_to_first_token_ms"`
					FirstTokenToDoneMS  float64 `json:"first_token_to_done_ms"`
				} `json:"request_latency"`
				Cache *struct {
					MatchedTokens int `json:"matched_tokens"`
				} `json:"cache"`
			} `json:"metrics"`
			// llama.cpp and Ollama. prompt_n excludes what cache_n served.
			Timings *struct {
				CacheN      int     `json:"cache_n"`
				PromptN     int     `json:"prompt_n"`
				PromptMS    float64 `json:"prompt_ms"`
				PredictedN  int     `json:"predicted_n"`
				PredictedMS float64 `json:"predicted_ms"`
			} `json:"timings"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		if len(chunk.Choices) > 0 {
			text := cmp.Or(chunk.Choices[0].Delta.Content, chunk.Choices[0].Delta.Reasoning, chunk.Choices[0].Delta.ReasoningContent)
			if text != "" {
				ttftOnce.Do(func() {
					metrics.TTFT = time.Since(start)
				})
				chunkTokens++
				if *fOpt.debug {
					fmt.Fprint(os.Stderr, text)
				}
			}
		}

		if chunk.Usage != nil {
			metrics.PromptTokens = chunk.Usage.PromptTokens
			metrics.CompletionTokens = chunk.Usage.CompletionTokens
			if d := chunk.Usage.PromptTokensDetails; d != nil {
				metrics.CachedTokens = d.CachedTokens
				metrics.HasCachedTokens = true
			}
		}

		// Only the server can separate a cache hit from real prompt processing.
		switch {
		case chunk.Timings != nil:
			metrics.ServerPromptTokens = chunk.Timings.PromptN
			metrics.ServerPromptMS = chunk.Timings.PromptMS
			metrics.ServerPredictedTokens = chunk.Timings.PredictedN
			metrics.ServerPredictedMS = chunk.Timings.PredictedMS
			metrics.CachedTokens = chunk.Timings.CacheN
			metrics.HasCachedTokens = true
			metrics.HasTimings = true
		case chunk.Metrics != nil && chunk.Metrics.RequestLatency != nil:
			if p := chunk.Metrics.Prefill; p != nil {
				metrics.ServerPromptTokens = p.Tokens
			}
			metrics.ServerPromptMS = chunk.Metrics.RequestLatency.StartToFirstTokenMS
			if d := chunk.Metrics.Decode; d != nil {
				metrics.ServerPredictedTokens = d.Tokens
			}
			metrics.ServerPredictedMS = chunk.Metrics.RequestLatency.FirstTokenToDoneMS
			if c := chunk.Metrics.Cache; c != nil {
				metrics.CachedTokens = c.MatchedTokens
				metrics.HasCachedTokens = true
			}
			metrics.HasTimings = true
		}
	}

	metrics.TotalDuration = time.Since(start)

	if *fOpt.debug {
		fmt.Fprintln(os.Stderr)
	}

	if metrics.CompletionTokens == 0 {
		metrics.CompletionTokens = chunkTokens
	}

	return metrics, nil
}

// Capped at one token: calibration only needs the prompt side.
func measureOpenAIPromptTokens(ctx context.Context, baseURL, apiKey, model string, fOpt flagOptions, plan promptPlan) (int, error) {
	maxTokens := 1
	fOpt.maxTokens = &maxTokens
	m, err := openaiGenerate(ctx, baseURL, apiKey, model, fOpt, 0, plan, nil)
	if err != nil {
		return 0, err
	}
	if m.PromptTokens == 0 {
		return 0, errors.New("server reported no prompt_tokens; cannot calibrate prompt size")
	}
	return m.PromptTokens, nil
}

func benchmarkOpenAI(fOpt flagOptions, models []string, out io.Writer) error {
	apiKey := *fOpt.apiKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	keepAliveForever := float64(-1)
	keepAliveUnload := float64(0)

	for _, model := range models {
		info := ModelInfo{Name: model}
		outputModelInfo(out, *fOpt.format, info)

		// Size the generated prompt against this server, over the same API the
		// timed requests use, so no Ollama endpoint is required.
		var plan promptPlan
		if *fOpt.promptTokens > 0 {
			calCtx, calCancel := context.WithTimeout(context.Background(), time.Duration(*fOpt.timeout)*time.Second)
			p, err := calibratePrompt(func(p promptPlan) (int, error) {
				return measureOpenAIPromptTokens(calCtx, *fOpt.openaiURL, apiKey, model, fOpt, p)
			}, model, fOpt)
			calCancel()
			if err != nil {
				fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
				return err
			}
			plan = p
			if *fOpt.debug {
				fmt.Fprintf(os.Stderr, "Prompt resolved to %d tokens for %s: %d problems plus %d pad tokens\n",
					*fOpt.promptTokens, model, len(codePromptProblems(plan.words, 0)), plan.pad)
			}
		}

		for i := range *fOpt.warmup {
			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*fOpt.timeout)*time.Second)
			_, err := openaiGenerate(ctx, *fOpt.openaiURL, apiKey, model, fOpt, i, plan, &keepAliveForever)
			cancel()
			if err != nil {
				fmt.Fprintf(os.Stderr, "WARNING: Warmup %d/%d for %s failed: %v\n", i+1, *fOpt.warmup, model, err)
			} else if *fOpt.debug {
				fmt.Fprintf(os.Stderr, "Warmup %d/%d for %s complete\n", i+1, *fOpt.warmup, model)
			}
		}

		hasTimings := false
		shortCount := 0
		for epoch := range *fOpt.epochs {
			var oaiMetrics *openaiMetrics
			var err error
			short := false

			isLast := epoch == *fOpt.epochs-1
			ka := &keepAliveForever
			if isLast {
				ka = &keepAliveUnload
			}

			const maxRetries = 3
			for attempt := range maxRetries + 1 {
				variation := benchmarkPromptVariation(*fOpt.warmup, *fOpt.epochs, epoch, attempt)
				ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*fOpt.timeout)*time.Second)
				oaiMetrics, err = openaiGenerate(ctx, *fOpt.openaiURL, apiKey, model, fOpt, variation, plan, ka)
				cancel()

				if err != nil {
					fmt.Fprintf(os.Stderr, "ERROR: Couldn't generate with model '%s': %v\n", model, err)
					break
				}

				short = *fOpt.maxTokens > 0 && oaiMetrics.CompletionTokens < *fOpt.maxTokens
				if !short || attempt == maxRetries {
					break
				}

				if *fOpt.debug {
					fmt.Fprintf(os.Stderr, "Short response (%d/%d tokens), retrying with different prompt (attempt %d/%d)\n",
						oaiMetrics.CompletionTokens, *fOpt.maxTokens, attempt+1, maxRetries)
				}
			}

			if err != nil || oaiMetrics == nil {
				continue
			}

			if short {
				shortCount++
				if *fOpt.debug {
					fmt.Fprintf(os.Stderr, "WARNING: Short response (%d/%d tokens) after %d retries for epoch %d\n",
						oaiMetrics.CompletionTokens, *fOpt.maxTokens, maxRetries, epoch+1)
				}
			}

			// Client-derived everywhere; within 0.1% of llama-server's own figure.
			metrics := []Metrics{
				{
					Model:    model,
					Step:     "generate",
					Count:    oaiMetrics.CompletionTokens,
					Duration: oaiMetrics.TotalDuration - oaiMetrics.TTFT,
				},
			}

			// Server-only: time-to-first-token also carries setup, queueing
			// and load. A silent server gets no prefill row, just ttft.
			if oaiMetrics.HasTimings {
				hasTimings = true
				prefill := Metrics{
					Model:    model,
					Step:     "prefill",
					Count:    oaiMetrics.ServerPromptTokens,
					Duration: time.Duration(oaiMetrics.ServerPromptMS * float64(time.Millisecond)),
				}
				if oaiMetrics.HasCachedTokens {
					cached := oaiMetrics.CachedTokens
					prefill.CachedPromptCount = &cached
				}
				metrics = append(metrics, prefill,
					Metrics{
						Model:    model,
						Step:     "generate_server",
						Count:    oaiMetrics.ServerPredictedTokens,
						Duration: time.Duration(oaiMetrics.ServerPredictedMS * float64(time.Millisecond)),
					},
				)
			}

			metrics = append(metrics,
				Metrics{
					Model:    model,
					Step:     "ttft",
					Count:    1,
					Duration: oaiMetrics.TTFT,
				},
				Metrics{
					Model:    model,
					Step:     "total",
					Count:    1,
					Duration: oaiMetrics.TotalDuration,
				},
			)

			OutputMetrics(out, *fOpt.format, metrics, *fOpt.verbose)

			if *fOpt.debug && *fOpt.promptTokens > 0 {
				fmt.Fprintf(os.Stderr, "Generated prompt targeting ~%d tokens (actual prompt_tokens: %d)\n",
					*fOpt.promptTokens, oaiMetrics.PromptTokens)
			}
		}

		if !hasTimings {
			fmt.Fprintf(os.Stderr, "# NOTE: Server reported no timings; no prefill rate available (see ttft)\n")
		}

		if shortCount > 0 {
			fmt.Fprintf(os.Stderr, "WARNING: %d/%d epochs for '%s' had short responses (<%d tokens). Generation metrics may be unreliable.\n",
				shortCount, *fOpt.epochs, model, *fOpt.maxTokens)
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
		promptTokens: flag.Int("prompt-tokens", 0, "Generate a prompt of exactly N tokens (0 = use -p prompt)"),
		numCtx:       flag.Int("num-ctx", 0, "Context size (0 = server default)"),
		openaiURL:    flag.String("openai", "", "OpenAI-compatible API base URL (e.g. http://localhost:11434/v1)"),
		apiKey:       flag.String("api-key", "", "API key for OpenAI endpoint (default: OPENAI_API_KEY env)"),
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
		fmt.Fprintf(os.Stderr, "  bench -model gemma3 -openai http://localhost:11434/v1\n")
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

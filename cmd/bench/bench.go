package main

import (
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

// buildChatRequest builds a single-message benchmark request through the
// model's chat template. plan is the calibrated prompt shape for generated
// prompts; ignored for -p prompts.
func buildChatRequest(model string, fOpt flagOptions, imgData api.ImageData, variation int, plan promptPlan) *api.ChatRequest {
	var content string
	if *fOpt.promptTokens > 0 {
		content = generateCodePrompt(plan, variation)
	} else {
		// A leading unique nonce defeats prefix cache reuse across runs.
		content = nonceHeader(promptNonce(nonceLetters)) + *fOpt.prompt
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
func calibratePrompt(ctx context.Context, client *api.Client, model string, fOpt flagOptions, imgData api.ImageData) (promptPlan, error) {
	targetTokens := *fOpt.promptTokens
	maxWords := fullCodePromptWords()

	measured := make(map[int]int)
	measure := func(words int) (int, error) {
		if tokens, ok := measured[words]; ok {
			return tokens, nil
		}
		tokens, err := measurePromptTokens(ctx, client, model, fOpt, imgData, promptPlan{words: words})
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
		actual, err := measurePromptTokens(ctx, client, model, fOpt, imgData, plan)
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
		fmt.Fprintf(os.Stderr, "Generated code prompt of exactly %d tokens (unique per request)\n", *fOpt.promptTokens)
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
			plan, err = calibratePrompt(calCtx, client, model, fOpt, imgData)
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

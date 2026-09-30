package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlxrunner/wire"
)

// Direct runner modes. prefill and decode produce single-phase workloads for
// clean profiler capture windows; both is a normal mixed run.
const (
	modePrefill = "prefill"
	modeDecode  = "decode"
	modeBoth    = "both"
)

// completionParams is the backend-agnostic description of one completion call.
type completionParams struct {
	prompt      string
	numPredict  int // -1 = generate to the context limit, 0 = prefill-only
	temperature float64
	seed        int // -1 = random
	ignoreEOS   bool
	stats       bool // ask the runner for per-request stats
	media       []llm.MediaData
	logprobs    bool
	format      json.RawMessage
	debug       bool
}

// completionResult carries the timing data every backend reports. Fields a
// backend cannot supply are left zero.
type completionResult struct {
	promptEvalCount    int
	cachedPromptCount  *int
	promptEvalDuration time.Duration
	evalCount          int
	evalDuration       time.Duration
	ttft               time.Duration
	loadDuration       time.Duration
	totalDuration      time.Duration
	stats              *wire.Stats // nil unless requested and supported
	content            string
}

// errNoMetrics signals that a completion finished without delivering a final
// metrics record, so its timings are unusable.
var errNoMetrics = errors.New("no metrics received")

// benchBackend is a runner driven directly: an MLX runner or a llama-server.
type benchBackend interface {
	Name() string
	ModelInfo(ctx context.Context, fOpt flagOptions) ModelInfo
	Complete(ctx context.Context, p completionParams) (completionResult, error)
	// Cleanup stops the runner if bench spawned it.
	Cleanup(timeout int)
}

func directParams(fOpt flagOptions, mode, prompt string) completionParams {
	numPredict := -1
	if *fOpt.maxTokens > 0 {
		numPredict = *fOpt.maxTokens
	}
	if mode == modePrefill {
		numPredict = 0
	}
	seed := -1
	if *fOpt.seed > 0 {
		seed = *fOpt.seed
	}
	return completionParams{
		prompt:      prompt,
		numPredict:  numPredict,
		temperature: *fOpt.temperature,
		seed:        seed,
		ignoreEOS:   *fOpt.ignoreEOS && mode != modePrefill,
		debug:       *fOpt.debug,
	}
}

// benchmarkDirect benchmarks one runner driven directly. The prompt is sent
// raw, without a chat template, so -prompt-tokens counts exactly what the
// runner evaluates.
//
// decode mode holds one prompt fixed across warmup and epochs, so the runner's
// prefix cache hits and each timed window is decode only. prefill and both use
// a unique prompt per request, forcing a cache miss.
func benchmarkDirect(fOpt flagOptions, out io.Writer) error {
	mode := *fOpt.mode
	if mode != modePrefill && mode != modeDecode && mode != modeBoth {
		return fmt.Errorf("unknown -mode %q (want prefill|decode|both)", mode)
	}

	backend, err := newDirectBackend(fOpt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		return err
	}
	defer backend.Cleanup(*fOpt.timeout)

	model := *fOpt.models
	timeout := time.Duration(*fOpt.timeout) * time.Second

	var plan promptPlan
	if *fOpt.promptTokens > 0 {
		plan, err = calibratePrompt(func(p promptPlan) (int, error) {
			ctx, cancel := context.WithTimeout(context.Background(), timeout)
			defer cancel()
			params := directParams(fOpt, modeBoth, generateCodePrompt(p, 0))
			params.numPredict = 1
			params.ignoreEOS = false
			res, err := backend.Complete(ctx, params)
			return res.promptEvalCount, err
		}, model, fOpt)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
			return err
		}
	}

	fixedPrompt := benchPromptContent(fOpt, 0, plan)
	promptFor := func(variation int) string {
		if mode == modeDecode {
			return fixedPrompt
		}
		return benchPromptContent(fOpt, variation, plan)
	}

	// In decode mode warmup also primes the prefix cache, so it always runs
	// at least once.
	warmups := *fOpt.warmup
	if mode == modeDecode {
		warmups = max(warmups, 1)
	}
	for i := range warmups {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		_, err := backend.Complete(ctx, directParams(fOpt, mode, promptFor(i)))
		cancel()
		if err != nil {
			fmt.Fprintf(os.Stderr, "WARNING: Warmup %d/%d for %s failed: %v\n", i+1, warmups, model, err)
		} else if *fOpt.debug {
			fmt.Fprintf(os.Stderr, "Warmup %d/%d for %s complete\n", i+1, warmups, model)
		}
	}

	infoCtx, infoCancel := context.WithTimeout(context.Background(), 10*time.Second)
	outputModelInfo(out, *fOpt.format, backend.ModelInfo(infoCtx, fOpt))
	infoCancel()

	shortCount := 0
	offTargetCount, offTargetExample := 0, 0
	for epoch := range *fOpt.epochs {
		var res completionResult
		var err error
		short := false

		// A retry only helps when a different prompt could run longer: not with
		// -ignore-eos, and not in decode mode, whose prompt is fixed.
		for attempt := range maxShortResponseRetries + 1 {
			p := directParams(fOpt, mode, promptFor(benchmarkPromptVariation(warmups, *fOpt.epochs, epoch, attempt)))
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

			short = p.numPredict > 0 && res.evalCount < p.numPredict
			if !short || p.ignoreEOS || mode == modeDecode || attempt == maxShortResponseRetries {
				break
			}
			if *fOpt.debug {
				fmt.Fprintf(os.Stderr, "Short response (%d/%d tokens), retrying with different prompt (attempt %d/%d)\n",
					res.evalCount, p.numPredict, attempt+1, maxShortResponseRetries)
			}
		}
		if err != nil {
			continue
		}
		if short {
			shortCount++
		}

		cached := 0
		if res.cachedPromptCount != nil {
			cached = *res.cachedPromptCount
		}
		OutputMetrics(out, *fOpt.format, []Metrics{
			{Model: model, Step: "prefill", Count: max(0, res.promptEvalCount-cached), CachedPromptCount: res.cachedPromptCount, Duration: res.promptEvalDuration},
			{Model: model, Step: "generate", Count: res.evalCount, Duration: res.evalDuration},
			{Model: model, Step: "ttft", Count: 1, Duration: res.ttft},
			{Model: model, Step: "load", Count: 1, Duration: res.loadDuration},
			{Model: model, Step: "total", Count: 1, Duration: res.totalDuration},
		}, *fOpt.verbose)

		if *fOpt.promptTokens > 0 && res.promptEvalCount != *fOpt.promptTokens {
			offTargetCount++
			offTargetExample = res.promptEvalCount
		}
	}

	if shortCount > 0 {
		fmt.Fprintf(os.Stderr, "WARNING: %d/%d epochs for '%s' had short responses (<%d tokens). Use -ignore-eos for exact counts.\n",
			shortCount, *fOpt.epochs, model, *fOpt.maxTokens)
	}
	if offTargetCount > 0 {
		fmt.Fprintf(os.Stderr, "WARNING: %d/%d epochs for '%s' ran at a prompt size other than the requested %d tokens (e.g. %d). Prefill comparisons across prompt sizes are not valid.\n",
			offTargetCount, *fOpt.epochs, model, *fOpt.promptTokens, offTargetExample)
	}
	return nil
}

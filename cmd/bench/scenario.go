package main

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/llm"
)

// Cache scenarios drive a runner through request sequences whose prefix-cache
// behavior is known in advance. Every step checks that the cache did what the
// sequence requires, from the cached-token count and stats the runner reports.
// An epoch that fails a check is void and emits no metrics, so a broken cache
// reads as a failure rather than a slow number.

// hitSlack is how far below a full match a hit may land. The runner holds one
// token back to seed decode, and caches that cannot rewind (recurrent, sliding
// window) restore from the nearest snapshot, which sits a few tokens earlier.
const hitSlack = 8

// coldMax bounds the tokens a cold step may reuse: the nonce header's fixed
// preamble precedes the nonce and is shared across epochs.
const coldMax = 64

// snapshotInterval mirrors the runner's periodic prefill snapshot spacing.
// Caches that cannot rewind restore a mid-prompt divergence from the last of
// these at or before it.
const snapshotInterval = 8192

type scenario struct {
	name string
	desc string
	run  func(*scenarioRun) error
}

var scenarios = []scenario{
	{"repeat", "same prompt twice: restore cost and decode on a full hit", runRepeat},
	{"extend", "prompt, then the prompt plus a new turn: reuse up to the divergence", runExtend},
	{"continue", "prompt, then prompt plus the model's reply plus a new turn: reuse through generated tokens", runContinue},
	{"branch", "a prefix, two turns over it, then back to the first: branch snapshots and path switching", runBranch},
	{"midbranch", "long prompt, then a divergence mid-prompt, then a second divergence there: periodic and branch-point snapshots", runMidBranch},
	{"longgen", "long generation from a hit: cache growth across many blocks", runLongGen},
	{"sweep", "hits at 2k, 8k, 16k and 32k tokens: restore and decode against context length", runSweep},
	{"evict", "distinct long prompts past the snapshot limit: bounded cold storage, oldest evicted, newest kept", runEvict},
	{"churn", "two conversations repeated in turn: memory stays flat across path switches", runChurn},
	{"cancel", "prefill cancelled midway, then retried: the retry resumes from the cancelled progress", runCancel},
	{"interleave", "two conversations alternating turns: every request switches path and restores", runInterleave},
	{"concurrent", "two sequences submitted together, then again: each hits its own prefix", runConcurrent},
	{"short", "prompts of a few tokens: the seed back-off and snapshot edge cases", runShort},
	{"media", "image and audio prompts: media folded into the cache key, extension past an image, back-off with the image last", runMedia},
	{"modes", "logprobs, then plain, then structured output, then plain over one conversation: parked drafting and the grammar decoder", runModes},
}

// tokenCounter is implemented by backends that can tokenize without evaluating.
type tokenCounter interface {
	Tokenize(ctx context.Context, text string) (int, error)
}

var (
	errVoid = errors.New("void")
	errSkip = errors.New("skip")
)

// promptSizer finds the word budget that lands a prompt near a token target and
// remembers it across epochs; the nonce keeps each prompt distinct.
type promptSizer struct {
	words map[int]int
}

type scenarioRun struct {
	backend benchBackend
	tokens  tokenCounter
	sizer   *promptSizer
	fOpt    flagOptions
	model   string
	name    string
	epoch   int
	timeout time.Duration
	ctxLen  int
	target  int // base prompt size in tokens

	seq   int // distinct prompt material per call within an epoch
	mu    sync.Mutex
	lines []string
}

func (s *scenarioRun) count(text string) (int, error) {
	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()
	return s.tokens.Tokenize(ctx, text)
}

func (s *scenarioRun) variation() int {
	s.seq++
	return s.epoch*1000 + s.seq
}

// prompt returns a new prompt of about target tokens and its exact size.
func (s *scenarioRun) prompt(target int) (string, int, error) {
	words, ok := s.sizer.words[target]
	if !ok {
		words = max(smallestCodePromptWords(), int(float64(target)/tokensPerWordSeed))
		for range 10 {
			n, err := s.count(nonceHeader(promptNonce(nonceLetters)) + longCodeBody(words, 0))
			if err != nil {
				return "", 0, err
			}
			if abs(n-target) <= max(8, target/100) {
				break
			}
			words = max(1, words*target/n)
		}
		s.sizer.words[target] = words
	}
	text := nonceHeader(promptNonce(nonceLetters)) + longCodeBody(words, s.variation())
	n, err := s.count(text)
	return text, n, err
}

// tail returns a new turn to append after a prompt.
func (s *scenarioRun) tail() string {
	return scenarioTail(max(smallestCodePromptWords(), s.target/16), s.variation())
}

type stepSpec struct {
	name       string
	prompt     string
	media      []llm.MediaData
	numPredict int           // 0 uses -max-tokens
	cancel     time.Duration // cancel the request after this long; the step then checks nothing
	min, max   int           // bounds on cached tokens
	quiet      bool          // check but emit no line
	logprobs   bool          // request logprobs, which parks speculative drafting
	format     string        // structured output; stop tokens stay on and generation length is not checked
}

// step sends one request and checks it: cached tokens within [min, max], never
// the whole prompt, and exactly the requested number of generated tokens.
func (s *scenarioRun) step(spec stepSpec) (completionResult, error) {
	p := directParams(s.fOpt, modeBoth, spec.prompt)
	if spec.numPredict > 0 {
		p.numPredict = spec.numPredict
	}
	p.ignoreEOS = true
	p.stats = true
	p.media = spec.media
	p.logprobs = spec.logprobs
	if spec.format != "" {
		p.format = json.RawMessage(spec.format)
		p.ignoreEOS = false
	}

	timeout := s.timeout
	if spec.cancel > 0 {
		timeout = spec.cancel
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	res, err := s.backend.Complete(ctx, p)
	cancel()
	if spec.cancel > 0 {
		switch {
		case err == nil:
			return res, fmt.Errorf("step %s: finished before the %v cancel: %w", spec.name, spec.cancel, errVoid)
		case errors.Is(err, context.DeadlineExceeded):
			return res, nil
		default:
			return res, fmt.Errorf("step %s: %w", spec.name, err)
		}
	}
	if err != nil {
		return res, fmt.Errorf("step %s: %w", spec.name, err)
	}

	if res.cachedPromptCount == nil || res.stats == nil {
		return res, fmt.Errorf("step %s: runner did not report cached tokens and stats: %w", spec.name, errVoid)
	}
	cached := *res.cachedPromptCount
	switch {
	case cached < spec.min || cached > spec.max:
		return res, fmt.Errorf("step %s: cached %d of %d tokens, want %d..%d: %w", spec.name, cached, res.promptEvalCount, spec.min, spec.max, errVoid)
	case cached >= res.promptEvalCount:
		return res, fmt.Errorf("step %s: cached all %d tokens; one must be evaluated to seed decode: %w", spec.name, res.promptEvalCount, errVoid)
	case p.ignoreEOS && p.numPredict > 0 && res.evalCount != p.numPredict:
		return res, fmt.Errorf("step %s: generated %d tokens, want %d: %w", spec.name, res.evalCount, p.numPredict, errVoid)
	}
	if !spec.quiet {
		s.emit(spec.name, res)
	}
	return res, nil
}

func (s *scenarioRun) emit(name string, res completionResult) {
	cached := *res.cachedPromptCount
	st := res.stats
	var b strings.Builder
	fmt.Fprintf(&b, "BenchmarkCache/model=%s/scenario=%s/step=%s 1 %d ns/op", s.model, s.name, name, res.ttft.Nanoseconds())
	fmt.Fprintf(&b, " %.2f prefill-tok/s %.2f decode-tok/s", rate(res.promptEvalCount-cached, res.promptEvalDuration), rate(res.evalCount, res.evalDuration))
	fmt.Fprintf(&b, " %d prompt-tokens %d cached-tokens %d matched-tokens", res.promptEvalCount, cached, st.MatchedTokens)
	fmt.Fprintf(&b, " %d peak-B %d active-B %d buffer-cache-B %d cold-B", st.PeakBytes, st.ActiveBytes, st.CacheBytes, st.ColdBytes)
	if st.DraftTokens > 0 {
		fmt.Fprintf(&b, " %.3f draft-acceptance", float64(st.AcceptedDraft)/float64(st.DraftTokens))
	}
	s.mu.Lock()
	s.lines = append(s.lines, b.String())
	s.mu.Unlock()
}

func rate(n int, d time.Duration) float64 {
	if n <= 0 || d <= 0 {
		return 0
	}
	return float64(n) / d.Seconds()
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// tailPreamble opens every tail, so two tails over one base share it.
const tailPreamble = "\n\n\n# turn "

// scenarioTail is a new turn after a prompt. The nonce makes each tail diverge
// from every other right after the preamble.
func scenarioTail(words, variation int) string {
	return tailPreamble + promptNonce(8) + "\n" + codePromptBody(words, variation)
}

func selectScenarios(spec string) ([]scenario, error) {
	if spec == "all" {
		return scenarios, nil
	}
	var selected []scenario
	for name := range strings.SplitSeq(spec, ",") {
		i := slices.IndexFunc(scenarios, func(sc scenario) bool { return sc.name == name })
		if i < 0 {
			return nil, fmt.Errorf("unknown scenario %q (want all or %s)", name, strings.Join(scenarioNames(), ","))
		}
		selected = append(selected, scenarios[i])
	}
	return selected, nil
}

func scenarioNames() []string {
	names := make([]string, len(scenarios))
	for i, sc := range scenarios {
		names[i] = sc.name
	}
	return names
}

// benchmarkScenarios runs the selected cache scenarios against one directly
// driven runner. It returns an error if any epoch was void.
func benchmarkScenarios(fOpt flagOptions, out io.Writer) error {
	selected, err := selectScenarios(*fOpt.scenario)
	if err != nil {
		return err
	}
	if *fOpt.format != "benchstat" {
		return errors.New("-scenario supports only -format benchstat")
	}

	backend, err := newDirectBackend(fOpt)
	if err != nil {
		return err
	}
	defer backend.Cleanup(*fOpt.timeout)

	tokens, ok := backend.(tokenCounter)
	if !ok {
		return fmt.Errorf("-scenario needs a runner that can tokenize; %s cannot", backend.Name())
	}

	model := directModel(fOpt)
	timeout := time.Duration(*fOpt.timeout) * time.Second
	target := cmp.Or(*fOpt.promptTokens, 4096)

	infoCtx, infoCancel := context.WithTimeout(context.Background(), 10*time.Second)
	info := backend.ModelInfo(infoCtx, fOpt)
	infoCancel()
	outputModelInfo(out, *fOpt.format, info)
	fmt.Fprintf(out, "# Scenarios: %s | prompt-tokens: %d | max-tokens: %d\n", *fOpt.scenario, target, *fOpt.maxTokens)

	sizer := &promptSizer{words: map[int]int{}}
	voids := 0
	for _, sc := range selected {
		for epoch := range *fOpt.warmup + *fOpt.epochs {
			run := &scenarioRun{
				backend: backend,
				tokens:  tokens,
				sizer:   sizer,
				fOpt:    fOpt,
				model:   model,
				name:    sc.name,
				epoch:   epoch,
				timeout: timeout,
				ctxLen:  int(info.NumCtx),
				target:  target,
			}
			start := time.Now()
			err := sc.run(run)
			warm := epoch < *fOpt.warmup
			switch {
			case errors.Is(err, errSkip):
				fmt.Fprintf(out, "# SKIP scenario=%s: %v\n", sc.name, err)
			case errors.Is(err, errVoid):
				voids++
				fmt.Fprintf(out, "# VOID scenario=%s epoch=%d: %v\n", sc.name, epoch, err)
				fmt.Fprintf(os.Stderr, "ERROR: void scenario=%s epoch=%d: %v\n", sc.name, epoch, err)
			case err != nil:
				return fmt.Errorf("scenario %s epoch %d: %w", sc.name, epoch, err)
			case !warm:
				for _, line := range run.lines {
					fmt.Fprintln(out, line)
				}
			}
			if *fOpt.debug {
				fmt.Fprintf(os.Stderr, "scenario %s epoch %d took %v\n", sc.name, epoch, time.Since(start).Round(time.Millisecond))
			}
			if errors.Is(err, errSkip) {
				break
			}
		}
	}
	if voids > 0 {
		return fmt.Errorf("%d void scenario epochs", voids)
	}
	return nil
}

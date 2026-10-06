package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/decisiontest"
)

type decisionSample struct {
	Case     decisiontest.Case
	Epoch    int
	Response decisiontest.Response
	Duration time.Duration
	Wrong    []string
	Err      error
}

func benchmarkDecision(opts flagOptions, models []string, out io.Writer) error {
	if *opts.openaiURL != "" || *opts.imageFile != "" || *opts.promptTokens != 0 || *opts.numCtx != 0 {
		return fmt.Errorf("-decision cannot be combined with -openai, -image, -prompt-tokens, or -num-ctx; put images in the corpus")
	}
	if *opts.epochs < 1 || *opts.warmup < 0 || *opts.timeout <= 0 || *opts.concurrency < 1 {
		return fmt.Errorf("decision benchmark requires positive epochs, timeout, concurrency and nonnegative warmup")
	}
	if *opts.keepAlive > 0 && *opts.concurrency > 1 {
		return fmt.Errorf("cold-load measurements (-k > 0) require -concurrency 1")
	}
	for _, name := range models {
		if strings.TrimSpace(name) == "" {
			return fmt.Errorf("empty model name")
		}
	}
	mode, err := decisiontest.Mode(*opts.decisionMode)
	if err != nil {
		return err
	}
	corpus, err := decisiontest.Load(*opts.decisionFile)
	if err != nil {
		return err
	}
	if *opts.epochs > math.MaxInt/len(corpus.Cases) {
		return fmt.Errorf("epochs times corpus length exceeds the maximum request count")
	}
	if opts.outputFile != nil && *opts.outputFile != "" {
		// Check inode identity too: a symlink or hardlink must not destroy input.
		if err := checkDecisionOutput(*opts.decisionFile, *opts.outputFile, corpus); err != nil {
			return err
		}
		f, err := os.Create(*opts.outputFile)
		if err != nil {
			return err
		}
		defer f.Close()
		out = f
	}
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}
	buffered := bufio.NewWriter(out)
	out = buffered
	var csvOut *csv.Writer
	if *opts.format == "csv" {
		csvOut = csv.NewWriter(out)
		defer csvOut.Flush()
		if err := csvOut.Write([]string{"model", "corpus_sha256", "dataset", "case", "epoch", "concurrency", "input_tokens", "output_tokens", "state_bytes", "questions", "options", "images", "image_bytes", "image_pixels", "latency_ns", "correct_questions", "correct_case", "error", "total_duration_ns", "load_duration_ns", "mismatches"}); err != nil {
			return err
		}
	} else {
		outputFormatHeader(out, *opts.format, *opts.verbose)
	}
	var failures []error
	for _, name := range models {
		name = strings.TrimSpace(name)
		if err := benchmarkDecisionModel(opts, client, name, corpus, mode, out, csvOut); err != nil {
			failures = append(failures, err)
		}
		if err := unloadModel(client, name, *opts.timeout); err != nil {
			failures = append(failures, fmt.Errorf("%s unload: %w", name, err))
			break
		}
	}
	if csvOut != nil {
		csvOut.Flush()
		failures = append(failures, csvOut.Error())
	}
	failures = append(failures, buffered.Flush())
	return errors.Join(failures...)
}

func benchmarkDecisionModel(opts flagOptions, client *api.Client, name string, corpus decisiontest.Corpus, mode string, out io.Writer, csvOut *csv.Writer) error {
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(nil)
	endpoint := envconfig.Host().JoinPath("v1/systemone").String()
	keepAlive := benchmarkKeepAlive(opts)
	one := func(index int) decisionSample {
		c := corpus.Cases[index%len(corpus.Cases)]
		if *opts.keepAlive > 0 {
			// Include the first measured request: warmup or a previous run may
			// have left the model loaded. Waiting for expiry alone is racy.
			if err := unloadModel(client, name, *opts.timeout); err != nil {
				return decisionSample{Case: c, Epoch: index / len(corpus.Cases), Err: fmt.Errorf("unload before cold request: %w", err)}
			}
		}
		req := c.Request
		req.Model, req.KeepAlive = name, keepAlive
		requestCtx, cancel := context.WithTimeout(ctx, time.Duration(*opts.timeout)*time.Second)
		defer cancel()
		response, duration, err := decisiontest.Do(requestCtx, endpoint, req)
		s := decisionSample{Case: c, Epoch: index / len(corpus.Cases), Response: response, Duration: duration, Err: err}
		if err == nil {
			s.Wrong = c.Mismatches(response)
		}
		return s
	}
	for i := range *opts.warmup {
		if sample := one(i); sample.Err != nil {
			return fmt.Errorf("%s warmup: %w", name, sample.Err)
		}
	}
	if csvOut == nil {
		fmt.Fprintf(out, "# API: systemone | Corpus: %s | SHA256: %s | Mode: %s | Concurrency: %d | Cache: unchanged corpus replay\n", *opts.decisionFile, corpus.SHA256, mode, *opts.concurrency)
	}
	// Workers submit their next request immediately after a response. This is a
	// closed-loop concurrency test, not an offered-rate/open-loop load generator.
	total := len(corpus.Cases) * *opts.epochs
	concurrency := min(*opts.concurrency, total)
	results := make(chan decisionSample, concurrency)
	var next atomic.Int64
	var workers sync.WaitGroup
	start := time.Now()
	for range concurrency {
		workers.Go(func() {
			for ctx.Err() == nil {
				i := int(next.Add(1)) - 1
				if i >= total {
					return
				}
				sample := one(i)
				results <- sample
				if sample.Err != nil {
					cancel(sample.Err)
					return
				}
				if *opts.keepAlive > 0 && i+1 < total {
					time.Sleep(time.Duration(*opts.keepAlive*float64(time.Second)) + 200*time.Millisecond)
				}
			}
		})
	}
	go func() { workers.Wait(); close(results) }()
	var samples []decisionSample
	for s := range results {
		samples = append(samples, s)
	}
	elapsed := time.Since(start)
	wrongCases := 0
	for _, s := range samples {
		if len(s.Wrong) > 0 {
			wrongCases++
		}
		if csvOut != nil {
			var message string
			correct := s.Case.Request.Questions.Len() - len(s.Wrong)
			if s.Err != nil {
				message, correct = s.Err.Error(), 0
			}
			row := []string{name, corpus.SHA256, s.Case.Dataset, s.Case.ID, strconv.Itoa(s.Epoch), strconv.Itoa(*opts.concurrency), strconv.Itoa(s.Response.Usage.InputTokens), strconv.Itoa(s.Response.Usage.OutputTokens), strconv.Itoa(len(s.Case.Request.State)), strconv.Itoa(s.Case.Request.Questions.Len()), strconv.Itoa(s.Case.Options()), strconv.Itoa(len(s.Case.Request.Images)), strconv.Itoa(s.Case.ImageBytes), strconv.FormatInt(s.Case.ImagePixels, 10), strconv.FormatInt(s.Duration.Nanoseconds(), 10), strconv.Itoa(correct), strconv.FormatBool(s.Err == nil && len(s.Wrong) == 0), message}
			for _, duration := range []*time.Duration{s.Response.TotalDuration, s.Response.LoadDuration} {
				value := ""
				if duration != nil {
					value = strconv.FormatInt(duration.Nanoseconds(), 10)
				}
				row = append(row, value)
			}
			row = append(row, strings.Join(s.Wrong, "; "))
			if err := csvOut.Write(row); err != nil {
				return err
			}
		} else if s.Err == nil {
			step := fmt.Sprintf("decision/case=%s/input=%d/questions=%d/images=%d/concurrency=%d", url.PathEscape(s.Case.ID), s.Response.Usage.InputTokens, s.Case.Request.Questions.Len(), len(s.Case.Request.Images), *opts.concurrency)
			for _, metric := range []struct {
				name     string
				duration *time.Duration
			}{{"http", &s.Duration}, {"total", s.Response.TotalDuration}, {"load", s.Response.LoadDuration}} {
				if metric.duration != nil {
					OutputMetrics(out, *opts.format, []Metrics{{Model: name, Step: step + "/metric=" + metric.name, Count: 1, Duration: *metric.duration}}, *opts.verbose)
				}
			}
		}
	}
	writeDecisionSummary(os.Stderr, name, samples, total, elapsed)
	if err := context.Cause(ctx); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	if mode == "regression" && wrongCases > 0 {
		return fmt.Errorf("%s: %d/%d cases failed", name, wrongCases, total)
	}
	return nil
}

func writeDecisionSummary(out io.Writer, model string, samples []decisionSample, total int, elapsed time.Duration) {
	type accuracy struct {
		decisiontest.Tally
		CaseAccuracy     float64 `json:"case_accuracy_pct"`
		QuestionAccuracy float64 `json:"question_accuracy_pct"`
	}
	type summary struct {
		accuracy
		Model              string              `json:"model"`
		Expected           int                 `json:"expected_requests"`
		Complete           bool                `json:"complete"`
		RequestsPerSecond  float64             `json:"requests_per_second"`
		QuestionsPerSecond float64             `json:"questions_per_second"`
		P50                float64             `json:"p50_ms"`
		P95                float64             `json:"p95_ms"`
		P99                float64             `json:"p99_ms"`
		Datasets           map[string]accuracy `json:"datasets"`
	}
	s := summary{Model: model, Expected: total, Datasets: make(map[string]accuracy)}
	var durations []time.Duration
	var questions int
	for _, sample := range samples {
		s.Tally.Add(sample.Case, sample.Wrong, sample.Err)
		d := s.Datasets[sample.Case.Dataset]
		d.Tally.Add(sample.Case, sample.Wrong, sample.Err)
		d.CaseAccuracy, d.QuestionAccuracy = d.Tally.CaseAccuracy(), d.Tally.QuestionAccuracy()
		s.Datasets[sample.Case.Dataset] = d
		if sample.Err != nil {
			continue
		}
		durations = append(durations, sample.Duration)
		questions += len(sample.Case.Expected)
	}
	s.Complete = len(samples) == total && s.Errors == 0
	s.CaseAccuracy, s.QuestionAccuracy = s.Tally.CaseAccuracy(), s.Tally.QuestionAccuracy()
	if elapsed > 0 {
		s.RequestsPerSecond = float64(len(durations)) / elapsed.Seconds()
		s.QuestionsPerSecond = float64(questions) / elapsed.Seconds()
	}
	slices.Sort(durations)
	if len(durations) > 0 {
		percentile := func(p float64) float64 {
			return float64(durations[int(math.Ceil(p*float64(len(durations))))-1]) / float64(time.Millisecond)
		}
		s.P50, s.P95, s.P99 = percentile(.5), percentile(.95), percentile(.99)
	}
	data, _ := json.Marshal(s)
	fmt.Fprintln(out, string(data))
}

func checkDecisionOutput(input, output string, corpus decisiontest.Corpus) error {
	dest, err := os.Stat(output)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	paths := []string{input}
	for _, c := range corpus.Cases {
		for _, image := range c.ImageFiles {
			paths = append(paths, filepath.Join(filepath.Dir(input), image))
		}
	}
	for _, path := range paths {
		source, err := os.Stat(path)
		if err != nil {
			return err
		}
		if os.SameFile(source, dest) {
			return fmt.Errorf("output would overwrite decision input %s", path)
		}
	}
	return nil
}

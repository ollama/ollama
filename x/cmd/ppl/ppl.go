// Command ppl measures perplexity of an MLX-loaded language model on a text
// corpus. It supports two scoring methodologies:
//
//   - "harness" (default): reproduces EleutherAI lm-evaluation-harness'
//     wikitext task. Document-level rolling loglikelihood with context_len=1.
//     Default corpus is fetched from the canonical Hugging Face dataset
//     (EleutherAI/wikitext_document_level, wikitext-2-raw-v1, test split).
//
//   - "llamacpp": reproduces llama.cpp's llama-perplexity tool. Concatenates
//     the corpus into one stream, splits into non-overlapping n_ctx-token
//     chunks with BOS substituted at chunk position 0, scores only the
//     second half of each chunk. Default corpus is the wiki.test.raw flat
//     file from the ggml-org/ci dataset (the source the llama.cpp community
//     uses).
//
// In both modes, -corpus FILE overrides the default fetch with a local file.
// In harness mode, the file is treated as one document per line; in llamacpp
// mode, the entire file is one stream.
//
// The CLI imports the mlxrunner package transitively to register all
// supported model architectures. Models are loaded via -model NAME (an
// ollama tag from the local store) or -model-dir PATH (a HuggingFace-format
// directory of weights and configs).
//
// Usage examples:
//
//	# Harness-mode (default) on an already-pulled ollama model
//	go run ./x/cmd/ppl -model mymodel:base-mlx-bf16
//
//	# Llama.cpp-compatible mode for direct comparison with llama-perplexity
//	go run ./x/cmd/ppl -model mymodel:base-mlx-bf16 -mode llamacpp
//
//	# Load straight from a safetensors directory (no ollama tag required)
//	go run ./x/cmd/ppl -model-dir models/mymodel-Base
//
//	# Tighter context for quicker dev iteration
//	go run ./x/cmd/ppl -model mymodel:base-mlx-bf16 -max-length 512 -max-docs 5
package main

import (
	"archive/zip"
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	_ "github.com/ollama/ollama/x/mlxrunner" // register model architectures
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/testutil"
)

const (
	// harnessDataset is the dataset lm-eval-harness uses for its `wikitext`
	// task. Hosted on the HuggingFace datasets-server.
	harnessDataset = "EleutherAI/wikitext_document_level"
	harnessConfig  = "wikitext-2-raw-v1"
	harnessSplit   = "test"

	// llamacppCorpusURL is the flat-text wiki.test.raw corpus llama.cpp's
	// llama-perplexity uses, distributed as a zip with a single file inside.
	llamacppCorpusURL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
)

func main() {
	var (
		modelName    = flag.String("model", "", "ollama model tag to load")
		modelDir     = flag.String("model-dir", "", "Safetensors model directory (alternative to -model)")
		modeFlag     = flag.String("mode", "harness", "scoring methodology: harness | llamacpp")
		corpusPath   = flag.String("corpus", "", "local corpus file (one doc per line in harness mode, single stream in llamacpp mode); default fetches the canonical dataset for the chosen mode")
		maxLength    = flag.Int("max-length", 0, "context window in tokens (default 2048 for harness, 512 for llamacpp)")
		maxDocs      = flag.Int("max-docs", 0, "limit number of documents (harness) or chunks (llamacpp); 0 = process all")
		baselinePath = flag.String("baseline", "", "optional path to a Python baseline JSON for comparison")
		cacheDirFlag = flag.String("cache-dir", "", "cache directory for downloaded corpora (default: $OLLAMA_PPL_CACHE_DIR or user cache)")
		maxRelDelta  = flag.Float64("max-rel-delta", 0, "optional maximum absolute relative token PPL delta vs -baseline; 0 disables")
		outFormat    = flag.String("format", "text", "output format: text | json")
		verbose      = flag.Bool("verbose", false, "enable per-document/chunk progress logging")
	)
	flag.Parse()

	if *modelName == "" && *modelDir == "" {
		fmt.Fprintln(os.Stderr, "ERROR: one of -model or -model-dir is required")
		flag.Usage()
		os.Exit(2)
	}
	if *modelName != "" && *modelDir != "" {
		fmt.Fprintln(os.Stderr, "ERROR: -model and -model-dir are mutually exclusive")
		os.Exit(2)
	}

	mode, err := testutil.ParseMode(*modeFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(2)
	}

	cdir, err := resolveCacheDir(*cacheDirFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: cannot resolve cache dir: %v\n", err)
		os.Exit(1)
	}
	if err := os.MkdirAll(cdir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: cannot create cache dir %s: %v\n", cdir, err)
		os.Exit(1)
	}

	// Logger writes progress to stderr so stdout stays clean for results.
	var log testutil.Logger = testutil.NewWriterLogger(os.Stderr)
	if !*verbose {
		log = testutil.NewWriterLogger(nil) // discard
	}

	// Load the model.
	startLoad := time.Now()
	var (
		model    base.Model
		modelTag string
		cleanup  func()
	)
	if *modelName != "" {
		modelTag = *modelName
		fmt.Fprintf(os.Stderr, "Loading ollama model %q...\n", *modelName)
		m, c, err := testutil.LoadModelByNameOrErr(*modelName)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
			os.Exit(1)
		}
		model = m
		cleanup = c
	} else {
		modelTag = filepath.Base(*modelDir)
		fmt.Fprintf(os.Stderr, "Loading model from %q...\n", *modelDir)
		blobDir, err := os.MkdirTemp("", "ollama-ppl-blobs-*")
		if err != nil {
			fmt.Fprintf(os.Stderr, "ERROR: tempdir: %v\n", err)
			os.Exit(1)
		}
		cleanup = func() { os.RemoveAll(blobDir) }
		m, err := testutil.LoadModelFromDirOrErr(*modelDir, blobDir)
		if err != nil {
			cleanup()
			fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
			os.Exit(1)
		}
		model = m
	}
	if cleanup != nil {
		defer cleanup()
	}
	fmt.Fprintf(os.Stderr, "Loaded in %.1fs\n", time.Since(startLoad).Seconds())

	// Fetch / load the corpus.
	docs, corpusSource, err := loadCorpus(mode, *corpusPath, cdir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR loading corpus: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Corpus: %d documents from %s\n", len(docs), corpusSource)

	// Run perplexity.
	opts := testutil.PPLOptions{
		Mode:            mode,
		MaxLength:       *maxLength,
		MaxDocs:         *maxDocs,
		BOSSwapLlamaCpp: mode == testutil.ModeLlamaCpp,
	}
	startEval := time.Now()
	result, err := testutil.RunPerplexity(model, docs, opts, log)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR running perplexity: %v\n", err)
		os.Exit(1)
	}
	evalDur := time.Since(startEval)

	baseline, err := loadBaseline(*baselinePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR loading baseline: %v\n", err)
		os.Exit(1)
	}
	delta := compareBaseline(result, baseline)

	// Print results.
	if *outFormat == "json" {
		printJSON(os.Stdout, modelTag, mode, opts, result, evalDur, delta)
	} else if *outFormat == "text" {
		printText(os.Stdout, modelTag, mode, opts, result, evalDur, delta)
	} else {
		fmt.Fprintf(os.Stderr, "ERROR: unknown -format %q (valid: text, json)\n", *outFormat)
		os.Exit(2)
	}

	if *maxRelDelta > 0 && delta != nil && math.Abs(delta.TokenPerplexityRel) > *maxRelDelta {
		fmt.Fprintf(os.Stderr, "ERROR: relative token PPL delta %.6f exceeds threshold %.6f\n", delta.TokenPerplexityRel, *maxRelDelta)
		os.Exit(1)
	}
}

// resolveCacheDir returns the corpus cache directory. Explicit CLI value wins,
// then OLLAMA_PPL_CACHE_DIR, then the user's standard cache directory.
func resolveCacheDir(flagValue string) (string, error) {
	if flagValue != "" {
		return flagValue, nil
	}
	if env := os.Getenv("OLLAMA_PPL_CACHE_DIR"); env != "" {
		return env, nil
	}
	userCache, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(userCache, "ollama", "ppl"), nil
}

// loadCorpus returns documents for the chosen mode. If localPath is non-empty,
// the local file is used and a humanized source string is returned. Otherwise
// the canonical dataset for the mode is fetched (with on-disk cache).
func loadCorpus(mode testutil.Mode, localPath, cacheDir string) ([]testutil.Document, string, error) {
	if localPath != "" {
		text, err := os.ReadFile(localPath)
		if err != nil {
			return nil, "", err
		}
		switch mode {
		case testutil.ModeHarness:
			// One document per line. Empty lines are skipped so users can
			// keep blank separators if they like.
			var docs []testutil.Document
			scanner := bufio.NewScanner(bytes.NewReader(text))
			scanner.Buffer(make([]byte, 1<<20), 1<<27)
			for scanner.Scan() {
				line := strings.TrimSpace(scanner.Text())
				if line == "" {
					continue
				}
				docs = append(docs, testutil.Document{Text: line})
			}
			if err := scanner.Err(); err != nil {
				return nil, "", err
			}
			return docs, fmt.Sprintf("local file %s (%d lines)", localPath, len(docs)), nil
		case testutil.ModeLlamaCpp:
			return []testutil.Document{{Text: string(text)}}, fmt.Sprintf("local file %s", localPath), nil
		}
	}

	switch mode {
	case testutil.ModeHarness:
		docs, src, err := fetchHarnessCorpus(cacheDir)
		return docs, src, err
	case testutil.ModeLlamaCpp:
		text, src, err := fetchLlamaCppCorpus(cacheDir)
		if err != nil {
			return nil, "", err
		}
		return []testutil.Document{{Text: text}}, src, nil
	}
	return nil, "", fmt.Errorf("unhandled mode: %v", mode)
}

// fetchHarnessCorpus fetches the wikitext-2 test split via the HuggingFace
// datasets-server JSON API. It returns 62 documents (entire test split).
// Cached on disk after the first fetch.
func fetchHarnessCorpus(cacheDir string) ([]testutil.Document, string, error) {
	cachePath := filepath.Join(cacheDir, "wikitext-2-raw-v1-test-harness.json")
	if data, err := os.ReadFile(cachePath); err == nil {
		var docs []testutil.Document
		if err := json.Unmarshal(data, &docs); err == nil && len(docs) > 0 {
			return docs, fmt.Sprintf("cache %s", cachePath), nil
		}
		// fall through and refetch
	}

	q := url.Values{}
	q.Set("dataset", harnessDataset)
	q.Set("config", harnessConfig)
	q.Set("split", harnessSplit)
	q.Set("offset", "0")
	q.Set("length", "100") // wikitext-2 test has 62 docs; 100 fits in one request
	endpoint := "https://datasets-server.huggingface.co/rows?" + q.Encode()

	fmt.Fprintf(os.Stderr, "Fetching corpus from %s ...\n", harnessDataset)
	body, err := httpGet(endpoint)
	if err != nil {
		return nil, "", fmt.Errorf("fetch wikitext: %w", err)
	}

	var resp struct {
		Rows []struct {
			Row struct {
				Page string `json:"page"`
			} `json:"row"`
			TruncatedCells []string `json:"truncated_cells"`
		} `json:"rows"`
		NumRowsTotal int `json:"num_rows_total"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, "", fmt.Errorf("parse wikitext json: %w", err)
	}
	if len(resp.Rows) == 0 {
		return nil, "", fmt.Errorf("no rows returned from datasets-server")
	}
	if len(resp.Rows) < resp.NumRowsTotal {
		return nil, "", fmt.Errorf("partial fetch: got %d of %d rows; pagination not implemented", len(resp.Rows), resp.NumRowsTotal)
	}

	docs := make([]testutil.Document, 0, len(resp.Rows))
	for _, r := range resp.Rows {
		if len(r.TruncatedCells) > 0 {
			return nil, "", fmt.Errorf("dataset row was truncated by datasets-server; cannot proceed")
		}
		// Apply the wikitext detokenizer to what the model sees, but
		// keep Text (= the original page) for word/byte denominators.
		// Matches lm-evaluation-harness preprocess_wikitext exactly.
		page := r.Row.Page
		docs = append(docs, testutil.Document{
			Text:        page,
			ScoringText: testutil.WikitextDetokenize(page),
		})
	}

	// Persist cache.
	if data, err := json.Marshal(docs); err == nil {
		_ = os.WriteFile(cachePath, data, 0o644)
	}

	return docs, fmt.Sprintf("HuggingFace %s/%s/%s (%d docs)", harnessDataset, harnessConfig, harnessSplit, len(docs)), nil
}

// fetchLlamaCppCorpus fetches the wikitext-2-raw-v1.zip from ggml-org/ci on
// HuggingFace and extracts wiki.test.raw. The text is cached on disk after
// the first fetch.
func fetchLlamaCppCorpus(cacheDir string) (string, string, error) {
	cachePath := filepath.Join(cacheDir, "wiki.test.raw")
	if data, err := os.ReadFile(cachePath); err == nil && len(data) > 0 {
		return string(data), fmt.Sprintf("cache %s", cachePath), nil
	}

	fmt.Fprintf(os.Stderr, "Fetching corpus from %s ...\n", llamacppCorpusURL)
	body, err := httpGet(llamacppCorpusURL)
	if err != nil {
		return "", "", fmt.Errorf("fetch llamacpp corpus: %w", err)
	}

	zr, err := zip.NewReader(bytes.NewReader(body), int64(len(body)))
	if err != nil {
		return "", "", fmt.Errorf("open zip: %w", err)
	}
	for _, f := range zr.File {
		if !strings.HasSuffix(f.Name, "wiki.test.raw") {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return "", "", err
		}
		text, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			return "", "", err
		}
		_ = os.WriteFile(cachePath, text, 0o644)
		return string(text), fmt.Sprintf("HuggingFace ggml-org/ci %s", filepath.Base(f.Name)), nil
	}
	return "", "", fmt.Errorf("wiki.test.raw not found in zip")
}

// httpGet performs a one-shot GET with a sane timeout.
func httpGet(rawURL string) ([]byte, error) {
	client := &http.Client{Timeout: 5 * time.Minute}
	req, err := http.NewRequest("GET", rawURL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("HTTP %d %s", resp.StatusCode, resp.Status)
	}
	return io.ReadAll(resp.Body)
}

type pplBaseline struct {
	Model           string  `json:"model"`
	Mode            string  `json:"mode"`
	MaxLength       int     `json:"max_length"`
	TokenPerplexity float64 `json:"token_perplexity"`
}

type pplBaselineDelta struct {
	BaselineModel           string  `json:"baseline_model,omitempty"`
	BaselineMode            string  `json:"baseline_mode,omitempty"`
	BaselineMaxLength       int     `json:"baseline_max_length,omitempty"`
	BaselineTokenPerplexity float64 `json:"baseline_token_perplexity"`
	TokenPerplexityAbs      float64 `json:"token_perplexity_abs"`
	TokenPerplexityRel      float64 `json:"token_perplexity_rel"`
}

func loadBaseline(path string) (*pplBaseline, error) {
	if path == "" {
		return nil, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var baseline pplBaseline
	if err := json.Unmarshal(data, &baseline); err != nil {
		return nil, err
	}
	if baseline.TokenPerplexity <= 0 {
		return nil, fmt.Errorf("%s has invalid token_perplexity: %f", path, baseline.TokenPerplexity)
	}
	return &baseline, nil
}

func compareBaseline(r testutil.PPLResult, baseline *pplBaseline) *pplBaselineDelta {
	if baseline == nil {
		return nil
	}
	abs := r.TokenPerplexity - baseline.TokenPerplexity
	rel := abs / baseline.TokenPerplexity
	return &pplBaselineDelta{
		BaselineModel:           baseline.Model,
		BaselineMode:            baseline.Mode,
		BaselineMaxLength:       baseline.MaxLength,
		BaselineTokenPerplexity: baseline.TokenPerplexity,
		TokenPerplexityAbs:      abs,
		TokenPerplexityRel:      rel,
	}
}

func printText(w io.Writer, modelTag string, mode testutil.Mode, opts testutil.PPLOptions, r testutil.PPLResult, dur time.Duration, delta *pplBaselineDelta) {
	fmt.Fprintf(w, "\nModel:        %s\n", modelTag)
	fmt.Fprintf(w, "Mode:         %s\n", mode)
	fmt.Fprintf(w, "Max length:   %d\n", opts.MaxLength)
	fmt.Fprintf(w, "Eval time:    %.1fs\n", dur.Seconds())
	fmt.Fprintf(w, "Tokens:       %d scored\n", r.TotalTokens)
	if r.TotalWords > 0 {
		fmt.Fprintf(w, "Words:        %d\n", r.TotalWords)
	}
	if r.TotalChars > 0 {
		fmt.Fprintf(w, "Bytes:        %d\n", r.TotalChars)
	}
	fmt.Fprintln(w)
	fmt.Fprintf(w, "Token PPL:        %12.4f  +/- %.4f\n", r.TokenPerplexity, r.StderrTokenPPL)
	if r.WordPerplexity > 0 {
		fmt.Fprintf(w, "Word PPL:         %12.4f\n", r.WordPerplexity)
	}
	if r.BytePerplexity > 0 {
		fmt.Fprintf(w, "Byte PPL:         %12.4f\n", r.BytePerplexity)
		fmt.Fprintf(w, "Bits per byte:    %12.4f\n", r.BitsPerByte)
	}
	if delta != nil {
		fmt.Fprintln(w)
		fmt.Fprintf(w, "Baseline token PPL: %10.4f\n", delta.BaselineTokenPerplexity)
		fmt.Fprintf(w, "Token PPL delta:    %+10.4f (%+.4f%%)\n",
			delta.TokenPerplexityAbs,
			delta.TokenPerplexityRel*100,
		)
	}
	fmt.Fprintln(w)
}

func printJSON(w io.Writer, modelTag string, mode testutil.Mode, opts testutil.PPLOptions, r testutil.PPLResult, dur time.Duration, delta *pplBaselineDelta) {
	out := map[string]any{
		"model":            modelTag,
		"mode":             mode.String(),
		"max_length":       opts.MaxLength,
		"eval_seconds":     dur.Seconds(),
		"total_tokens":     r.TotalTokens,
		"total_words":      r.TotalWords,
		"total_bytes":      r.TotalChars,
		"token_perplexity": r.TokenPerplexity,
		"stderr_token_ppl": r.StderrTokenPPL,
		"word_perplexity":  r.WordPerplexity,
		"byte_perplexity":  r.BytePerplexity,
		"bits_per_byte":    r.BitsPerByte,
	}
	if delta != nil {
		out["baseline_delta"] = delta
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(out)
}

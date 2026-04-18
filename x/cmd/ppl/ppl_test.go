package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/x/models/testutil"
)

func TestResolveCacheDirPrecedence(t *testing.T) {
	t.Setenv("OLLAMA_PPL_CACHE_DIR", "/env/cache")
	got, err := resolveCacheDir("/flag/cache")
	if err != nil {
		t.Fatal(err)
	}
	if got != "/flag/cache" {
		t.Fatalf("resolveCacheDir flag = %q, want /flag/cache", got)
	}

	got, err = resolveCacheDir("")
	if err != nil {
		t.Fatal(err)
	}
	if got != "/env/cache" {
		t.Fatalf("resolveCacheDir env = %q, want /env/cache", got)
	}
}

func TestResolveCacheDirDefault(t *testing.T) {
	t.Setenv("OLLAMA_PPL_CACHE_DIR", "")
	got, err := resolveCacheDir("")
	if err != nil {
		t.Fatal(err)
	}
	wantSuffix := filepath.Join("ollama", "ppl")
	if !strings.HasSuffix(got, wantSuffix) {
		t.Fatalf("resolveCacheDir default = %q, want suffix %q", got, wantSuffix)
	}
}

func TestLoadAndCompareBaseline(t *testing.T) {
	path := filepath.Join(t.TempDir(), "baseline.json")
	if err := os.WriteFile(path, []byte(`{"model":"hf","mode":"harness","max_length":128,"token_perplexity":10}`), 0o644); err != nil {
		t.Fatal(err)
	}

	baseline, err := loadBaseline(path)
	if err != nil {
		t.Fatal(err)
	}
	delta := compareBaseline(testutil.PPLResult{TokenPerplexity: 11}, baseline)
	if delta == nil {
		t.Fatal("expected delta")
	}
	if delta.TokenPerplexityAbs != 1 {
		t.Fatalf("abs delta = %f, want 1", delta.TokenPerplexityAbs)
	}
	if delta.TokenPerplexityRel != 0.1 {
		t.Fatalf("rel delta = %f, want 0.1", delta.TokenPerplexityRel)
	}
}

func TestLoadBaselineRejectsInvalidPPL(t *testing.T) {
	path := filepath.Join(t.TempDir(), "baseline.json")
	if err := os.WriteFile(path, []byte(`{"token_perplexity":0}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadBaseline(path); err == nil {
		t.Fatal("expected invalid baseline error")
	}
}

func TestPrintJSONIncludesBaselineDelta(t *testing.T) {
	var buf bytes.Buffer
	delta := &pplBaselineDelta{
		BaselineModel:           "hf",
		BaselineMode:            "harness",
		BaselineMaxLength:       128,
		BaselineTokenPerplexity: 10,
		TokenPerplexityAbs:      1,
		TokenPerplexityRel:      0.1,
	}
	printJSON(
		&buf,
		"mlx",
		testutil.ModeHarness,
		testutil.PPLOptions{MaxLength: 128},
		testutil.PPLResult{TokenPerplexity: 11, TotalTokens: 20},
		time.Second,
		delta,
	)

	var out map[string]any
	if err := json.Unmarshal(buf.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	rawDelta, ok := out["baseline_delta"].(map[string]any)
	if !ok {
		t.Fatalf("baseline_delta missing from JSON: %#v", out)
	}
	if rawDelta["token_perplexity_rel"] != 0.1 {
		t.Fatalf("rel delta = %#v, want 0.1", rawDelta["token_perplexity_rel"])
	}
}

func TestPrintTextIncludesBaselineDelta(t *testing.T) {
	var buf bytes.Buffer
	printText(
		&buf,
		"mlx",
		testutil.ModeHarness,
		testutil.PPLOptions{MaxLength: 128},
		testutil.PPLResult{TokenPerplexity: 11, StderrTokenPPL: 0.2, TotalTokens: 20},
		time.Second,
		&pplBaselineDelta{
			BaselineTokenPerplexity: 10,
			TokenPerplexityAbs:      1,
			TokenPerplexityRel:      0.1,
		},
	)
	text := buf.String()
	if !strings.Contains(text, "Baseline token PPL") {
		t.Fatalf("baseline summary missing from text output:\n%s", text)
	}
	if !strings.Contains(text, "+10.0000%") {
		t.Fatalf("relative delta missing from text output:\n%s", text)
	}
}

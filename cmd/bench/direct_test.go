package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/mlxrunner/wire"
)

// mockRunner is an MLX runner stand-in that records every completion request.
type mockRunner struct {
	mu       sync.Mutex
	requests []wire.CompletionRequest
	cached   int

	// prefix fakes a prefix cache over bytes, one byte per token: a prompt
	// reuses its longest common prefix with any stored prompt, less one
	// held-back token. Images fold into the key by content. Prefill runs in
	// timed chunks and a cancelled request stores what it processed. Stored
	// prompts other than the latest count as cold bytes, evicted oldest first
	// past coldLimit.
	prefix    bool
	coldLimit int64
	evicted   int64
	seen      []string
}

var mockImgTag = regexp.MustCompile(`\[img-(\d+)\]`)

func mockKey(req wire.CompletionRequest) string {
	return mockImgTag.ReplaceAllStringFunc(req.Prompt, func(tag string) string {
		id, _ := strconv.Atoi(mockImgTag.FindStringSubmatch(tag)[1])
		for _, m := range req.Media {
			if m.ID == id {
				return fmt.Sprintf("<img:%x>", sha256.Sum256(m.Data))
			}
		}
		return tag
	})
}

// store records key as the latest prompt and evicts cold prompts past the
// limit, returning the cold bytes that remain.
func (m *mockRunner) store(key string) int64 {
	m.seen = slices.DeleteFunc(m.seen, func(s string) bool { return s == key })
	m.seen = append(m.seen, key)
	cold := func() (n int64) {
		for _, s := range m.seen[:len(m.seen)-1] {
			n += int64(len(s))
		}
		return n
	}
	for m.coldLimit > 0 && cold() > m.coldLimit && len(m.seen) > 1 {
		m.evicted += int64(len(m.seen[0]))
		m.seen = m.seen[1:]
	}
	return cold()
}

func commonPrefix(a, b string) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}

func (m *mockRunner) start(t *testing.T) string {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/status", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"Status": 0, "Progress": 100, "ContextLength": 65536, "Memory": 1 << 30})
	})
	mux.HandleFunc("POST /v1/completions", func(w http.ResponseWriter, r *http.Request) {
		var req wire.CompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if len(req.Format) > 0 && string(req.Format) != "null" && !strings.Contains(string(req.Format), `"structural_tag"`) {
			http.Error(w, "invalid format: expected a structural tag", http.StatusBadRequest)
			return
		}
		m.mu.Lock()
		defer m.mu.Unlock()
		start := time.Now()
		m.requests = append(m.requests, req)
		cached := m.cached
		promptEval := 100
		var coldBytes int64
		if m.prefix {
			key := mockKey(req)
			promptEval = len(key) + 1
			cached = 0
			for _, prev := range m.seen {
				cached = max(cached, commonPrefix(prev, key))
			}
			cached = min(cached, promptEval-1)
			const chunk = 2048
			for done := cached; done < len(key); done += chunk {
				select {
				case <-r.Context().Done():
					m.store(key[:done])
					return
				case <-time.After(12 * time.Millisecond):
				}
			}
			coldBytes = m.store(key)
		}

		enc := json.NewEncoder(w)
		evalCount := max(req.Options.NumPredict, 0)
		if evalCount > 0 {
			enc.Encode(wire.CompletionResponse{Content: "x"})
		}
		enc.Encode(wire.CompletionResponse{
			Done:                  true,
			PromptEvalCount:       promptEval,
			PromptEvalCachedCount: &cached,
			PromptEvalDuration:    max(time.Since(start), time.Millisecond),
			EvalCount:             evalCount,
			EvalDuration:          20 * time.Millisecond,
			Stats:                 &wire.Stats{MatchedTokens: cached, ColdBytes: coldBytes, ColdLimit: m.coldLimit, ColdEvicted: m.evicted},
		})
	})
	mux.HandleFunc("POST /v1/tokenize", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.NewEncoder(w).Encode(make([]int32, len(body)+1))
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	return strings.TrimPrefix(server.URL, "http://")
}

func directTestOptions(t *testing.T, mode string, m *mockRunner) flagOptions {
	fOpt := createTestFlagOptions()
	addr := m.start(t)
	fOpt.runner = &addr
	fOpt.mode = &mode
	epochs := 3
	fOpt.epochs = &epochs
	return fOpt
}

func TestDirectParams(t *testing.T) {
	fOpt := createTestFlagOptions()
	ignoreEOS := true
	fOpt.ignoreEOS = &ignoreEOS

	p := directParams(fOpt, modeBoth, "p")
	if p.numPredict != 50 || !p.ignoreEOS || p.seed != 42 {
		t.Errorf("both: got numPredict=%d ignoreEOS=%v seed=%d", p.numPredict, p.ignoreEOS, p.seed)
	}

	p = directParams(fOpt, modePrefill, "p")
	if p.numPredict != 0 || p.ignoreEOS {
		t.Errorf("prefill: got numPredict=%d ignoreEOS=%v, want 0 false", p.numPredict, p.ignoreEOS)
	}

	seed := 0
	fOpt.seed = &seed
	if p := directParams(fOpt, modeBoth, "p"); p.seed != -1 {
		t.Errorf("seed 0 should map to random (-1), got %d", p.seed)
	}
}

func TestBenchmarkDirect_DecodeModeReusesPrompt(t *testing.T) {
	m := &mockRunner{}
	fOpt := directTestOptions(t, modeDecode, m)

	captureOutput(func() {
		if err := benchmarkDirect(fOpt, &strings.Builder{}); err != nil {
			t.Fatal(err)
		}
	})

	// One priming warmup plus three epochs, all on the same prompt.
	if len(m.requests) != 4 {
		t.Fatalf("got %d requests, want 4", len(m.requests))
	}
	for i, req := range m.requests {
		if req.Prompt != m.requests[0].Prompt {
			t.Errorf("request %d prompt differs from the first", i)
		}
	}
}

func TestBenchmarkDirect_PrefillModeUniquePrefillOnly(t *testing.T) {
	m := &mockRunner{}
	fOpt := directTestOptions(t, modePrefill, m)

	captureOutput(func() {
		if err := benchmarkDirect(fOpt, &strings.Builder{}); err != nil {
			t.Fatal(err)
		}
	})

	seen := map[string]bool{}
	for i, req := range m.requests {
		if req.Options.NumPredict != 0 {
			t.Errorf("request %d num_predict=%d, want 0", i, req.Options.NumPredict)
		}
		if seen[req.Prompt] {
			t.Errorf("request %d repeats an earlier prompt", i)
		}
		seen[req.Prompt] = true
	}
}

func TestBenchmarkDirect_ReportsCachedTokens(t *testing.T) {
	m := &mockRunner{cached: 60}
	fOpt := directTestOptions(t, modeDecode, m)
	ignoreEOS := true
	fOpt.ignoreEOS = &ignoreEOS

	var out strings.Builder
	captureOutput(func() {
		if err := benchmarkDirect(fOpt, &out); err != nil {
			t.Fatal(err)
		}
	})

	if !strings.Contains(out.String(), "40 processed-prompt-token 60 cached-prompt-token") {
		t.Errorf("prefill line missing cached split:\n%s", out.String())
	}
	for i, req := range m.requests {
		if !req.IgnoreEOS {
			t.Errorf("request %d did not forward IgnoreEOS", i)
		}
		if req.Options.TopK != 40 {
			t.Errorf("request %d TopK=%d, want API default 40", i, req.Options.TopK)
		}
	}
}

func TestBenchmarkDirect_RejectsUnknownMode(t *testing.T) {
	m := &mockRunner{}
	fOpt := directTestOptions(t, "sideways", m)
	if err := benchmarkDirect(fOpt, &strings.Builder{}); err == nil {
		t.Fatal("expected an error for an unknown mode")
	}
}

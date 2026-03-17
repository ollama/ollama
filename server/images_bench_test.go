package server

import (
	"bytes"
	"cmp"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/google/go-cmp/cmp/cmpopts"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/convert"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/template"

	gocmp "github.com/google/go-cmp/cmp"
)

// benchModelSink prevents the compiler from eliminating GetModel calls
// in benchmark loops via dead-code elimination.
var benchModelSink *Model

// createBinFileBench mirrors createBinFile but accepts testing.TB
// so it works in both Test and Benchmark contexts.
func createBinFileBench(tb testing.TB, kv map[string]any, ti []*ggml.Tensor) (string, string) {
	tb.Helper()
	tb.Setenv("OLLAMA_MODELS", cmp.Or(os.Getenv("OLLAMA_MODELS"), tb.TempDir()))

	modelDir := envconfig.Models()

	f, err := os.CreateTemp(tb.TempDir(), "")
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()

	var base convert.KV = map[string]any{"general.architecture": "test"}
	maps.Copy(base, kv)

	if err := ggml.WriteGGUF(f, base, ti); err != nil {
		tb.Fatal(err)
	}

	if _, err := f.Seek(0, 0); err != nil {
		tb.Fatal(err)
	}

	digest, _ := GetSHA256Digest(f)
	if err := f.Close(); err != nil {
		tb.Fatal(err)
	}

	if err := createLink(f.Name(), fmt.Sprintf("%s/blobs/sha256-%s", modelDir, strings.TrimPrefix(digest, "sha256:"))); err != nil {
		tb.Fatal(err)
	}

	return f.Name(), digest
}

// createRequestBench mirrors createRequest but accepts testing.TB.
func createRequestBench(tb testing.TB, fn func(*gin.Context), body any) *httptest.ResponseRecorder {
	tb.Helper()
	tb.Setenv("OLLAMA_MODELS", cmp.Or(os.Getenv("OLLAMA_MODELS"), tb.TempDir()))

	w := NewRecorder()
	c, _ := gin.CreateTestContext(w)

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(body); err != nil {
		tb.Fatal(err)
	}

	c.Request = &http.Request{
		Body: io.NopCloser(&b),
	}

	fn(c)
	return w.ResponseRecorder
}

// setupBenchModel creates a model on disk via CreateHandler, returns the model name.
// When full is true, includes template, system prompt, parameters, messages, and license layers.
// When full is false, creates a minimal model with only the GGUF binary.
func setupBenchModel(tb testing.TB, full bool) string {
	tb.Helper()
	gin.SetMode(gin.TestMode)

	p := tb.TempDir()
	tb.Setenv("OLLAMA_MODELS", p)

	var s Server
	_, digest := createBinFileBench(tb, nil, nil)

	name := "benchmodel"

	if full {
		w := createRequestBench(tb, s.CreateHandler, api.CreateRequest{
			Name:     name,
			Files:    map[string]string{"model.gguf": digest},
			Template: "{{ .System }}\n{{ .Prompt }}",
			System:   "You are a helpful assistant.",
			Parameters: map[string]any{
				"temperature": 0.7,
				"top_k":       40,
				"top_p":       0.9,
			},
			Messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi! How can I help you today?"},
			},
			License: []string{"MIT License test benchmark"},
			Stream:  &stream,
		})
		if w.Code != http.StatusOK {
			tb.Fatalf("setupBenchModel full: expected 200, got %d: %s", w.Code, w.Body.String())
		}
	} else {
		w := createRequestBench(tb, s.CreateHandler, api.CreateRequest{
			Name:   name,
			Files:  map[string]string{"model.gguf": digest},
			Stream: &stream,
		})
		if w.Code != http.StatusOK {
			tb.Fatalf("setupBenchModel minimal: expected 200, got %d: %s", w.Code, w.Body.String())
		}
	}

	return name
}

// BenchmarkGetModel measures the per-call cost of GetModel, which performs
// 6-7 disk I/O operations (manifest, config, template, system, params,
// messages, license). The OS page cache warms after setup, so these
// benchmarks measure application-layer cost: JSON decode, SHA256 digest
// computation, and template parsing — not raw disk latency.
func BenchmarkGetModel(b *testing.B) {
	b.Run("Minimal", func(b *testing.B) {
		name := setupBenchModel(b, false)
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			m, err := GetModel(name)
			if err != nil {
				b.Fatal(err)
			}
			benchModelSink = m
		}
	})

	b.Run("FullLayers", func(b *testing.B) {
		name := setupBenchModel(b, true)
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			m, err := GetModel(name)
			if err != nil {
				b.Fatal(err)
			}
			benchModelSink = m
		}
	})
}

// BenchmarkGetModelDoubleCall simulates the ChatHandler hot path where
// GetModel is called twice per request: once for validation (routes.go:2114)
// and once inside scheduleRunner (routes.go:149). After implementing an LRU
// cache, the second call should drop from milliseconds to microseconds.
func BenchmarkGetModelDoubleCall(b *testing.B) {
	b.Run("Minimal", func(b *testing.B) {
		name := setupBenchModel(b, false)
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			m1, err := GetModel(name)
			if err != nil {
				b.Fatal(err)
			}
			m2, err := GetModel(name)
			if err != nil {
				b.Fatal(err)
			}
			benchModelSink = m1
			benchModelSink = m2
		}
	})

	b.Run("FullLayers", func(b *testing.B) {
		name := setupBenchModel(b, true)
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			m1, err := GetModel(name)
			if err != nil {
				b.Fatal(err)
			}
			m2, err := GetModel(name)
			if err != nil {
				b.Fatal(err)
			}
			benchModelSink = m1
			benchModelSink = m2
		}
	})
}

// TestGetModelConsistency verifies that consecutive GetModel calls on the
// same model return structurally identical results. This guards correctness
// for future cache implementations: a cache hit must produce the same Model
// as a cache miss.
func TestGetModelConsistency(t *testing.T) {
	gin.SetMode(gin.TestMode)
	name := setupBenchModel(t, true)

	m1, err := GetModel(name)
	if err != nil {
		t.Fatalf("first GetModel: %v", err)
	}

	m2, err := GetModel(name)
	if err != nil {
		t.Fatalf("second GetModel: %v", err)
	}

	// Compare all fields except Template, which is a parsed *template.Template
	// and not comparable via DeepEqual. We verify template equivalence by
	// executing both with identical input and comparing output.
	opts := gocmp.Options{
		cmpopts.IgnoreFields(Model{}, "Template"),
		cmpopts.EquateEmpty(),
	}
	if diff := gocmp.Diff(m1, m2, opts...); diff != "" {
		t.Errorf("GetModel consistency mismatch (-first +second):\n%s", diff)
	}

	// Verify templates produce identical output
	templateInput := template.Values{
		Messages: []api.Message{
			{Role: "system", Content: "test system"},
			{Role: "user", Content: "test prompt"},
		},
	}

	var buf1, buf2 bytes.Buffer
	if err := m1.Template.Execute(&buf1, templateInput); err != nil {
		t.Fatalf("template execute m1: %v", err)
	}
	if err := m2.Template.Execute(&buf2, templateInput); err != nil {
		t.Fatalf("template execute m2: %v", err)
	}

	if buf1.String() != buf2.String() {
		t.Errorf("template output mismatch:\n  first:  %q\n  second: %q", buf1.String(), buf2.String())
	}
}

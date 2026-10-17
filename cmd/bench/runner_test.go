package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestDirectRunnerErrorsInvalidateMeasurement(t *testing.T) {
	for _, tt := range []struct{ name, body string }{
		{"malformed response", "{invalid}\n"},
		{"runner error", "{\"error\":{\"error\":\"backend failed\"}}\n"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			mux := http.NewServeMux()
			mux.HandleFunc("GET /v1/status", func(w http.ResponseWriter, _ *http.Request) {
				fmt.Fprintln(w, `{}`)
			})
			mux.HandleFunc("POST /v1/completions", func(w http.ResponseWriter, _ *http.Request) {
				fmt.Fprint(w, tt.body)
				fmt.Fprintln(w, `{"Done":true,"EvalCount":50}`)
			})
			server := httptest.NewServer(mux)
			t.Cleanup(server.Close)
			options := createTestFlagOptions()
			addr := strings.TrimPrefix(server.URL, "http://")
			options.runner = &addr
			var output strings.Builder
			if err := benchmarkDirect(options, &output); err == nil {
				t.Fatal("failed completion reported a successful benchmark")
			}
			if strings.Contains(output.String(), "Benchmark") {
				t.Fatalf("failed completion emitted measurements: %s", output.String())
			}
		})
	}
}

func TestScenarioCancellationRejectsServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "backend failed", http.StatusInternalServerError)
	}))
	t.Cleanup(server.Close)
	run := scenarioRun{
		backend: &runnerBackend{addr: strings.TrimPrefix(server.URL, "http://"), client: server.Client()},
		fOpt:    createTestFlagOptions(),
	}
	if _, err := run.step(stepSpec{name: "cancel", prompt: "test", cancel: time.Minute}); err == nil {
		t.Fatal("server failure counted as successful cancellation")
	}
}

func TestLlamaServerRejectsMalformedEvent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, "data: {invalid}\n\ndata: {\"stop\":true}\n")
	}))
	t.Cleanup(server.Close)
	b := llamaServerBackend{addr: strings.TrimPrefix(server.URL, "http://"), client: server.Client()}
	if _, err := b.Complete(t.Context(), completionParams{prompt: "test"}); err == nil {
		t.Fatal("malformed event was silently discarded")
	}
}

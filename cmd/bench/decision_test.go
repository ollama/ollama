package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/decision"
	"github.com/ollama/ollama/internal/decisiontest"
)

func TestDecisionBenchmarkPolicies(t *testing.T) {
	for _, test := range []struct {
		name, mode, answer string
		wantErr            bool
		cold               bool
		unloadStatus       int
	}{
		{"correct regression", "regression", `"noul":1`, false, false, 0},
		{"wrong regression", "regression", `"noul":0`, true, false, 0},
		{"wrong score", "score", `"noul":0`, false, false, 0},
		{"broken score response", "score", `"noul":null`, true, false, 0},
		{"cold after warmup", "regression", `"noul":1`, false, true, 0},
		{"unload failure", "score", `"noul":1`, true, false, http.StatusInternalServerError},
	} {
		t.Run(test.name, func(t *testing.T) {
			var requests, unloads atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/generate":
					unloads.Add(1)
					if test.unloadStatus != 0 {
						http.Error(w, "unload failed", test.unloadStatus)
						return
					}
					writeJSON(w, api.GenerateResponse{Done: true})
				case "/v1/systemone":
					requests.Add(1)
					if test.cold && unloads.Load() != requests.Load() {
						t.Error("cold request was not preceded by an unload")
					}
					var req decision.Request
					if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
						t.Error(err)
					}
					if string(req.State) != `"An ordinary message."` || req.Model != "test-model" {
						t.Errorf("request changed: %+v", req)
					}
					if !test.cold && req.KeepAlive != nil {
						t.Error("warm request overrode the server keep_alive default")
					}
					fmt.Fprintf(w, `{"model":"test-model","total_duration":10000000,"load_duration":2000000,"usage":{"input_tokens":42,"output_tokens":0},"answers":{"result":{"type":"noul",%s}}}`, test.answer)
				default:
					t.Errorf("unexpected endpoint %s", r.URL.Path)
				}
			}))
			defer server.Close()
			t.Setenv("OLLAMA_HOST", server.URL)
			path := filepath.Join(t.TempDir(), "cases.jsonl")
			if err := os.WriteFile(path, []byte(`{"id":"test,case","dataset":"test","request":{"state":"An ordinary message.","questions":{"result":{"type":"noul","instructions":"Is this a message?"}}},"expected":{"result":{"noul":true}}}`), 0o600); err != nil {
				t.Fatal(err)
			}
			opts := createTestFlagOptions()
			concurrency := 1
			opts.decisionFile, opts.decisionMode, opts.concurrency = &path, &test.mode, &concurrency
			*opts.format, *opts.epochs = "csv", 2
			if test.cold {
				*opts.keepAlive, *opts.epochs, *opts.warmup = .001, 1, 1
			}
			var out bytes.Buffer
			err := benchmarkDecision(opts, []string{"test-model"}, &out)
			if (err != nil) != test.wantErr {
				t.Fatalf("error=%v, wantErr=%v", err, test.wantErr)
			}
			rows, err := csv.NewReader(&out).ReadAll()
			if err != nil {
				t.Fatal(err)
			}
			wantUnloads := int32(1)
			if test.cold {
				wantUnloads += requests.Load()
			}
			if unloads.Load() != wantUnloads || len(rows) != int(requests.Load())-*opts.warmup+1 {
				t.Fatalf("unloads=%d requests=%d csv rows=%d", unloads.Load(), requests.Load(), len(rows))
			}
			if rows[1][3] != "test,case" || rows[1][6] != "42" {
				t.Fatalf("missing per-request metadata or CSV escaping: %v", rows[1])
			}
			if rows[0][18] != "total_duration_ns" || rows[0][19] != "load_duration_ns" || rows[1][18] != "10000000" || rows[1][19] != "2000000" {
				t.Fatalf("server timings were not preserved: %v", rows)
			}
			if test.answer == `"noul":0` && (rows[1][17] != "" || len(rows[1]) <= 20 || rows[1][20] == "") {
				t.Fatalf("label mismatch conflated with request error: %v", rows[1])
			}
		})
	}
}

func TestDecisionBenchmarkOutputError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/generate" {
			writeJSON(w, api.GenerateResponse{Done: true})
			return
		}
		fmt.Fprint(w, `{"model":"test-model","usage":{"input_tokens":20,"output_tokens":0},"answers":{"result":{"type":"noul","noul":1}}}`)
	}))
	t.Cleanup(server.Close)
	t.Setenv("OLLAMA_HOST", server.URL)
	path := filepath.Join(t.TempDir(), "cases.jsonl")
	if err := os.WriteFile(path, []byte(`{"id":"one","dataset":"test","request":{"state":"Message.","questions":{"result":{"type":"noul"}}},"expected":{"result":{"noul":true}}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	output, err := os.CreateTemp(t.TempDir(), "closed-output")
	if err != nil {
		t.Fatal(err)
	}
	if err := output.Close(); err != nil {
		t.Fatal(err)
	}
	for _, format := range []string{"csv", "benchstat"} {
		t.Run(format, func(t *testing.T) {
			opts := createTestFlagOptions()
			concurrency, mode := 1, "score"
			opts.decisionFile, opts.decisionMode, opts.concurrency = &path, &mode, &concurrency
			*opts.format, *opts.epochs = format, 1
			if err := benchmarkDecision(opts, []string{"test-model"}, output); !errors.Is(err, os.ErrClosed) {
				t.Fatalf("got %v, want output write error", err)
			}
		})
	}
}

func TestDecisionOutputPreservesInputs(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "cases.jsonl")
	data := []byte(`{"id":"one","dataset":"test","request":{"state":"Message.","questions":{"result":{"type":"noul"}}},"expected":{"result":{"noul":true}}}`)
	if err := os.WriteFile(input, data, 0o600); err != nil {
		t.Fatal(err)
	}
	alias := filepath.Join(dir, "alias")
	if err := os.Link(input, alias); err != nil {
		t.Fatal(err)
	}
	for _, output := range []string{input, alias} {
		opts := createTestFlagOptions()
		mode, concurrency := "score", 1
		opts.decisionFile, opts.decisionMode, opts.concurrency, opts.outputFile = &input, &mode, &concurrency, &output
		if err := BenchmarkModel(opts); err == nil || !strings.Contains(err.Error(), "overwrite") {
			t.Fatalf("output %s: got %v, want overwrite error", output, err)
		}
		got, err := os.ReadFile(input)
		if err != nil || !bytes.Equal(got, data) {
			t.Fatalf("input changed: %s, %v", got, err)
		}
	}
}

func TestDecisionSummaryPartialRun(t *testing.T) {
	c := decisiontest.Case{Dataset: "a", Expected: map[string]decisiontest.Expected{"one": {}, "two": {}}}
	samples := []decisionSample{
		{Case: c, Duration: 10 * time.Millisecond},
		{Case: c, Duration: 30 * time.Millisecond, Wrong: []string{"two"}},
		{Case: decisiontest.Case{Dataset: "b", Expected: map[string]decisiontest.Expected{"one": {}}}, Err: errors.New("HTTP 500")},
	}
	var out bytes.Buffer
	writeDecisionSummary(&out, "test", samples, 5, time.Second)
	var got struct {
		decisiontest.Tally
		Expected         int                           `json:"expected_requests"`
		Complete         bool                          `json:"complete"`
		QuestionAccuracy float64                       `json:"question_accuracy_pct"`
		P50              float64                       `json:"p50_ms"`
		P95              float64                       `json:"p95_ms"`
		Rate             float64                       `json:"requests_per_second"`
		QuestionsRate    float64                       `json:"questions_per_second"`
		Datasets         map[string]decisiontest.Tally `json:"datasets"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Complete || got.Expected != 5 || got.Requests != 3 || got.Errors != 1 || got.Questions != 5 || got.CorrectQuestions != 3 || got.QuestionAccuracy != 60 || got.P50 != 10 || got.P95 != 30 || got.Rate != 2 || got.QuestionsRate != 4 || got.Datasets["a"].CorrectCases != 1 || got.Datasets["b"].Errors != 1 {
		t.Fatalf("incorrect partial-run summary: %s", out.String())
	}
}

func TestDecisionBenchmarkConcurrency(t *testing.T) {
	var requests atomic.Int32
	both := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/generate":
			writeJSON(w, api.GenerateResponse{Done: true})
		case "/v1/systemone":
			_, _ = io.Copy(io.Discard, r.Body)
			if requests.Add(1) == 2 {
				close(both)
			}
			select {
			case <-both:
			case <-r.Context().Done():
				return
			case <-t.Context().Done():
				return
			}
			fmt.Fprint(w, `{"model":"test-model","usage":{"input_tokens":20,"output_tokens":0},"answers":{"result":{"type":"noul","noul":1}}}`)
		}
	}))
	t.Cleanup(server.Close)
	t.Setenv("OLLAMA_HOST", server.URL)
	path := filepath.Join(t.TempDir(), "cases.jsonl")
	if err := os.WriteFile(path, []byte(`{"id":"one","dataset":"test","request":{"state":"Message.","questions":{"result":{"type":"noul"}}},"expected":{"result":{"noul":true}}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	opts := createTestFlagOptions()
	concurrency, mode := 2, "regression"
	opts.decisionFile, opts.decisionMode, opts.concurrency = &path, &mode, &concurrency
	*opts.epochs, *opts.timeout = 4, 5
	var out bytes.Buffer
	if err := benchmarkDecision(opts, []string{"test-model"}, &out); err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 4 || strings.Count(out.String(), "step=decision/") != 4 {
		t.Fatalf("requests=%d output=%s", requests.Load(), out.String())
	}
}

func TestDecisionBenchmarkCancelsPeers(t *testing.T) {
	var requests atomic.Int32
	started, canceled := make(chan struct{}), make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/generate":
			writeJSON(w, api.GenerateResponse{Done: true})
		case "/v1/systemone":
			_, _ = io.Copy(io.Discard, r.Body)
			if requests.Add(1) == 1 {
				close(started)
				select {
				case <-r.Context().Done():
					close(canceled)
				case <-t.Context().Done():
				}
				return
			}
			<-started
			http.Error(w, "runner failed", http.StatusInternalServerError)
		}
	}))
	t.Cleanup(server.Close)
	t.Setenv("OLLAMA_HOST", server.URL)
	path := filepath.Join(t.TempDir(), "cases.jsonl")
	if err := os.WriteFile(path, []byte(`{"id":"one","dataset":"test","request":{"state":"Message.","questions":{"result":{"type":"noul"}}},"expected":{"result":{"noul":true}}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	opts := createTestFlagOptions()
	concurrency, mode := 2, "score"
	opts.decisionFile, opts.decisionMode, opts.concurrency = &path, &mode, &concurrency
	*opts.epochs, *opts.timeout = 100, 5
	var out bytes.Buffer
	err := benchmarkDecision(opts, []string{"test-model"}, &out)
	if err == nil || !strings.Contains(err.Error(), "runner failed") || requests.Load() != 2 {
		t.Fatalf("error=%v requests=%d, want original failure and only two requests", err, requests.Load())
	}
	select {
	case <-canceled:
	case <-time.After(5 * time.Second):
		t.Fatal("in-flight peer was not canceled")
	}
}

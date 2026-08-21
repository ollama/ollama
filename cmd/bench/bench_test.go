package main

import (
	"bytes"
	"crypto/rand"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func createTestFlagOptions() flagOptions {
	models := "test-model"
	format := "benchstat"
	epochs := 1
	maxTokens := 50
	temperature := 0.7
	seed := 42
	timeout := 30
	prompt := "test prompt"
	imageFile := ""
	keepAlive := 0.0
	verbose := false
	debug := false
	warmup := 0
	promptTokens := 0

	return flagOptions{
		models:       &models,
		format:       &format,
		epochs:       &epochs,
		maxTokens:    &maxTokens,
		temperature:  &temperature,
		seed:         &seed,
		timeout:      &timeout,
		prompt:       &prompt,
		imageFile:    &imageFile,
		keepAlive:    &keepAlive,
		verbose:      &verbose,
		debug:        &debug,
		warmup:       &warmup,
		promptTokens: &promptTokens,
	}
}

func captureOutput(f func()) string {
	oldStdout := os.Stdout
	oldStderr := os.Stderr
	defer func() {
		os.Stdout = oldStdout
		os.Stderr = oldStderr
	}()

	r, w, _ := os.Pipe()
	os.Stdout = w
	os.Stderr = w

	f()

	w.Close()
	var buf bytes.Buffer
	io.Copy(&buf, r)
	return buf.String()
}

type mockServerOptions struct {
	chatResponses []api.ChatResponse
	showResponse  *api.ShowResponse
	psResponse    *api.ProcessResponse
	// onChat, when set, is called with each decoded chat request.
	onChat func(req api.ChatRequest)
	// promptEvalCount, when set, simulates model-specific prompt tokenization.
	promptEvalCount func(req api.ChatRequest) int
}

// mockPromptEvalCount stands in for a real tokenizer well enough for
// calibration tests. It reproduces the two properties calibration relies on:
// a single character costs exactly one token (so pad letters move the count by
// one), and code words cost roughly one token per four characters, which puts
// HumanEval near the 2.0-2.5 tokens/word real tokenizers charge. The trailing
// constant stands in for the chat template.
func mockPromptEvalCount(req api.ChatRequest) int {
	tokens := 0
	for _, m := range req.Messages {
		for field := range strings.FieldsSeq(m.Content) {
			if len(field) == 1 {
				tokens++
				continue
			}
			tokens += 1 + len(field)/4
		}
	}
	return tokens + 25
}

func writeJSON(w http.ResponseWriter, v any) {
	jsonData, err := json.Marshal(v)
	if err != nil {
		return
	}
	w.Write(jsonData)
	w.Write([]byte("\n"))
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

func createMockOllamaServer(t *testing.T, opts mockServerOptions) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		switch r.URL.Path {
		case "/api/chat":
			if r.Method != "POST" {
				t.Errorf("Expected POST method for /api/chat, got %s", r.Method)
				http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
				return
			}

			var req api.ChatRequest
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &req)

			if opts.onChat != nil {
				opts.onChat(req)
			}

			w.WriteHeader(http.StatusOK)
			for _, resp := range opts.chatResponses {
				resp.Model = req.Model
				if resp.Done && opts.promptEvalCount != nil {
					resp.Metrics.PromptEvalCount = opts.promptEvalCount(req)
				}
				writeJSON(w, resp)
				time.Sleep(10 * time.Millisecond)
			}

		case "/api/generate":
			// Only used by unloadModel
			w.WriteHeader(http.StatusOK)
			writeJSON(w, api.GenerateResponse{Done: true})

		case "/api/show":
			if opts.showResponse != nil {
				json.NewEncoder(w).Encode(opts.showResponse)
			} else {
				json.NewEncoder(w).Encode(api.ShowResponse{
					Details: api.ModelDetails{
						ParameterSize:     "4.3B",
						QuantizationLevel: "Q4_K_M",
						Family:            "testfamily",
					},
				})
			}

		case "/api/ps":
			if opts.psResponse != nil {
				json.NewEncoder(w).Encode(opts.psResponse)
			} else {
				json.NewEncoder(w).Encode(api.ProcessResponse{
					Models: []api.ProcessModelResponse{
						{
							Name:     "test-model",
							Model:    "test-model",
							Size:     4080218931, // ~3.80 GB total
							SizeVRAM: 4080218931, // ~3.80 GB on GPU
						},
					},
				})
			}

		default:
			http.Error(w, "Not found", http.StatusNotFound)
		}
	}))
}

func defaultChatResponses() []api.ChatResponse {
	return []api.ChatResponse{
		{
			Model:   "test-model",
			Message: api.Message{Role: "assistant", Content: "test response part 1"},
			Done:    false,
		},
		{
			Model:   "test-model",
			Message: api.Message{Role: "assistant", Content: "test response part 2"},
			Done:    true,
			Metrics: api.Metrics{
				PromptEvalCount:    10,
				PromptEvalDuration: 100 * time.Millisecond,
				EvalCount:          50,
				EvalDuration:       500 * time.Millisecond,
				TotalDuration:      600 * time.Millisecond,
				LoadDuration:       50 * time.Millisecond,
			},
		},
	}
}

func TestBenchmarkModel_Success(t *testing.T) {
	fOpt := createTestFlagOptions()

	server := createMockOllamaServer(t, mockServerOptions{
		chatResponses: defaultChatResponses(),
	})
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	if !strings.Contains(output, "BenchmarkModel/name=test-model/step=prefill") {
		t.Errorf("Expected output to contain prefill metrics, got: %s", output)
	}
	if !strings.Contains(output, "BenchmarkModel/name=test-model/step=generate") {
		t.Errorf("Expected output to contain generate metrics, got: %s", output)
	}
	if !strings.Contains(output, "ns/token") {
		t.Errorf("Expected output to contain ns/token metric, got: %s", output)
	}
	if !strings.Contains(output, "BenchmarkModel/name=test-model/step=ttft") {
		t.Errorf("Expected output to contain ttft metrics, got: %s", output)
	}
}

func TestBenchmarkModel_ServerError(t *testing.T) {
	fOpt := createTestFlagOptions()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "Internal server error", http.StatusInternalServerError)
	}))
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected error to be handled internally, got returned error: %v", err)
		}
	})

	if !strings.Contains(output, "ERROR: Couldn't generate with model") {
		t.Errorf("Expected error message about generate failure, got: %s", output)
	}
}

func TestBenchmarkModel_Timeout(t *testing.T) {
	fOpt := createTestFlagOptions()
	shortTimeout := 1
	fOpt.timeout = &shortTimeout

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" || r.URL.Path == "/api/ps" || r.URL.Path == "/api/generate" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{})
			return
		}
		// Simulate a long delay that will cause timeout
		time.Sleep(2 * time.Second)

		w.Header().Set("Content-Type", "application/json")
		writeJSON(w, api.ChatResponse{
			Model:   "test-model",
			Message: api.Message{Role: "assistant", Content: "test response"},
			Done:    true,
			Metrics: api.Metrics{
				PromptEvalCount:    10,
				PromptEvalDuration: 100 * time.Millisecond,
				EvalCount:          50,
				EvalDuration:       500 * time.Millisecond,
				TotalDuration:      600 * time.Millisecond,
				LoadDuration:       50 * time.Millisecond,
			},
		})
	}))
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected timeout to be handled internally, got returned error: %v", err)
		}
	})

	if !strings.Contains(output, "ERROR: Request timed out") {
		t.Errorf("Expected timeout error message, got: %s", output)
	}
}

func TestBenchmarkModel_NoMetrics(t *testing.T) {
	fOpt := createTestFlagOptions()

	server := createMockOllamaServer(t, mockServerOptions{
		chatResponses: []api.ChatResponse{
			{
				Model:   "test-model",
				Message: api.Message{Role: "assistant", Content: "test response"},
				Done:    false, // Never sends Done=true
			},
		},
	})
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	if !strings.Contains(output, "ERROR: No metrics received") {
		t.Errorf("Expected no metrics error message, got: %s", output)
	}
}

func TestBenchmarkModel_MultipleModels(t *testing.T) {
	fOpt := createTestFlagOptions()
	models := "model1,model2"
	epochs := 2
	fOpt.models = &models
	fOpt.epochs = &epochs

	chatCallCount := 0
	server := createMockOllamaServer(t, mockServerOptions{
		onChat:        func(req api.ChatRequest) { chatCallCount++ },
		chatResponses: defaultChatResponses(),
	})
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	// Should be called 4 times (2 models x 2 epochs)
	if chatCallCount != 4 {
		t.Errorf("Expected 4 API calls, got %d", chatCallCount)
	}

	if !strings.Contains(output, "BenchmarkModel/name=model1") || !strings.Contains(output, "BenchmarkModel/name=model2") {
		t.Errorf("Expected output for both models, got: %s", output)
	}
}

func TestBenchmarkModel_WithImage(t *testing.T) {
	fOpt := createTestFlagOptions()

	tmpfile, err := os.CreateTemp(t.TempDir(), "testimage")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpfile.Name())

	content := []byte("fake image data")
	if _, err := tmpfile.Write(content); err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	tmpfile.Close()

	tmpfileName := tmpfile.Name()
	fOpt.imageFile = &tmpfileName

	server := createMockOllamaServer(t, mockServerOptions{
		onChat: func(req api.ChatRequest) {
			if len(req.Messages) == 0 || len(req.Messages[0].Images) == 0 {
				t.Error("Expected request to contain images")
			}
		},
		chatResponses: defaultChatResponses(),
	})
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	if !strings.Contains(output, "BenchmarkModel/name=test-model") {
		t.Errorf("Expected benchmark output, got: %s", output)
	}
}

func TestBenchmarkModel_ImageError(t *testing.T) {
	randFileName := func() string {
		const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
		const length = 8

		result := make([]byte, length)
		rand.Read(result)

		for i := range result {
			result[i] = charset[result[i]%byte(len(charset))]
		}

		return string(result) + ".txt"
	}

	fOpt := createTestFlagOptions()
	imageFile := randFileName()
	fOpt.imageFile = &imageFile

	output := captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err == nil {
			t.Error("Expected error from image reading, got nil")
		}
	})

	if !strings.Contains(output, "ERROR: Couldn't read image") {
		t.Errorf("Expected image read error message, got: %s", output)
	}
}

func TestReadImage_Success(t *testing.T) {
	tmpfile, err := os.CreateTemp(t.TempDir(), "testimage")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpfile.Name())

	content := []byte("fake image data for testing")
	if _, err := tmpfile.Write(content); err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	tmpfile.Close()

	imgData, err := readImage(tmpfile.Name())
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if len(imgData) != len(content) {
		t.Errorf("Expected image data length %d, got %d", len(content), len(imgData))
	}
}

func TestReadImage_FileNotFound(t *testing.T) {
	_, err := readImage("/nonexistent/path/to/image.png")
	if err == nil {
		t.Error("Expected error for non-existent file, got nil")
	}
}

func TestOptionsMapCreation(t *testing.T) {
	fOpt := createTestFlagOptions()

	options := benchmarkOptions(fOpt)

	if options["num_predict"] != *fOpt.maxTokens {
		t.Errorf("Expected num_predict %d, got %v", *fOpt.maxTokens, options["num_predict"])
	}
	if options["temperature"] != *fOpt.temperature {
		t.Errorf("Expected temperature %f, got %v", *fOpt.temperature, options["temperature"])
	}
	if options["seed"] != *fOpt.seed {
		t.Errorf("Expected seed %d, got %v", *fOpt.seed, options["seed"])
	}
}

// --- Feature tests ---

func TestBenchmarkModel_Warmup(t *testing.T) {
	fOpt := createTestFlagOptions()
	warmup := 2
	fOpt.warmup = &warmup
	debug := true
	fOpt.debug = &debug

	chatCallCount := 0
	server := createMockOllamaServer(t, mockServerOptions{
		onChat:        func(req api.ChatRequest) { chatCallCount++ },
		chatResponses: defaultChatResponses(),
	})
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	// 2 warmup + 1 epoch = 3 total chat calls
	if chatCallCount != 3 {
		t.Errorf("Expected 3 chat calls (2 warmup + 1 epoch), got %d", chatCallCount)
	}

	if !strings.Contains(output, "Warmup 1/2 for test-model complete") {
		t.Errorf("Expected warmup debug output, got: %s", output)
	}
	if !strings.Contains(output, "Warmup 2/2 for test-model complete") {
		t.Errorf("Expected warmup debug output for 2/2, got: %s", output)
	}
}

func TestBenchmarkModel_TTFT(t *testing.T) {
	fOpt := createTestFlagOptions()

	server := createMockOllamaServer(t, mockServerOptions{
		chatResponses: defaultChatResponses(),
	})
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	if !strings.Contains(output, "step=ttft") {
		t.Errorf("Expected TTFT metric in output, got: %s", output)
	}
}

func TestBenchmarkModel_ModelInfo(t *testing.T) {
	fOpt := createTestFlagOptions()

	server := createMockOllamaServer(t, mockServerOptions{
		chatResponses: defaultChatResponses(),
		showResponse: &api.ShowResponse{
			Details: api.ModelDetails{
				ParameterSize:     "4.3B",
				QuantizationLevel: "Q4_K_M",
				Family:            "gemma3",
			},
		},
	})
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	if !strings.Contains(output, "Params: 4.3B") {
		t.Errorf("Expected model info with parameter size, got: %s", output)
	}
	if !strings.Contains(output, "Quant: Q4_K_M") {
		t.Errorf("Expected model info with quant level, got: %s", output)
	}
	if !strings.Contains(output, "Family: gemma3") {
		t.Errorf("Expected model info with family, got: %s", output)
	}
}

func TestBenchmarkModel_VRAM(t *testing.T) {
	fOpt := createTestFlagOptions()

	server := createMockOllamaServer(t, mockServerOptions{
		chatResponses: defaultChatResponses(),
		psResponse: &api.ProcessResponse{
			Models: []api.ProcessModelResponse{
				{
					Name:     "test-model",
					Model:    "test-model",
					Size:     4080218931,
					SizeVRAM: 4080218931,
				},
			},
		},
	})
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	// VRAM should appear in model info header
	if !strings.Contains(output, "VRAM: 4080218931") {
		t.Errorf("Expected VRAM in model info header, got: %s", output)
	}
}

func TestBenchmarkModel_PromptTokens(t *testing.T) {
	fOpt := createTestFlagOptions()
	promptTokens := 1000
	fOpt.promptTokens = &promptTokens

	var receivedContents []string
	server := createMockOllamaServer(t, mockServerOptions{
		onChat: func(req api.ChatRequest) {
			if len(req.Messages) > 0 {
				receivedContents = append(receivedContents, req.Messages[0].Content)
			}
		},
		chatResponses:   defaultChatResponses(),
		promptEvalCount: mockPromptEvalCount,
	})
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	if len(receivedContents) == 0 {
		t.Fatal("Expected at least one chat request")
	}
	content := receivedContents[len(receivedContents)-1]

	if !strings.HasPrefix(content, "# -*- coding: utf-8 -*-") {
		t.Errorf("Expected generated code prompt, got: %.80s...", content)
	}
	if strings.Contains(content, "test prompt") {
		t.Error("Expected generated prompt when promptTokens is set")
	}
	if !strings.Contains(content, "def ") {
		t.Error("Expected HumanEval function signatures in generated prompt")
	}
	if got := mockPromptEvalCount(api.ChatRequest{Messages: []api.Message{{Content: content}}}); got != promptTokens {
		t.Errorf("timed request measured %d prompt tokens, want exactly %d", got, promptTokens)
	}
}

// TestBenchmarkModel_PromptTokensExact pins the contract -prompt-tokens now
// carries: the target is a ceiling that every benchmark request lands on
// exactly, so a run can be placed on a kernel tile boundary instead of drifting
// a few percent past one.
func TestBenchmarkModel_PromptTokensExact(t *testing.T) {
	for _, target := range []int{256, 512, 1024, 4096, 4097} {
		t.Run(strconv.Itoa(target), func(t *testing.T) {
			fOpt := createTestFlagOptions()
			epochs, warmups := 3, 1
			fOpt.epochs = &epochs
			fOpt.warmup = &warmups
			fOpt.promptTokens = &target

			var counts []int
			server := createMockOllamaServer(t, mockServerOptions{
				onChat: func(req api.ChatRequest) {
					counts = append(counts, mockPromptEvalCount(req))
				},
				chatResponses:   defaultChatResponses(),
				promptEvalCount: mockPromptEvalCount,
			})
			defer server.Close()

			t.Setenv("OLLAMA_HOST", server.URL)

			output := captureOutput(func() {
				if err := BenchmarkModel(fOpt); err != nil {
					t.Fatalf("BenchmarkModel() error = %v", err)
				}
			})

			// Calibration probes are free to overshoot; the warmup and timed
			// requests that follow are not.
			benchmarked := warmups + epochs
			if len(counts) < benchmarked {
				t.Fatalf("got %d chat requests, want at least %d", len(counts), benchmarked)
			}
			for i, count := range counts[len(counts)-benchmarked:] {
				if count != target {
					t.Errorf("benchmark request %d measured %d prompt tokens, want exactly %d", i, count, target)
				}
			}
			if strings.Contains(output, "prompt size other than the requested") {
				t.Errorf("unexpected off-target warning: %s", output)
			}
		})
	}
}

// TestBenchmarkModel_PromptSizeDriftWarning covers the safety net: the plan is
// calibrated once, so a size that moves afterwards must be reported rather than
// silently benchmarked.
func TestBenchmarkModel_PromptSizeDriftWarning(t *testing.T) {
	fOpt := createTestFlagOptions()
	epochs, warmups, promptTokens := 2, 0, 1024
	fOpt.epochs = &epochs
	fOpt.warmup = &warmups
	fOpt.promptTokens = &promptTokens

	server := createMockOllamaServer(t, mockServerOptions{
		chatResponses: defaultChatResponses(),
		promptEvalCount: func(req api.ChatRequest) int {
			count := mockPromptEvalCount(req)
			// Calibration probes ask for a single token; drift the count only
			// for the benchmark requests that follow them.
			if predict, ok := req.Options["num_predict"].(float64); !ok || predict != 1 {
				count++
			}
			return count
		},
	})
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		if err := BenchmarkModel(fOpt); err != nil {
			t.Fatalf("BenchmarkModel() error = %v", err)
		}
	})

	if !strings.Contains(output, "prompt size other than the requested 1024 tokens") {
		t.Errorf("Expected off-target warning, got: %s", output)
	}
}

func TestBenchmarkModel_GeneratedPromptVariesByRequest(t *testing.T) {
	fOpt := createTestFlagOptions()
	epochs := 3
	warmups := 1
	promptTokens := 1000
	fOpt.epochs = &epochs
	fOpt.warmup = &warmups
	fOpt.promptTokens = &promptTokens

	var receivedContents []string
	server := createMockOllamaServer(t, mockServerOptions{
		onChat: func(req api.ChatRequest) {
			if len(req.Messages) > 0 {
				receivedContents = append(receivedContents, req.Messages[0].Content)
			}
		},
		chatResponses:   defaultChatResponses(),
		promptEvalCount: mockPromptEvalCount,
	})
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	captureOutput(func() {
		if err := BenchmarkModel(fOpt); err != nil {
			t.Fatalf("BenchmarkModel() error = %v", err)
		}
	})

	requestCount := warmups + epochs
	if len(receivedContents) < requestCount {
		t.Fatalf("got %d chat requests, want at least %d", len(receivedContents), requestCount)
	}

	seen := make(map[string]bool)
	for i, content := range receivedContents[len(receivedContents)-requestCount:] {
		_, body, ok := strings.Cut(content, "\n\n\n")
		if !ok {
			t.Fatalf("request %d has no generated prompt body", i)
		}
		if seen[body] {
			t.Errorf("request %d reused a HumanEval window", i)
		}
		seen[body] = true
	}
}

func TestBenchmarkModel_PromptCalibrationFailure(t *testing.T) {
	fOpt := createTestFlagOptions()
	promptTokens := 2000
	fOpt.promptTokens = &promptTokens

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			json.NewEncoder(w).Encode(api.ShowResponse{})
		case "/api/chat":
			http.Error(w, "calibration failed", http.StatusInternalServerError)
		default:
			http.Error(w, "not found", http.StatusNotFound)
		}
	}))
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err == nil {
			t.Error("Expected error when calibration chat fails")
		}
	})

	if !strings.Contains(output, "cannot measure prompt tokens") {
		t.Errorf("Expected prompt measurement error, got: %s", output)
	}
}

func TestBenchmarkModel_PromptBelowMinimum(t *testing.T) {
	fOpt := createTestFlagOptions()
	promptTokens := 2
	fOpt.promptTokens = &promptTokens

	server := createMockOllamaServer(t, mockServerOptions{
		chatResponses: defaultChatResponses(),
	})
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err == nil {
			t.Error("Expected error for prompt target below minimum")
		}
	})

	if !strings.Contains(output, "below the minimum") {
		t.Errorf("Expected minimum-prompt error, got: %s", output)
	}
}

func TestBenchmarkModel_PromptAboveMaximum(t *testing.T) {
	fOpt := createTestFlagOptions()
	promptTokens := 50000
	fOpt.promptTokens = &promptTokens

	server := createMockOllamaServer(t, mockServerOptions{
		chatResponses:   defaultChatResponses(),
		promptEvalCount: mockPromptEvalCount,
	})
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected run to proceed with full-set prompt, got %v", err)
		}
	})

	if !strings.Contains(output, "exceeds the problem set") {
		t.Errorf("Expected over-maximum warning, got: %s", output)
	}
}

func TestBenchmarkModel_ChatTransport(t *testing.T) {
	fOpt := createTestFlagOptions()

	chatCalls := 0
	server := createMockOllamaServer(t, mockServerOptions{
		onChat: func(req api.ChatRequest) {
			chatCalls++
			if len(req.Messages) != 1 || req.Messages[0].Role != "user" {
				t.Errorf("Expected a single user message, got %+v", req.Messages)
			}
		},
		chatResponses: defaultChatResponses(),
	})
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	if chatCalls == 0 {
		t.Error("Expected benchmark traffic on /api/chat")
	}
}

func TestBenchmarkModel_PromptUniquePerRequest(t *testing.T) {
	fOpt := createTestFlagOptions()
	epochs := 3
	fOpt.epochs = &epochs

	var receivedContents []string
	server := createMockOllamaServer(t, mockServerOptions{
		onChat: func(req api.ChatRequest) {
			if len(req.Messages) > 0 {
				receivedContents = append(receivedContents, req.Messages[0].Content)
			}
		},
		chatResponses: defaultChatResponses(),
	})
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	if len(receivedContents) != 3 {
		t.Fatalf("Expected 3 requests, got %d", len(receivedContents))
	}

	// Every request must carry a unique nonce so prefix caches cannot serve timed epochs
	for i := range receivedContents {
		if !strings.HasPrefix(receivedContents[i], "# -*- coding: utf-8 -*-") {
			t.Errorf("Expected nonce prefix on request %d, got: %.50s...", i, receivedContents[i])
		}
		for j := i + 1; j < len(receivedContents); j++ {
			if receivedContents[i] == receivedContents[j] {
				t.Errorf("Expected unique prompts for requests %d and %d", i, j)
			}
		}
		// ...while the measured workload stays identical
		if !strings.HasSuffix(receivedContents[i], "test prompt") {
			t.Errorf("Expected identical workload body across requests, got: %.50s", receivedContents[i])
		}
	}
}

func TestBenchmarkModel_ShortResponseRetry(t *testing.T) {
	fOpt := createTestFlagOptions()
	maxTokens := 100
	fOpt.maxTokens = &maxTokens

	chatCallCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		switch r.URL.Path {
		case "/api/chat":
			var req api.ChatRequest
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &req)

			chatCallCount++

			// First 3 attempts return short responses, 4th returns full
			evalCount := 20
			if chatCallCount == 4 {
				evalCount = 100
			}

			writeJSON(w, api.ChatResponse{
				Model:   "test-model",
				Message: api.Message{Role: "assistant", Content: "response"},
				Done:    true,
				Metrics: api.Metrics{
					PromptEvalCount:    10,
					PromptEvalDuration: 100 * time.Millisecond,
					EvalCount:          evalCount,
					EvalDuration:       500 * time.Millisecond,
					TotalDuration:      600 * time.Millisecond,
					LoadDuration:       50 * time.Millisecond,
				},
			})

		case "/api/generate":
			writeJSON(w, api.GenerateResponse{Done: true})
		case "/api/show":
			json.NewEncoder(w).Encode(api.ShowResponse{})
		case "/api/ps":
			json.NewEncoder(w).Encode(api.ProcessResponse{})
		}
	}))
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	// 1 epoch: 3 short retries + 1 successful = 4 chat calls
	if chatCallCount != 4 {
		t.Errorf("Expected 4 chat calls (3 retries + 1 success), got %d", chatCallCount)
	}
}

func TestBenchmarkModel_ShortResponseWarning(t *testing.T) {
	fOpt := createTestFlagOptions()
	maxTokens := 100
	fOpt.maxTokens = &maxTokens

	// Always return short responses to trigger the warning
	server := createMockOllamaServer(t, mockServerOptions{
		chatResponses: []api.ChatResponse{
			{
				Model:   "test-model",
				Message: api.Message{Role: "assistant", Content: "response"},
				Done:    true,
				Metrics: api.Metrics{
					PromptEvalCount:    10,
					PromptEvalDuration: 100 * time.Millisecond,
					EvalCount:          20, // Always short
					EvalDuration:       500 * time.Millisecond,
					TotalDuration:      600 * time.Millisecond,
					LoadDuration:       50 * time.Millisecond,
				},
			},
		},
	})
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	// Should still produce metrics (uses best attempt)
	if !strings.Contains(output, "BenchmarkModel/name=test-model") {
		t.Errorf("Expected benchmark output even with short responses, got: %s", output)
	}

	// Should warn about short responses
	if !strings.Contains(output, "WARNING") || !strings.Contains(output, "short responses") {
		t.Errorf("Expected warning about short responses, got: %s", output)
	}
}

func TestBenchmarkModel_NoRetryWhenMaxTokensZero(t *testing.T) {
	fOpt := createTestFlagOptions()
	maxTokens := 0
	fOpt.maxTokens = &maxTokens

	chatCallCount := 0
	server := createMockOllamaServer(t, mockServerOptions{
		onChat: func(req api.ChatRequest) { chatCallCount++ },
		chatResponses: []api.ChatResponse{
			{
				Model:   "test-model",
				Message: api.Message{Role: "assistant", Content: "response"},
				Done:    true,
				Metrics: api.Metrics{
					PromptEvalCount:    10,
					PromptEvalDuration: 100 * time.Millisecond,
					EvalCount:          5, // Very short, but maxTokens=0 so no retry
					EvalDuration:       500 * time.Millisecond,
					TotalDuration:      600 * time.Millisecond,
					LoadDuration:       50 * time.Millisecond,
				},
			},
		},
	})
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	// With maxTokens=0, no retries should happen: exactly 1 call for 1 epoch
	if chatCallCount != 1 {
		t.Errorf("Expected 1 chat call (no retries when maxTokens=0), got %d", chatCallCount)
	}
}

func TestBenchmarkModel_CSVFormat(t *testing.T) {
	fOpt := createTestFlagOptions()
	format := "csv"
	fOpt.format = &format

	server := createMockOllamaServer(t, mockServerOptions{
		chatResponses: defaultChatResponses(),
	})
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	if !strings.Contains(output, "NAME,STEP,COUNT,NS_PER_COUNT,TOKEN_PER_SEC") {
		t.Errorf("Expected CSV header, got: %s", output)
	}
	if !strings.Contains(output, "test-model,prefill,") {
		t.Errorf("Expected CSV prefill row, got: %s", output)
	}
	if !strings.Contains(output, "test-model,ttft,") {
		t.Errorf("Expected CSV ttft row, got: %s", output)
	}
}

// --- Unit tests for helper functions ---

func TestGenerateCodePrompt(t *testing.T) {
	prompt := generateCodePrompt(promptPlan{words: 800}, 0)

	if !strings.HasPrefix(prompt, "# -*- coding: utf-8 -*-\n# checksum: ") {
		t.Errorf("Expected session prefix, got: %.50s...", prompt)
	}
	if !strings.Contains(prompt, "def ") {
		t.Error("Expected HumanEval function signatures in prompt")
	}

	// The word budget covers the problems alone, and repeated best-fit packs to
	// within the smallest problem of it.
	_, body, _ := strings.Cut(prompt, "\n\n\n")
	if got, floor := len(strings.Fields(body)), 800-smallestCodePromptWords(); got > 800 || got <= floor {
		t.Errorf("body packed to %d words, want (%d, 800]", got, floor)
	}
}

func TestPromptNonce(t *testing.T) {
	for _, n := range []int{1, 12, 40} {
		fields := strings.Fields(promptNonce(n))
		if len(fields) != n {
			t.Errorf("promptNonce(%d) produced %d fields, want %d", n, len(fields), n)
		}
		// Single lowercase letters are what cost exactly one token each; any
		// wider unit would make the pad unable to hit the target.
		for _, field := range fields {
			if len(field) != 1 || field[0] < 'a' || field[0] > 'z' {
				t.Errorf("promptNonce(%d) field %q is not a single lowercase letter", n, field)
			}
		}
	}
	first, second := promptNonce(12), promptNonce(12)
	if first == second {
		t.Errorf("Expected a fresh nonce per call to defeat prefix caches, got %q twice", first)
	}
}

func TestGenerateCodePrompt_PadOnlyExtendsHeader(t *testing.T) {
	baseHeader, baseBody, _ := strings.Cut(generateCodePrompt(promptPlan{words: 800}, 0), "\n\n\n")
	padHeader, padBody, _ := strings.Cut(generateCodePrompt(promptPlan{words: 800, pad: 7}, 0), "\n\n\n")

	if baseBody != padBody {
		t.Error("pad changed the coding request; it must only extend the header nonce")
	}
	if got := len(strings.Fields(padHeader)) - len(strings.Fields(baseHeader)); got != 7 {
		t.Errorf("pad added %d header words, want 7", got)
	}
}

func TestGenerateCodePrompt_WholeProblemsOnly(t *testing.T) {
	problems := codePromptProblems(800, 0)
	if len(problems) == 0 {
		t.Fatal("expected at least one window problem")
	}
	seen := make(map[string]bool)
	for _, problem := range problems {
		if seen[problem.TaskID] {
			t.Fatal("prompt repeats a HumanEval problem")
		}
		seen[problem.TaskID] = true
	}
}

func TestGenerateCodePrompt_FullSetCap(t *testing.T) {
	problems := codePromptProblems(1<<20, 0)
	if len(problems) != len(humanEvalProblems()) {
		t.Fatalf("full-set prompt contains %d problems, want %d", len(problems), len(humanEvalProblems()))
	}
	seen := make(map[string]bool)
	for _, problem := range problems {
		if seen[problem.TaskID] {
			t.Fatal("full-set prompt repeats a HumanEval problem")
		}
		seen[problem.TaskID] = true
	}
}

func TestGenerateCodePrompt_TinyTarget(t *testing.T) {
	prompt := generateCodePrompt(promptPlan{words: 10}, 0)
	if !strings.HasPrefix(prompt, "# -*- coding: utf-8 -*-\n# checksum: ") {
		t.Errorf("Expected session prefix even for tiny targets, got: %.50s...", prompt)
	}
	if strings.Contains(prompt, "def ") {
		t.Error("Expected header-only prompt when no problem fits a tiny budget")
	}
}

func TestCodePromptBody_Deterministic(t *testing.T) {
	first, second := codePromptBody(800, 0), codePromptBody(800, 0)
	if first != second {
		t.Error("Expected identical bodies for identical inputs")
	}
}

func TestCodePromptBody_VariationsPreserveProblemSet(t *testing.T) {
	reference := codePromptBody(800, 0)
	want := codePromptProblems(800, 0)
	wantSet := make(map[string]bool)
	for _, problem := range want {
		wantSet[problem.TaskID] = true
	}
	seenPrompts := make(map[string]bool)
	for i := range 6 {
		prompt := codePromptBody(800, i)
		if seenPrompts[prompt] {
			t.Errorf("variation %d reused a prompt", i)
		}
		seenPrompts[prompt] = true

		problems := codePromptProblems(800, i)
		if len(problems) != len(wantSet) {
			t.Errorf("variation %d contains %d problems, want %d", i, len(problems), len(wantSet))
		}
		seen := make(map[string]bool)
		for _, problem := range problems {
			if !wantSet[problem.TaskID] {
				t.Errorf("variation %d changed the calibrated problem set", i)
			}
			if seen[problem.TaskID] {
				t.Errorf("variation %d repeats problem %s", i, problem.TaskID)
			}
			seen[problem.TaskID] = true
		}
		if got, want := len(strings.Fields(prompt)), len(strings.Fields(reference)); got != want {
			t.Errorf("variation %d contains %d words, want %d", i, got, want)
		}
	}
}

func TestBenchmarkPromptVariation(t *testing.T) {
	const (
		warmups = 2
		epochs  = 3
	)
	seen := make(map[int]bool)
	for epoch := range epochs {
		for attempt := range maxShortResponseRetries + 1 {
			variation := benchmarkPromptVariation(warmups, epochs, epoch, attempt)
			if variation < warmups {
				t.Errorf("benchmarkPromptVariation(%d, %d, %d, %d) = %d, overlaps warmups", warmups, epochs, epoch, attempt, variation)
			}
			if seen[variation] {
				t.Errorf("benchmarkPromptVariation(%d, %d, %d, %d) = %d, already used", warmups, epochs, epoch, attempt, variation)
			}
			seen[variation] = true
		}
	}
}

func TestGenerateCodePrompt_UniquePerRequest(t *testing.T) {
	plan := promptPlan{words: 800}
	first, second := generateCodePrompt(plan, 0), generateCodePrompt(plan, 0)
	if first == second {
		t.Error("Expected a fresh nonce per request to defeat prefix caches")
	}
}

func TestBuildChatRequest(t *testing.T) {
	fOpt := createTestFlagOptions()
	req := buildChatRequest("test-model", fOpt, nil, 0, promptPlan{})

	if req.Model != "test-model" {
		t.Errorf("Expected model 'test-model', got '%s'", req.Model)
	}
	if len(req.Messages) != 1 || req.Messages[0].Role != "user" {
		t.Fatalf("Expected single user message, got %+v", req.Messages)
	}
	if !strings.Contains(req.Messages[0].Content, "test prompt") {
		t.Errorf("Expected message to contain 'test prompt', got '%s'", req.Messages[0].Content)
	}
	if !strings.HasPrefix(req.Messages[0].Content, "# -*- coding: utf-8 -*-\n# checksum: ") {
		t.Errorf("Expected nonce prefix, got: %.50s...", req.Messages[0].Content)
	}
}

func TestBuildChatRequest_WithPromptTokens(t *testing.T) {
	fOpt := createTestFlagOptions()
	promptTokens := 2000
	fOpt.promptTokens = &promptTokens

	req := buildChatRequest("test-model", fOpt, nil, 0, promptPlan{words: 1500})
	content := req.Messages[0].Content

	if strings.Contains(content, "test prompt") {
		t.Error("Expected generated prompt when promptTokens is set")
	}
	if !strings.HasPrefix(content, "# -*- coding: utf-8 -*-") {
		t.Errorf("Expected code prompt, got: %.50s...", content)
	}
}

func TestBuildChatRequest_WithImage(t *testing.T) {
	fOpt := createTestFlagOptions()
	imgData := api.ImageData([]byte("fake image"))

	req := buildChatRequest("test-model", fOpt, imgData, 0, promptPlan{})
	if len(req.Messages[0].Images) != 1 {
		t.Errorf("Expected 1 image, got %d", len(req.Messages[0].Images))
	}
}

func TestBuildChatRequest_VariesByAttempt(t *testing.T) {
	fOpt := createTestFlagOptions()
	promptTokens := 2000
	fOpt.promptTokens = &promptTokens

	plan := promptPlan{words: 1500}
	_, body0, _ := strings.Cut(buildChatRequest("test-model", fOpt, nil, 0, plan).Messages[0].Content, "\n\n\n")
	_, body1, _ := strings.Cut(buildChatRequest("test-model", fOpt, nil, 1, plan).Messages[0].Content, "\n\n\n")

	if body0 == body1 {
		t.Error("Expected different prompts for different attempts")
	}
}

func TestOutputMetrics_Benchstat(t *testing.T) {
	var buf bytes.Buffer
	metrics := []Metrics{
		{Model: "m1", Step: "prefill", Count: 10, Duration: 100 * time.Millisecond},
		{Model: "m1", Step: "generate", Count: 50, Duration: 500 * time.Millisecond},
		{Model: "m1", Step: "ttft", Count: 1, Duration: 50 * time.Millisecond},
		{Model: "m1", Step: "load", Count: 1, Duration: 50 * time.Millisecond},
		{Model: "m1", Step: "total", Count: 1, Duration: 600 * time.Millisecond},
	}

	OutputMetrics(&buf, "benchstat", metrics, false)
	output := buf.String()

	if !strings.Contains(output, "step=prefill") {
		t.Errorf("Expected prefill in output, got: %s", output)
	}
	if !strings.Contains(output, "token/sec") {
		t.Errorf("Expected token/sec in output, got: %s", output)
	}
	if !strings.Contains(output, "ns/token") {
		t.Errorf("Expected ns/token in output, got: %s", output)
	}
}

func TestOutputMetrics_BenchstatFormat(t *testing.T) {
	var buf bytes.Buffer
	metrics := []Metrics{
		{Model: "m1", Step: "prefill", Count: 10, Duration: 100 * time.Millisecond},
	}

	OutputMetrics(&buf, "benchstat", metrics, false)
	output := buf.String()

	// Verify benchstat format: BenchmarkModel/name=m1/step=prefill 1 <ns/token> ns/token <tokens/sec> token/sec
	if !strings.Contains(output, "BenchmarkModel/name=m1/step=prefill 1") {
		t.Errorf("Expected benchstat format, got: %s", output)
	}
	if !strings.Contains(output, "10000000.00 ns/token") {
		t.Errorf("Expected correct ns/token value, got: %s", output)
	}
	if !strings.Contains(output, "100.00 token/sec") {
		t.Errorf("Expected correct token/sec value, got: %s", output)
	}
}

func TestOutputModelInfo(t *testing.T) {
	var buf bytes.Buffer
	info := ModelInfo{
		Name:              "test-model",
		ParameterSize:     "7B",
		QuantizationLevel: "Q4_0",
		Family:            "llama",
		SizeBytes:         4000000000,
		VRAMBytes:         3800000000,
		NumCtx:            8192,
	}

	outputModelInfo(&buf, "benchstat", info)
	output := buf.String()

	if !strings.Contains(output, "test-model") {
		t.Errorf("Expected model name in output, got: %s", output)
	}
	if !strings.Contains(output, "7B") {
		t.Errorf("Expected parameter size in output, got: %s", output)
	}
	if !strings.Contains(output, "Size: 4000000000") {
		t.Errorf("Expected memory size in output, got: %s", output)
	}
	if !strings.Contains(output, "VRAM: 3800000000") {
		t.Errorf("Expected VRAM in output, got: %s", output)
	}
	if !strings.Contains(output, "NumCtx: 8192") {
		t.Errorf("Expected context length in output, got: %s", output)
	}
}

func TestOutputModelInfo_Unknown(t *testing.T) {
	var buf bytes.Buffer
	info := ModelInfo{Name: "test-model"}

	outputModelInfo(&buf, "benchstat", info)
	output := buf.String()

	if !strings.Contains(output, "unknown") {
		t.Errorf("Expected 'unknown' for missing fields, got: %s", output)
	}
}

func TestFetchMemoryUsage_PrefixMatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/api/ps" {
			json.NewEncoder(w).Encode(api.ProcessResponse{
				Models: []api.ProcessModelResponse{
					{
						Name:     "gemma3:27b-it-qat",
						Model:    "gemma3:27b-it-qat",
						Size:     21634043438,
						SizeVRAM: 21634043438,
					},
				},
			})
		}
	}))
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	client, err := api.ClientFromEnvironment()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	size, vram := fetchMemoryUsage(t.Context(), client, "gemma3:27b")

	if size != 21634043438 {
		t.Errorf("Expected size 21634043438, got %d", size)
	}
	if vram != 21634043438 {
		t.Errorf("Expected VRAM 21634043438, got %d", vram)
	}
}

func TestFetchMemoryUsage_CPUSpill(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/api/ps" {
			json.NewEncoder(w).Encode(api.ProcessResponse{
				Models: []api.ProcessModelResponse{
					{
						Name:     "qwen3-coder:30b-a3b-q4_K_M",
						Model:    "qwen3-coder:30b-a3b-q4_K_M",
						Size:     22000000000,
						SizeVRAM: 18000000000, // 4GB spilled to CPU
					},
				},
			})
		}
	}))
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	client, err := api.ClientFromEnvironment()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	size, vram := fetchMemoryUsage(t.Context(), client, "qwen3-coder:30b-a3b-q4_K_M")

	if size != 22000000000 {
		t.Errorf("Expected total size 22000000000, got %d", size)
	}
	if vram != 18000000000 {
		t.Errorf("Expected VRAM 18000000000, got %d", vram)
	}
}

func TestOutputFormatHeader(t *testing.T) {
	tests := []struct {
		format   string
		verbose  bool
		contains []string
	}{
		{"csv", false, []string{"NAME,STEP,COUNT,NS_PER_COUNT,TOKEN_PER_SEC"}},
		{"benchstat", true, []string{"goos:", "goarch:"}},
		{"benchstat", false, []string{}},
	}

	for _, tt := range tests {
		t.Run(tt.format, func(t *testing.T) {
			var buf bytes.Buffer
			outputFormatHeader(&buf, tt.format, tt.verbose)
			output := buf.String()

			for _, expected := range tt.contains {
				if !strings.Contains(output, expected) {
					t.Errorf("Expected output to contain %q, got: %s", expected, output)
				}
			}
		})
	}
}

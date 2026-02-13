package main

import (
	"bytes"
	"crypto/rand"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func createTestFlagOptions() flagOptions {
	models := "test-model"
	format := "benchstat"
	epochs := 1
	maxTokens := 100
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
	generateResponses []api.GenerateResponse
	showResponse      *api.ShowResponse
	psResponse        *api.ProcessResponse
}

func createMockOllamaServer(t *testing.T, opts mockServerOptions) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		switch r.URL.Path {
		case "/api/generate":
			if r.Method != "POST" {
				t.Errorf("Expected POST method for /api/generate, got %s", r.Method)
				http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
				return
			}

			w.WriteHeader(http.StatusOK)
			for _, resp := range opts.generateResponses {
				jsonData, err := json.Marshal(resp)
				if err != nil {
					t.Errorf("Failed to marshal response: %v", err)
					return
				}
				w.Write(jsonData)
				w.Write([]byte("\n"))
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
				time.Sleep(10 * time.Millisecond)
			}

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

func defaultGenerateResponses() []api.GenerateResponse {
	return []api.GenerateResponse{
		{
			Model:    "test-model",
			Response: "test response part 1",
			Done:     false,
		},
		{
			Model:    "test-model",
			Response: "test response part 2",
			Done:     true,
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
		generateResponses: defaultGenerateResponses(),
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
		if r.URL.Path == "/api/show" || r.URL.Path == "/api/ps" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{})
			return
		}
		// Simulate a long delay that will cause timeout
		time.Sleep(2 * time.Second)

		w.Header().Set("Content-Type", "application/json")
		response := api.GenerateResponse{
			Model:    "test-model",
			Response: "test response",
			Done:     true,
			Metrics: api.Metrics{
				PromptEvalCount:    10,
				PromptEvalDuration: 100 * time.Millisecond,
				EvalCount:          50,
				EvalDuration:       500 * time.Millisecond,
				TotalDuration:      600 * time.Millisecond,
				LoadDuration:       50 * time.Millisecond,
			},
		}
		jsonData, _ := json.Marshal(response)
		w.Write(jsonData)
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
		generateResponses: []api.GenerateResponse{
			{
				Model:    "test-model",
				Response: "test response",
				Done:     false, // Never sends Done=true
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

	generateCallCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		switch r.URL.Path {
		case "/api/generate":
			generateCallCount++
			var req api.GenerateRequest
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &req)

			response := api.GenerateResponse{
				Model:    req.Model,
				Response: "test response for " + req.Model,
				Done:     true,
				Metrics: api.Metrics{
					PromptEvalCount:    10,
					PromptEvalDuration: 100 * time.Millisecond,
					EvalCount:          50,
					EvalDuration:       500 * time.Millisecond,
					TotalDuration:      600 * time.Millisecond,
					LoadDuration:       50 * time.Millisecond,
				},
			}
			jsonData, _ := json.Marshal(response)
			w.Write(jsonData)

		case "/api/show":
			json.NewEncoder(w).Encode(api.ShowResponse{
				Details: api.ModelDetails{
					ParameterSize:     "7B",
					QuantizationLevel: "Q4_0",
					Family:            "llama",
				},
			})

		case "/api/ps":
			json.NewEncoder(w).Encode(api.ProcessResponse{})

		default:
			http.Error(w, "Not found", http.StatusNotFound)
		}
	}))
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	// Should be called 4 times (2 models x 2 epochs)
	if generateCallCount != 4 {
		t.Errorf("Expected 4 API calls, got %d", generateCallCount)
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

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		switch r.URL.Path {
		case "/api/generate":
			var req api.GenerateRequest
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &req)

			if len(req.Images) == 0 {
				t.Error("Expected request to contain images")
			}

			response := api.GenerateResponse{
				Model:    "test-model",
				Response: "test response with image",
				Done:     true,
				Metrics: api.Metrics{
					PromptEvalCount:    10,
					PromptEvalDuration: 100 * time.Millisecond,
					EvalCount:          50,
					EvalDuration:       500 * time.Millisecond,
					TotalDuration:      600 * time.Millisecond,
					LoadDuration:       50 * time.Millisecond,
				},
			}
			jsonData, _ := json.Marshal(response)
			w.Write(jsonData)

		case "/api/show":
			json.NewEncoder(w).Encode(api.ShowResponse{})

		case "/api/ps":
			json.NewEncoder(w).Encode(api.ProcessResponse{})

		default:
			http.Error(w, "Not found", http.StatusNotFound)
		}
	}))
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

	content := []byte("fake image data")
	if _, err := tmpfile.Write(content); err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	tmpfile.Close()

	imgData, err := readImage(tmpfile.Name())
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	if imgData == nil {
		t.Error("Expected image data, got nil")
	}

	expected := api.ImageData(content)
	if string(imgData) != string(expected) {
		t.Errorf("Expected image data %v, got %v", expected, imgData)
	}
}

func TestReadImage_FileNotFound(t *testing.T) {
	imgData, err := readImage("nonexistentfile.jpg")
	if err == nil {
		t.Error("Expected error for non-existent file, got nil")
	}
	if imgData != nil {
		t.Error("Expected nil image data for non-existent file")
	}
}

func TestOptionsMapCreation(t *testing.T) {
	fOpt := createTestFlagOptions()

	options := make(map[string]interface{})
	if *fOpt.maxTokens > 0 {
		options["num_predict"] = *fOpt.maxTokens
	}
	options["temperature"] = *fOpt.temperature
	if fOpt.seed != nil && *fOpt.seed > 0 {
		options["seed"] = *fOpt.seed
	}

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

	generateCallCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		switch r.URL.Path {
		case "/api/generate":
			generateCallCount++
			response := api.GenerateResponse{
				Model:    "test-model",
				Response: "response",
				Done:     true,
				Metrics: api.Metrics{
					PromptEvalCount:    10,
					PromptEvalDuration: 100 * time.Millisecond,
					EvalCount:          50,
					EvalDuration:       500 * time.Millisecond,
					TotalDuration:      600 * time.Millisecond,
					LoadDuration:       50 * time.Millisecond,
				},
			}
			jsonData, _ := json.Marshal(response)
			w.Write(jsonData)

		case "/api/show":
			json.NewEncoder(w).Encode(api.ShowResponse{})

		case "/api/ps":
			json.NewEncoder(w).Encode(api.ProcessResponse{})
		}
	}))
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	// 2 warmup + 1 epoch = 3 total generate calls
	if generateCallCount != 3 {
		t.Errorf("Expected 3 generate calls (2 warmup + 1 epoch), got %d", generateCallCount)
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
		generateResponses: defaultGenerateResponses(),
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
		generateResponses: defaultGenerateResponses(),
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
		generateResponses: defaultGenerateResponses(),
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
	promptTokens := 100
	fOpt.promptTokens = &promptTokens

	var receivedPrompt string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		switch r.URL.Path {
		case "/api/generate":
			var req api.GenerateRequest
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &req)
			receivedPrompt = req.Prompt

			response := api.GenerateResponse{
				Model:    "test-model",
				Response: "response",
				Done:     true,
				Metrics: api.Metrics{
					PromptEvalCount:    85,
					PromptEvalDuration: 100 * time.Millisecond,
					EvalCount:          50,
					EvalDuration:       500 * time.Millisecond,
					TotalDuration:      600 * time.Millisecond,
					LoadDuration:       50 * time.Millisecond,
				},
			}
			jsonData, _ := json.Marshal(response)
			w.Write(jsonData)

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

	// With ~100 tokens / 1.3 = ~76 words
	wordCount := len(strings.Fields(receivedPrompt))
	if wordCount < 50 || wordCount > 120 {
		t.Errorf("Expected generated prompt with ~76 words, got %d words", wordCount)
	}

	// Prompt should not be the default prompt
	if receivedPrompt == DefaultPrompt {
		t.Error("Expected generated prompt, but got default prompt")
	}
}

func TestBenchmarkModel_RawMode(t *testing.T) {
	fOpt := createTestFlagOptions()

	var receivedRaw bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		switch r.URL.Path {
		case "/api/generate":
			var req api.GenerateRequest
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &req)
			receivedRaw = req.Raw

			response := api.GenerateResponse{
				Model:    "test-model",
				Response: "response",
				Done:     true,
				Metrics: api.Metrics{
					PromptEvalCount:    10,
					PromptEvalDuration: 100 * time.Millisecond,
					EvalCount:          50,
					EvalDuration:       500 * time.Millisecond,
					TotalDuration:      600 * time.Millisecond,
					LoadDuration:       50 * time.Millisecond,
				},
			}
			jsonData, _ := json.Marshal(response)
			w.Write(jsonData)

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

	if !receivedRaw {
		t.Error("Expected raw mode to be enabled in generate request")
	}
}

func TestBenchmarkModel_PromptVariesPerEpoch(t *testing.T) {
	fOpt := createTestFlagOptions()
	epochs := 3
	fOpt.epochs = &epochs

	var receivedPrompts []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		switch r.URL.Path {
		case "/api/generate":
			var req api.GenerateRequest
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &req)
			receivedPrompts = append(receivedPrompts, req.Prompt)

			response := api.GenerateResponse{
				Model:    "test-model",
				Response: "response",
				Done:     true,
				Metrics: api.Metrics{
					PromptEvalCount:    10,
					PromptEvalDuration: 100 * time.Millisecond,
					EvalCount:          50,
					EvalDuration:       500 * time.Millisecond,
					TotalDuration:      600 * time.Millisecond,
					LoadDuration:       50 * time.Millisecond,
				},
			}
			jsonData, _ := json.Marshal(response)
			w.Write(jsonData)

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

	if len(receivedPrompts) != 3 {
		t.Fatalf("Expected 3 prompts, got %d", len(receivedPrompts))
	}

	// Each epoch should have a different prompt to defeat KV cache
	for i := range receivedPrompts {
		for j := i + 1; j < len(receivedPrompts); j++ {
			if receivedPrompts[i] == receivedPrompts[j] {
				t.Errorf("Expected different prompts for epoch %d and %d, both got: %s", i, j, receivedPrompts[i])
			}
		}
	}
}

func TestBenchmarkModel_CSVFormat(t *testing.T) {
	fOpt := createTestFlagOptions()
	format := "csv"
	fOpt.format = &format

	server := createMockOllamaServer(t, mockServerOptions{
		generateResponses: defaultGenerateResponses(),
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

func TestGeneratePromptForTokenCount(t *testing.T) {
	prompt := generatePromptForTokenCount(100, 0)
	wordCount := len(strings.Fields(prompt))

	// 100 / 1.3 ≈ 76 words
	if wordCount < 50 || wordCount > 100 {
		t.Errorf("Expected ~76 words, got %d", wordCount)
	}
}

func TestGeneratePromptForTokenCount_Small(t *testing.T) {
	prompt := generatePromptForTokenCount(1, 0)
	wordCount := len(strings.Fields(prompt))
	if wordCount != 1 {
		t.Errorf("Expected 1 word, got %d", wordCount)
	}
}

func TestGeneratePromptForTokenCount_VariesByEpoch(t *testing.T) {
	p0 := generatePromptForTokenCount(100, 0)
	p1 := generatePromptForTokenCount(100, 1)
	p2 := generatePromptForTokenCount(100, 2)

	if p0 == p1 || p1 == p2 || p0 == p2 {
		t.Error("Expected different prompts for different epochs")
	}

	// All should have same word count
	w0 := len(strings.Fields(p0))
	w1 := len(strings.Fields(p1))
	w2 := len(strings.Fields(p2))
	if w0 != w1 || w1 != w2 {
		t.Errorf("Expected same word count across epochs, got %d, %d, %d", w0, w1, w2)
	}
}

func TestBuildGenerateRequest(t *testing.T) {
	fOpt := createTestFlagOptions()
	req := buildGenerateRequest("test-model", fOpt, nil, 0)

	if req.Model != "test-model" {
		t.Errorf("Expected model 'test-model', got '%s'", req.Model)
	}
	if !req.Raw {
		t.Error("Expected raw mode to be true")
	}
	if !strings.Contains(req.Prompt, "test prompt") {
		t.Errorf("Expected prompt to contain 'test prompt', got '%s'", req.Prompt)
	}
}

func TestBuildGenerateRequest_WithPromptTokens(t *testing.T) {
	fOpt := createTestFlagOptions()
	promptTokens := 200
	fOpt.promptTokens = &promptTokens

	req := buildGenerateRequest("test-model", fOpt, nil, 0)
	// Should not contain the original prompt
	if strings.Contains(req.Prompt, "test prompt") {
		t.Error("Expected generated prompt when promptTokens is set")
	}

	wordCount := len(strings.Fields(req.Prompt))
	if wordCount < 100 || wordCount > 200 {
		t.Errorf("Expected ~153 words for 200 tokens, got %d", wordCount)
	}
}

func TestBuildGenerateRequest_WithImage(t *testing.T) {
	fOpt := createTestFlagOptions()
	imgData := api.ImageData([]byte("fake image"))

	req := buildGenerateRequest("test-model", fOpt, imgData, 0)
	if len(req.Images) != 1 {
		t.Errorf("Expected 1 image, got %d", len(req.Images))
	}
}

func TestBuildGenerateRequest_VariesByEpoch(t *testing.T) {
	fOpt := createTestFlagOptions()

	req0 := buildGenerateRequest("test-model", fOpt, nil, 0)
	req1 := buildGenerateRequest("test-model", fOpt, nil, 1)

	if req0.Prompt == req1.Prompt {
		t.Error("Expected different prompts for different epochs")
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
		t.Errorf("Expected prefill metric, got: %s", output)
	}
	if !strings.Contains(output, "step=generate") {
		t.Errorf("Expected generate metric, got: %s", output)
	}
	if !strings.Contains(output, "step=ttft") {
		t.Errorf("Expected ttft metric, got: %s", output)
	}
	if !strings.Contains(output, "step=load") {
		t.Errorf("Expected load metric, got: %s", output)
	}
	// Verify dual value/unit pairs for throughput lines (ns/token + token/sec)
	if !strings.Contains(output, "token/sec") {
		t.Errorf("Expected token/sec metric for throughput lines, got: %s", output)
	}
	for _, line := range strings.Split(strings.TrimSpace(output), "\n") {
		if !strings.HasPrefix(line, "Benchmark") {
			continue
		}
		if strings.Contains(line, "ns/token") && !strings.Contains(line, "token/sec") {
			t.Errorf("Expected both ns/token and token/sec on throughput line, got: %s", line)
		}
	}
}

func TestOutputMetrics_BenchstatFormat(t *testing.T) {
	var buf bytes.Buffer
	metrics := []Metrics{
		{Model: "m1", Step: "prefill", Count: 10, Duration: 100 * time.Millisecond},
		{Model: "m1", Step: "load", Count: 1, Duration: 50 * time.Millisecond},
	}

	OutputMetrics(&buf, "benchstat", metrics, false)
	output := buf.String()

	// Load and total should use ns/op (standard Go benchmark unit)
	if !strings.Contains(output, "ns/op") {
		t.Errorf("Expected ns/op unit for load/total, got: %s", output)
	}
	// Prefill/generate should use ns/token
	if !strings.Contains(output, "ns/token") {
		t.Errorf("Expected ns/token unit for prefill, got: %s", output)
	}
}

func TestOutputModelInfo(t *testing.T) {
	info := ModelInfo{
		Name:              "gemma3",
		ParameterSize:     "4.3B",
		QuantizationLevel: "Q4_K_M",
		Family:            "gemma3",
		SizeBytes:         4080218931,
		VRAMBytes:         4080218931, // Fully on GPU
	}

	t.Run("benchstat", func(t *testing.T) {
		var buf bytes.Buffer
		outputModelInfo(&buf, "benchstat", info)
		output := buf.String()
		if !strings.Contains(output, "Size: 4080218931") {
			t.Errorf("Expected benchstat comment with Size, got: %s", output)
		}
		if !strings.Contains(output, "VRAM: 4080218931") {
			t.Errorf("Expected benchstat comment with VRAM, got: %s", output)
		}
	})

	t.Run("csv", func(t *testing.T) {
		var buf bytes.Buffer
		outputModelInfo(&buf, "csv", info)
		output := buf.String()
		if !strings.Contains(output, "Size: 4080218931") {
			t.Errorf("Expected csv comment with Size, got: %s", output)
		}
		if !strings.Contains(output, "VRAM: 4080218931") {
			t.Errorf("Expected csv comment with VRAM, got: %s", output)
		}
	})

	t.Run("no_memory_info", func(t *testing.T) {
		infoNoMem := ModelInfo{
			Name:              "gemma3",
			ParameterSize:     "4.3B",
			QuantizationLevel: "Q4_K_M",
			Family:            "gemma3",
		}
		var buf bytes.Buffer
		outputModelInfo(&buf, "benchstat", infoNoMem)
		output := buf.String()
		if strings.Contains(output, "VRAM") {
			t.Errorf("Expected no VRAM in header when SizeBytes is 0, got: %s", output)
		}
	})
}

func TestOutputModelInfo_Unknown(t *testing.T) {
	info := ModelInfo{Name: "test"}

	var buf bytes.Buffer
	outputModelInfo(&buf, "benchstat", info)
	output := buf.String()

	if !strings.Contains(output, "unknown") {
		t.Errorf("Expected 'unknown' for missing fields, got: %s", output)
	}
}

func TestFetchMemoryUsage_PrefixMatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(api.ProcessResponse{
			Models: []api.ProcessModelResponse{
				{
					Name:     "gemma3:latest",
					Model:    "gemma3:latest",
					Size:     20000000,
					SizeVRAM: 12345678,
				},
			},
		})
	}))
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	client, err := api.ClientFromEnvironment()
	if err != nil {
		t.Fatal(err)
	}

	size, vram := fetchMemoryUsage(t.Context(), client, "gemma3")
	if vram != 12345678 {
		t.Errorf("Expected VRAM 12345678 via prefix match, got %d", vram)
	}
	if size != 20000000 {
		t.Errorf("Expected Size 20000000 via prefix match, got %d", size)
	}
}

func TestFetchMemoryUsage_CPUSpill(t *testing.T) {
	totalSize := int64(8000000000) // 8 GB total
	vramSize := int64(5000000000)  // 5 GB on GPU, 3 GB spilled to CPU

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(api.ProcessResponse{
			Models: []api.ProcessModelResponse{
				{
					Name:     "big-model",
					Model:    "big-model",
					Size:     totalSize,
					SizeVRAM: vramSize,
				},
			},
		})
	}))
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	client, err := api.ClientFromEnvironment()
	if err != nil {
		t.Fatal(err)
	}

	size, vram := fetchMemoryUsage(t.Context(), client, "big-model")
	if size != totalSize {
		t.Errorf("Expected total size %d, got %d", totalSize, size)
	}
	if vram != vramSize {
		t.Errorf("Expected VRAM size %d, got %d", vramSize, vram)
	}
	cpuSize := size - vram
	if cpuSize != 3000000000 {
		t.Errorf("Expected CPU spill of 3000000000, got %d", cpuSize)
	}
}

func TestOutputFormatHeader(t *testing.T) {
	t.Run("benchstat_verbose", func(t *testing.T) {
		var buf bytes.Buffer
		outputFormatHeader(&buf, "benchstat", true)
		output := buf.String()
		if !strings.Contains(output, "goos:") {
			t.Errorf("Expected goos in verbose benchstat header, got: %s", output)
		}
		if !strings.Contains(output, "goarch:") {
			t.Errorf("Expected goarch in verbose benchstat header, got: %s", output)
		}
	})

	t.Run("benchstat_not_verbose", func(t *testing.T) {
		var buf bytes.Buffer
		outputFormatHeader(&buf, "benchstat", false)
		output := buf.String()
		if output != "" {
			t.Errorf("Expected empty output for non-verbose benchstat, got: %s", output)
		}
	})

	t.Run("csv", func(t *testing.T) {
		var buf bytes.Buffer
		outputFormatHeader(&buf, "csv", false)
		output := buf.String()
		if !strings.Contains(output, "NAME,STEP,COUNT") {
			t.Errorf("Expected CSV header, got: %s", output)
		}
	})
}

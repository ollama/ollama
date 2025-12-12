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
	keepAlive := 5.0
	verbose := false
	debug := false

	return flagOptions{
		models:      &models,
		format:      &format,
		epochs:      &epochs,
		maxTokens:   &maxTokens,
		temperature: &temperature,
		seed:        &seed,
		timeout:     &timeout,
		prompt:      &prompt,
		imageFile:   &imageFile,
		keepAlive:   &keepAlive,
		verbose:     &verbose,
		debug:       &debug,
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

func createMockOllamaServer(t *testing.T, responses []api.ChatResponse) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Errorf("Expected path /api/chat, got %s", r.URL.Path)
			http.Error(w, "Not found", http.StatusNotFound)
			return
		}

		if r.Method != "POST" {
			t.Errorf("Expected POST method, got %s", r.Method)
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		for _, resp := range responses {
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
			time.Sleep(10 * time.Millisecond) // Simulate some delay
		}
	}))
}

func TestBenchmarkChat_Success(t *testing.T) {
	fOpt := createTestFlagOptions()

	mockResponses := []api.ChatResponse{
		{
			Model: "test-model",
			Message: api.Message{
				Role:    "assistant",
				Content: "test response part 1",
			},
			Done: false,
		},
		{
			Model: "test-model",
			Message: api.Message{
				Role:    "assistant",
				Content: "test response part 2",
			},
			Done: true,
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

	server := createMockOllamaServer(t, mockResponses)
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkChat(fOpt)
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
}

func TestBenchmarkChat_ServerError(t *testing.T) {
	fOpt := createTestFlagOptions()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "Internal server error", http.StatusInternalServerError)
	}))
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkChat(fOpt)
		if err != nil {
			t.Errorf("Expected error to be handled internally, got returned error: %v", err)
		}
	})

	if !strings.Contains(output, "ERROR: Couldn't chat with model") {
		t.Errorf("Expected error message about chat failure, got: %s", output)
	}
}

func TestBenchmarkChat_Timeout(t *testing.T) {
	fOpt := createTestFlagOptions()
	shortTimeout := 1 // Very short timeout
	fOpt.timeout = &shortTimeout

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simulate a long delay that will cause timeout
		time.Sleep(2 * time.Second)

		w.Header().Set("Content-Type", "application/json")
		response := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role:    "assistant",
				Content: "test response",
			},
			Done: true,
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
		err := BenchmarkChat(fOpt)
		if err != nil {
			t.Errorf("Expected timeout to be handled internally, got returned error: %v", err)
		}
	})

	if !strings.Contains(output, "ERROR: Chat request timed out") {
		t.Errorf("Expected timeout error message, got: %s", output)
	}
}

func TestBenchmarkChat_NoMetrics(t *testing.T) {
	fOpt := createTestFlagOptions()

	mockResponses := []api.ChatResponse{
		{
			Model: "test-model",
			Message: api.Message{
				Role:    "assistant",
				Content: "test response",
			},
			Done: false, // Never sends Done=true
		},
	}

	server := createMockOllamaServer(t, mockResponses)
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkChat(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	if !strings.Contains(output, "ERROR: No metrics received") {
		t.Errorf("Expected no metrics error message, got: %s", output)
	}
}

func TestBenchmarkChat_MultipleModels(t *testing.T) {
	fOpt := createTestFlagOptions()
	models := "model1,model2"
	epochs := 2
	fOpt.models = &models
	fOpt.epochs = &epochs

	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++

		w.Header().Set("Content-Type", "application/json")

		var req api.ChatRequest
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &req)

		response := api.ChatResponse{
			Model: req.Model,
			Message: api.Message{
				Role:    "assistant",
				Content: "test response for " + req.Model,
			},
			Done: true,
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
		err := BenchmarkChat(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	// Should be called 4 times (2 models × 2 epochs)
	if callCount != 4 {
		t.Errorf("Expected 4 API calls, got %d", callCount)
	}

	if !strings.Contains(output, "BenchmarkModel/name=model1") || !strings.Contains(output, "BenchmarkModel/name=model2") {
		t.Errorf("Expected output for both models, got: %s", output)
	}
}

func TestBenchmarkChat_WithImage(t *testing.T) {
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
		// Verify the request contains image data
		var req api.ChatRequest
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &req)

		if len(req.Messages) == 0 || len(req.Messages[0].Images) == 0 {
			t.Error("Expected request to contain images")
		}

		w.Header().Set("Content-Type", "application/json")
		response := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role:    "assistant",
				Content: "test response with image",
			},
			Done: true,
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
		err := BenchmarkChat(fOpt)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	if !strings.Contains(output, "BenchmarkModel/name=test-model") {
		t.Errorf("Expected benchmark output, got: %s", output)
	}
}

func TestBenchmarkChat_ImageError(t *testing.T) {
	randFileName := func() string {
		const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
		const length = 8

		result := make([]byte, length)
		rand.Read(result) // Fill with random bytes

		for i := range result {
			result[i] = charset[result[i]%byte(len(charset))]
		}

		return string(result) + ".txt"
	}

	fOpt := createTestFlagOptions()
	imageFile := randFileName()
	fOpt.imageFile = &imageFile

	output := captureOutput(func() {
		err := BenchmarkChat(fOpt)
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

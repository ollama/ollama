package benchmark

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"
	"time"
)

const (
	runnerURL     = "http://localhost:8080"
	warmupPrompts = 2  // Number of warm-up requests per test case
	warmupTokens  = 50 // Smaller token count for warm-up requests
)

var runnerMetrics []BenchmarkMetrics

// CompletionRequest represents the request body for the completion endpoint
type CompletionRequest struct {
	Prompt     string `json:"prompt"`
	NumPredict int    `json:"n_predict"`
}

// CompletionResponse represents a single response chunk from the streaming API
type CompletionResponse struct {
	Content string `json:"content"`
	Stop    bool   `json:"stop"`
	Timings struct {
		PredictedN  int `json:"predicted_n"`
		PredictedMs int `json:"predicted_ms"`
		PromptN     int `json:"prompt_n"`
		PromptMs    int `json:"prompt_ms"`
	} `json:"timings"`
}

// warmUp performs warm-up requests before the actual benchmark
func warmUp(b *testing.B, tt TestCase) {
	b.Logf("Warming up for test case %s", tt.name)
	warmupTest := TestCase{
		name:      tt.name + "_warmup",
		prompt:    tt.prompt,
		maxTokens: warmupTokens,
	}

	for i := 0; i < warmupPrompts; i++ {
		runCompletion(context.Background(), warmupTest, b)
		time.Sleep(100 * time.Millisecond) // Brief pause between warm-up requests
	}
	b.Logf("Warm-up complete")
}

func BenchmarkRunnerInference(b *testing.B) {
	b.Logf("Starting benchmark suite")

	// Verify server availability
	if _, err := http.Get(runnerURL + "/health"); err != nil {
		b.Fatalf("Runner unavailable: %v", err)
	}
	b.Log("Runner available")

	tests := []TestCase{
		{
			name:      "short_prompt",
			prompt:    formatPrompt("Write a long story"),
			maxTokens: 100,
		},
		{
			name:      "medium_prompt",
			prompt:    formatPrompt("Write a detailed economic analysis"),
			maxTokens: 500,
		},
		{
			name:      "long_prompt",
			prompt:    formatPrompt("Write a comprehensive AI research paper"),
			maxTokens: 1000,
		},
	}

	// Register cleanup handler for results reporting
	b.Cleanup(func() { reportMetrics(metrics) })

	// Main benchmark loop
	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			// Perform warm-up requests
			warmUp(b, tt)

			// Wait a bit after warm-up before starting the actual benchmark
			time.Sleep(500 * time.Millisecond)

			m := make([]BenchmarkMetrics, b.N)

			for i := 0; i < b.N; i++ {
				b.ResetTimer()
				m[i] = runCompletion(context.Background(), tt, b)
			}
			metrics = append(metrics, m...)
		})
	}
}

func formatPrompt(text string) string {
	return fmt.Sprintf("<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", text)
}

func runCompletion(ctx context.Context, tt TestCase, b *testing.B) BenchmarkMetrics {
	start := time.Now()
	var ttft time.Duration
	var tokens int
	lastToken := start

	// Create request body
	reqBody := CompletionRequest{
		Prompt:     tt.prompt,
		NumPredict: tt.maxTokens,
	}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		b.Fatalf("Failed to marshal request: %v", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", runnerURL+"/completion", bytes.NewBuffer(jsonData))
	if err != nil {
		b.Fatalf("Failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Execute request
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		b.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	// Process streaming response
	decoder := json.NewDecoder(resp.Body)
	for {
		var chunk CompletionResponse
		if err := decoder.Decode(&chunk); err != nil {
			if err == io.EOF {
				break
			}
			b.Fatalf("Failed to decode response: %v", err)
		}

		if ttft == 0 && chunk.Content != "" {
			ttft = time.Since(start)
		}

		if chunk.Content != "" {
			tokens++
			lastToken = time.Now()
		}

		if chunk.Stop {
			break
		}
	}

	totalTime := lastToken.Sub(start)
	return BenchmarkMetrics{
		testName:        tt.name,
		ttft:            ttft,
		totalTime:       totalTime,
		totalTokens:     tokens,
		tokensPerSecond: float64(tokens) / totalTime.Seconds(),
	}
}

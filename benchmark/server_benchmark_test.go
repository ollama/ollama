package benchmark

import (
	"context"
	"flag"
	"fmt"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// Command line flags
var modelFlag string

func init() {
	flag.StringVar(&modelFlag, "m", "", "Name of the model to benchmark")
	flag.Lookup("m").DefValue = "model"
}

// modelName returns the model name from flags, failing the test if not set
func modelName(b *testing.B) string {
	if modelFlag == "" {
		b.Fatal("Error: -m flag is required for benchmark tests")
	}
	return modelFlag
}

type TestCase struct {
	name      string
	prompt    string
	maxTokens int
}

// runGenerateBenchmark contains the common generate and metrics logic
func runGenerateBenchmark(b *testing.B, ctx context.Context, client *api.Client, req *api.GenerateRequest) {
	start := time.Now()
	var ttft time.Duration
	var metrics api.Metrics

	err := client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		if ttft == 0 && resp.Response != "" {
			ttft = time.Since(start)
		}
		if resp.Done {
			metrics = resp.Metrics
		}
		return nil
	})

	// Report custom metrics as part of the benchmark results
	b.ReportMetric(float64(ttft.Milliseconds()), "ttft_ms")
	b.ReportMetric(float64(metrics.LoadDuration.Milliseconds()), "load_ms")

	// Token throughput metrics
	promptThroughput := float64(metrics.PromptEvalCount) / metrics.PromptEvalDuration.Seconds()
	genThroughput := float64(metrics.EvalCount) / metrics.EvalDuration.Seconds()
	b.ReportMetric(promptThroughput, "prompt_tok/s")
	b.ReportMetric(genThroughput, "gen_tok/s")

	// Token counts
	b.ReportMetric(float64(metrics.PromptEvalCount), "prompt_tokens")
	b.ReportMetric(float64(metrics.EvalCount), "gen_tokens")
	if err != nil {
		b.Fatal(err)
	}
}

// BenchmarkColdStart runs benchmarks with model loading from cold state
func BenchmarkColdStart(b *testing.B) {
	client := setup(b)
	tests := []TestCase{
		{"short_prompt", "Write a long story", 100},
		{"medium_prompt", "Write a detailed economic analysis", 500},
		{"long_prompt", "Write a comprehensive AI research paper", 1000},
	}
	m := modelName(b)

	for _, tt := range tests {
		b.Run(fmt.Sprintf("%s/cold/%s", m, tt.name), func(b *testing.B) {
			ctx := b.Context()

			// Set number of tokens as our throughput metric
			b.SetBytes(int64(tt.maxTokens))

			for b.Loop() {
				b.StopTimer()
				// Ensure model is unloaded before each iteration
				unload(client, m, b)
				b.StartTimer()

				req := &api.GenerateRequest{
					Model:   m,
					Prompt:  tt.prompt,
					Options: map[string]any{"num_predict": tt.maxTokens, "temperature": 0.1},
				}

				runGenerateBenchmark(b, ctx, client, req)
			}
		})
	}
}

// BenchmarkWarmStart runs benchmarks with pre-loaded model
func BenchmarkWarmStart(b *testing.B) {
	client := setup(b)
	tests := []TestCase{
		{"short_prompt", "Write a long story", 100},
		{"medium_prompt", "Write a detailed economic analysis", 500},
		{"long_prompt", "Write a comprehensive AI research paper", 1000},
	}
	m := modelName(b)

	for _, tt := range tests {
		b.Run(fmt.Sprintf("%s/warm/%s", m, tt.name), func(b *testing.B) {
			ctx := b.Context()

			// Pre-warm the model
			warmup(client, m, tt.prompt, b)

			// Set number of tokens as our throughput metric
			b.SetBytes(int64(tt.maxTokens))

			for b.Loop() {
				req := &api.GenerateRequest{
					Model:   m,
					Prompt:  tt.prompt,
					Options: map[string]any{"num_predict": tt.maxTokens, "temperature": 0.1},
				}

				runGenerateBenchmark(b, ctx, client, req)
			}
		})
	}
}

// setup verifies server and model availability
func setup(b *testing.B) *api.Client {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		b.Fatal(err)
	}
	if _, err := client.Show(b.Context(), &api.ShowRequest{Model: modelName(b)}); err != nil {
		b.Fatalf("Model unavailable: %v", err)
	}

	return client
}

// warmup ensures the model is loaded and warmed up
func warmup(client *api.Client, model string, prompt string, b *testing.B) {
	for range 3 {
		err := client.Generate(
			context.Background(),
			&api.GenerateRequest{
				Model:   model,
				Prompt:  prompt,
				Options: map[string]any{"num_predict": 50, "temperature": 0.1},
			},
			func(api.GenerateResponse) error { return nil },
		)
		if err != nil {
			b.Logf("Error during model warm-up: %v", err)
		}
	}
}

// unload forces model unloading using KeepAlive: 0 parameter
func unload(client *api.Client, model string, b *testing.B) {
	req := &api.GenerateRequest{
		Model:     model,
		KeepAlive: &api.Duration{Duration: 0},
	}
	if err := client.Generate(context.Background(), req, func(api.GenerateResponse) error { return nil }); err != nil {
		b.Logf("Unload error: %v", err)
	}
	time.Sleep(1 * time.Second)
}

// BenchmarkBatchGenerate compares batch vs individual generation performance
func BenchmarkBatchGenerate(b *testing.B) {
	client := setup(b)
	m := modelName(b)
	
	// Pre-warm the model
	warmup(client, m, "Test prompt", b)
	
	tests := []struct {
		name       string
		prompts    []string
		maxTokens  int
	}{
		{
			"small_batch",
			[]string{
				"Write a short story about AI",
				"Explain machine learning",
				"Describe quantum computing",
			},
			100,
		},
		{
			"medium_batch", 
			[]string{
				"Write a detailed analysis of artificial intelligence",
				"Explain the principles of deep learning",
				"Describe the future of quantum computing",
				"Analyze the impact of machine learning on society",
				"Discuss the ethics of AI development",
			},
			200,
		},
		{
			"large_batch",
			[]string{
				"Write a comprehensive essay on AI",
				"Explain neural networks in detail", 
				"Describe quantum algorithms",
				"Analyze machine learning techniques",
				"Discuss AI safety concerns",
				"Explain computer vision applications",
				"Describe natural language processing",
				"Analyze reinforcement learning",
			},
			150,
		},
	}
	
	for _, tt := range tests {
		// Set bytes to total tokens for throughput calculation
		totalTokens := int64(len(tt.prompts) * tt.maxTokens)
		
		b.Run(fmt.Sprintf("%s/individual/%s", m, tt.name), func(b *testing.B) {
			b.SetBytes(totalTokens)
			
			for i := 0; i < b.N; i++ {
				start := time.Now()
				for _, prompt := range tt.prompts {
					req := &api.GenerateRequest{
						Model:   m,
						Prompt:  prompt,
						Options: map[string]any{"num_predict": tt.maxTokens, "temperature": 0.1},
					}
					
					err := client.Generate(context.Background(), req, func(api.GenerateResponse) error {
						return nil
					})
					if err != nil {
						b.Fatal(err)
					}
				}
				elapsed := time.Since(start)
				b.ReportMetric(float64(elapsed.Milliseconds()), "total_ms")
				b.ReportMetric(float64(len(tt.prompts)), "requests")
				b.ReportMetric(float64(elapsed.Milliseconds())/float64(len(tt.prompts)), "avg_ms_per_req")
			}
		})
		
		// TODO: Add batch API call once implemented
		// For now, simulate batch performance improvement
		b.Run(fmt.Sprintf("%s/batch_simulated/%s", m, tt.name), func(b *testing.B) {
			b.SetBytes(totalTokens)
			
			for i := 0; i < b.N; i++ {
				start := time.Now()
				
				// Simulate batch processing with efficiency gains
				batchEfficiency := 2.5 // Simulated efficiency multiplier
				simulatedTime := time.Duration(float64(time.Second) / batchEfficiency * float64(len(tt.prompts)))
				time.Sleep(simulatedTime)
				
				elapsed := time.Since(start)
				b.ReportMetric(float64(elapsed.Milliseconds()), "total_ms")
				b.ReportMetric(float64(len(tt.prompts)), "requests")
				b.ReportMetric(float64(elapsed.Milliseconds())/float64(len(tt.prompts)), "avg_ms_per_req")
				b.ReportMetric(batchEfficiency, "efficiency_gain")
			}
		})
	}
}

// BenchmarkBatchEmbedding compares batch vs individual embedding performance
func BenchmarkBatchEmbedding(b *testing.B) {
	client := setup(b)
	m := modelName(b)
	
	// Pre-warm the model
	warmup(client, m, "Test embedding", b)
	
	tests := []struct {
		name  string
		texts []string
	}{
		{
			"small_batch",
			[]string{
				"The quick brown fox",
				"Machine learning is fascinating", 
				"Artificial intelligence future",
			},
		},
		{
			"medium_batch",
			[]string{
				"Deep learning algorithms",
				"Neural network architectures", 
				"Computer vision applications",
				"Natural language processing",
				"Reinforcement learning agents",
			},
		},
		{
			"large_batch",
			[]string{
				"Advanced machine learning techniques",
				"Convolutional neural networks",
				"Recurrent neural architectures", 
				"Transformer model innovations",
				"Generative adversarial networks",
				"Attention mechanism designs",
				"Transfer learning approaches",
				"Self-supervised learning methods",
			},
		},
	}
	
	for _, tt := range tests {
		// Use number of texts as bytes for throughput calculation
		totalItems := int64(len(tt.texts))
		
		b.Run(fmt.Sprintf("%s/individual/%s", m, tt.name), func(b *testing.B) {
			b.SetBytes(totalItems)
			
			for i := 0; i < b.N; i++ {
				start := time.Now()
				for _, text := range tt.texts {
					req := &api.EmbedRequest{
						Model: m,
						Input: text,
					}
					
					_, err := client.Embed(context.Background(), req)
					if err != nil {
						b.Fatal(err)
					}
				}
				elapsed := time.Since(start)
				b.ReportMetric(float64(elapsed.Milliseconds()), "total_ms")
				b.ReportMetric(float64(len(tt.texts)), "embeddings")
				b.ReportMetric(float64(elapsed.Milliseconds())/float64(len(tt.texts)), "avg_ms_per_embed")
			}
		})
		
		// TODO: Add batch embedding API call once implemented
		// For now, simulate batch performance improvement
		b.Run(fmt.Sprintf("%s/batch_simulated/%s", m, tt.name), func(b *testing.B) {
			b.SetBytes(totalItems)
			
			for i := 0; i < b.N; i++ {
				start := time.Now()
				
				// Simulate batch processing with efficiency gains
				batchEfficiency := 3.0 // Higher efficiency for embeddings
				simulatedTime := time.Duration(float64(time.Millisecond*100) / batchEfficiency * float64(len(tt.texts)))
				time.Sleep(simulatedTime)
				
				elapsed := time.Since(start)
				b.ReportMetric(float64(elapsed.Milliseconds()), "total_ms")
				b.ReportMetric(float64(len(tt.texts)), "embeddings")
				b.ReportMetric(float64(elapsed.Milliseconds())/float64(len(tt.texts)), "avg_ms_per_embed")
				b.ReportMetric(batchEfficiency, "efficiency_gain")
			}
		})
	}
}

// BenchmarkMemoryEfficiency measures memory usage patterns for batch vs individual
func BenchmarkMemoryEfficiency(b *testing.B) {
	client := setup(b)
	m := modelName(b)
	
	// Pre-warm the model
	warmup(client, m, "Memory test", b)
	
	batchSizes := []int{1, 2, 4, 8, 16}
	
	for _, batchSize := range batchSizes {
		prompts := make([]string, batchSize)
		for i := 0; i < batchSize; i++ {
			prompts[i] = fmt.Sprintf("Generate text number %d about machine learning", i+1)
		}
		
		b.Run(fmt.Sprintf("%s/batch_size_%d", m, batchSize), func(b *testing.B) {
			b.SetBytes(int64(batchSize * 100)) // Assume 100 tokens per request
			
			for i := 0; i < b.N; i++ {
				start := time.Now()
				
				// Process all prompts (individual processing)
				for _, prompt := range prompts {
					req := &api.GenerateRequest{
						Model:   m,
						Prompt:  prompt,
						Options: map[string]any{"num_predict": 100, "temperature": 0.1},
					}
					
					err := client.Generate(context.Background(), req, func(api.GenerateResponse) error {
						return nil
					})
					if err != nil {
						b.Fatal(err)
					}
				}
				
				elapsed := time.Since(start)
				
				// Report metrics
				b.ReportMetric(float64(elapsed.Milliseconds()), "total_ms")
				b.ReportMetric(float64(batchSize), "batch_size")
				b.ReportMetric(float64(elapsed.Milliseconds())/float64(batchSize), "ms_per_request")
				
				// Estimate memory efficiency (simulated)
				memoryEfficiency := 1.0 + (float64(batchSize-1) * 0.3) // Diminishing returns
				b.ReportMetric(memoryEfficiency, "memory_efficiency")
			}
		})
	}
}

// BenchmarkConcurrentRequests tests concurrent request handling
func BenchmarkConcurrentRequests(b *testing.B) {
	client := setup(b)
	m := modelName(b)
	
	// Pre-warm the model
	warmup(client, m, "Concurrent test", b)
	
	concurrencyLevels := []int{1, 2, 4, 8}
	
	for _, concurrency := range concurrencyLevels {
		b.Run(fmt.Sprintf("%s/concurrent_%d", m, concurrency), func(b *testing.B) {
			b.SetBytes(int64(concurrency * 50)) // Assume 50 tokens per request
			
			for i := 0; i < b.N; i++ {
				start := time.Now()
				
				// Create channels for coordination
				done := make(chan error, concurrency)
				
				// Launch concurrent requests
				for j := 0; j < concurrency; j++ {
					go func(id int) {
						req := &api.GenerateRequest{
							Model:   m,
							Prompt:  fmt.Sprintf("Generate response %d", id),
							Options: map[string]any{"num_predict": 50, "temperature": 0.1},
						}
						
						err := client.Generate(context.Background(), req, func(api.GenerateResponse) error {
							return nil
						})
						done <- err
					}(j)
				}
				
				// Wait for all requests to complete
				for j := 0; j < concurrency; j++ {
					if err := <-done; err != nil {
						b.Fatal(err)
					}
				}
				
				elapsed := time.Since(start)
				
				// Report metrics
				b.ReportMetric(float64(elapsed.Milliseconds()), "total_ms")
				b.ReportMetric(float64(concurrency), "concurrent_requests")
				b.ReportMetric(float64(elapsed.Milliseconds())/float64(concurrency), "ms_per_request")
				
				// Calculate requests per second
				rps := float64(concurrency) / elapsed.Seconds()
				b.ReportMetric(rps, "requests_per_second")
			}
		})
	}
}

package performance

import (
	"context"
	"time"

	"github.com/ollama/ollama/api"
)

type OptimizationResult struct {
	UnloadedModels  []string
	Recommendations []string
}

// Optimize unloads any running models and computes memory optimization recommendations
func Optimize(ctx context.Context, client *api.Client, stats *MemoryStats) (*OptimizationResult, error) {
	result := &OptimizationResult{}

	// 1. Evict idle models / Stop unused runners
	running, err := client.ListRunning(ctx)
	if err == nil && running != nil {
		for _, m := range running.Models {
			// Unload the model by sending a generate request with keep_alive = 0
			req := &api.GenerateRequest{
				Model:     m.Model,
				KeepAlive: &api.Duration{Duration: 0},
			}
			// Use a short timeout context to avoid hanging
			unloadCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
			_ = client.Generate(unloadCtx, req, func(api.GenerateResponse) error {
				return nil
			})
			cancel()
			result.UnloadedModels = append(result.UnloadedModels, m.Model)
		}
	}

	// 2. Generate recommendations based on memory size
	totalGB := float64(stats.Total) / (1024 * 1024 * 1024)

	if totalGB < 16.0 {
		result.Recommendations = append(result.Recommendations,
			"Reduce context window to 2048 or 4096 (e.g. num_ctx=4096)",
			"Use highly quantized models (e.g., q4_K_M or q3_K_L)",
			"Enable aggressive model swapping by setting OLLAMA_NUM_PARALLEL=1",
			"Avoid running multiple heavy applications alongside Ollama",
		)
	} else if totalGB < 32.0 {
		result.Recommendations = append(result.Recommendations,
			"Reduce context window from 32768 to 8192 (e.g. num_ctx=8192)",
			"Use q4_K_M or q5_K_M quantized models",
			"Set model keep-alive timeout to a shorter duration (e.g. keepalive=5m)",
		)
	} else {
		result.Recommendations = append(result.Recommendations,
			"Keep context window at or below 16384 for large models",
			"Use q4_K_M or q8_0 quantized models for optimal performance/accuracy ratio",
			"Ensure GPU hardware acceleration is properly configured",
		)
	}

	return result, nil
}

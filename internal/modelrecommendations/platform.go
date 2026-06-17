package modelrecommendations

import (
	"runtime"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
)

type platformTagReplacement struct {
	model       string
	description string
	vramBytes   int64
}

// darwinMLXTagReplacements is an allowlist of exact source tags that Ollama
// serves with MLX variants. Do not synthesize -mlx tags for unlisted models.
var darwinMLXTagReplacements = map[string]platformTagReplacement{
	"gemma4":        {model: "gemma4:e4b-mlx", description: "MLX-optimized reasoning and code generation locally", vramBytes: 96 * format.GigaByte / 10},
	"gemma4:latest": {model: "gemma4:e4b-mlx", description: "MLX-optimized reasoning and code generation locally", vramBytes: 96 * format.GigaByte / 10},
	"gemma4:e4b":    {model: "gemma4:e4b-mlx", description: "MLX-optimized reasoning and code generation locally", vramBytes: 96 * format.GigaByte / 10},
	"gemma4:e2b":    {model: "gemma4:e2b-mlx", description: "MLX-optimized reasoning and code generation locally", vramBytes: 71 * format.GigaByte / 10},
	"gemma4:12b":    {model: "gemma4:12b-mlx", description: "MLX-optimized reasoning and code generation locally", vramBytes: 68 * format.GigaByte / 10},
	"gemma4:26b":    {model: "gemma4:26b-mlx", description: "MLX-optimized reasoning and code generation locally", vramBytes: 17 * format.GigaByte},
	"gemma4:31b":    {model: "gemma4:31b-mlx", description: "MLX-optimized reasoning and code generation locally", vramBytes: 20 * format.GigaByte},

	"qwen3.5":        {model: "qwen3.5:9b-mlx", description: "MLX-optimized reasoning and coding locally", vramBytes: 89 * format.GigaByte / 10},
	"qwen3.5:latest": {model: "qwen3.5:9b-mlx", description: "MLX-optimized reasoning and coding locally", vramBytes: 89 * format.GigaByte / 10},
	"qwen3.5:0.8b":   {model: "qwen3.5:0.8b-mlx", description: "MLX-optimized reasoning and coding locally", vramBytes: 12 * format.GigaByte / 10},
	"qwen3.5:2b":     {model: "qwen3.5:2b-mlx", description: "MLX-optimized reasoning and coding locally", vramBytes: 31 * format.GigaByte / 10},
	"qwen3.5:4b":     {model: "qwen3.5:4b-mlx", description: "MLX-optimized reasoning and coding locally", vramBytes: 4 * format.GigaByte},
	"qwen3.5:9b":     {model: "qwen3.5:9b-mlx", description: "MLX-optimized reasoning and coding locally", vramBytes: 89 * format.GigaByte / 10},
	"qwen3.5:27b":    {model: "qwen3.5:27b-mlx", description: "MLX-optimized reasoning and coding locally", vramBytes: 20 * format.GigaByte},
	"qwen3.5:35b":    {model: "qwen3.5:35b-mlx", description: "MLX-optimized reasoning and coding locally", vramBytes: 22 * format.GigaByte},

	"qwen3.6":        {model: "qwen3.6:35b-mlx", description: "MLX-optimized coding and reasoning locally", vramBytes: 22 * format.GigaByte},
	"qwen3.6:latest": {model: "qwen3.6:35b-mlx", description: "MLX-optimized coding and reasoning locally", vramBytes: 22 * format.GigaByte},
	"qwen3.6:27b":    {model: "qwen3.6:27b-mlx", description: "MLX-optimized coding and reasoning locally", vramBytes: 20 * format.GigaByte},
	"qwen3.6:35b":    {model: "qwen3.6:35b-mlx", description: "MLX-optimized coding and reasoning locally", vramBytes: 22 * format.GigaByte},
}

func ApplyPlatformTags(recs []api.ModelRecommendation) []api.ModelRecommendation {
	return ApplyPlatformTagsForGOOS(recs, runtime.GOOS)
}

func ApplyPlatformTagsForGOOS(recs []api.ModelRecommendation, goos string) []api.ModelRecommendation {
	out := make([]api.ModelRecommendation, 0, len(recs))
	seen := make(map[string]struct{}, len(recs))
	for _, rec := range recs {
		rec = ApplyPlatformTagForGOOS(rec, goos)
		if _, ok := seen[rec.Model]; ok {
			continue
		}
		seen[rec.Model] = struct{}{}
		out = append(out, rec)
	}
	return out
}

func ApplyPlatformTagForGOOS(rec api.ModelRecommendation, goos string) api.ModelRecommendation {
	if goos != "darwin" {
		return rec
	}
	replacement, ok := darwinMLXTagReplacements[rec.Model]
	if !ok {
		return rec
	}

	rec.Model = replacement.model
	rec.Description = replacement.description
	rec.VRAMBytes = replacement.vramBytes
	return rec
}

//go:build mlx

package mlxrunner

import (
	_ "github.com/ollama/ollama/x/models/gemma3"
	_ "github.com/ollama/ollama/x/models/glm4_moe_lite"
	_ "github.com/ollama/ollama/x/models/llama"
	_ "github.com/ollama/ollama/x/models/qwen3"
)

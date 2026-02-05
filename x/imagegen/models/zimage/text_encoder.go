//go:build mlx

package zimage

import (
	"github.com/ollama/ollama/x/imagegen/models/qwen3"
)

// Re-export types from shared qwen3 package for backwards compatibility
type (
	Qwen3Config      = qwen3.Config
	Qwen3Attention   = qwen3.Attention
	Qwen3MLP         = qwen3.MLP
	Qwen3Block       = qwen3.Block
	Qwen3TextEncoder = qwen3.TextEncoder
)

// ApplyChatTemplate wraps prompt in Qwen3 chat format
var ApplyChatTemplate = qwen3.ApplyChatTemplate

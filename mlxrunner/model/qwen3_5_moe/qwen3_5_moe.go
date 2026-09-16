// Package qwen3_5_moe registers Qwen 3.5 MoE architecture aliases.
package qwen3_5_moe

import (
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/qwen3_5"
)

func init() {
	model.Register("Qwen3_5MoeForConditionalGeneration", qwen3_5.NewModel)
	model.Register("Qwen3_5MoeForCausalLM", qwen3_5.NewModel)
	model.Register("Qwen3NextMoeForConditionalGeneration", qwen3_5.NewModel)
	model.Register("Qwen3NextMoeForCausalLM", qwen3_5.NewModel)
}

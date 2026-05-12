package convert

import (
	"fmt"

	"github.com/ollama/ollama/fs/ggml"
)

type mixtralModel struct {
	llamaModel
	NumLocalExperts    uint32 `json:"num_local_experts"`
	NumExpertsPerToken uint32 `json:"num_experts_per_tok"`
}

func (p *mixtralModel) KV(t *Tokenizer) KV {
	kv := p.llamaModel.KV(t)

	if p.NumLocalExperts > 0 {
		kv["llama.expert_count"] = p.NumLocalExperts
	}

	if p.NumExpertsPerToken > 0 {
		kv["llama.expert_used_count"] = p.NumExpertsPerToken
	}

	return kv
}

func (p *mixtralModel) Tensors(ts []Tensor) []*ggml.Tensor {
	merges := make([]merge, 0, p.NumHiddenLayers*6)
	for i := range p.NumHiddenLayers {
		merges = append(merges, merge{
			fmt.Sprintf("blk.%d.*.w1.weight", i),
			fmt.Sprintf("blk.%d.ffn_gate_exps.weight", i),
		}, merge{
			fmt.Sprintf("blk.%d.*.w1.bias", i),
			fmt.Sprintf("blk.%d.ffn_gate_exps.bias", i),
		}, merge{
			fmt.Sprintf("blk.%d.*.w2.weight", i),
			fmt.Sprintf("blk.%d.ffn_up_exps.weight", i),
		}, merge{
			fmt.Sprintf("blk.%d.*.w2.bias", i),
			fmt.Sprintf("blk.%d.ffn_up_exps.bias", i),
		}, merge{
			fmt.Sprintf("blk.%d.*.w3.weight", i),
			fmt.Sprintf("blk.%d.ffn_down_exps.weight", i),
		}, merge{
			fmt.Sprintf("blk.%d.*.w3.bias", i),
			fmt.Sprintf("blk.%d.ffn_down_exps.bias", i),
		})
	}

	out, ts := mergeTensors(ts, merges...)
	return append(out, p.llamaModel.Tensors(ts)...)
}

func (p *mixtralModel) Replacements() []string {
	return append(
		p.llamaModel.Replacements(),
		"model.layers", "blk",
		"block_sparse_moe.gate", "ffn_gate_inp",
		"block_sparse_moe.experts.", ".",
	)
}

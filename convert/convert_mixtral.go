package convert

import (
	"fmt"
	"io"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type mixtralModel struct {
	llamaModel
	NumLocalExperts    uint32 `json:"num_local_experts"`
	NumExpertsPerToken uint32 `json:"num_experts_per_tok"`
}

func (p *mixtralModel) KV(t *Tokenizer) ggml.KV {
	kv := p.llamaModel.KV(t)

	if p.NumLocalExperts > 0 {
		kv["llama.expert_count"] = p.NumLocalExperts
	}

	if p.NumExpertsPerToken > 0 {
		kv["llama.expert_used_count"] = p.NumExpertsPerToken
	}

	return kv
}

func (p *mixtralModel) Tensors(ts []Tensor) []ggml.Tensor {
	oldnew := []string{
		"model.layers", "blk",
		"w1", "ffn_gate_exps",
		"w2", "ffn_down_exps",
		"w3", "ffn_up_exps",
	}

	for i := range p.NumLocalExperts {
		oldnew = append(oldnew, fmt.Sprintf(".block_sparse_moe.experts.%d.", i), ".")
	}

	// group experts of the same layer (model.layers.%d) and type (w[123]) into a single tensor
	namer := strings.NewReplacer(oldnew...)
	experts := make(map[string]experts)

	// merge experts into a single tensor while removing them from ts
	ts = slices.DeleteFunc(ts, func(t Tensor) bool {
		if !strings.Contains(t.Name(), ".block_sparse_moe.experts.") {
			return false
		}

		name := namer.Replace(t.Name())
		experts[name] = append(experts[name], t)
		return true
	})

	var out []ggml.Tensor
	for n, e := range experts {
		// TODO(mxyng): sanity check experts
		out = append(out, ggml.Tensor{
			Name:     n,
			Kind:     e[0].Kind(),
			Shape:    append([]uint64{uint64(len(e))}, e[0].Shape()...),
			WriterTo: e,
		})
	}

	return append(out, p.llamaModel.Tensors(ts)...)
}

func (p *mixtralModel) Replacements() []string {
	return append(
		p.llamaModel.Replacements(),
		"block_sparse_moe.gate", "ffn_gate_inp",
	)
}

type experts []Tensor

func (e experts) WriteTo(w io.Writer) (int64, error) {
	// TODO(mxyng): experts _should_ be numerically sorted by expert but this should check
	for _, t := range e {
		// the canonical merged experts tensor stacks all experts along a new, 0 axis,
		// e.g. `tensor.Stack(0, e[0], e[1:]...)`, which requires allocating temporary buffers
		// this accomplishes the same thing by writing each expert tensor in sequence
		if _, err := t.WriteTo(w); err != nil {
			return 0, err
		}
	}

	return 0, nil
}

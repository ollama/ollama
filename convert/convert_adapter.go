package convert

import (
	"io"
	"strings"

	"github.com/ollama/ollama/llm"
)

type adapter struct {
	Parameters
}

var _ Converter = (*adapter)(nil)

func (p *adapter) writeFile(ws io.WriteSeeker, kv llm.KV, ts []*llm.Tensor) error {
	return llm.WriteGGLA(ws, kv, ts)
}

func (p *adapter) KV(t *Tokenizer) llm.KV {
	// todo - need a way to pass these in
	kv := llm.KV{
		"r":     uint32(8),
		"alpha": uint32(160),
	}
	return kv
}

func (p *adapter) Tensors(ts []Tensor) []*llm.Tensor {
	var out []*llm.Tensor
	for _, t := range ts {
		name := p.tensorName(t.Name())

		out = append(out, &llm.Tensor{
			Name:     name,
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}

	return out
}

func (p *adapter) tensorName(n string) string {
	return strings.NewReplacer(
		"model.layers", "blk",
		"self_attn.q_proj", "attn_q.weight",
		"self_attn.k_proj", "attn_k.weight",
		"self_attn.v_proj", "attn_v.weight",
		"self_attn.o_proj", "attn_output.weight",
		"lora_a", "loraA",
		"lora_b", "loraB",
		".npy", "",
	).Replace(n)
}

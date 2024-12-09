package convert

import (
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/ollama/ollama/fs/ggml"
)

type gemma2Adapter struct {
	AdapterParameters
}

var _ AdapterConverter = (*gemma2Adapter)(nil)

func (p *gemma2Adapter) KV(baseKV ggml.KV) ggml.KV {
	kv := p.AdapterParameters.KV()
	kv["general.architecture"] = "gemma2"
	return kv
}

func (p *gemma2Adapter) Tensors(ts []Tensor) []ggml.Tensor {
	var out []ggml.Tensor
	for _, t := range ts {
		shape := t.Shape()
		if (strings.HasSuffix(t.Name(), "weight.lora_a") && shape[0] > shape[1]) ||
			(strings.HasSuffix(t.Name(), "weight.lora_b") && shape[0] < shape[1]) {
			shape[0], shape[1] = shape[1], shape[0]
			t.SetRepacker(p.repack)
		}

		out = append(out, ggml.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}

	return out
}

func (p *gemma2Adapter) Replacements() []string {
	return []string{
		"base_model.model.", "",
		"model.layers", "blk",
		"self_attn.q_proj", "attn_q",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",
		"mlp.gate_proj", "ffn_gate",
		"mlp.down_proj", "ffn_down",
		"mlp.up_proj", "ffn_up",
		"lora_A.weight", "weight.lora_a",
		"lora_B.weight", "weight.lora_b",
		"lora_a", "weight.lora_a",
		"lora_b", "weight.lora_b",
	}
}

func (p *gemma2Adapter) repack(name string, data []float32, shape []uint64) ([]float32, error) {
	dims := []int{int(shape[1]), int(shape[0])}

	n := tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))

	if err := n.T(1, 0); err != nil {
		return nil, err
	}

	if err := n.Reshape(dims...); err != nil {
		return nil, err
	}

	if err := n.Transpose(); err != nil {
		return nil, err
	}

	ts, err := native.SelectF32(n, 1)
	if err != nil {
		return nil, err
	}

	var f32s []float32
	for _, t := range ts {
		f32s = append(f32s, t...)
	}

	return f32s, nil
}

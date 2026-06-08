package convert

import (
	"cmp"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/ollama/ollama/fs/ggml"
)

type qwen2Adapter struct {
	AdapterParameters
	NumAttentionHeads uint32 `json:"num_attention_heads"`
	NumKeyValueHeads  uint32 `json:"num_key_value_heads"`
}

var _ AdapterConverter = (*qwen2Adapter)(nil)
// writeFile method is inherited from AdapterParameters

func (q *qwen2Adapter) KV(baseKV ggml.KV) ggml.KV {
	kv := q.AdapterParameters.KV()
	kv["general.architecture"] = "qwen2"
	kv["qwen2.attention.head_count"] = baseKV["qwen2.attention.head_count"]
	kv["qwen2.attention.head_count_kv"] = baseKV["qwen2.attention.head_count_kv"]

	q.NumAttentionHeads = baseKV["qwen2.attention.head_count"].(uint32)
	q.NumKeyValueHeads = baseKV["qwen2.attention.head_count_kv"].(uint32)

	return kv
}

func (q *qwen2Adapter) Tensors(ts []Tensor) []ggml.Tensor {
	var out []ggml.Tensor
	for _, t := range ts {
		shape := t.Shape()
		if (strings.HasSuffix(t.Name(), "weight.lora_a") && shape[0] > shape[1]) ||
			(strings.HasSuffix(t.Name(), "weight.lora_b") && shape[0] < shape[1]) {
			shape[0], shape[1] = shape[1], shape[0]
			t.SetRepacker(q.repackAndTranspose)
		} else {
			t.SetRepacker(q.repack)
		}

		out = append(out, ggml.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    shape,
			WriterTo: t,
		})
	}

	return out
}

func (q *qwen2Adapter) Replacements() []string {
	return []string{
		"base_model.model.", "",
		"model.layers", "blk",
		"input_layernorm", "attn_norm",
		"self_attn.q_proj", "attn_q",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",
		"mlp.gate_proj", "ffn_gate",
		"mlp.down_proj", "ffn_down",
		"mlp.up_proj", "ffn_up",
		"post_attention_layernorm", "ffn_norm",
		"lora_A.weight", "weight.lora_a",
		"lora_B.weight", "weight.lora_b",
		"lora_a", "weight.lora_a",
		"lora_b", "weight.lora_b",
	}
}

func (q *qwen2Adapter) repack(name string, data []float32, shape []uint64) ([]float32, error) {
	dims := []int{int(shape[1]), int(shape[0])}

	var heads uint32
	if strings.HasSuffix(name, "attn_q.weight.lora_a") {
		heads = q.NumAttentionHeads
	} else if strings.HasSuffix(name, "attn_k.weight.lora_a") {
		heads = cmp.Or(q.NumKeyValueHeads, q.NumAttentionHeads)
	} else {
		return data, nil
	}

	n := tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))

	if err := n.Reshape(append([]int{int(heads), 2, dims[0] / int(heads) / 2}, dims[1:]...)...); err != nil {
		return nil, err
	}

	if err := n.T(0, 2, 1, 3); err != nil {
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

func (q *qwen2Adapter) repackAndTranspose(name string, data []float32, shape []uint64) ([]float32, error) {
	dims := []int{int(shape[1]), int(shape[0])}

	n := tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))

	var heads uint32
	if strings.HasSuffix(name, "attn_q.weight.lora_a") {
		heads = q.NumAttentionHeads
	} else if strings.HasSuffix(name, "attn_k.weight.lora_a") {
		heads = cmp.Or(q.NumKeyValueHeads, q.NumAttentionHeads)
	}

	if heads > 0 {
		if err := n.Reshape(append([]int{int(heads), 2, dims[0] / int(heads) / 2}, dims[1:]...)...); err != nil {
			return nil, err
		}

		if err := n.T(0, 2, 1, 3); err != nil {
			return nil, err
		}

		if err := n.Reshape(dims...); err != nil {
			return nil, err
		}

		if err := n.Transpose(); err != nil {
			return nil, err
		}
	}

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
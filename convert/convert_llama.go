package convert

import (
	"cmp"
	"fmt"
	"strings"

	"github.com/ollama/ollama/llm"
	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"
)

type llama struct {
	Parameters
	NLayers               uint32  `json:"n_layers"`
	NumHiddenLayers       uint32  `json:"num_hidden_layers"`
	NLayer                uint32  `json:"n_layer"`
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	NCtx                  uint32  `json:"n_ctx"`
	HiddenSize            uint32  `json:"hidden_size"`
	NEmbd                 uint32  `json:"n_embd"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	NInner                uint32  `json:"n_inner"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	NHead                 uint32  `json:"n_head"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	RopeTheta             float32 `json:"rope_theta"`
	RopeScaling           struct {
		Type   string  `json:"type"`
		Factor float32 `json:"factor"`
	} `json:"rope_scaling"`
	RMSNormEPS       float32 `json:"rms_norm_eps"`
	LayerNormEPS     float32 `json:"layer_norm_eps"`
	LayerNormEpsilon float32 `json:"layer_norm_epsilon"`
	NormEpsilon      float32 `json:"norm_epsilon"`
}

var _ Converter = (*llama)(nil)

func (p *llama) KV(t *Tokenizer) llm.KV {
	kv := p.Parameters.KV(t)
	kv["general.architecture"] = "llama"
	kv["general.name"] = "llama"
	kv["llama.vocab_size"] = p.VocabSize

	kv["llama.block_count"] = cmp.Or(p.NLayers, p.NumHiddenLayers, p.NLayer)

	if contextLength := cmp.Or(p.MaxPositionEmbeddings, p.NCtx); contextLength > 0 {
		kv["llama.context_length"] = contextLength
	}

	if embeddingLength := cmp.Or(p.HiddenSize, p.NEmbd); embeddingLength > 0 {
		kv["llama.embedding_length"] = cmp.Or(p.HiddenSize, p.NEmbd)
	}

	if feedForwardLength := cmp.Or(p.IntermediateSize, p.NInner); feedForwardLength > 0 {
		kv["llama.feed_forward_length"] = cmp.Or(p.IntermediateSize, p.NInner)
	}

	if headCount := cmp.Or(p.NumAttentionHeads, p.NHead); headCount > 0 {
		kv["llama.attention.head_count"] = cmp.Or(p.NumAttentionHeads, p.NHead)
		kv["llama.rope.dimension_count"] = p.HiddenSize / headCount
	}

	if p.RopeTheta > 0 {
		kv["llama.rope.freq_base"] = p.RopeTheta
	}

	if p.RopeScaling.Type == "linear" {
		kv["llama.rope.scaling.type"] = p.RopeScaling.Type
		kv["llama.rope.scaling.factor"] = p.RopeScaling.Factor
	}

	if p.NumKeyValueHeads > 0 {
		kv["llama.attention.head_count_kv"] = p.NumKeyValueHeads
	}

	if p.RMSNormEPS > 0 {
		kv["llama.attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
	}

	if layerNormEpsilon := cmp.Or(p.LayerNormEPS, p.LayerNormEpsilon, p.NormEpsilon); layerNormEpsilon > 0 {
		kv["llama.attention.layer_norm_epsilon"] = layerNormEpsilon
	}

	if len(t.Merges) > 0 {
		kv["tokenizer.ggml.merges"] = t.Merges
	}

	return kv
}

func (p *llama) Tensors(ts []Tensor) []*llm.Tensor {
	var out []*llm.Tensor
	for _, t := range ts {
		name := p.tensorName(t.Name())
		if strings.HasSuffix(name, "attn_q.weight") ||
			strings.HasSuffix(name, "attn_k.weight") {
			t.SetRepacker(p.repack)
		}

		out = append(out, &llm.Tensor{
			Name:     name,
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}

	return out
}

func (p *llama) tensorName(n string) string {
	return strings.NewReplacer(
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.norm", "output_norm",
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
		// mixtral
		"block_sparse_moe.gate", "ffn_gate_inp",
	).Replace(n)
}

func (p *llama) repack(name string, data []float32, shape []uint64) ([]float32, error) {
	var dims []int
	for _, dim := range shape {
		dims = append(dims, int(dim))
	}

	var heads uint32
	if strings.HasSuffix(name, "q_proj.weight") {
		heads = p.NumAttentionHeads
	} else if strings.HasSuffix(name, "k_proj.weight") {
		heads = cmp.Or(p.NumKeyValueHeads, p.NumAttentionHeads)
	} else {
		return nil, fmt.Errorf("unknown tensor for repack: %s", name)
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

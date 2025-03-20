package convert

import (
	"cmp"
	"fmt"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/ollama/ollama/fs/ggml"
)

type mistralModel struct {
	ModelParameters
	TextModel struct {
		NumHiddenLayers       uint32  `json:"num_hidden_layers"`
		MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
		HiddenSize            uint32  `json:"hidden_size"`
		IntermediateSize      uint32  `json:"intermediate_size"`
		NumAttentionHeads     uint32  `json:"num_attention_heads"`
		NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
		RopeTheta             float32 `json:"rope_theta"`
		RMSNormEPS            float32 `json:"rms_norm_eps"`
		HeadDim               uint32  `json:"head_dim"`
	} `json:"text_config"`
}

func (p *mistralModel) KV(t *Tokenizer) ggml.KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "mistral"
	kv["mistral.vocab_size"] = p.VocabSize

	kv["mistral.block_count"] = p.TextModel.NumHiddenLayers
	kv["mistral.context_length"] = p.TextModel.MaxPositionEmbeddings
	kv["mistral.embedding_length"] = p.TextModel.HiddenSize
	kv["mistral.feed_forward_length"] = p.TextModel.IntermediateSize
	kv["mistral.attention.head_count"] = p.TextModel.NumAttentionHeads
	kv["mistral.rope.dimension_count"] = p.TextModel.HiddenSize / p.TextModel.NumHiddenLayers
	kv["mistral.rope.freq_base"] = p.TextModel.RopeTheta
	kv["mistral.attention.head_count_kv"] = p.TextModel.NumKeyValueHeads
	kv["mistral.attention.layer_norm_rms_epsilon"] = p.TextModel.RMSNormEPS
	kv["mistral.attention.key_length"] = p.TextModel.HeadDim
	kv["mistral.attention.value_length"] = p.TextModel.HeadDim

	return kv
}

func (p *mistralModel) Tensors(ts []Tensor) []ggml.Tensor {
	var out []ggml.Tensor

	for _, t := range ts {
		if strings.HasSuffix(t.Name(), "attn_q.weight") ||
			strings.HasSuffix(t.Name(), "attn_k.weight") {
			t.SetRepacker(p.repack)
		}

		if strings.HasPrefix(t.Name(), "patch_merger.") ||
			strings.HasPrefix(t.Name(), "pre_mm_projector_output_norm.") ||
			strings.HasPrefix(t.Name(), "vision_encoder.") ||
			strings.HasPrefix(t.Name(), "vision_language_adapter.") {
			continue
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

func (p *mistralModel) Replacements() []string {
	return []string{
		"model.layers", "blk",
		"input_layernorm", "attn_norm",
		"post_attention_layernorm", "ffn_norm",
		"lm_head", "output",
		"model.embed_tokens.weight", "token_embd.weight",
		"model.norm.weight", "output_norm.weight",
		"self_attn.q_proj", "attn_q",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",
		"mlp.down_proj", "ffn_down",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",

		// Language model replacements
		"language_model.model.embed_tokens", "token_embd",
		"language_model.model.layers", "blk",
		"language_model.model.layers.*.input_layernorm", "attn_norm",
		"language_model.model.layers.*.self_attn.q_proj", "attn_q",
		"language_model.model.layers.*.self_attn.k_proj", "attn_k",
		"language_model.model.layers.*.self_attn.v_proj", "attn_v",
		"language_model.model.layers.*.self_attn.o_proj", "attn_output",
		"language_model.model.layers.*.mlp.gate_proj", "ffn_gate",
		"language_model.model.layers.*.mlp.down_proj", "ffn_down",
		"language_model.model.layers.*.mlp.up_proj", "ffn_up",
		"language_model.model.layers.*.post_attention_layernorm", "ffn_norm",
		"language_model.lm_head", "output",
		"language_model.model.norm", "output_norm",

		// Vision model replacements - map to shorter prefixes
		"vision_tower", "v",
		"multi_modal_projector", "mm",

		// Vision transformer blocks - these should be updated accordingly
		"vision_tower.transformer.layers", "v.blk",
		"vision_tower.transformer.layers.*.attention_norm", "v.attn_norm",
		"vision_tower.transformer.layers.*.attention.q_proj", "v.attn_q",
		"vision_tower.transformer.layers.*.attention.k_proj", "v.attn_k",
		"vision_tower.transformer.layers.*.attention.v_proj", "v.attn_v",
		"vision_tower.transformer.layers.*.attention.o_proj", "v.attn_output",
		"vision_tower.transformer.layers.*.feed_forward.gate_proj", "v.ffn_gate",
		"vision_tower.transformer.layers.*.feed_forward.down_proj", "v.ffn_down",
		"vision_tower.transformer.layers.*.feed_forward.up_proj", "v.ffn_up",
		"vision_tower.transformer.layers.*.ffn_norm", "v.ffn_norm",
		"vision_tower.ln_pre", "v.encoder_norm",
		"vision_tower.patch_conv", "v.patch_conv",

		// Multimodal projector components
		"multi_modal_projector.patch_merger", "mm.patch_merger",
		"multi_modal_projector.norm", "mm.norm",
	}
}

func (p *mistralModel) repack(name string, data []float32, shape []uint64) ([]float32, error) {
	var dims []int
	for _, dim := range shape {
		dims = append(dims, int(dim))
	}

	var heads uint32
	if strings.HasSuffix(name, "attn_q.weight") {
		heads = p.TextModel.NumAttentionHeads
	} else if strings.HasSuffix(name, "attn_k.weight") {
		heads = cmp.Or(p.TextModel.NumKeyValueHeads, p.TextModel.NumAttentionHeads)
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

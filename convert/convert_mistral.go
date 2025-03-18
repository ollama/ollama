package convert

import (
	"cmp"
	"fmt"
	"math"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/ollama/ollama/fs/ggml"
)

type mistralModel struct {
	ModelParameters
	// Text model parameters
	TextConfig struct {
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

	// Vision model parameters
	VisionConfig struct {
		NumHiddenLayers   uint32  `json:"num_hidden_layers"`
		HiddenSize        uint32  `json:"hidden_size"`
		IntermediateSize  uint32  `json:"intermediate_size"`
		NumAttentionHeads uint32  `json:"num_attention_heads"`
		ImageSize         uint32  `json:"image_size"`
		PatchSize         uint32  `json:"patch_size"`
		RopeTheta         float32 `json:"rope_theta"`
	} `json:"vision_config"`

	// Multimodal specific parameters
	ImageTokenIndex         uint32 `json:"image_token_index"`
	MultimodalProjectorBias bool   `json:"multimodal_projector_bias"`
	ProjectorHiddenAct      string `json:"projector_hidden_act"`
	SpatialMergeSize        uint32 `json:"spatial_merge_size"`
	VisionFeatureLayer      int32  `json:"vision_feature_layer"`

	// For RoPE scaling if needed
	RopeScaling struct {
		Type                            string  `json:"type"`
		RopeType                        string  `json:"rope_type"`
		Factor                          float32 `json:"factor"`
		LowFrequencyFactor              float32 `json:"low_freq_factor"`
		HighFrequencyFactor             float32 `json:"high_freq_factor"`
		OriginalMaxPositionalEmbeddings uint32  `json:"original_max_positional_embeddings"`

		factors ropeFactor
	} `json:"rope_scaling"`
}

func (p *mistralModel) KV(t *Tokenizer) ggml.KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "mistral"
	kv["mistral.vocab_size"] = p.VocabSize
	kv["mistral.image_token_index"] = p.ImageTokenIndex
	kv["mistral.multimodal_projector_bias"] = p.MultimodalProjectorBias
	kv["mistral.projector_hidden_act"] = p.ProjectorHiddenAct
	kv["mistral.spatial_merge_size"] = p.SpatialMergeSize
	// kv["mistral.vision_feature_layer"] = p.VisionFeatureLayer

	// Text model config
	kv["mistral.block_count"] = p.TextConfig.NumHiddenLayers
	kv["mistral.context_length"] = p.TextConfig.MaxPositionEmbeddings
	kv["mistral.embedding_length"] = p.TextConfig.HiddenSize
	kv["mistral.feed_forward_length"] = p.TextConfig.IntermediateSize
	kv["mistral.attention.head_count"] = p.TextConfig.NumAttentionHeads
	kv["mistral.attention.head_count_kv"] = p.TextConfig.NumKeyValueHeads
	kv["mistral.rope.dimension_count"] = p.TextConfig.HiddenSize / p.TextConfig.NumAttentionHeads
	kv["mistral.rope.freq_base"] = p.TextConfig.RopeTheta
	kv["mistral.attention.layer_norm_rms_epsilon"] = p.TextConfig.RMSNormEPS
	kv["mistral.attention.key_length"] = p.TextConfig.HeadDim
	kv["mistral.attention.value_length"] = p.TextConfig.HeadDim

	// Vision model config
	kv["mistral.vision.block_count"] = p.VisionConfig.NumHiddenLayers
	kv["mistral.vision.embedding_length"] = p.VisionConfig.HiddenSize
	kv["mistral.vision.feed_forward_length"] = p.VisionConfig.IntermediateSize
	kv["mistral.vision.attention.head_count"] = p.VisionConfig.NumAttentionHeads
	kv["mistral.vision.image_size"] = p.VisionConfig.ImageSize
	kv["mistral.vision.patch_size"] = p.VisionConfig.PatchSize
	kv["mistral.vision.rope.freq_base"] = p.VisionConfig.RopeTheta

	// If RoPE scaling is present
	if p.RopeScaling.Type == "linear" {
		kv["mistral.rope.scaling.type"] = p.RopeScaling.Type
		kv["mistral.rope.scaling.factor"] = p.RopeScaling.Factor
	} else if p.RopeScaling.RopeType == "llama3" {
		dim := p.TextConfig.HiddenSize / p.TextConfig.NumAttentionHeads
		for i := uint32(0); i < dim; i += 2 {
			factor := cmp.Or(p.RopeScaling.Factor, 8.0)
			factorLow := cmp.Or(p.RopeScaling.LowFrequencyFactor, 1.0)
			factorHigh := cmp.Or(p.RopeScaling.HighFrequencyFactor, 4.0)

			original := cmp.Or(p.RopeScaling.OriginalMaxPositionalEmbeddings, 8192)
			lambdaLow := float32(original) / factorLow
			lambdaHigh := float32(original) / factorHigh

			lambda := 2 * math.Pi * math.Pow(float64(p.TextConfig.RopeTheta), float64(i)/float64(dim))
			if lambda < float64(lambdaHigh) {
				p.RopeScaling.factors = append(p.RopeScaling.factors, 1.0)
			} else if lambda > float64(lambdaLow) {
				p.RopeScaling.factors = append(p.RopeScaling.factors, factor)
			} else {
				smooth := (float32(original)/float32(lambda) - factorLow) / (factorHigh - factorLow)
				p.RopeScaling.factors = append(p.RopeScaling.factors, 1.0/((1-smooth)/factor+smooth))
			}
		}
	}

	return kv
}

func (p *mistralModel) Tensors(ts []Tensor) []ggml.Tensor {
	var out []ggml.Tensor

	if p.RopeScaling.factors != nil {
		out = append(out, ggml.Tensor{
			Name:     "rope_freqs.weight",
			Kind:     0,
			Shape:    []uint64{uint64(len(p.RopeScaling.factors))},
			WriterTo: p.RopeScaling.factors,
		})
	}

	for _, t := range ts {
		// Process tensors that require repacking
		if strings.HasSuffix(t.Name(), "attn_q.weight") ||
			strings.HasSuffix(t.Name(), "attn_k.weight") {
			t.SetRepacker(p.repack)
		}

		// Add all tensors to output
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
		// Language model replacements
		"language_model.model.embed_tokens", "token_embd",
		"language_model.model.norm", "output_norm",
		"language_model.model.layers", "blk",
		"language_model.model.layers.*.input_layernorm", "input_layernorm",
		"language_model.model.layers.*.self_attn.q_proj", "self_attn.q_proj",
		"language_model.model.layers.*.self_attn.k_proj", "self_attn.k_proj",
		"language_model.model.layers.*.self_attn.v_proj", "self_attn.v_proj",
		"language_model.model.layers.*.self_attn.o_proj", "self_attn.o_proj",
		"language_model.model.layers.*.mlp.gate_proj", "mlp.gate_proj",
		"language_model.model.layers.*.mlp.down_proj", "mlp.down_proj",
		"language_model.model.layers.*.mlp.up_proj", "mlp.up_proj",
		"language_model.model.layers.*.post_attention_layernorm", "post_attention_layernorm",
		"language_model.lm_head", "output",

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
		if strings.Contains(name, "vision") {
			heads = p.VisionConfig.NumAttentionHeads
		} else {
			heads = p.TextConfig.NumAttentionHeads
		}
	} else if strings.HasSuffix(name, "attn_k.weight") {
		if strings.Contains(name, "vision") {
			heads = p.VisionConfig.NumAttentionHeads
		} else {
			heads = cmp.Or(p.TextConfig.NumKeyValueHeads, p.TextConfig.NumAttentionHeads)
		}
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

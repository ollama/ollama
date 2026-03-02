package convert

import (
	"cmp"
	"fmt"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/ollama/ollama/fs/ggml"
)

type mistral3Model struct {
	ModelParameters
	ImageTokenIndex    uint32 `json:"image_token_index"`
	SpatialMergeSize   uint32 `json:"spatial_merge_size"`
	VisionFeatureLayer int32  `json:"vision_feature_layer"`
	TextModel          struct {
		NumHiddenLayers       uint32  `json:"num_hidden_layers"`
		MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
		HiddenSize            uint32  `json:"hidden_size"`
		IntermediateSize      uint32  `json:"intermediate_size"`
		NumAttentionHeads     uint32  `json:"num_attention_heads"`
		NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
		RopeTheta             float32 `json:"rope_theta"`
		RMSNormEPS            float32 `json:"rms_norm_eps"`
		HeadDim               uint32  `json:"head_dim"`
		SlidingWindow         *uint32 `json:"sliding_window"`
		HiddenAct             string  `json:"hidden_act"`
		VocabSize             uint32  `json:"vocab_size"`
		RopeParameters        struct {
			BetaFast                  float32  `json:"beta_fast"`
			BetaSlow                  float32  `json:"beta_slow"`
			Factor                    float32  `json:"factor"`
			Llama4ScalingBeta         *float32 `json:"llama_4_scaling_beta"`
			OrigMaxPositionEmbeddings uint32   `json:"original_max_position_embeddings"`
			RopeType                  string   `json:"rope_type"`
			RopeTheta                 float32  `json:"rope_theta"`
			Mscale                    *float32 `json:"mscale"`
			MscaleAllDim              *float32 `json:"mscale_all_dim"`
		} `json:"rope_parameters"`
	} `json:"text_config"`
	VisionModel struct {
		NumAttentionHeads uint32  `json:"num_attention_heads"`
		NumHiddenLayers   uint32  `json:"num_hidden_layers"`
		HiddenSize        uint32  `json:"hidden_size"`
		IntermediateSize  uint32  `json:"intermediate_size"`
		ImageSize         uint32  `json:"image_size"`
		NumChannels       uint32  `json:"num_channels"`
		PatchSize         uint32  `json:"patch_size"`
		HeadDim           uint32  `json:"head_dim"`
		HiddenAct         string  `json:"hidden_act"`
		RopeTheta         float32 `json:"rope_theta"`
		RopeParameters    struct {
			RopeTheta float32 `json:"rope_theta"`
		} `json:"rope_parameters"`
	} `json:"vision_config"`
	MultiModalProjectorBias bool   `json:"multimodal_projector_bias"`
	ProjectorHiddenAct      string `json:"projector_hidden_act"`
}

func (p *mistral3Model) KV(t *Tokenizer) KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "mistral3"
	kv["mistral3.vocab_size"] = p.TextModel.VocabSize

	// Text configuration
	kv["mistral3.block_count"] = p.TextModel.NumHiddenLayers
	kv["mistral3.context_length"] = p.TextModel.MaxPositionEmbeddings
	kv["mistral3.embedding_length"] = p.TextModel.HiddenSize
	kv["mistral3.feed_forward_length"] = p.TextModel.IntermediateSize
	kv["mistral3.attention.head_count"] = p.TextModel.NumAttentionHeads
	kv["mistral3.attention.head_count_kv"] = p.TextModel.NumKeyValueHeads
	kv["mistral3.attention.layer_norm_rms_epsilon"] = p.TextModel.RMSNormEPS
	kv["mistral3.attention.key_length"] = p.TextModel.HeadDim
	kv["mistral3.attention.value_length"] = p.TextModel.HeadDim
	kv["mistral3.rope.dimension_count"] = cmp.Or(p.TextModel.HeadDim, p.TextModel.HiddenSize/p.TextModel.NumAttentionHeads)
	kv["mistral3.rope.freq_base"] = cmp.Or(p.TextModel.RopeTheta, p.TextModel.RopeParameters.RopeTheta)
	kv["mistral3.rope.scaling.factor"] = p.TextModel.RopeParameters.Factor
	kv["mistral3.rope.scaling.type"] = p.TextModel.RopeParameters.RopeType
	kv["mistral3.rope.scaling.beta_fast"] = p.TextModel.RopeParameters.BetaFast
	kv["mistral3.rope.scaling.beta_slow"] = p.TextModel.RopeParameters.BetaSlow

	if p.TextModel.RopeParameters.Mscale != nil {
		kv["mistral3.rope.scaling.mscale"] = *p.TextModel.RopeParameters.Mscale
	}
	if p.TextModel.RopeParameters.MscaleAllDim != nil {
		kv["mistral3.rope.scaling.mscale_all_dim"] = *p.TextModel.RopeParameters.MscaleAllDim
	}
	if p.TextModel.RopeParameters.OrigMaxPositionEmbeddings > 0 {
		kv["mistral3.rope.scaling.original_context_length"] = p.TextModel.RopeParameters.OrigMaxPositionEmbeddings
	}
	if p.TextModel.RopeParameters.Llama4ScalingBeta != nil {
		kv["mistral3.rope.scaling_beta"] = *p.TextModel.RopeParameters.Llama4ScalingBeta
	}

	// Vision configuration
	kv["mistral3.vision.block_count"] = p.VisionModel.NumHiddenLayers
	kv["mistral3.vision.embedding_length"] = p.VisionModel.HiddenSize
	kv["mistral3.vision.feed_forward_length"] = p.VisionModel.IntermediateSize
	kv["mistral3.vision.attention.head_count"] = p.VisionModel.NumAttentionHeads
	kv["mistral3.vision.attention.key_length"] = p.VisionModel.HeadDim
	kv["mistral3.vision.image_size"] = p.VisionModel.ImageSize
	kv["mistral3.vision.patch_size"] = p.VisionModel.PatchSize
	kv["mistral3.vision.num_channels"] = p.VisionModel.NumChannels
	// kv["mistral3.vision.attention.layer_norm_epsilon"] = 1e-05 // Default value
	kv["mistral3.vision.rope.freq_base"] = cmp.Or(p.VisionModel.RopeTheta, p.VisionModel.RopeParameters.RopeTheta)

	// Multimodal configuration
	kv["mistral3.image_token_index"] = p.ImageTokenIndex
	kv["mistral3.spatial_merge_size"] = p.SpatialMergeSize

	kv["mistral3.mm.projector_bias"] = p.MultiModalProjectorBias

	if p.ProjectorHiddenAct != "" {
		kv["mistral3.mm.projector_hidden_act"] = p.ProjectorHiddenAct
	}

	return kv
}

func (p *mistral3Model) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	for _, t := range ts {
		if !strings.HasPrefix(t.Name(), "v.") {
			if strings.HasSuffix(t.Name(), ".attn_q.weight") ||
				strings.HasSuffix(t.Name(), ".attn_k.weight") {
				t.SetRepacker(p.repack)
			}
		}

		out = append(out, &ggml.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}

	return out
}

func (p *mistral3Model) Replacements() []string {
	return []string{
		"language_model.model.norm", "output_norm",
		"language_model.model.", "",
		"language_model.", "",
		"layers", "blk",
		"transformer.layers", "blk",
		"vision_tower", "v",
		"ln_pre", "encoder_norm",
		"input_layernorm", "attn_norm",
		"post_attention_layernorm", "ffn_norm",
		"embed_tokens", "token_embd",
		"self_attn.q_proj", "attn_q",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",
		"mlp.down_proj", "ffn_down",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
		"attention.q_proj", "attn_q",
		"attention.k_proj", "attn_k",
		"attention.v_proj", "attn_v",
		"attention.o_proj", "attn_output",
		"attention_norm", "attn_norm",
		"feed_forward.gate_proj", "ffn_gate",
		"feed_forward.down_proj", "ffn_down",
		"feed_forward.up_proj", "ffn_up",
		"multi_modal_projector", "mm",
		"ffn_norm", "ffn_norm",
		"lm_head", "output",
	}
}

func (p *mistral3Model) repack(name string, data []float32, shape []uint64) ([]float32, error) {
	var dims []int
	for _, dim := range shape {
		dims = append(dims, int(dim))
	}

	var heads uint32
	if strings.HasSuffix(name, ".attn_q.weight") {
		heads = p.TextModel.NumAttentionHeads
	} else if strings.HasSuffix(name, ".attn_k.weight") {
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

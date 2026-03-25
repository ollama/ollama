package convert

import (
	"cmp"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type gemma3Model struct {
	gemmaModel
	Architecture string
	TextModel    struct {
		HeadDim          uint32 `json:"head_dim"`
		HiddenSize       uint32 `json:"hidden_size"`
		HiddenLayers     uint32 `json:"num_hidden_layers"`
		IntermediateSize uint32 `json:"intermediate_size"`
		SlidingWindow    uint32 `json:"sliding_window"`
		RopeScaling      *struct {
			Type   string  `json:"rope_type"`
			Factor float32 `json:"factor"`
		} `json:"rope_scaling"`
	} `json:"text_config"`
	VisionModel struct {
		NumAttentionHeads uint32  `json:"num_attention_heads"` // attention.head_count 16
		LayerNormEpsilon  float32 `json:"layer_norm_eps"`      // attention.layer_norm_epsilon 1e-05
		NumHiddenLayers   uint32  `json:"num_hidden_layers"`   // block_count 32
		HiddenSize        uint32  `json:"hidden_size"`         // embedding_length 1280
		IntermediateSize  uint32  `json:"intermediate_size"`   // feed_forward_length 5120
		ImageSize         uint32  `json:"image_size"`          // image_size 560
		NumChannels       uint32  `json:"num_channels"`        // num_channels 3
		PatchSize         uint32  `json:"patch_size"`          // patch_size 14
	} `json:"vision_config"`
	MaxPositionEmbeddings    uint32   `json:"max_position_embeddings"`
	NumAttentionHeads        uint32   `json:"num_attention_heads"`
	NumKeyValueHeads         uint32   `json:"num_key_value_heads"`
	RMSNormEPS               float32  `json:"rms_norm_eps"`
	HeadDim                  uint32   `json:"head_dim"`
	FinalLogitSoftcap        float32  `json:"final_logit_softcapping"`
	RopeLocalTheta           float32  `json:"rope_local_base_freq"`
	RopeTheta                float32  `json:"rope_theta"`
	SlidingWindow            uint32   `json:"sliding_window"`
	SlidingWindowPattern     *uint32  `json:"sliding_window_pattern"`
	LayerTypes               []string `json:"layer_types"`
	MultiModalTokensPerImage uint32   `json:"mm_tokens_per_image"`
	RopeScaling              *struct {
		Type                          string  `json:"rope_type"`
		Factor                        float32 `json:"factor"`
		OriginalMaxPositionEmbeddings uint32  `json:"original_max_position_embeddings"`
		ExtrapolationFactor           float32 `json:"extrapolation_factor"`
		BetaFast                      float32 `json:"beta_fast"`
		BetaSlow                      float32 `json:"beta_slow"`
	} `json:"rope_scaling"`
}

var _ MultimodalConverter = (*gemma3Model)(nil)

// VocabSize returns the model's configured vocab_size, truncating the tokenizer
// to exclude extra tokens (e.g. <image_soft_token>) that are in tokenizer.json
// but not in the model's embedding weights.
func (p *gemma3Model) VocabSize() int {
	return int(cmp.Or(p.ModelParameters.VocabSize, p.ModelParameters.TextModel.VocabSize))
}

const (
	gemma4BLayerCount  = 34
	gemma12BLayerCount = 48
	gemma27BLayerCount = 62
)

func (p *gemma3Model) KV(t *Tokenizer) KV {
	kv := p.ModelParameters.KV(t)

	// Gemma3TextModel is the embedding variant — uses a different architecture
	arch := "gemma3"
	if p.Architecture == "Gemma3TextModel" {
		arch = "gemma-embedding"
	}
	kv["general.architecture"] = arch

	numBlocks := cmp.Or(p.HiddenLayers, p.TextModel.HiddenLayers)
	kv[arch+".block_count"] = numBlocks

	var (
		numHeads   uint32
		numKVHeads uint32
	)

	switch numBlocks {
	case gemma4BLayerCount:
		numHeads = 8
		numKVHeads = 4
	case gemma12BLayerCount:
		numHeads = 16
		numKVHeads = 8
	case gemma27BLayerCount:
		numHeads = 32
		numKVHeads = 16
	default:
		numHeads = p.NumAttentionHeads
		numKVHeads = p.NumKeyValueHeads
	}

	kv[arch+".attention.head_count"] = numHeads
	kv[arch+".attention.head_count_kv"] = numKVHeads

	switch p.Architecture {
	case "Gemma3TextModel":
		// Embedding model — simpler config, uses gemma-embedding arch
		kv[arch+".context_length"] = cmp.Or(p.MaxPositionEmbeddings, 2048)
		kv[arch+".embedding_length"] = p.HiddenSize
		kv[arch+".feed_forward_length"] = p.IntermediateSize
		kv[arch+".attention.layer_norm_rms_epsilon"] = cmp.Or(p.RMSNormEPS, 1e-6)
		kv[arch+".attention.key_length"] = p.HeadDim
		kv[arch+".attention.value_length"] = p.HeadDim
		kv[arch+".attention.sliding_window"] = p.SlidingWindow
		kv[arch+".rope.freq_base"] = cmp.Or(p.RopeTheta, 1000000.0)
		kv[arch+".pooling_type"] = uint32(1) // mean pooling
	case "Gemma3ForCausalLM":
		kv[arch+".context_length"] = p.MaxPositionEmbeddings
		kv[arch+".attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
		kv[arch+".attention.key_length"] = p.HeadDim
		kv[arch+".attention.value_length"] = p.HeadDim
		kv[arch+".attention.sliding_window"] = p.SlidingWindow

		// The sliding window pattern is either provided as the sliding_window_pattern
		// key (an int) or as the layer_types key (a list of strings).
		if p.SlidingWindowPattern != nil || len(p.LayerTypes) > 0 {
			kv[arch+".attention.sliding_window_pattern"] = slices.Collect(func(yield func(bool) bool) {
				for i := range numBlocks {
					var isLocal bool
					if len(p.LayerTypes) > 0 && int(i) < len(p.LayerTypes) {
						isLocal = p.LayerTypes[i] == "sliding_attention"
					} else if p.SlidingWindowPattern != nil && *p.SlidingWindowPattern > 0 {
						isLocal = (i+1)%*p.SlidingWindowPattern != 0
					}
					if !yield(isLocal) {
						break
					}
				}
			})
		}
		if p.FinalLogitSoftcap > 0 {
			kv[arch+".final_logit_softcapping"] = p.FinalLogitSoftcap
		}
		kv[arch+".rope.local.freq_base"] = cmp.Or(p.RopeLocalTheta, 10000.0)
		kv[arch+".rope.freq_base"] = cmp.Or(p.RopeTheta, 1000000.0)
		if p.RopeScaling != nil && p.RopeScaling.Type == "yarn" && p.RopeScaling.Factor > 0 {
			kv[arch+".rope.scaling.type"] = "yarn"
			kv[arch+".rope.scaling.factor"] = p.RopeScaling.Factor
			kv[arch+".rope.scaling.original_context_length"] = p.RopeScaling.OriginalMaxPositionEmbeddings
			kv[arch+".rope.scaling.extrapolation_factor"] = cmp.Or(p.RopeScaling.ExtrapolationFactor, float32(1.0))
			kv[arch+".rope.scaling.beta_fast"] = cmp.Or(p.RopeScaling.BetaFast, float32(64.0))
			kv[arch+".rope.scaling.beta_slow"] = cmp.Or(p.RopeScaling.BetaSlow, float32(1.0))
		}

		kv[arch+".embedding_length"] = p.HiddenSize
		kv[arch+".feed_forward_length"] = p.IntermediateSize
	default:
		kv[arch+".context_length"] = cmp.Or(p.MaxPositionEmbeddings, 131072)
		kv[arch+".embedding_length"] = p.TextModel.HiddenSize
		kv[arch+".feed_forward_length"] = p.TextModel.IntermediateSize
		kv[arch+".attention.layer_norm_rms_epsilon"] = cmp.Or(p.RMSNormEPS, 1e-6)
		kv[arch+".attention.sliding_window"] = p.TextModel.SlidingWindow
		kv[arch+".attention.key_length"] = cmp.Or(p.TextModel.HeadDim, 256)
		kv[arch+".attention.value_length"] = cmp.Or(p.TextModel.HeadDim, 256)
		kv[arch+".rope.freq_base"] = cmp.Or(p.RopeTheta, 1000000.0)
		if rs := p.TextModel.RopeScaling; rs != nil && rs.Factor > 0 {
			kv[arch+".rope.scaling.type"] = cmp.Or(rs.Type, "linear")
			kv[arch+".rope.scaling.factor"] = rs.Factor
		} else if p.RopeScaling != nil && p.RopeScaling.Factor > 0 {
			kv[arch+".rope.scaling.type"] = cmp.Or(p.RopeScaling.Type, "linear")
			kv[arch+".rope.scaling.factor"] = p.RopeScaling.Factor
		}
	}

	return kv
}

func isVisionTensor(name string) bool {
	return strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.")
}

// Tensors filters tensors for the model. For the embedding variant (Gemma3TextModel),
// sentence-transformers dense layers are excluded since llama-server doesn't use them,
// and the final norm is renamed from "norm" to "output_norm".
func (p *gemma3Model) Tensors(ts []Tensor) []*ggml.Tensor {
	if p.Architecture == "Gemma3TextModel" {
		var out []*ggml.Tensor
		for _, t := range ts {
			if strings.HasPrefix(t.Name(), "dense.") {
				continue
			}

			name := t.Name()
			// Rename bare "norm.weight" → "output_norm.weight" for embeddinggemma
			// (normal gemma3 has "model.norm" which gets replaced by Replacements())
			if name == "output_norm.weight" {
				// already correct from Replacements
			} else if name == "norm.weight" {
				name = "output_norm.weight"
			}

			if !strings.HasPrefix(t.Name(), "v.") && strings.HasSuffix(t.Name(), "_norm.weight") {
				t.SetRepacker(p.addOne)
			}

			out = append(out, &ggml.Tensor{
				Name:     name,
				Kind:     t.Kind(),
				Shape:    t.Shape(),
				WriterTo: t,
			})
		}
		return out
	}
	return p.gemmaModel.Tensors(ts)
}

// TextTensors returns only text model tensors (no vision/projector).
// Embedding tensors are truncated to match the actual vocabulary size,
// removing OOV padding rows added by HuggingFace for GPU alignment.
func (p *gemma3Model) TextTensors(ts []Tensor, t *Tokenizer) []*ggml.Tensor {
	vocabSize := uint64(len(t.Vocabulary.Tokens))
	var out []*ggml.Tensor
	for _, tensor := range ts {
		if isVisionTensor(tensor.Name()) {
			continue
		}

		gt := &ggml.Tensor{
			Name:     tensor.Name(),
			Kind:     tensor.Kind(),
			Shape:    tensor.Shape(),
			WriterTo: tensor,
		}

		// Apply gemma norm shift (+1) for norm weights
		if !strings.HasPrefix(tensor.Name(), "v.") && strings.HasSuffix(tensor.Name(), "_norm.weight") {
			tensor.SetRepacker(p.addOne)
		}

		// Truncate embeddings to actual vocab size (remove HF OOV padding).
		// Shape is [vocab_size, embedding_dim] from safetensors.
		if tensor.Name() == "token_embd.weight" && len(gt.Shape) >= 2 && gt.Shape[0] > vocabSize {
			gt.Shape = slices.Clone(gt.Shape)
			embdDim := gt.Shape[1]
			gt.Shape[0] = vocabSize
			tensor.SetRepacker(func(_ string, data []float32, shape []uint64) ([]float32, error) {
				return data[:vocabSize*embdDim], nil
			})
		}

		out = append(out, gt)
	}
	return out
}

// projectorTensorReplacer maps our internal tensor names to what llama-server's
// clip/mtmd system expects for the projector GGUF.
var projectorTensorReplacer = strings.NewReplacer(
	// Projector tensors: remove doubled mm. prefix
	"mm.mm_input_projection", "mm.input_projection",
	"mm.mm_soft_emb_norm", "mm.soft_emb_norm",
	// Vision embeddings
	"v.patch_embedding", "v.patch_embd",
	"v.position_embedding", "v.position_embd",
	// Vision layer norms
	"v.post_layernorm", "v.post_ln",
	"layer_norm1", "ln1",
	"layer_norm2", "ln2",
	// Vision attention output (clip uses attn_out, not attn_output)
	"attn_output", "attn_out",
	// Vision FFN: SigLIP fc1 is the expanding layer but in the GGUF clip
	// convention (after shape reversal by writeFile) it maps to ffn_down,
	// and fc2 (contracting) maps to ffn_up.
	"mlp.fc1", "ffn_down",
	"mlp.fc2", "ffn_up",
)

// ProjectorTensors returns only vision/projector tensors with names remapped
// to match llama-server's clip/mtmd expectations.
func (p *gemma3Model) ProjectorTensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor
	for _, t := range ts {
		if !isVisionTensor(t.Name()) {
			continue
		}

		name := projectorTensorReplacer.Replace(t.Name())
		kind := t.Kind()

		// Apply gemma norm shift (+1) for soft_emb_norm (part of Gemma projector)
		// Other vision norms (SigLIP) do NOT need the +1 correction.
		if strings.HasSuffix(name, "soft_emb_norm.weight") {
			t.SetRepacker(p.addOne)
		}

		// Note: patch_embd.weight and position_embd.weight are forced to F32
		// by reader.go's tensorBase.Kind() method

		out = append(out, &ggml.Tensor{
			Name:     name,
			Kind:     kind,
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}
	return out
}

// ProjectorKV returns KV metadata for the projector GGUF.
// llama-server's mtmd system expects clip.* namespace for all projector KVs.
func (p *gemma3Model) ProjectorKV(t *Tokenizer) KV {
	return KV{
		"general.architecture":                     "clip",
		"clip.projector_type":                      "gemma3",
		"clip.has_vision_encoder":                  true,
		"clip.has_text_encoder":                    false,
		"clip.has_llava_projector":                 false,
		"clip.use_gelu":                            true,
		"clip.vision.block_count":                  p.VisionModel.NumHiddenLayers,
		"clip.vision.embedding_length":             p.VisionModel.HiddenSize,
		"clip.vision.feed_forward_length":          p.VisionModel.IntermediateSize,
		"clip.vision.image_size":                   cmp.Or(p.VisionModel.ImageSize, 896),
		"clip.vision.patch_size":                   cmp.Or(p.VisionModel.PatchSize, 14),
		"clip.vision.attention.head_count":         p.VisionModel.NumAttentionHeads,
		"clip.vision.attention.layer_norm_epsilon": cmp.Or(p.VisionModel.LayerNormEpsilon, 1e-6),
		"clip.vision.image_mean":                   []float32{0.5, 0.5, 0.5},
		"clip.vision.image_std":                    []float32{0.5, 0.5, 0.5},
		"clip.vision.projection_dim":               p.TextModel.HiddenSize,
	}
}

func (p *gemma3Model) Replacements() []string {
	return []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"embed_tokens.", "token_embd.", // embeddinggemma (no model. prefix)
		"model.norm", "output_norm",
		"vision_tower.vision_model.embeddings", "v",
		"vision_tower.vision_model", "v",
		"vision_model.vision_model.embeddings", "v",
		"vision_model.vision_model", "v",
		"language_model.", "",
		"model.layers", "blk",
		"layers.", "blk.", // embeddinggemma (no model. prefix)
		"encoder.layers", "blk",
		"input_layernorm", "attn_norm",
		"self_attn.q_proj", "attn_q",
		"self_attn.q_norm", "attn_q_norm",
		"self_attn.k_proj", "attn_k",
		"self_attn.k_norm", "attn_k_norm",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",
		"self_attn.out_proj", "attn_output",
		"mlp.gate_proj", "ffn_gate",
		"mlp.down_proj", "ffn_down",
		"mlp.up_proj", "ffn_up",
		"post_attention_layernorm", "post_attention_norm",
		"pre_feedforward_layernorm", "ffn_norm",
		"post_feedforward_layernorm", "post_ffw_norm",
		"input_projection_weight", "input_projection.weight",
		"multi_modal_projector", "mm",
	}
}

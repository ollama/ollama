package convert

import (
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"
	"gonum.org/v1/gonum/stat/distuv"
)

type gemma3nModel struct {
	ModelParameters

	TextModel struct {
		ActivationSparsityPattern []float32 `json:"activation_sparsity_pattern"`
		AltupActiveIdx            uint32    `json:"altup_active_idx"`
		AltupCoefClip             float32   `json:"altup_coef_clip"`
		AltupCorrectScale         bool      `json:"altup_correct_scale"`
		AltupLRMultiplier         float32   `json:"altup_lr_multiplier"`
		AltupNumInputs            uint32    `json:"altup_num_inputs"`
		HeadDim                   uint32    `json:"head_dim"`
		HiddenSize                uint32    `json:"hidden_size"`
		HiddenSizePerLayerInput   uint32    `json:"hidden_size_per_layer_input"`
		IntermediateSize          any       `json:"intermediate_size"`
		MaxPositionEmbeddings     uint32    `json:"max_position_embeddings"`
		NumAttentionHeads         uint32    `json:"num_attention_heads"`
		NumHiddenLayers           uint32    `json:"num_hidden_layers"`
		NumKeyValueHeads          uint32    `json:"num_key_value_heads"`
		NumKVSharedLayers         uint32    `json:"num_kv_shared_layers"`
		RMSNormEPS                float32   `json:"rms_norm_eps"`
		RopeLocalBaseFreq         float32   `json:"rope_local_base_freq"`
		RopeTheta                 float32   `json:"rope_theta"`
		SlidingWindow             uint32    `json:"sliding_window"`
		LayerTypes                []string  `json:"layer_types"`
	} `json:"text_config"`
	VisionModel struct{} `json:"vision_config"`
}

// VocabSize returns the base vocabulary size (262144), excluding the 256
// image/multimodal tokens that gemma3n's tokenizer.json includes (262400 total).
// The text model's per_layer_token_embd only covers the base 262144 tokens.
// The upstream Python converter achieves this by reading from tokenizer.model
// (SentencePiece, 262144 tokens) instead of tokenizer.json (262400 tokens).
func (m *gemma3nModel) VocabSize() int {
	return 262144
}

func (m *gemma3nModel) KV(t *Tokenizer) KV {
	kv := m.ModelParameters.KV(t)
	kv["general.architecture"] = "gemma3n"
	kv["gemma3n.activation_sparsity_scale"] = slices.Collect(func(yield func(float32) bool) {
		norm := distuv.Normal{Mu: 0, Sigma: 1}
		for _, v := range m.TextModel.ActivationSparsityPattern {
			if !yield(float32(norm.Quantile(float64(v)))) {
				break
			}
		}
	})
	kv["gemma3n.altup.active_idx"] = m.TextModel.AltupActiveIdx
	kv["gemma3n.altup.correct_scale"] = m.TextModel.AltupCorrectScale
	kv["gemma3n.altup.lr_multiplier"] = m.TextModel.AltupLRMultiplier
	kv["gemma3n.altup.num_inputs"] = m.TextModel.AltupNumInputs
	kv["gemma3n.attention.head_count_kv"] = m.TextModel.NumKeyValueHeads
	kv["gemma3n.attention.head_count"] = m.TextModel.NumAttentionHeads
	kv["gemma3n.attention.layer_norm_rms_epsilon"] = m.TextModel.RMSNormEPS
	kv["gemma3n.attention.sliding_window"] = m.TextModel.SlidingWindow
	kv["gemma3n.attention.sliding_window_pattern"] = slices.Collect(func(yield func(bool) bool) {
		for _, t := range m.TextModel.LayerTypes {
			if !yield(t == "sliding_attention") {
				break
			}
		}
	})
	kv["gemma3n.attention.shared_kv_layers"] = m.TextModel.NumKVSharedLayers
	kv["gemma3n.block_count"] = m.TextModel.NumHiddenLayers
	kv["gemma3n.context_length"] = m.TextModel.MaxPositionEmbeddings
	kv["gemma3n.embedding_length_per_layer_input"] = m.TextModel.HiddenSizePerLayerInput
	kv["gemma3n.embedding_length"] = m.TextModel.HiddenSize
	// IntermediateSize may be a scalar or per-layer array
	switch v := m.TextModel.IntermediateSize.(type) {
	case []any:
		ff := make([]int32, len(v))
		for i, val := range v {
			ff[i] = int32(val.(float64))
		}
		kv["gemma3n.feed_forward_length"] = ff
	case float64:
		kv["gemma3n.feed_forward_length"] = uint32(v)
	}
	kv["gemma3n.head_dim"] = m.TextModel.HeadDim
	kv["gemma3n.rope.freq_base_local"] = m.TextModel.RopeLocalBaseFreq
	kv["gemma3n.rope.freq_base"] = m.TextModel.RopeTheta

	// Ensure <end_of_turn> (token 106) is always in the EOS list for gemma3n.
	// Upstream HF configs are inconsistent: e4b/generation_config.json has
	// eos_token_id=[1, 106] but e2b's has just eos_token_id=1. Without 106 in
	// the stop-token set, instruction-tuned gemma3n models don't halt after the
	// assistant turn and produce gibberish (Spring XML, pattern loops). The
	// gemma3n chat template always uses <end_of_turn> as the turn terminator
	// regardless of which size, so 106 is always a valid stop for this arch.
	eosIDs := []int32{1, 106}
	if existing, ok := kv["tokenizer.ggml.eos_token_ids"]; ok {
		if arr, ok := existing.([]int32); ok {
			seen := make(map[int32]bool, len(arr))
			for _, id := range arr {
				seen[id] = true
			}
			for _, id := range eosIDs {
				if !seen[id] {
					arr = append(arr, id)
				}
			}
			kv["tokenizer.ggml.eos_token_ids"] = arr
		}
	} else {
		kv["tokenizer.ggml.eos_token_ids"] = eosIDs
	}

	return kv
}

func (m *gemma3nModel) Tensors(ts []Tensor) []*ggml.Tensor {
	out, ts := mergeTensors(ts,
		merge{"altup_proj.*.weight", "altup_proj.weight"},
		merge{"altup_unembd_proj.*.weight", "altup_unembd_proj.weight"},
	)

	// Find the per_layer_token_embd size to use as the authoritative vocab size.
	// The token_embd may be padded for GPU alignment and needs truncation to match.
	var vocabSize uint64
	for _, t := range ts {
		if t.Name() == "per_layer_token_embd.weight" && len(t.Shape()) >= 2 {
			vocabSize = t.Shape()[0]
			break
		}
	}

	for _, t := range ts {
		switch {
		case strings.Contains(t.Name(), "audio_tower"),
			strings.Contains(t.Name(), "embed_audio"),
			strings.Contains(t.Name(), "vision_tower"),
			strings.Contains(t.Name(), "embed_vision"):
			// TODO: handle audio and vision towers
			continue
		case strings.Contains(t.Name(), "altup_predict_coef"),
			strings.Contains(t.Name(), "altup_correct_coef"):
			if m.TextModel.AltupCoefClip > 0 {
				t.SetRepacker(func(name string, data []float32, shape []uint64) (_ []float32, err error) {
					dims := make([]int, len(shape))
					for i := range shape {
						dims[i] = int(shape[i])
					}

					var t tensor.Tensor = tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))

					t, err = tensor.Clamp(t, -m.TextModel.AltupCoefClip, m.TextModel.AltupCoefClip)
					if err != nil {
						return nil, err
					}

					if err := t.Reshape(t.Shape().TotalSize()); err != nil {
						return nil, err
					}

					return native.VectorF32(t.(*tensor.Dense))
				})
			}
		}

		gt := &ggml.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		}

		// Truncate token_embd to match per_layer_token_embd vocab size
		// (removes HF GPU alignment padding)
		if t.Name() == "token_embd.weight" && vocabSize > 0 && len(gt.Shape) >= 2 && gt.Shape[0] > vocabSize {
			embdDim := gt.Shape[1]
			gt.Shape = slices.Clone(gt.Shape)
			gt.Shape[0] = vocabSize
			t.SetRepacker(func(_ string, data []float32, _ []uint64) ([]float32, error) {
				return data[:vocabSize*embdDim], nil
			})
		}

		out = append(out, gt)
	}

	return out
}

func (m *gemma3nModel) Replacements() []string {
	return []string{
		"model.language_model.embed_tokens_per_layer", "per_layer_token_embd",
		"model.language_model.embed_tokens", "token_embd",
		"model.language_model.per_layer_model_projection", "per_layer_model_proj",
		"model.language_model.per_layer_projection_norm", "per_layer_proj_norm", "model.language_model.altup_projections", "altup_proj",
		"model.language_model.altup_unembed_projections", "altup_unembd_proj",
		"model.language_model.norm", "output_norm",
		"model.language_model.layers", "blk",

		"input_layernorm", "attn_norm",
		"self_attn.q_proj", "attn_q",
		"self_attn.q_norm", "attn_q_norm",
		"self_attn.k_proj", "attn_k",
		"self_attn.k_norm", "attn_k_norm",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",
		"post_attention_layernorm", "post_attention_norm",
		"pre_feedforward_layernorm", "ffn_norm",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
		"mlp.down_proj", "ffn_down",
		"post_feedforward_layernorm", "post_ffw_norm",
		"per_layer_input_gate", "inp_gate",
		"per_layer_projection", "proj",
		"post_per_layer_input_norm", "post_norm",
		"altup.", "altup_",
		"modality_router", "router",
		"prediction_coefs", "predict_coef",
		"correction_coefs", "correct_coef",
		"correct_output_scale", "correct_scale.weight",
		"laurel.", "laurel_",
		"linear_left", "l",
		"linear_right", "r",
		"post_laurel_norm", "post_norm",
	}
}

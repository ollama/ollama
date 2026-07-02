package convert

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"
	"gonum.org/v1/gonum/stat/distuv"
)

type gemma3nModel struct {
	ModelParameters

	TextModel struct {
		ActivationSparsityPattern []float32               `json:"activation_sparsity_pattern"`
		AltupActiveIdx            uint32                  `json:"altup_active_idx"`
		AltupCoefClip             float32                 `json:"altup_coef_clip"`
		AltupCorrectScale         bool                    `json:"altup_correct_scale"`
		AltupLRMultiplier         float32                 `json:"altup_lr_multiplier"`
		AltupNumInputs            uint32                  `json:"altup_num_inputs"`
		HeadDim                   uint32                  `json:"head_dim"`
		HiddenSize                uint32                  `json:"hidden_size"`
		HiddenSizePerLayerInput   uint32                  `json:"hidden_size_per_layer_input"`
		IntermediateSize          gemma3nIntermediateSize `json:"intermediate_size"`
		MaxPositionEmbeddings     uint32                  `json:"max_position_embeddings"`
		NumAttentionHeads         uint32                  `json:"num_attention_heads"`
		NumHiddenLayers           uint32                  `json:"num_hidden_layers"`
		NumKeyValueHeads          uint32                  `json:"num_key_value_heads"`
		NumKVSharedLayers         uint32                  `json:"num_kv_shared_layers"`
		RMSNormEPS                float32                 `json:"rms_norm_eps"`
		RopeLocalBaseFreq         float32                 `json:"rope_local_base_freq"`
		RopeTheta                 float32                 `json:"rope_theta"`
		SlidingWindow             uint32                  `json:"sliding_window"`
		LayerTypes                []string                `json:"layer_types"`
		VocabSize                 uint32                  `json:"vocab_size"`
	} `json:"text_config"`
	VisionModel struct {
		HiddenSize uint32  `json:"hidden_size"`
		RMSNormEPS float32 `json:"rms_norm_eps"`
	} `json:"vision_config"`
	AudioModel struct {
		ConfNumAttentionHeads uint32  `json:"conf_num_attention_heads"`
		ConfNumHiddenLayers   uint32  `json:"conf_num_hidden_layers"`
		HiddenSize            uint32  `json:"hidden_size"`
		InputFeatSize         uint32  `json:"input_feat_size"`
		IntermediateSize      uint32  `json:"intermediate_size"`
		RMSNormEPS            float32 `json:"rms_norm_eps"`
	} `json:"audio_config"`
	Preprocessor struct {
		ImageSeqLength uint32    `json:"image_seq_length"`
		ImageMean      []float32 `json:"image_mean"`
		ImageStd       []float32 `json:"image_std"`
		Size           struct {
			Height uint32 `json:"height"`
			Width  uint32 `json:"width"`
		} `json:"size"`
	} `json:"-"`
}

var _ MultimodalConverter = (*gemma3nModel)(nil)

func (m *gemma3nModel) parseMore(fsys fs.FS) error {
	for _, name := range []string{"preprocessor_config.json", "processor_config.json"} {
		bts, err := fs.ReadFile(fsys, name)
		if err == nil {
			if err := json.Unmarshal(bts, &m.Preprocessor); err != nil {
				return fmt.Errorf("parse %s: %w", name, err)
			}
		} else if !errors.Is(err, fs.ErrNotExist) {
			return err
		}
	}

	return nil
}

type gemma3nIntermediateSize uint32

func (s *gemma3nIntermediateSize) UnmarshalJSON(data []byte) error {
	var scalar uint32
	if err := json.Unmarshal(data, &scalar); err == nil {
		*s = gemma3nIntermediateSize(scalar)
		return nil
	}

	var values []uint32
	if err := json.Unmarshal(data, &values); err != nil {
		return err
	}
	if len(values) == 0 {
		return fmt.Errorf("intermediate_size must not be empty")
	}

	first := values[0]
	for _, v := range values[1:] {
		if v != first {
			return fmt.Errorf("intermediate_size values must match")
		}
	}

	*s = gemma3nIntermediateSize(first)
	return nil
}

func (m *gemma3nModel) KV(t *Tokenizer) KV {
	kv := m.ModelParameters.KV(t)
	kv["general.architecture"] = "gemma3n"
	kv["tokenizer.ggml.model"] = "llama"
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
	kv["gemma3n.feed_forward_length"] = uint32(m.TextModel.IntermediateSize)
	kv["gemma3n.head_dim"] = m.TextModel.HeadDim
	kv["gemma3n.rope.freq_base_local"] = m.TextModel.RopeLocalBaseFreq
	kv["gemma3n.rope.freq_base"] = m.TextModel.RopeTheta
	return kv
}

func (m *gemma3nModel) TextKV(t *Tokenizer) KV {
	return m.KV(t)
}

func (m *gemma3nModel) ProjectorKV(*Tokenizer) KV {
	imageSize := m.gemma3nImageSize()
	imageSeqLength := cmp.Or(m.Preprocessor.ImageSeqLength, uint32(256))
	patchSize := imageSize / imageSeqLength
	if patchSize == 0 {
		patchSize = 3
	}

	kv := KV{
		"general.architecture":                     "clip",
		"general.type":                             "mmproj",
		"general.file_type":                        uint32(1),
		"general.quantization_version":             uint32(2),
		"clip.has_vision_encoder":                  true,
		"clip.vision.projector_type":               "gemma3nv",
		"clip.vision.block_count":                  uint32(128),
		"clip.vision.embedding_length":             cmp.Or(m.VisionModel.HiddenSize, uint32(2048)),
		"clip.vision.feed_forward_length":          cmp.Or(m.VisionModel.HiddenSize, uint32(2048)) * 4,
		"clip.vision.attention.head_count":         uint32(8),
		"clip.vision.attention.layer_norm_epsilon": cmp.Or(m.VisionModel.RMSNormEPS, float32(1e-6)),
		"clip.vision.image_size":                   imageSize,
		"clip.vision.patch_size":                   patchSize,
		"clip.vision.projection_dim":               m.TextModel.HiddenSize,
		"clip.vision.image_mean":                   []float32{0, 0, 0},
		"clip.vision.image_std":                    []float32{1, 1, 1},
	}

	if m.AudioModel.HiddenSize > 0 {
		kv["clip.has_audio_encoder"] = true
		kv["clip.audio.projector_type"] = "gemma3na"
		kv["clip.audio.projection_dim"] = m.TextModel.HiddenSize
		kv["clip.audio.embedding_length"] = m.AudioModel.HiddenSize
		kv["clip.audio.feed_forward_length"] = cmp.Or(m.AudioModel.IntermediateSize, m.AudioModel.HiddenSize*4, uint32(6144))
		kv["clip.audio.block_count"] = cmp.Or(m.AudioModel.ConfNumHiddenLayers, uint32(12))
		kv["clip.audio.attention.head_count"] = cmp.Or(m.AudioModel.ConfNumAttentionHeads, uint32(8))
		kv["clip.audio.num_mel_bins"] = cmp.Or(m.AudioModel.InputFeatSize, uint32(128))
		kv["clip.audio.attention.layer_norm_epsilon"] = float32(1e-5)
	}

	return kv
}

func (m *gemma3nModel) gemma3nImageSize() uint32 {
	if m.Preprocessor.Size.Height > 0 {
		return m.Preprocessor.Size.Height
	}
	if m.Preprocessor.Size.Width > 0 {
		return m.Preprocessor.Size.Width
	}
	return 768
}

func (m *gemma3nModel) Tensors(ts []Tensor) []*ggml.Tensor {
	out, ts := mergeTensors(ts,
		merge{"altup_proj.*.weight", "altup_proj.weight"},
		merge{"altup_unembd_proj.*.weight", "altup_unembd_proj.weight"},
	)

	for _, t := range ts {
		switch {
		case gemma3nProjectorTensor(t.Name()):
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

		shape := slices.Clone(t.Shape())
		if m.TextModel.VocabSize > 0 && (t.Name() == "token_embd.weight" || t.Name() == "per_layer_token_embd.weight") && len(shape) > 0 && shape[0] < uint64(m.TextModel.VocabSize) {
			t.SetRepacker(padTensorRows(m.TextModel.VocabSize))
			shape[0] = uint64(m.TextModel.VocabSize)
		}

		out = append(out, &ggml.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    shape,
			WriterTo: t,
		})
	}

	return out
}

func (m *gemma3nModel) TextTensors(ts []Tensor, _ *Tokenizer) []*ggml.Tensor {
	return m.Tensors(ts)
}

func (m *gemma3nModel) ProjectorTensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	for _, t := range ts {
		name, ok := gemma3nProjectorTensorName(t.Name())
		if !ok {
			continue
		}

		shape := slices.Clone(t.Shape())
		var repacker Repacker
		switch {
		case (name == "v.conv_stem.conv.bias" || strings.HasSuffix(name, ".layer_scale.gamma")) && len(shape) == 1:
			shape = []uint64{1, shape[0], 1, 1}
		case strings.Contains(name, ".conv_dw.") && strings.HasSuffix(name, ".weight") && len(shape) == 3:
			shape = []uint64{shape[0], shape[2]}
			repacker = squeezeMiddleDim
		}

		kind := t.Kind()
		var writer io.WriterTo = t
		if gemma3nForceF32Tensor(t.Name(), name) || len(t.Shape()) <= 1 || !strings.HasSuffix(name, ".weight") {
			kind = tensorKindFP32
			writer = tensorFloat32Writer{tensor: t, repacker: repacker}
		} else if sourceDType(t) == "BF16" || kind == tensorKindBF16 {
			kind = tensorKindFP16
			writer = tensorFloat16Writer{tensor: t, repacker: repacker}
		} else if repacker != nil {
			t.SetRepacker(repacker)
		}

		out = append(out, &ggml.Tensor{
			Name:     name,
			Kind:     kind,
			Shape:    shape,
			WriterTo: writer,
		})
	}

	return out
}

func padTensorRows(rows uint32) Repacker {
	return func(_ string, data []float32, shape []uint64) ([]float32, error) {
		if len(shape) == 0 || shape[0] >= uint64(rows) {
			return data, nil
		}

		rowSize := uint64(1)
		for _, dim := range shape[1:] {
			rowSize *= dim
		}
		expected := shape[0] * rowSize
		if uint64(len(data)) != expected {
			return nil, fmt.Errorf("tensor data size %d, expected %d for shape %v", len(data), expected, shape)
		}

		out := make([]float32, uint64(rows)*rowSize)
		copy(out, data)
		return out, nil
	}
}

func gemma3nProjectorTensor(name string) bool {
	return strings.HasPrefix(name, "model.embed_vision.") ||
		strings.HasPrefix(name, "model.vision_tower.") ||
		strings.HasPrefix(name, "model.embed_audio.") ||
		strings.HasPrefix(name, "model.audio_tower.")
}

func gemma3nProjectorTensorName(name string) (string, bool) {
	switch {
	case strings.HasPrefix(name, "model.embed_vision."):
		return gemma3nProjectorReplace(name, []struct{ from, to string }{
			{"model.embed_vision.embedding_projection", "mm.input_projection"},
			{"model.embed_vision.soft_embedding_norm", "mm.soft_emb_norm"},
			{"model.embed_vision.embedding", "mm.embedding"},
			{"model.embed_vision.hard_embedding_norm", "mm.hard_emb_norm"},
		})
	case strings.HasPrefix(name, "model.vision_tower.timm_model.blocks."):
		return gemma3nVisionBlockTensorName(name)
	case strings.HasPrefix(name, "model.vision_tower."):
		return gemma3nProjectorReplace(name, []struct{ from, to string }{
			{"model.vision_tower.timm_model.conv_stem.conv", "v.conv_stem.conv"},
			{"model.vision_tower.timm_model.conv_stem.bn", "v.conv_stem.bn"},
			{"model.vision_tower.timm_model.msfa.ffn.pw_exp.conv", "v.msfa.ffn.pw_exp.conv"},
			{"model.vision_tower.timm_model.msfa.ffn.pw_exp.bn", "v.msfa.ffn.pw_exp.bn"},
			{"model.vision_tower.timm_model.msfa.ffn.pw_proj.conv", "v.msfa.ffn.pw_proj.conv"},
			{"model.vision_tower.timm_model.msfa.ffn.pw_proj.bn", "v.msfa.ffn.pw_proj.bn"},
			{"model.vision_tower.timm_model.msfa.norm", "v.msfa.norm"},
		})
	case strings.HasPrefix(name, "model.embed_audio."):
		return gemma3nProjectorReplace(name, []struct{ from, to string }{
			{"model.embed_audio.embedding_projection", "mm.a.input_projection"},
			{"model.embed_audio.soft_embedding_norm", "mm.a.soft_emb_norm"},
			{"model.embed_audio.embedding", "mm.a.embedding"},
			{"model.embed_audio.hard_embedding_norm", "mm.a.hard_emb_norm"},
		})
	case strings.HasPrefix(name, "model.audio_tower.subsample_conv_projection."):
		return gemma3nProjectorReplace(name, []struct{ from, to string }{
			{"model.audio_tower.subsample_conv_projection.conv_0.conv", "a.conv1d.0"},
			{"model.audio_tower.subsample_conv_projection.conv_0.norm", "a.conv1d.0.norm"},
			{"model.audio_tower.subsample_conv_projection.conv_1.conv", "a.conv1d.1"},
			{"model.audio_tower.subsample_conv_projection.conv_1.norm", "a.conv1d.1.norm"},
			{"model.audio_tower.subsample_conv_projection.input_proj_linear", "a.pre_encode.out"},
		})
	case strings.HasPrefix(name, "model.audio_tower.conformer."):
		return gemma3nAudioBlockTensorName(name)
	default:
		return "", false
	}
}

func gemma3nProjectorReplace(name string, replacements []struct{ from, to string }) (string, bool) {
	for _, r := range replacements {
		if strings.HasPrefix(name, r.from) {
			return strings.Replace(name, r.from, r.to, 1), true
		}
	}
	return "", false
}

func gemma3nVisionBlockTensorName(name string) (string, bool) {
	parts := strings.Split(name, ".")
	if len(parts) < 7 {
		return "", false
	}
	if _, err := strconv.ParseUint(parts[4], 10, 32); err != nil {
		return "", false
	}
	if _, err := strconv.ParseUint(parts[5], 10, 32); err != nil {
		return "", false
	}

	suffix := strings.Join(parts[6:], ".")
	switch suffix {
	case "conv_exp.weight", "bn1.weight", "conv_pwl.weight", "bn2.weight",
		"dw_start.conv.weight", "dw_start.bn.weight", "dw_mid.conv.weight", "dw_mid.bn.weight",
		"pw_exp.conv.weight", "pw_exp.bn.weight", "pw_proj.conv.weight", "pw_proj.bn.weight",
		"layer_scale.gamma", "attn.query.proj.weight", "attn.key.proj.weight",
		"attn.value.proj.weight", "attn.output.proj.weight", "attn.key.down_conv.weight",
		"attn.key.norm.weight", "attn.value.down_conv.weight", "attn.value.norm.weight",
		"norm.weight":
		return fmt.Sprintf("v.blk.%s.%s.%s", parts[4], parts[5], suffix), true
	default:
		return "", false
	}
}

func gemma3nAudioBlockTensorName(name string) (string, bool) {
	rest := strings.TrimPrefix(name, "model.audio_tower.conformer.")
	parts := strings.SplitN(rest, ".", 2)
	if len(parts) != 2 {
		return "", false
	}
	if _, err := strconv.ParseUint(parts[0], 10, 32); err != nil {
		return "", false
	}

	replacements := []struct{ from, to string }{
		{"attention.attn.q_proj", "attn_q"},
		{"attention.attn.k_proj", "attn_k"},
		{"attention.attn.v_proj", "attn_v"},
		{"attention.attn.relative_position_embedding.pos_proj", "linear_pos"},
		{"attention.attn.per_dim_scale", "per_dim_scale"},
		{"attention.pre_attn_norm", "ln1"},
		{"attention.post_norm", "ln2"},
		{"attention.post", "attn_out"},
		{"ffw_layer_start.pre_layer_norm", "ffn_norm"},
		{"ffw_layer_start.post_layer_norm", "ffn_post_norm"},
		{"ffw_layer_start.ffw_layer_1", "ffn_up"},
		{"ffw_layer_start.ffw_layer_2", "ffn_down"},
		{"ffw_layer_end.pre_layer_norm", "ffn_norm_1"},
		{"ffw_layer_end.post_layer_norm", "ffn_post_norm_1"},
		{"ffw_layer_end.ffw_layer_1", "ffn_up_1"},
		{"ffw_layer_end.ffw_layer_2", "ffn_down_1"},
		{"lconv1d.depthwise_conv1d", "conv_dw"},
		{"lconv1d.pre_layer_norm", "conv_norm"},
		{"lconv1d.linear_start", "conv_pw1"},
		{"lconv1d.linear_end", "conv_pw2"},
		{"lconv1d.conv_norm", "norm_conv"},
		{"norm", "layer_pre_norm"},
	}
	for _, r := range replacements {
		if strings.HasPrefix(parts[1], r.from) {
			return fmt.Sprintf("a.blk.%s.%s", parts[0], strings.Replace(parts[1], r.from, r.to, 1)), true
		}
	}
	return "", false
}

func gemma3nForceF32Tensor(sourceName, mappedName string) bool {
	if strings.Contains(sourceName, "conv_stem") || strings.Contains(mappedName, ".conv_dw.") {
		return true
	}
	return strings.HasPrefix(sourceName, "model.audio_tower.") &&
		strings.Contains(sourceName, "conv") &&
		strings.HasSuffix(sourceName, ".weight")
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

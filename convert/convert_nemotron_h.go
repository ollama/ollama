package convert

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type hybridPattern string

func (p *hybridPattern) UnmarshalJSON(data []byte) error {
	if string(data) == "null" {
		*p = ""
		return nil
	}

	var single string
	if err := json.Unmarshal(data, &single); err == nil {
		*p = hybridPattern(strings.TrimSpace(single))
		return nil
	}

	var parts []string
	if err := json.Unmarshal(data, &parts); err == nil {
		*p = hybridPattern(strings.Join(parts, ""))
		return nil
	}

	return fmt.Errorf("hybrid_override_pattern must be a string or string array")
}

type nemotronHModel struct {
	ModelParameters
	MaxPositionEmbeddings uint32        `json:"max_position_embeddings"`
	HiddenSize            uint32        `json:"hidden_size"`
	NumHiddenLayers       uint32        `json:"num_hidden_layers"`
	NumAttentionHeads     uint32        `json:"num_attention_heads"`
	NumKeyValueHeads      uint32        `json:"num_key_value_heads"`
	HeadDim               uint32        `json:"head_dim"`
	LayerNormEpsilon      float32       `json:"layer_norm_epsilon"`
	NormEpsilon           float32       `json:"norm_eps"`
	RopeTheta             float32       `json:"rope_theta"`
	PartialRotaryFactor   float32       `json:"partial_rotary_factor"`
	ConvKernel            uint32        `json:"conv_kernel"`
	SSMStateSize          uint32        `json:"ssm_state_size"`
	MambaNumHeads         uint32        `json:"mamba_num_heads"`
	MambaHeadDim          uint32        `json:"mamba_head_dim"`
	NGroups               uint32        `json:"n_groups"`
	IntermediateSize      uint32        `json:"intermediate_size"`
	HybridOverridePattern hybridPattern `json:"hybrid_override_pattern"`

	// MoE
	NumExperts                  uint32  `json:"num_experts"`
	NumSharedExperts            uint32  `json:"num_shared_experts"`
	NRoutedExperts              uint32  `json:"n_routed_experts"`
	NSharedExperts              uint32  `json:"n_shared_experts"`
	NumExpertsPerTok            uint32  `json:"num_experts_per_tok"`
	MoEIntermediateSize         uint32  `json:"moe_intermediate_size"`
	MoESharedExpertIntermediate uint32  `json:"moe_shared_expert_intermediate_size"`
	NormTopKProb                bool    `json:"norm_topk_prob"`
	RoutedScalingFactor         float32 `json:"routed_scaling_factor"`
	ExpertGroupCount            uint32  `json:"n_group"`
	ExpertGroupUsedCount        uint32  `json:"topk_group"`
}

type nemotronHNanoVLModel struct {
	ModelParameters
	MaxSequenceLength   uint32         `json:"max_sequence_length"`
	ForceImageSize      uint32         `json:"force_image_size"`
	DownsampleRatio     float32        `json:"downsample_ratio"`
	PatchSize           uint32         `json:"patch_size"`
	UseThumbnail        *bool          `json:"use_thumbnail"`
	ImgContextTokenID   uint32         `json:"img_context_token_id"`
	ImgContextToken     string         `json:"img_context_token"`
	ImgStartToken       string         `json:"img_start_token"`
	ImgEndToken         string         `json:"img_end_token"`
	VitHiddenSize       uint32         `json:"vit_hidden_size"`
	ProjectorHidden     uint32         `json:"projector_hidden_size"`
	SoundContextTokenID uint32         `json:"sound_context_token_id"`
	SoundContextToken   string         `json:"sound_context_token"`
	NormMean            []float32      `json:"norm_mean"`
	NormStd             []float32      `json:"norm_std"`
	VisionConfig        radioConfig    `json:"vision_config"`
	SoundConfig         soundConfig    `json:"sound_config"`
	LLMConfig           nemotronHModel `json:"llm_config"`
	Preprocessor        struct {
		ImageSize       uint32    `json:"image_size"`
		PatchSize       uint32    `json:"patch_size"`
		DownsampleRatio float32   `json:"downsample_ratio"`
		MaxNumTiles     uint32    `json:"max_num_tiles"`
		UseThumbnail    *bool     `json:"use_thumbnail"`
		NormMean        []float32 `json:"norm_mean"`
		NormStd         []float32 `json:"norm_std"`
	}
}

type soundConfig struct {
	ModelType                 string `json:"model_type"`
	HiddenSize                uint32 `json:"hidden_size"`
	NumAttentionHeads         uint32 `json:"num_attention_heads"`
	NumHiddenLayers           uint32 `json:"num_hidden_layers"`
	IntermediateSize          uint32 `json:"intermediate_size"`
	ConvKernelSize            uint32 `json:"conv_kernel_size"`
	SubsamplingConvChannels   uint32 `json:"subsampling_conv_channels"`
	SubsamplingConvKernelSize uint32 `json:"subsampling_conv_kernel_size"`
	SubsamplingConvStride     uint32 `json:"subsampling_conv_stride"`
	SubsamplingFactor         uint32 `json:"subsampling_factor"`
	NumMelBins                uint32 `json:"num_mel_bins"`
	ProjectionHiddenSize      uint32 `json:"projection_hidden_size"`
	SamplingRate              uint32 `json:"sampling_rate"`
	ScaleInput                bool   `json:"scale_input"`
}

type radioConfig struct {
	Version               string `json:"version"`
	PatchSize             uint32 `json:"patch_size"`
	MaxResolution         uint32 `json:"max_resolution"`
	MinNumPatches         uint32 `json:"min_num_patches"`
	MaxNumPatches         uint32 `json:"max_num_patches"`
	SeparateVideoEmbedder bool   `json:"separate_video_embedder"`
	Args                  struct {
		MinNumPatches uint32 `json:"min_num_patches"`
		MaxNumPatches uint32 `json:"max_num_patches"`
	} `json:"args"`
}

var _ ModelConverter = (*nemotronHModel)(nil)
var _ ModelConverter = (*nemotronHNanoVLModel)(nil)

func (n *nemotronHNanoVLModel) parseMore(fsys fs.FS) error {
	if n.MaxSequenceLength > 0 {
		n.LLMConfig.MaxPositionEmbeddings = n.MaxSequenceLength
	}

	if err := n.LLMConfig.parseMore(fsys); err != nil {
		return err
	}

	if bts, err := fs.ReadFile(fsys, "preprocessor_config.json"); err == nil {
		if err := json.Unmarshal(bts, &n.Preprocessor); err != nil {
			return fmt.Errorf("nemotron_h_omni: parse preprocessor_config.json: %w", err)
		}
	} else if !errors.Is(err, fs.ErrNotExist) {
		return err
	}

	if version := strings.TrimSpace(n.VisionConfig.Version); version != "" && version != "radio_v2.5-h" {
		return fmt.Errorf("nemotron_h_omni: unsupported RADIO version %q", version)
	}
	if patchSize := n.visionPatchSize(); patchSize != 16 {
		return fmt.Errorf("nemotron_h_omni: unsupported vision patch_size=%d", patchSize)
	}
	if scale := n.visionProjectorScaleFactor(); scale != 2 {
		return fmt.Errorf("nemotron_h_omni: unsupported vision projector scale factor=%d", scale)
	}

	if n.SoundConfig.NumHiddenLayers > 0 {
		if modelType := strings.TrimSpace(n.SoundConfig.ModelType); modelType != "" && modelType != "parakeet" {
			return fmt.Errorf("nemotron_h_omni: unsupported sound model_type %q", modelType)
		}
		if n.soundHiddenSize() == 0 {
			return fmt.Errorf("nemotron_h_omni: sound hidden_size must be set")
		}
		if n.soundAttentionHeads() == 0 {
			return fmt.Errorf("nemotron_h_omni: sound num_attention_heads must be set")
		}
		if n.soundSubsamplingFactor() != 8 {
			return fmt.Errorf("nemotron_h_omni: unsupported sound subsampling_factor=%d", n.soundSubsamplingFactor())
		}
		if n.soundMelBins() != 128 {
			return fmt.Errorf("nemotron_h_omni: unsupported sound num_mel_bins=%d", n.soundMelBins())
		}
	}

	return nil
}

func (n *nemotronHNanoVLModel) KV(t *Tokenizer) KV {
	kv := n.LLMConfig.KV(t)
	kv["general.architecture"] = "nemotron_h_omni"

	kv["vision.block_count"] = n.visionBlockCount()
	kv["vision.embedding_length"] = n.visionEmbeddingLength()
	kv["vision.feed_forward_length"] = n.visionFeedForwardLength()
	kv["vision.attention.head_count"] = n.visionAttentionHeads()
	kv["vision.attention.layer_norm_epsilon"] = float32(1e-6)
	kv["vision.patch_size"] = n.visionPatchSize()
	kv["vision.image_size"] = n.visionImageSize()
	kv["vision.max_tiles"] = n.visionMaxTiles()
	kv["vision.use_thumbnail"] = n.visionUseThumbnail()
	if minPatches := n.visionMinNumPatches(); minPatches > 0 {
		kv["vision.min_num_patches"] = minPatches
	}
	if maxPatches := n.visionMaxNumPatches(); maxPatches > 0 {
		kv["vision.max_num_patches"] = maxPatches
	}
	kv["vision.num_channels"] = uint32(3)
	kv["vision.image_mean"] = slices.Clone(defaultFloat32Slice(n.visionMean(), imageNetStandardMean))
	kv["vision.image_std"] = slices.Clone(defaultFloat32Slice(n.visionStd(), imageNetStandardSTD))
	kv["vision.projector.scale_factor"] = n.visionProjectorScaleFactor()

	setTokenID := func(key string, explicit uint32, token string) {
		if explicit > 0 {
			kv[key] = explicit
			return
		}
		if t == nil || t.Vocabulary == nil {
			return
		}
		for i, v := range t.Vocabulary.Tokens {
			if v == token {
				kv[key] = uint32(i)
				return
			}
		}
	}

	setTokenID("vision.image_token_id", n.ImgContextTokenID, cmp.Or(n.ImgContextToken, "<image>"))
	setTokenID("vision.image_start_token_id", 0, cmp.Or(n.ImgStartToken, "<img>"))
	setTokenID("vision.image_end_token_id", 0, cmp.Or(n.ImgEndToken, "</img>"))

	if n.SoundConfig.NumHiddenLayers > 0 {
		kv["audio.block_count"] = n.SoundConfig.NumHiddenLayers
		kv["audio.embedding_length"] = n.soundHiddenSize()
		kv["audio.feed_forward_length"] = n.soundFeedForwardLength()
		kv["audio.attention.head_count"] = n.soundAttentionHeads()
		kv["audio.attention.layer_norm_epsilon"] = float32(1e-5)
		kv["audio.conv_kernel_size"] = n.soundConvKernelSize()
		kv["audio.num_mel_bins"] = n.soundMelBins()
		kv["audio.sample_rate"] = n.soundSampleRate()
		kv["audio.subsampling_factor"] = n.soundSubsamplingFactor()
		kv["audio.subsampling_conv_channels"] = n.soundSubsamplingConvChannels()
		kv["audio.subsampling_conv_kernel_size"] = n.soundSubsamplingConvKernelSize()
		kv["audio.subsampling_conv_stride"] = n.soundSubsamplingConvStride()
		kv["audio.projection_hidden_size"] = n.soundProjectionHiddenSize()
		kv["audio.scale_input"] = n.SoundConfig.ScaleInput
		setTokenID("audio.sound_token_id", n.SoundContextTokenID, cmp.Or(n.SoundContextToken, "<so_embedding>"))
	}

	return kv
}

func (n *nemotronHNanoVLModel) Tensors(ts []Tensor) []*ggml.Tensor {
	var textTensors []Tensor
	var out []*ggml.Tensor

	for _, t := range ts {
		switch {
		case isNemotronHNanoVLOmittedTensor(t.Name()):
			continue
		case strings.Contains(t.Name(), ".attn_qkv"):
			out = append(out, slices.Collect(splitDim(t, 0,
				split{Replacer: strings.NewReplacer("attn_qkv", "attn_q")},
				split{Replacer: strings.NewReplacer("attn_qkv", "attn_k")},
				split{Replacer: strings.NewReplacer("attn_qkv", "attn_v")},
			))...)
		case t.Name() == "v.position_embd":
			shape := t.Shape()
			if len(shape) == 3 && shape[0] == 1 {
				shape = shape[1:]
			}
			out = append(out, &ggml.Tensor{
				Name:     t.Name(),
				Kind:     t.Kind(),
				Shape:    shape,
				WriterTo: t,
			})
		case strings.HasPrefix(t.Name(), "a.") || strings.HasPrefix(t.Name(), "v.") || strings.HasPrefix(t.Name(), "mm."):
			name := t.Name()
			shape := slices.Clone(t.Shape())
			if strings.HasPrefix(name, "a.blk.") && strings.Contains(name, ".conv_dw.") && strings.HasSuffix(name, ".weight") && len(shape) == 3 {
				t.SetRepacker(squeezeMiddleDim)
				shape = []uint64{shape[0], shape[2]}
			}
			if strings.HasPrefix(name, "a.blk.") && (strings.Contains(name, ".conv_pw1.") || strings.Contains(name, ".conv_pw2.")) && strings.HasSuffix(name, ".weight") && len(shape) == 3 && shape[2] == 1 {
				t.SetRepacker(squeezeLastDim)
				shape = shape[:2]
			}
			out = append(out, &ggml.Tensor{
				Name:     name,
				Kind:     t.Kind(),
				Shape:    shape,
				WriterTo: t,
			})
		default:
			textTensors = append(textTensors, t)
		}
	}

	return append(n.LLMConfig.Tensors(textTensors), out...)
}

func (n *nemotronHNanoVLModel) Replacements() []string {
	return append([]string{
		"language_model.", "",
		"vision_model.radio_model.model.patch_generator.embedder", "v.patch_embd",
		"vision_model.radio_model.model.patch_generator.pos_embed", "v.position_embd",
		"vision_model.radio_model.model.patch_generator.cls_token.token", "v.cls_embd",
		"vision_model.radio_model.model.blocks", "v.blk",
		"attn.qkv", "attn_qkv",
		"attn.proj", "attn_out",
		"mlp.fc1", "ffn_up",
		"mlp.fc2", "ffn_down",
		"norm1", "ln1",
		"norm2", "ln2",
		"mlp1.0", "mm.norm",
		"mlp1.1", "mm.1",
		"mlp1.3", "mm.2",
		"sound_encoder.encoder.feature_extractor.featurizer.fb", "a.feature_extractor.fb",
		"sound_encoder.encoder.feature_extractor.featurizer.window", "a.feature_extractor.window",
		"sound_encoder.encoder.subsampling.layers.0", "a.subsampling.conv0",
		"sound_encoder.encoder.subsampling.layers.2", "a.subsampling.dw1",
		"sound_encoder.encoder.subsampling.layers.3", "a.subsampling.pw1",
		"sound_encoder.encoder.subsampling.layers.5", "a.subsampling.dw2",
		"sound_encoder.encoder.subsampling.layers.6", "a.subsampling.pw2",
		"sound_encoder.encoder.subsampling.linear", "a.subsampling.linear",
		"sound_encoder.encoder.layers", "a.blk",
		"feed_forward1.linear1", "ffn1_up",
		"feed_forward1.linear2", "ffn1_down",
		"feed_forward2.linear1", "ffn2_up",
		"feed_forward2.linear2", "ffn2_down",
		"norm_feed_forward1", "ffn1_norm",
		"norm_feed_forward2", "ffn2_norm",
		"norm_self_att", "attn_norm",
		"norm_conv", "conv_norm",
		"norm_out", "out_norm",
		"self_attn.q_proj", "attn_q",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_out",
		"self_attn.relative_k_proj", "attn_rel_k",
		"self_attn.bias_u", "attn_bias_u",
		"self_attn.bias_v", "attn_bias_v",
		"conv.pointwise_conv1", "conv_pw1",
		"conv.pointwise_conv2", "conv_pw2",
		"conv.depthwise_conv", "conv_dw",
		"conv.norm", "conv_bn",
		"sound_projection.norm", "mm.a.norm",
		"sound_projection.linear1", "mm.a.1",
		"sound_projection.linear2", "mm.a.2",
	}, n.LLMConfig.Replacements()...)
}

func (n *nemotronHNanoVLModel) specialTokenTypes() []string {
	return n.LLMConfig.specialTokenTypes()
}

func isNemotronHNanoVLOmittedTensor(name string) bool {
	return strings.HasSuffix(name, ".conv_bn.num_batches_tracked") ||
		strings.HasPrefix(name, "vision_model.radio_model.input_conditioner.") ||
		strings.HasPrefix(name, "vision_model.radio_model.model.patch_generator.video_embedder")
}

func squeezeLastDim(_ string, data []float32, _ []uint64) ([]float32, error) {
	return data, nil
}

func (n *nemotronHNanoVLModel) visionImageSize() uint32 {
	return cmp.Or(n.ForceImageSize, n.Preprocessor.ImageSize, uint32(512))
}

func (n *nemotronHNanoVLModel) visionPatchSize() uint32 {
	return cmp.Or(n.PatchSize, n.Preprocessor.PatchSize, n.VisionConfig.PatchSize, uint32(16))
}

func (n *nemotronHNanoVLModel) visionProjectorScaleFactor() uint32 {
	ratio := cmp.Or(n.DownsampleRatio, n.Preprocessor.DownsampleRatio, float32(0.5))
	if ratio <= 0 {
		return 2
	}

	return max(uint32(1), uint32(math.Round(1.0/float64(ratio))))
}

func (n *nemotronHNanoVLModel) visionBlockCount() uint32 {
	return 32
}

func (n *nemotronHNanoVLModel) visionEmbeddingLength() uint32 {
	return cmp.Or(n.VitHiddenSize, uint32(1280))
}

func (n *nemotronHNanoVLModel) visionAttentionHeads() uint32 {
	return 16
}

func (n *nemotronHNanoVLModel) visionFeedForwardLength() uint32 {
	return 4 * n.visionEmbeddingLength()
}

func (n *nemotronHNanoVLModel) visionMaxTiles() uint32 {
	return cmp.Or(n.Preprocessor.MaxNumTiles, uint32(12))
}

func (n *nemotronHNanoVLModel) visionMinNumPatches() uint32 {
	return cmp.Or(n.VisionConfig.MinNumPatches, n.VisionConfig.Args.MinNumPatches)
}

func (n *nemotronHNanoVLModel) visionMaxNumPatches() uint32 {
	return cmp.Or(n.VisionConfig.MaxNumPatches, n.VisionConfig.Args.MaxNumPatches)
}

func (n *nemotronHNanoVLModel) visionUseThumbnail() bool {
	for _, v := range []*bool{n.UseThumbnail, n.Preprocessor.UseThumbnail} {
		if v != nil {
			return *v
		}
	}

	return true
}

func (n *nemotronHNanoVLModel) visionMean() []float32 {
	if len(n.NormMean) > 0 {
		return n.NormMean
	}
	return n.Preprocessor.NormMean
}

func (n *nemotronHNanoVLModel) visionStd() []float32 {
	if len(n.NormStd) > 0 {
		return n.NormStd
	}
	return n.Preprocessor.NormStd
}

func (n *nemotronHNanoVLModel) soundHiddenSize() uint32 {
	return cmp.Or(n.SoundConfig.HiddenSize, uint32(1024))
}

func (n *nemotronHNanoVLModel) soundAttentionHeads() uint32 {
	return cmp.Or(n.SoundConfig.NumAttentionHeads, uint32(8))
}

func (n *nemotronHNanoVLModel) soundFeedForwardLength() uint32 {
	return cmp.Or(n.SoundConfig.IntermediateSize, 4*n.soundHiddenSize())
}

func (n *nemotronHNanoVLModel) soundConvKernelSize() uint32 {
	return cmp.Or(n.SoundConfig.ConvKernelSize, uint32(9))
}

func (n *nemotronHNanoVLModel) soundMelBins() uint32 {
	return cmp.Or(n.SoundConfig.NumMelBins, uint32(128))
}

func (n *nemotronHNanoVLModel) soundSampleRate() uint32 {
	return cmp.Or(n.SoundConfig.SamplingRate, uint32(16000))
}

func (n *nemotronHNanoVLModel) soundSubsamplingFactor() uint32 {
	return cmp.Or(n.SoundConfig.SubsamplingFactor, uint32(8))
}

func (n *nemotronHNanoVLModel) soundSubsamplingConvChannels() uint32 {
	return cmp.Or(n.SoundConfig.SubsamplingConvChannels, uint32(256))
}

func (n *nemotronHNanoVLModel) soundSubsamplingConvKernelSize() uint32 {
	return cmp.Or(n.SoundConfig.SubsamplingConvKernelSize, uint32(3))
}

func (n *nemotronHNanoVLModel) soundSubsamplingConvStride() uint32 {
	return cmp.Or(n.SoundConfig.SubsamplingConvStride, uint32(2))
}

func (n *nemotronHNanoVLModel) soundProjectionHiddenSize() uint32 {
	return cmp.Or(n.SoundConfig.ProjectionHiddenSize, uint32(4096))
}

var (
	imageNetStandardMean = []float32{0.48145466, 0.4578275, 0.40821073}
	imageNetStandardSTD  = []float32{0.26862954, 0.26130258, 0.27577711}
)

func (n *nemotronHModel) parseMore(_ fs.FS) error {
	if n.NumHiddenLayers == 0 {
		return fmt.Errorf("nemotron_h: num_hidden_layers must be set")
	}
	if n.HiddenSize == 0 {
		return fmt.Errorf("nemotron_h: hidden_size must be set")
	}
	if n.NumAttentionHeads == 0 {
		return fmt.Errorf("nemotron_h: num_attention_heads must be set")
	}
	if n.HeadDim == 0 {
		if n.HiddenSize%n.NumAttentionHeads != 0 {
			return fmt.Errorf("nemotron_h: hidden_size (%d) must be divisible by num_attention_heads (%d)", n.HiddenSize, n.NumAttentionHeads)
		}
		n.HeadDim = n.HiddenSize / n.NumAttentionHeads
	}
	if n.NumKeyValueHeads == 0 {
		n.NumKeyValueHeads = n.NumAttentionHeads
	}
	if n.ConvKernel == 0 {
		return fmt.Errorf("nemotron_h: conv_kernel must be set")
	}
	if n.SSMStateSize == 0 {
		return fmt.Errorf("nemotron_h: ssm_state_size must be set")
	}
	if n.ssmHeadCount() == 0 {
		return fmt.Errorf("nemotron_h: mamba_num_heads must be set")
	}
	if n.MambaHeadDim == 0 {
		return fmt.Errorf("nemotron_h: mamba_head_dim must be set")
	}
	if n.NGroups == 0 {
		n.NGroups = 1
	}

	if _, _, err := n.layerArrays(); err != nil {
		return err
	}

	if n.isMoE() {
		if n.routedExpertCount() == 0 {
			return fmt.Errorf("nemotron_h: routed expert count must be set for MoE models")
		}
		if n.NumExpertsPerTok == 0 {
			return fmt.Errorf("nemotron_h: num_experts_per_tok must be set for MoE models")
		}
		if n.NumExpertsPerTok > n.routedExpertCount() {
			return fmt.Errorf("nemotron_h: num_experts_per_tok (%d) cannot exceed expert_count (%d)", n.NumExpertsPerTok, n.routedExpertCount())
		}
		if n.moeIntermediateSize() == 0 {
			return fmt.Errorf("nemotron_h: moe_intermediate_size must be set for MoE models")
		}
	}

	return nil
}

func (n *nemotronHModel) isMoE() bool {
	return cmp.Or(n.routedExpertCount(), n.NumExpertsPerTok, n.MoEIntermediateSize) > 0
}

func (n *nemotronHModel) routedExpertCount() uint32 {
	return cmp.Or(n.NRoutedExperts, n.NumExperts)
}

func (n *nemotronHModel) sharedExpertCount() uint32 {
	return cmp.Or(n.NSharedExperts, n.NumSharedExperts)
}

func (n *nemotronHModel) ssmHeadCount() uint32 {
	return n.MambaNumHeads
}

func (n *nemotronHModel) ssmInnerSize() uint32 {
	return n.MambaHeadDim * n.ssmHeadCount()
}

func (n *nemotronHModel) epsilon() float32 {
	return cmp.Or(n.NormEpsilon, n.LayerNormEpsilon, float32(1e-5))
}

func (n *nemotronHModel) moeIntermediateSize() uint32 {
	return cmp.Or(n.MoEIntermediateSize, n.IntermediateSize)
}

func (n *nemotronHModel) denseIntermediateSize() uint32 {
	return cmp.Or(n.IntermediateSize, n.MoEIntermediateSize)
}

func (n *nemotronHModel) layerArrays() (headCountKV []uint32, ffnLengths []uint32, err error) {
	pattern := strings.TrimSpace(string(n.HybridOverridePattern))
	if pattern == "" {
		return nil, nil, fmt.Errorf("nemotron_h: hybrid_override_pattern must be set")
	}

	runes := []rune(pattern)
	if len(runes) != int(n.NumHiddenLayers) {
		return nil, nil, fmt.Errorf("nemotron_h: hybrid_override_pattern length (%d) must match num_hidden_layers (%d)", len(runes), n.NumHiddenLayers)
	}

	headCountKV = make([]uint32, n.NumHiddenLayers)
	ffnLengths = make([]uint32, n.NumHiddenLayers)

	attnKVHeads := cmp.Or(n.NumKeyValueHeads, n.NumAttentionHeads)
	moeFFN := n.moeIntermediateSize()
	denseFFN := n.denseIntermediateSize()

	for i, layerType := range runes {
		switch layerType {
		case 'M':
			// Recurrent layer: no KV heads and no FFN.
		case '*', 'A':
			// Attention-only layer.
			headCountKV[i] = attnKVHeads
		case 'E':
			// MoE layer.
			if moeFFN == 0 {
				return nil, nil, fmt.Errorf("nemotron_h: moe layer at index %d but moe_intermediate_size is zero", i)
			}
			ffnLengths[i] = moeFFN
		case '-':
			// Dense FFN layer.
			if denseFFN == 0 {
				return nil, nil, fmt.Errorf("nemotron_h: dense FFN layer at index %d but intermediate_size is zero", i)
			}
			ffnLengths[i] = denseFFN
		default:
			return nil, nil, fmt.Errorf("nemotron_h: unsupported layer type %q in hybrid_override_pattern at index %d", layerType, i)
		}
	}

	return headCountKV, ffnLengths, nil
}

func (n *nemotronHModel) KV(t *Tokenizer) KV {
	kv := n.ModelParameters.KV(t)

	arch := "nemotron_h"
	if n.isMoE() {
		arch = "nemotron_h_moe"
	}
	kv["general.architecture"] = arch
	kv["block_count"] = n.NumHiddenLayers
	kv["context_length"] = n.MaxPositionEmbeddings
	kv["embedding_length"] = n.HiddenSize
	kv["attention.head_count"] = n.NumAttentionHeads
	kv["attention.key_length"] = n.HeadDim
	kv["attention.value_length"] = n.HeadDim
	kv["attention.layer_norm_epsilon"] = n.epsilon()
	kv["attention.layer_norm_rms_epsilon"] = n.epsilon()
	kv["rope.freq_base"] = cmp.Or(n.RopeTheta, float32(10000))
	if n.PartialRotaryFactor > 0 && n.PartialRotaryFactor <= 1 {
		kv["rope.dimension_count"] = uint32(float32(n.HeadDim) * n.PartialRotaryFactor)
	}

	if headCountKV, ffnLengths, err := n.layerArrays(); err == nil {
		kv["attention.head_count_kv"] = headCountKV
		kv["feed_forward_length"] = ffnLengths
	}

	kv["ssm.conv_kernel"] = n.ConvKernel
	kv["ssm.inner_size"] = n.ssmInnerSize()
	kv["ssm.state_size"] = n.SSMStateSize
	kv["ssm.group_count"] = n.NGroups
	kv["ssm.time_step_rank"] = n.ssmHeadCount()

	if n.isMoE() {
		kv["expert_count"] = n.routedExpertCount()
		kv["expert_used_count"] = n.NumExpertsPerTok
		kv["expert_feed_forward_length"] = n.moeIntermediateSize()
		if n.sharedExpertCount() > 0 {
			kv["expert_shared_count"] = n.sharedExpertCount()
		}
		if n.MoESharedExpertIntermediate > 0 {
			kv["expert_shared_feed_forward_length"] = n.MoESharedExpertIntermediate
		}
		kv["expert_weights_norm"] = n.NormTopKProb
		kv["expert_weights_scale"] = n.RoutedScalingFactor
		if n.ExpertGroupCount > 0 {
			kv["expert_group_count"] = n.ExpertGroupCount
		}
		if n.ExpertGroupUsedCount > 0 {
			kv["expert_group_used_count"] = n.ExpertGroupUsedCount
		}
	}

	return kv
}

func normalizeVectorShapeToColumn(shape []uint64) []uint64 {
	switch len(shape) {
	case 1:
		return []uint64{shape[0], 1}
	case 2:
		if shape[0] == 1 && shape[1] > 1 {
			return []uint64{shape[1], 1}
		}
		if shape[1] == 1 && shape[0] > 1 {
			return []uint64{shape[0], 1}
		}
	}

	return slices.Clone(shape)
}

func (n *nemotronHModel) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	remaining := ts
	if n.isMoE() {
		merges := make([]merge, 0, n.NumHiddenLayers*2)
		for i := range n.NumHiddenLayers {
			merges = append(merges, merge{
				fmt.Sprintf("blk.%d.mixer.experts.*.up_proj.weight", i),
				fmt.Sprintf("blk.%d.ffn_up_exps.weight", i),
			}, merge{
				fmt.Sprintf("blk.%d.mixer.experts.*.down_proj.weight", i),
				fmt.Sprintf("blk.%d.ffn_down_exps.weight", i),
			})
		}

		merged, rest := mergeTensors(ts, merges...)
		out = append(out, merged...)
		remaining = rest
	}

	nGroups := uint64(cmp.Or(n.NGroups, uint32(1)))
	for _, t := range remaining {
		name := t.Name()
		shape := slices.Clone(t.Shape())

		switch {
		case strings.HasSuffix(name, ".ssm_a"):
			shape = normalizeVectorShapeToColumn(shape)
			t.SetRepacker(func(_ string, data []float32, _ []uint64) ([]float32, error) {
				out := make([]float32, len(data))
				for i, v := range data {
					out[i] = -float32(math.Exp(float64(v)))
				}
				return out, nil
			})
		case strings.HasSuffix(name, ".ssm_d"):
			shape = normalizeVectorShapeToColumn(shape)
		case strings.HasSuffix(name, ".ssm_norm.weight"):
			switch len(shape) {
			case 1:
				if nGroups > 0 && shape[0]%nGroups == 0 {
					shape = []uint64{nGroups, shape[0] / nGroups}
				}
			case 2:
				if shape[0] == 1 && nGroups > 0 && shape[1]%nGroups == 0 {
					shape = []uint64{nGroups, shape[1] / nGroups}
				}
			}
		case strings.HasSuffix(name, ".ssm_conv1d.weight"):
			if len(shape) == 3 {
				if shape[0] == 1 {
					shape = []uint64{shape[1], shape[2]}
				} else if shape[1] == 1 {
					shape = []uint64{shape[0], shape[2]}
				}
			}
		}

		out = append(out, &ggml.Tensor{
			Name:     name,
			Kind:     t.Kind(),
			Shape:    shape,
			WriterTo: t,
		})
	}

	return out
}

func (n *nemotronHModel) Replacements() []string {
	return []string{
		// Embedding and output
		"lm_head", "output",
		"backbone.embeddings", "token_embd",
		"backbone.norm_f", "output_norm",
		"backbone.layers", "blk",

		// Recurrent (Mamba2) tensors
		"mixer.in_proj", "ssm_in",
		"mixer.out_proj", "ssm_out",
		"mixer.dt_bias", "ssm_dt.bias",
		"mixer.A_log", "ssm_a",
		"mixer.D", "ssm_d",
		"mixer.conv1d", "ssm_conv1d",
		"mixer.norm.weight", "ssm_norm.weight",

		// Attention tensors
		"mixer.q_proj", "attn_q",
		"mixer.k_proj", "attn_k",
		"mixer.v_proj", "attn_v",
		"mixer.o_proj", "attn_output",

		// FFN / MoE tensors
		"mixer.gate.e_score_correction_bias", "exp_probs_b.bias",
		"mixer.gate", "ffn_gate_inp",
		"mixer.fc1_latent_proj", "ffn_latent_in",
		"mixer.fc2_latent_proj", "ffn_latent_out",
		"mixer.shared_experts.up_proj", "ffn_up_shexp",
		"mixer.shared_experts.down_proj", "ffn_down_shexp",
		"mixer.up_proj", "ffn_up",
		"mixer.down_proj", "ffn_down",

		// Per-layer pre-norm
		".norm.weight", ".attn_norm.weight",
	}
}

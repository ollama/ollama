package convert

import (
	"cmp"
	"encoding/json"
	"io/fs"
	"log/slog"
	"regexp"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"
)

// normalToNeoXRepacker creates a repacker that permutes Q/K weights from interleaved (LLaMA)
// to NeoX ordering for compatibility with GGML's M-RoPE kernel.
//
// For weights: reshape [out, in] -> [n_heads, head_dim, in], permute rotary dims, reshape back
// For biases: reshape [out] -> [n_heads, head_dim], permute rotary dims, reshape back
func normalToNeoXRepacker(nHeads, headDim int, partialRotaryFactor float32) func(string, []float32, []uint64) ([]float32, error) {
	return func(_ string, data []float32, shape []uint64) ([]float32, error) {
		rotaryDim := int(float32(headDim) * partialRotaryFactor)
		if rotaryDim%2 != 0 {
			rotaryDim = (rotaryDim / 2) * 2 // Round down to even
		}

		// Handle 1D (bias) or 2D (weight) tensors
		is1D := len(shape) == 1
		var inFeatures int
		if is1D {
			inFeatures = 1
		} else {
			inFeatures = int(shape[1])
		}
		outFeatures := int(shape[0])
		nEffectiveHeads := outFeatures / headDim

		if nEffectiveHeads != nHeads {
			slog.Warn("normalToNeoX: unexpected head count", "effective", nEffectiveHeads, "expected", nHeads)
		}

		// Reshape to [n_heads, head_dim, in_features]
		reshaped := make([]float32, len(data))
		copy(reshaped, data)

		// Permute the rotary dimensions: even indices first, then odd
		// For each head, reorder [0,1,2,3,4,5...] to [0,2,4...,1,3,5...]
		result := make([]float32, len(data))
		halfRotary := rotaryDim / 2

		for h := range nEffectiveHeads {
			for f := range inFeatures {
				for i := range halfRotary {
					// Even dim (0, 2, 4, ...) -> position i
					srcIdx := h*headDim*inFeatures + (2*i)*inFeatures + f
					dstIdx := h*headDim*inFeatures + i*inFeatures + f
					result[dstIdx] = reshaped[srcIdx]

					// Odd dim (1, 3, 5, ...) -> position halfRotary + i
					srcIdx = h*headDim*inFeatures + (2*i+1)*inFeatures + f
					dstIdx = h*headDim*inFeatures + (halfRotary+i)*inFeatures + f
					result[dstIdx] = reshaped[srcIdx]
				}

				// Non-rotary part: copy as-is
				for i := rotaryDim; i < headDim; i++ {
					srcIdx := h*headDim*inFeatures + i*inFeatures + f
					result[srcIdx] = reshaped[srcIdx]
				}
			}
		}

		return result, nil
	}
}

type glmOcrModel struct {
	ModelParameters

	TextConfig struct {
		HiddenSize            uint32  `json:"hidden_size"`
		IntermediateSize      uint32  `json:"intermediate_size"`
		NumHiddenLayers       uint32  `json:"num_hidden_layers"`
		NumAttentionHeads     uint32  `json:"num_attention_heads"`
		NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
		HeadDim               uint32  `json:"head_dim"`
		MaxPositionEmbed      uint32  `json:"max_position_embeddings"`
		RMSNormEps            float32 `json:"rms_norm_eps"`
		PartialRotaryFactor   float32 `json:"partial_rotary_factor"`
		NumNextNPredictLayers uint32  `json:"num_nextn_predict_layers"`
		RopeParameters        struct {
			RopeType            string  `json:"rope_type"`
			MRopeSection        []int32 `json:"mrope_section"`
			RopeTheta           float32 `json:"rope_theta"`
			PartialRotaryFactor float32 `json:"partial_rotary_factor"`
		} `json:"rope_parameters"`
	} `json:"text_config"`

	VisionConfig struct {
		HiddenSize        uint32  `json:"hidden_size"`
		IntermediateSize  uint32  `json:"intermediate_size"`
		Depth             uint32  `json:"depth"`
		NumHeads          uint32  `json:"num_heads"`
		ImageSize         uint32  `json:"image_size"`
		PatchSize         uint32  `json:"patch_size"`
		OutHiddenSize     uint32  `json:"out_hidden_size"`
		RMSNormEps        float32 `json:"rms_norm_eps"`
		SpatialMergeSize  uint32  `json:"spatial_merge_size"`
		TemporalPatchSize uint32  `json:"temporal_patch_size"`
	} `json:"vision_config"`

	ImageStartTokenID uint32 `json:"image_start_token_id"`
	ImageEndTokenID   uint32 `json:"image_end_token_id"`
	VideoStartTokenID uint32 `json:"video_start_token_id"`
	VideoEndTokenID   uint32 `json:"video_end_token_id"`
	ImageTokenID      uint32 `json:"image_token_id"`
	VideoTokenID      uint32 `json:"video_token_id"`

	// Preprocessor config (preprocessor_config.json)
	Preprocessor struct {
		Size struct {
			ShortestEdge uint32 `json:"shortest_edge"`
			LongestEdge  uint32 `json:"longest_edge"`
		} `json:"size"`
		PatchSize         uint32    `json:"patch_size"`
		TemporalPatchSize uint32    `json:"temporal_patch_size"`
		MergeSize         uint32    `json:"merge_size"`
		ImageMean         []float32 `json:"image_mean"`
		ImageStd          []float32 `json:"image_std"`
	} `json:"-"`
}

var _ MultimodalConverter = (*glmOcrModel)(nil)

func (m *glmOcrModel) parseMore(fsys fs.FS) error {
	bts, err := fs.ReadFile(fsys, "preprocessor_config.json")
	if err != nil {
		return err
	}

	return json.Unmarshal(bts, &m.Preprocessor)
}

func (m *glmOcrModel) KV(t *Tokenizer) KV {
	kv := m.ModelParameters.KV(t)
	kv["general.architecture"] = "glm4"

	// Text model parameters — block_count includes NextN layers
	blockCount := m.TextConfig.NumHiddenLayers + m.TextConfig.NumNextNPredictLayers
	kv["block_count"] = cmp.Or(blockCount, 17)
	kv["embedding_length"] = cmp.Or(m.TextConfig.HiddenSize, 1536)
	kv["attention.head_count"] = cmp.Or(m.TextConfig.NumAttentionHeads, 16)
	kv["attention.head_count_kv"] = cmp.Or(m.TextConfig.NumKeyValueHeads, 8)
	headDim := cmp.Or(m.TextConfig.HeadDim, m.TextConfig.HiddenSize/m.TextConfig.NumAttentionHeads)
	kv["attention.key_length"] = headDim
	kv["attention.value_length"] = headDim
	kv["feed_forward_length"] = cmp.Or(m.TextConfig.IntermediateSize, 4608)
	kv["attention.layer_norm_rms_epsilon"] = cmp.Or(m.TextConfig.RMSNormEps, 1e-5)
	kv["context_length"] = cmp.Or(m.TextConfig.MaxPositionEmbed, 131072)
	kv["rope.freq_base"] = cmp.Or(m.TextConfig.RopeParameters.RopeTheta, float32(10000))

	partialRotaryFactor := cmp.Or(m.TextConfig.RopeParameters.PartialRotaryFactor, m.TextConfig.PartialRotaryFactor, float32(0.5))
	kv["rope.dimension_count"] = uint32(float32(headDim) * partialRotaryFactor)

	// LLM_ARCH_GLM4 reads rope dimension sections via
	// LLM_KV_ROPE_DIMENSION_SECTIONS = "%s.rope.dimension_sections" and
	// expects exactly 4 elements (llama-model.cpp:1703 get_key_or_arr
	// passes n=4). HF ships the M-RoPE section as a 3-element list
	// [t, h, w]; pad with a trailing 0 for the unused 4th (channel/time)
	// dimension to match what the loader expects.
	if len(m.TextConfig.RopeParameters.MRopeSection) > 0 {
		sections := append([]int32{}, m.TextConfig.RopeParameters.MRopeSection...)
		for len(sections) < 4 {
			sections = append(sections, 0)
		}
		kv["rope.dimension_sections"] = sections
	}

	if m.TextConfig.NumNextNPredictLayers > 0 {
		kv["nextn_predict_layers"] = m.TextConfig.NumNextNPredictLayers
	}

	return kv
}

// ProjectorKV returns KV metadata for the glm4v vision projector.
func (m *glmOcrModel) ProjectorKV(t *Tokenizer) KV {
	kv := KV{
		"general.architecture":                     "clip",
		"clip.projector_type":                      "glm4v",
		"clip.has_vision_encoder":                  true,
		"clip.vision.block_count":                  cmp.Or(m.VisionConfig.Depth, 24),
		"clip.vision.embedding_length":             cmp.Or(m.VisionConfig.HiddenSize, 1024),
		"clip.vision.attention.head_count":         cmp.Or(m.VisionConfig.NumHeads, 16),
		"clip.vision.image_size":                   cmp.Or(m.VisionConfig.ImageSize, 336),
		"clip.vision.patch_size":                   cmp.Or(m.VisionConfig.PatchSize, m.Preprocessor.PatchSize, 14),
		"clip.vision.spatial_merge_size":           cmp.Or(m.VisionConfig.SpatialMergeSize, m.Preprocessor.MergeSize, 2),
		"clip.vision.temporal_patch_size":          cmp.Or(m.VisionConfig.TemporalPatchSize, m.Preprocessor.TemporalPatchSize, 2),
		"clip.vision.out_hidden_size":              cmp.Or(m.VisionConfig.OutHiddenSize, 1536),
		"clip.vision.intermediate_size":            cmp.Or(m.VisionConfig.IntermediateSize, 4096),
		"clip.vision.feed_forward_length":          cmp.Or(m.VisionConfig.IntermediateSize, 4096),
		"clip.vision.projection_dim":               cmp.Or(m.VisionConfig.OutHiddenSize, 1536),
		"clip.vision.attention.layer_norm_epsilon": cmp.Or(m.VisionConfig.RMSNormEps, 1e-5),
	}

	if m.Preprocessor.Size.ShortestEdge > 0 {
		kv["clip.vision.min_pixels"] = m.Preprocessor.Size.ShortestEdge
	}
	if m.Preprocessor.Size.LongestEdge > 0 {
		kv["clip.vision.max_pixels"] = m.Preprocessor.Size.LongestEdge
	}
	if len(m.Preprocessor.ImageMean) == 3 {
		kv["clip.vision.image_mean"] = m.Preprocessor.ImageMean
	}
	if len(m.Preprocessor.ImageStd) == 3 {
		kv["clip.vision.image_std"] = m.Preprocessor.ImageStd
	}

	// Special tokens needed by the vision processor
	kv["clip.vision.image_token_id"] = m.ImageTokenID
	kv["clip.vision.image_start_token_id"] = m.ImageStartTokenID
	kv["clip.vision.image_end_token_id"] = m.ImageEndTokenID

	return kv
}

func isGlmOcrVisionTensor(name string) bool {
	return strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.")
}

// TextTensors returns only text model tensors (no vision/projector).
func (m *glmOcrModel) TextTensors(ts []Tensor, t *Tokenizer) []*ggml.Tensor {
	var textOnly []Tensor
	for _, tensor := range ts {
		if !isGlmOcrVisionTensor(tensor.Name()) {
			textOnly = append(textOnly, tensor)
		}
	}
	return m.Tensors(textOnly)
}

// ProjectorTensors returns only vision/projector tensors with names
// remapped for llama-server's clip/mtmd system.
func (m *glmOcrModel) ProjectorTensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor
	for _, t := range ts {
		if !isGlmOcrVisionTensor(t.Name()) {
			continue
		}

		name := t.Name()

		// Handle patch_embed temporal split for projector
		if strings.HasSuffix(name, "patch_embd.weight") {
			shape := t.Shape()
			if len(shape) == 5 && shape[2] == 2 {
				newShape := []uint64{shape[0], shape[1], shape[3], shape[4]}

				t0 := t.Clone()
				t0.SetRepacker(func(_ string, data []float32, shape []uint64) ([]float32, error) {
					dims := make([]int, len(shape))
					for i := range shape {
						dims[i] = int(shape[i])
					}
					var tt tensor.Tensor = tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))
					tt, err := tt.Slice(nil, nil, tensor.S(0, 1), nil, nil)
					if err != nil {
						return nil, err
					}
					tt = tensor.Materialize(tt)
					newDims := []int{int(shape[0]), int(shape[1]), int(shape[3]), int(shape[4])}
					if err := tt.Reshape(newDims...); err != nil {
						return nil, err
					}
					if err := tt.Reshape(tt.Shape().TotalSize()); err != nil {
						return nil, err
					}
					return native.VectorF32(tt.(*tensor.Dense))
				})
				out = append(out, &ggml.Tensor{
					Name:     strings.Replace(name, "patch_embd.weight", "patch_embd.weight", 1),
					Kind:     t.Kind(),
					Shape:    newShape,
					WriterTo: t0,
				})

				t1 := t.Clone()
				t1.SetRepacker(func(_ string, data []float32, shape []uint64) ([]float32, error) {
					dims := make([]int, len(shape))
					for i := range shape {
						dims[i] = int(shape[i])
					}
					var tt tensor.Tensor = tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))
					tt, err := tt.Slice(nil, nil, tensor.S(1, 2), nil, nil)
					if err != nil {
						return nil, err
					}
					tt = tensor.Materialize(tt)
					newDims := []int{int(shape[0]), int(shape[1]), int(shape[3]), int(shape[4])}
					if err := tt.Reshape(newDims...); err != nil {
						return nil, err
					}
					if err := tt.Reshape(tt.Shape().TotalSize()); err != nil {
						return nil, err
					}
					return native.VectorF32(tt.(*tensor.Dense))
				})
				out = append(out, &ggml.Tensor{
					Name:     strings.Replace(name, "patch_embd.weight", "patch_embd.weight.1", 1),
					Kind:     t.Kind(),
					Shape:    newShape,
					WriterTo: t1,
				})

				continue
			}

			if len(shape) == 4 {
				out = append(out, &ggml.Tensor{
					Name:     strings.Replace(name, "patch_embd.weight", "patch_embd.weight", 1),
					Kind:     t.Kind(),
					Shape:    t.Shape(),
					WriterTo: t,
				})
				continue
			}

			slog.Warn("glm-ocr: patch_embed weight has unexpected shape - not splitting", "shape", shape)
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

func (m *glmOcrModel) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	// Regex to extract block number from tensor names like "blk.16.nextn.eh_proj.weight"
	blkRe := regexp.MustCompile(`^blk\.(\d+)`)
	numLayers := int(cmp.Or(m.TextConfig.NumHiddenLayers, 16))

	// Rename NextN tensors: blk.N.eh_proj → blk.N.nextn.eh_proj, etc.
	renameNextN := func(name string, blkNum int) string {
		if blkNum < numLayers {
			return name
		}
		prefix := "blk." + strconv.Itoa(blkNum) + "."
		suffix := strings.TrimPrefix(name, prefix)
		// Map NextN tensor names
		switch {
		case strings.HasPrefix(suffix, "eh_proj"):
			return prefix + "nextn." + suffix
		case strings.HasPrefix(suffix, "embed_tokens"):
			return prefix + "nextn.embed_tokens" + strings.TrimPrefix(suffix, "embed_tokens")
		case strings.HasPrefix(suffix, "enorm"):
			return prefix + "nextn." + suffix
		case strings.HasPrefix(suffix, "hnorm"):
			return prefix + "nextn." + suffix
		case strings.HasPrefix(suffix, "shared_head.head"):
			return prefix + "nextn.shared_head_head" + strings.TrimPrefix(suffix, "shared_head.head")
		case strings.HasPrefix(suffix, "shared_head.norm"):
			return prefix + "nextn.shared_head_norm" + strings.TrimPrefix(suffix, "shared_head.norm")
		}
		return name
	}

	for _, t := range ts {
		name := t.Name()

		// Extract block number if present
		blkNum := -1
		if matches := blkRe.FindStringSubmatch(name); matches != nil {
			blkNum, _ = strconv.Atoi(matches[1])
		}

		// Rename NextN layer tensors
		name = renameNextN(name, blkNum)

		// Permute Q/K weights for M-RoPE compatibility (interleaved -> NeoX ordering)
		if len(m.TextConfig.RopeParameters.MRopeSection) > 0 &&
			blkNum >= 0 && blkNum < numLayers &&
			(strings.Contains(name, "attn_q.") || strings.Contains(name, "attn_k.")) {
			nHeads := int(cmp.Or(m.TextConfig.NumAttentionHeads, 16))
			nKVHeads := int(cmp.Or(m.TextConfig.NumKeyValueHeads, 8))
			hiddenSize := int(cmp.Or(m.TextConfig.HiddenSize, 1536))
			headDim := int(cmp.Or(m.TextConfig.HeadDim, uint32(hiddenSize/nHeads)))
			partialRotaryFactor := cmp.Or(m.TextConfig.PartialRotaryFactor, m.TextConfig.RopeParameters.PartialRotaryFactor, float32(0.5))

			effectiveHeads := nHeads
			if strings.Contains(name, "attn_k.") {
				effectiveHeads = nKVHeads
			}

			permutedT := t.Clone()
			permutedT.SetRepacker(normalToNeoXRepacker(effectiveHeads, headDim, partialRotaryFactor))
			out = append(out, &ggml.Tensor{
				Name:     name,
				Kind:     t.Kind(),
				Shape:    t.Shape(),
				WriterTo: permutedT,
			})
			continue
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

func (m *glmOcrModel) Replacements() []string {
	return []string{
		// Vision encoder
		"model.visual.patch_embed.proj", "v.patch_embd",
		"model.visual.blocks", "v.blk",
		// HF `post_layernorm` is the norm applied AFTER the vision transformer
		// output, which clip.cpp loads at clip.cpp:1566 via TN_LN_POST =
		// "%s.post_ln.%s" and uses in build_norm at line 497-498. Our prior
		// rename to `v.norm_embd` pointed it at TN_NORM_EMBD which is loaded
		// into a separate slot (norm_embd_w for patch-embedding norm), so the
		// post-transformer norm was dropped and patch-embedding norm was
		// populated with the wrong tensor. Reference community mmproj and
		// upstream convert_hf_to_gguf.py both use `v.post_ln`.
		"model.visual.post_layernorm", "v.post_ln",
		"model.visual.downsample", "mm.patch_merger",

		// Vision attention
		"attn.qkv", "attn_qkv",
		"attn.proj", "attn_out",
		"attn.q_norm", "attn_q_norm",
		"attn.k_norm", "attn_k_norm",

		// Vision norms
		"norm1", "ln1",
		"norm2", "ln2",

		// Vision MLP
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
		"mlp.down_proj", "ffn_down",

		// Merger (multimodal projector)
		"model.visual.merger.proj", "mm.model.fc",
		"model.visual.merger.post_projection_norm", "mm.post_norm",
		"model.visual.merger.gate_proj", "mm.gate",
		"model.visual.merger.up_proj", "mm.up",
		"model.visual.merger.down_proj", "mm.down",

		// Language model — strip "language_model." prefix like upstream does
		"model.language_model.embed_tokens", "token_embd",
		"model.language_model.layers", "blk",
		"model.language_model.norm", "output_norm",
		"lm_head", "output",

		// Language model attention
		"self_attn.q_proj", "attn_q",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",

		// Language model norms
		"input_layernorm", "attn_norm",
		"post_attention_layernorm", "ffn_norm",
		"post_self_attn_layernorm", "post_attention_norm",
		"post_mlp_layernorm", "post_ffw_norm",

		// Language model MLP (gate_up stays fused — GLM4 expects combined ffn_up)
		"mlp.gate_up_proj", "ffn_up",
		"mlp.down_proj", "ffn_down",
	}
}

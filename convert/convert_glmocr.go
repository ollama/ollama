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
		HiddenSize          uint32  `json:"hidden_size"`
		IntermediateSize    uint32  `json:"intermediate_size"`
		NumHiddenLayers     uint32  `json:"num_hidden_layers"`
		NumNextNPredict     uint32  `json:"num_nextn_predict_layers"`
		NumAttentionHeads   uint32  `json:"num_attention_heads"`
		NumKeyValueHeads    uint32  `json:"num_key_value_heads"`
		HeadDim             uint32  `json:"head_dim"`
		MaxPositionEmbed    uint32  `json:"max_position_embeddings"`
		RMSNormEps          float32 `json:"rms_norm_eps"`
		PartialRotaryFactor float32 `json:"partial_rotary_factor"`
		RopeParameters      struct {
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
	kv["general.architecture"] = "glmocr"
	applyGlmOcrTokenizerKV(kv, t)

	// Text model parameters
	numHiddenLayers := cmp.Or(m.TextConfig.NumHiddenLayers, 16)
	kv["glmocr.block_count"] = numHiddenLayers + m.TextConfig.NumNextNPredict
	if m.TextConfig.NumNextNPredict > 0 {
		kv["glmocr.nextn_predict_layers"] = m.TextConfig.NumNextNPredict
	}
	kv["glmocr.embedding_length"] = cmp.Or(m.TextConfig.HiddenSize, 1536)
	kv["glmocr.attention.head_count"] = cmp.Or(m.TextConfig.NumAttentionHeads, 16)
	kv["glmocr.attention.head_count_kv"] = cmp.Or(m.TextConfig.NumKeyValueHeads, 8)
	headDim := cmp.Or(m.TextConfig.HeadDim, m.TextConfig.HiddenSize/m.TextConfig.NumAttentionHeads)
	kv["glmocr.attention.key_length"] = headDim
	kv["glmocr.attention.value_length"] = headDim
	kv["glmocr.feed_forward_length"] = cmp.Or(m.TextConfig.IntermediateSize, 4608)
	kv["glmocr.attention.layer_norm_rms_epsilon"] = cmp.Or(m.TextConfig.RMSNormEps, 1e-5)
	kv["glmocr.context_length"] = cmp.Or(m.TextConfig.MaxPositionEmbed, 131072)
	kv["glmocr.rope.freq_base"] = cmp.Or(m.TextConfig.RopeParameters.RopeTheta, float32(10000))
	kv["glmocr.rope.partial_rotary_factor"] = cmp.Or(m.TextConfig.RopeParameters.PartialRotaryFactor, m.TextConfig.PartialRotaryFactor, float32(1.0))
	if len(m.TextConfig.RopeParameters.MRopeSection) > 0 {
		kv["glmocr.rope.mrope_section"] = m.TextConfig.RopeParameters.MRopeSection
	}

	// Vision model parameters
	kv["glmocr.vision.block_count"] = cmp.Or(m.VisionConfig.Depth, 24)
	kv["glmocr.vision.embedding_length"] = cmp.Or(m.VisionConfig.HiddenSize, 1024)
	kv["glmocr.vision.attention.head_count"] = cmp.Or(m.VisionConfig.NumHeads, 16)
	kv["glmocr.vision.image_size"] = cmp.Or(m.VisionConfig.ImageSize, 336)
	kv["glmocr.vision.patch_size"] = cmp.Or(m.VisionConfig.PatchSize, m.Preprocessor.PatchSize, 14)
	kv["glmocr.vision.spatial_merge_size"] = cmp.Or(m.VisionConfig.SpatialMergeSize, m.Preprocessor.MergeSize, 2)
	kv["glmocr.vision.temporal_patch_size"] = cmp.Or(m.VisionConfig.TemporalPatchSize, m.Preprocessor.TemporalPatchSize, 2)
	kv["glmocr.vision.out_hidden_size"] = cmp.Or(m.VisionConfig.OutHiddenSize, 1536)
	kv["glmocr.vision.intermediate_size"] = cmp.Or(m.VisionConfig.IntermediateSize, 4096)
	kv["glmocr.vision.attention.layer_norm_rms_epsilon"] = cmp.Or(m.VisionConfig.RMSNormEps, 1e-5)

	if m.Preprocessor.Size.ShortestEdge > 0 {
		kv["glmocr.vision.min_pixels"] = m.Preprocessor.Size.ShortestEdge
	}
	if m.Preprocessor.Size.LongestEdge > 0 {
		kv["glmocr.vision.max_pixels"] = m.Preprocessor.Size.LongestEdge
	}
	if len(m.Preprocessor.ImageMean) == 3 {
		kv["glmocr.vision.image_mean"] = m.Preprocessor.ImageMean
	}
	if len(m.Preprocessor.ImageStd) == 3 {
		kv["glmocr.vision.image_std"] = m.Preprocessor.ImageStd
	}

	kv["glmocr.image_token_id"] = m.ImageTokenID
	kv["glmocr.image_start_token_id"] = m.ImageStartTokenID
	kv["glmocr.image_end_token_id"] = m.ImageEndTokenID
	kv["glmocr.video_token_id"] = m.VideoTokenID
	kv["glmocr.video_start_token_id"] = m.VideoStartTokenID
	kv["glmocr.video_end_token_id"] = m.VideoEndTokenID

	return kv
}

func applyGlmOcrTokenizerKV(kv KV, t *Tokenizer) {
	kv["tokenizer.ggml.pre"] = "chatglm-bpe"
	if id, ok := glmOcrTokenID(t, "<|endoftext|>"); ok {
		kv["tokenizer.ggml.bos_token_id"] = uint32(id)
		kv["tokenizer.ggml.unknown_token_id"] = uint32(id)
	}
	if id, ok := glmOcrTokenID(t, "<|user|>"); ok {
		kv["tokenizer.ggml.eot_token_id"] = uint32(id)
	}
}

func (m *glmOcrModel) TextKV(t *Tokenizer) KV {
	kv := m.ModelParameters.KV(t)
	kv["general.architecture"] = "glm4"
	applyGlmOcrTokenizerKV(kv, t)

	numHiddenLayers := cmp.Or(m.TextConfig.NumHiddenLayers, 16)
	kv["block_count"] = numHiddenLayers + m.TextConfig.NumNextNPredict
	if m.TextConfig.NumNextNPredict > 0 {
		kv["nextn_predict_layers"] = m.TextConfig.NumNextNPredict
	}
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
	partialRotaryFactor := cmp.Or(m.TextConfig.RopeParameters.PartialRotaryFactor, m.TextConfig.PartialRotaryFactor, float32(1.0))
	kv["rope.dimension_count"] = uint32(float32(headDim) * partialRotaryFactor)
	if len(m.TextConfig.RopeParameters.MRopeSection) > 0 {
		sections := append([]int32(nil), m.TextConfig.RopeParameters.MRopeSection...)
		for len(sections) < 4 {
			sections = append(sections, 0)
		}
		kv["rope.dimension_sections"] = sections
	}

	return kv
}

func (m *glmOcrModel) ProjectorKV(*Tokenizer) KV {
	kv := KV{
		"general.architecture":                     "clip",
		"general.type":                             "mmproj",
		"general.file_type":                        uint32(1),
		"general.quantization_version":             uint32(2),
		"clip.has_vision_encoder":                  true,
		"clip.projector_type":                      "glm4v",
		"clip.use_silu":                            true,
		"clip.vision.block_count":                  cmp.Or(m.VisionConfig.Depth, 24),
		"clip.vision.embedding_length":             cmp.Or(m.VisionConfig.HiddenSize, 1024),
		"clip.vision.attention.head_count":         cmp.Or(m.VisionConfig.NumHeads, 16),
		"clip.vision.image_size":                   cmp.Or(m.VisionConfig.ImageSize, 336),
		"clip.vision.patch_size":                   cmp.Or(m.VisionConfig.PatchSize, m.Preprocessor.PatchSize, 14),
		"clip.vision.spatial_merge_size":           cmp.Or(m.VisionConfig.SpatialMergeSize, m.Preprocessor.MergeSize, 2),
		"clip.vision.temporal_patch_size":          cmp.Or(m.VisionConfig.TemporalPatchSize, m.Preprocessor.TemporalPatchSize, 2),
		"clip.vision.projection_dim":               cmp.Or(m.VisionConfig.OutHiddenSize, 1536),
		"clip.vision.out_hidden_size":              cmp.Or(m.VisionConfig.OutHiddenSize, 1536),
		"clip.vision.feed_forward_length":          cmp.Or(m.VisionConfig.IntermediateSize, 4096),
		"clip.vision.intermediate_size":            cmp.Or(m.VisionConfig.IntermediateSize, 4096),
		"clip.vision.attention.layer_norm_epsilon": cmp.Or(m.VisionConfig.RMSNormEps, 1e-5),
		"clip.vision.image_token_id":               m.ImageTokenID,
		"clip.vision.image_start_token_id":         m.ImageStartTokenID,
		"clip.vision.image_end_token_id":           m.ImageEndTokenID,
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

	return kv
}

func glmOcrTokenID(t *Tokenizer, token string) (int, bool) {
	if t == nil || t.Vocabulary == nil {
		return 0, false
	}
	for i, candidate := range t.Vocabulary.Tokens {
		if candidate == token {
			return i, true
		}
	}
	return 0, false
}

func isGlmOcrVisionTensor(name string) bool {
	return strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.")
}

func (m *glmOcrModel) TextTensors(ts []Tensor, t *Tokenizer) []*ggml.Tensor {
	textOnly := make([]Tensor, 0, len(ts))
	for _, tensor := range ts {
		if !isGlmOcrVisionTensor(tensor.Name()) {
			textOnly = append(textOnly, tensor)
		}
	}
	return m.Tensors(textOnly)
}

func (m *glmOcrModel) ProjectorTensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor
	for _, t := range ts {
		if !isGlmOcrVisionTensor(t.Name()) {
			continue
		}

		name := t.Name()
		switch {
		case strings.HasSuffix(name, "patch_embd_0.weight"):
			name = strings.Replace(name, "patch_embd_0.weight", "patch_embd.weight", 1)
		case strings.HasSuffix(name, "patch_embd_1.weight"):
			name = strings.Replace(name, "patch_embd_1.weight", "patch_embd.weight.1", 1)
		case strings.HasSuffix(name, "patch_embd.weight.0"):
			name = strings.Replace(name, "patch_embd.weight.0", "patch_embd.weight", 1)
		}
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

	numLayers := int(cmp.Or(m.TextConfig.NumHiddenLayers, 16))
	maxLayers := numLayers + int(m.TextConfig.NumNextNPredict)
	layerRe := regexp.MustCompile(`^blk\.(\d+)`)
	layerIndex := func(name string) (int, bool) {
		matches := layerRe.FindStringSubmatch(name)
		if matches == nil {
			return 0, false
		}
		blkNum, err := strconv.Atoi(matches[1])
		if err != nil {
			return 0, false
		}
		return blkNum, true
	}

	for _, t := range ts {
		name := t.Name()

		blkNum, hasLayer := layerIndex(name)
		if hasLayer && blkNum >= maxLayers {
			continue
		}
		if hasLayer && blkNum >= numLayers {
			switch {
			case strings.HasSuffix(name, ".embed_tokens.weight"):
				name = strings.Replace(name, ".embed_tokens.weight", ".nextn.embed_tokens.weight", 1)
			case strings.HasSuffix(name, ".eh_proj.weight"):
				name = strings.Replace(name, ".eh_proj.weight", ".nextn.eh_proj.weight", 1)
			case strings.HasSuffix(name, ".enorm.weight"):
				name = strings.Replace(name, ".enorm.weight", ".nextn.enorm.weight", 1)
			case strings.HasSuffix(name, ".hnorm.weight"):
				name = strings.Replace(name, ".hnorm.weight", ".nextn.hnorm.weight", 1)
			case strings.HasSuffix(name, ".shared_head.head.weight"):
				name = strings.Replace(name, ".shared_head.head.weight", ".nextn.shared_head_head.weight", 1)
			case strings.HasSuffix(name, ".shared_head.norm.weight"):
				name = strings.Replace(name, ".shared_head.norm.weight", ".nextn.shared_head_norm.weight", 1)
			}
		}

		// Split ffn_gate_up into separate gate and up projections
		if strings.Contains(name, "ffn_gate_up") {
			for t := range splitDim(t, 0,
				split{Replacer: strings.NewReplacer("ffn_gate_up", "ffn_gate")},
				split{Replacer: strings.NewReplacer("ffn_gate_up", "ffn_up")},
			) {
				out = append(out, t)
			}
			continue
		}

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
					Name:     strings.Replace(name, "patch_embd.weight", "patch_embd_0.weight", 1),
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
					Name:     strings.Replace(name, "patch_embd.weight", "patch_embd_1.weight", 1),
					Kind:     t.Kind(),
					Shape:    newShape,
					WriterTo: t1,
				})

				continue
			}

			if len(shape) == 4 {
				out = append(out, &ggml.Tensor{
					Name:     strings.Replace(name, "patch_embd.weight", "patch_embd_0.weight", 1),
					Kind:     t.Kind(),
					Shape:    t.Shape(),
					WriterTo: t,
				})
				continue
			}

			slog.Warn("glmocr: patch_embed weight has unexpected shape - not splitting", "shape", shape)
			// Fall through to default handling
		}

		// Handle pre-split patch embedding weights
		// Pattern 1: v.patch_embd.0.weight, v.patch_embd.1.weight -> patch_embd_0.weight, patch_embd_1.weight
		// Pattern 2: v.patch_embd.weight.0, v.patch_embd.weight.1 -> patch_embd_0.weight, patch_embd_1.weight
		if strings.Contains(name, "patch_embd.0.") {
			out = append(out, &ggml.Tensor{
				Name:     strings.Replace(name, "patch_embd.0.", "patch_embd_0.", 1),
				Kind:     t.Kind(),
				Shape:    t.Shape(),
				WriterTo: t,
			})
			continue
		}
		if strings.Contains(name, "patch_embd.1.") {
			out = append(out, &ggml.Tensor{
				Name:     strings.Replace(name, "patch_embd.1.", "patch_embd_1.", 1),
				Kind:     t.Kind(),
				Shape:    t.Shape(),
				WriterTo: t,
			})
			continue
		}
		// Handle .weight.0 and .weight.1 suffix patterns
		if strings.HasSuffix(name, "patch_embd.weight.0") {
			out = append(out, &ggml.Tensor{
				Name:     strings.Replace(name, "patch_embd.weight.0", "patch_embd_0.weight", 1),
				Kind:     t.Kind(),
				Shape:    t.Shape(),
				WriterTo: t,
			})
			continue
		}
		if strings.HasSuffix(name, "patch_embd.weight.1") {
			out = append(out, &ggml.Tensor{
				Name:     strings.Replace(name, "patch_embd.weight.1", "patch_embd_1.weight", 1),
				Kind:     t.Kind(),
				Shape:    t.Shape(),
				WriterTo: t,
			})
			continue
		}

		// Permute Q/K weights for M-RoPE compatibility (interleaved -> NeoX ordering)
		// GGML's M-RoPE kernel uses NeoX-style rotation, but GLM-OCR uses interleaved (LLaMA-style)
		// We permute at conversion time so the weights work correctly with GGML's kernel
		// This aligns Q/K rotary dimensions with GGML's NeoX-style rotation
		if len(m.TextConfig.RopeParameters.MRopeSection) > 0 &&
			strings.Contains(name, "blk.") && (strings.Contains(name, "attn_q.") || strings.Contains(name, "attn_k.")) {
			// Get config values for permutation
			nHeads := int(cmp.Or(m.TextConfig.NumAttentionHeads, 16))
			nKVHeads := int(cmp.Or(m.TextConfig.NumKeyValueHeads, 8))
			hiddenSize := int(cmp.Or(m.TextConfig.HiddenSize, 1536))
			headDim := int(cmp.Or(m.TextConfig.HeadDim, uint32(hiddenSize/nHeads)))
			partialRotaryFactor := cmp.Or(m.TextConfig.PartialRotaryFactor, m.TextConfig.RopeParameters.PartialRotaryFactor, float32(1.0))

			// Use appropriate head count: nHeads for Q, nKVHeads for K
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
		"model.visual.patch_embed.proj_1", "v.patch_embd_1", // Second temporal split
		"model.visual.patch_embed.proj", "v.patch_embd",
		"model.visual.blocks", "v.blk",
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

		// Language model
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

		// Language model MLP
		"mlp.gate_up_proj", "ffn_up",
		"mlp.down_proj", "ffn_down",
	}
}

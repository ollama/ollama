package compatmigrate

import (
	"fmt"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/fs/gguf"
)

type mistralPixtralMigrator struct{}

func (mistralPixtralMigrator) NeedsMigration(src *SourceModel) bool {
	if src.GGUF.KeyValue("general.architecture").String() != "mistral3" {
		return false
	}

	return sourceTensorHasPrefix(src, "v.") ||
		sourceTensorHasPrefix(src, "mm.") ||
		src.GGUF.KeyValue("rope.scaling.beta_fast").Valid() ||
		src.GGUF.KeyValue("rope.scaling.mscale_all_dim").Valid() ||
		src.GGUF.KeyValue("rope.scaling_beta").Valid()
}

func (mistralPixtralMigrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	imageToken := src.GGUF.KeyValue("image_token_index")
	if !imageToken.Valid() {
		return nil, fmt.Errorf("%s has no mistral3.image_token_index", src.Source.DisplayShortest())
	}

	visionHeads := src.GGUF.KeyValue("vision.attention.head_count").Uint()
	if visionHeads == 0 {
		return nil, fmt.Errorf("%s has no mistral3.vision.attention.head_count", src.Source.DisplayShortest())
	}

	var tokenEmbedding *sourceTensor
	var modelTensors []*ggml.Tensor
	var projectorTensors []*ggml.Tensor
	for _, tensor := range tensors {
		if tensor.name == "token_embd.weight" {
			tokenEmbedding = tensor
		}

		if isMistralPixtralProjectorTensor(tensor.name) {
			projectorTensors = append(projectorTensors, copyMistralPixtralProjectorTensor(tensor, visionHeads))
			continue
		}

		modelTensors = append(modelTensors, copyTensor(tensor.name, tensor))
	}
	if tokenEmbedding == nil {
		return nil, fmt.Errorf("%s has no token_embd.weight tensor", src.Source.DisplayShortest())
	}

	// The clean pull/re-create path gets this vector from the original HF
	// embedding matrix before quantization. Local migration is intentionally a
	// best-effort compatibility path, so we derive it from the user's installed
	// GGUF row. For q4/q6 installs this preserves the old local-model quality
	// instead of silently requiring a full model download on first load.
	imageBreak, err := copyTensorRowF32("v.token_embd.img_break", tokenEmbedding, imageToken.Uint())
	if err != nil {
		return nil, err
	}
	projectorTensors = append(projectorTensors, imageBreak)

	return &Result{
		ModelKV:          mistralPixtralModelKV(src),
		ModelTensors:     modelTensors,
		ProjectorKV:      mistralPixtralProjectorKV(src),
		ProjectorTensors: projectorTensors,
	}, nil
}

func mistralPixtralModelKV(src *SourceModel) ggml.KV {
	modelKV := ggml.KV{}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		key := keyValue.Key
		value := normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		switch {
		case key == "general.architecture":
			modelKV[key] = "mistral3"
		case strings.HasPrefix(key, "general."), strings.HasPrefix(key, "tokenizer."):
			modelKV[key] = value
		case strings.HasPrefix(key, "mistral3.vision."),
			strings.HasPrefix(key, "mistral3.mm."),
			key == "mistral3.image_token_index",
			key == "mistral3.spatial_merge_size":
			continue
		case key == "mistral3.rope.scaling.beta_fast":
			modelKV["mistral3.rope.scaling.yarn_beta_fast"] = value
		case key == "mistral3.rope.scaling.beta_slow":
			modelKV["mistral3.rope.scaling.yarn_beta_slow"] = value
		case key == "mistral3.rope.scaling.mscale":
			if _, ok := modelKV["mistral3.rope.scaling.yarn_log_multiplier"]; !ok {
				modelKV["mistral3.rope.scaling.yarn_log_multiplier"] = value
			}
		case key == "mistral3.rope.scaling.mscale_all_dim":
			modelKV["mistral3.rope.scaling.yarn_log_multiplier"] = value
		case key == "mistral3.rope.scaling_beta":
			modelKV["mistral3.attention.temperature_scale"] = value
		case strings.HasPrefix(key, "mistral3."):
			modelKV[key] = value
		}
	}
	if _, ok := modelKV["general.architecture"]; !ok {
		modelKV["general.architecture"] = "mistral3"
	}
	return modelKV
}

func mistralPixtralProjectorKV(src *SourceModel) ggml.KV {
	eps := float32(src.GGUF.KeyValue("vision.attention.layer_norm_epsilon").Float())
	if eps == 0 {
		eps = 1e-5
	}

	return ggml.KV{
		"general.architecture":                     "clip",
		"clip.has_vision_encoder":                  true,
		"clip.projector_type":                      "pixtral",
		"clip.vision.projection_dim":               uint32(src.GGUF.KeyValue("feed_forward_length").Uint()),
		"clip.vision.image_size":                   uint32(src.GGUF.KeyValue("vision.image_size").Uint()),
		"clip.vision.patch_size":                   uint32(src.GGUF.KeyValue("vision.patch_size").Uint()),
		"clip.vision.embedding_length":             uint32(src.GGUF.KeyValue("vision.embedding_length").Uint()),
		"clip.vision.feed_forward_length":          uint32(src.GGUF.KeyValue("vision.feed_forward_length").Uint()),
		"clip.vision.block_count":                  uint32(src.GGUF.KeyValue("vision.block_count").Uint()),
		"clip.vision.attention.head_count":         uint32(src.GGUF.KeyValue("vision.attention.head_count").Uint()),
		"clip.vision.attention.layer_norm_epsilon": eps,
		"clip.vision.image_mean":                   []float32{0.48145466, 0.4578275, 0.40821073},
		"clip.vision.image_std":                    []float32{0.26862954, 0.26130258, 0.27577711},
		"clip.rope.freq_base":                      float32(src.GGUF.KeyValue("vision.rope.freq_base").Float()),
		"clip.use_silu":                            true,
		"clip.vision.spatial_merge_size":           uint32(src.GGUF.KeyValue("spatial_merge_size").Uint()),
	}
}

func isMistralPixtralProjectorTensor(name string) bool {
	return strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.")
}

func copyMistralPixtralProjectorTensor(t *sourceTensor, visionHeads uint64) *ggml.Tensor {
	name := mistralPixtralProjectorTensorName(t.name)
	writer := t.Clone()
	outType := compatOutputTensorType(name, t.name, t.info.Type)

	// The generic compat policy preserves patch embeddings as F32 for several
	// projector families, but Pixtral/Mistral clean imports keep this kernel in
	// the installed F16 form. Preserve it here so the no-pull path matches the
	// clean pulled artifact.
	if name == "v.patch_embd.weight" && t.info.Type == gguf.TensorTypeF16 {
		outType = gguf.TensorTypeF16
	}

	if isMistralPixtralVisionQK(name) {
		writer.SetRepacker(func(_ string, data []float32, shape []uint64) ([]float32, error) {
			return mistralPixtralVisionQKRepack(data, shape, visionHeads)
		})
	}
	if outType != t.info.Type {
		writer.SetOutputType(outType)
	}

	return &ggml.Tensor{
		Name:     name,
		Kind:     uint32(outType),
		Shape:    slices.Clone(t.shape),
		WriterTo: writer,
	}
}

func isMistralPixtralVisionQK(name string) bool {
	return strings.HasPrefix(name, "v.blk.") &&
		(strings.HasSuffix(name, ".attn_q.weight") || strings.HasSuffix(name, ".attn_k.weight"))
}

func mistralPixtralVisionQKRepack(data []float32, shape []uint64, heads uint64) ([]float32, error) {
	if heads == 0 {
		return nil, fmt.Errorf("mistral pixtral q/k repack has zero attention heads")
	}
	if len(shape) != 2 {
		return nil, fmt.Errorf("mistral pixtral q/k repack requires a 2D tensor, got %v", shape)
	}
	if shape[0]%(heads*2) != 0 {
		return nil, fmt.Errorf("mistral pixtral q/k shape %v is not divisible by heads*2=%d", shape, heads*2)
	}

	cols := int(shape[0])
	rows := int(shape[1])
	if len(data) != cols*rows {
		return nil, fmt.Errorf("mistral pixtral q/k tensor has %d values, expected %d for shape %v", len(data), cols*rows, shape)
	}

	inner := cols / int(heads) / 2
	out := make([]float32, 0, len(data))
	for head := range int(heads) {
		for innerIndex := range inner {
			for pair := range 2 {
				start := (((head*2 + pair) * inner) + innerIndex) * rows
				out = append(out, data[start:start+rows]...)
			}
		}
	}
	return out, nil
}

func mistralPixtralProjectorTensorName(name string) string {
	name = strings.Replace(name, "v.patch_conv", "v.patch_embd", 1)
	name = strings.Replace(name, "v.encoder_norm", "v.pre_ln", 1)
	name = strings.Replace(name, "attn_output", "attn_out", 1)
	name = strings.Replace(name, "attn_norm", "ln1", 1)
	name = strings.Replace(name, "ffn_norm", "ln2", 1)
	name = strings.Replace(name, "mm.linear_1", "mm.1", 1)
	name = strings.Replace(name, "mm.linear_2", "mm.2", 1)
	name = strings.Replace(name, "mm.norm", "mm.input_norm", 1)
	name = strings.Replace(name, "mm.patch_merger.merging_layer", "mm.patch_merger", 1)
	return name
}

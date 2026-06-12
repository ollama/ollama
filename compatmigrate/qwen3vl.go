package compatmigrate

import (
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/fs/gguf"
)

type qwen3VLMigrator struct{}

func (qwen3VLMigrator) NeedsMigration(src *SourceModel) bool {
	arch := src.GGUF.KeyValue("general.architecture").String()
	if arch != "qwen3vl" && arch != "qwen3vlmoe" {
		return false
	}
	return !src.GGUF.KeyValue("rope.dimension_sections").Valid()
}

func (qwen3VLMigrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	deepstackIndexes := qwen3VLDeepstackIndexes(src)
	numChannels := src.GGUF.KeyValue("vision.num_channels").Uint()
	if numChannels == 0 {
		numChannels = 3
	}

	var modelTensors []*ggml.Tensor
	var projectorTensors []*ggml.Tensor
	projectorQKVWeights := map[string]map[string]*sourceTensor{}
	projectorQKVBiases := map[string]map[string]*sourceTensor{}
	for _, tensor := range tensors {
		if !qwen3VLVisionTensor(tensor.name) {
			modelTensors = append(modelTensors, copyTensor(tensor.name, tensor))
			continue
		}

		name := qwen3VLProjectorTensorName(tensor.name, deepstackIndexes)
		if qwen3VLPatchTensor(name, tensor) {
			split, err := qwen3VLSplitLegacyPatchTensor(tensor, numChannels)
			if err != nil {
				return nil, err
			}
			projectorTensors = append(projectorTensors, split...)
			continue
		}
		if fusedName, part, ok := qwen3VLProjectorQKVTarget(name); ok {
			projectorQKVWeights[fusedName] = ensureTensorParts(projectorQKVWeights[fusedName])
			projectorQKVWeights[fusedName][part] = tensor
			continue
		}
		if fusedName, part, ok := qwen3VLProjectorQKVBiasTarget(name); ok {
			projectorQKVBiases[fusedName] = ensureTensorParts(projectorQKVBiases[fusedName])
			projectorQKVBiases[fusedName][part] = tensor
			continue
		}

		projectorTensors = append(projectorTensors, copyTensor(name, tensor))
	}
	for _, name := range sortedTensorParts(projectorQKVWeights) {
		parts := projectorQKVWeights[name]
		if !tensorPartsComplete(parts) {
			for _, part := range []string{"q", "k", "v"} {
				if tensor := parts[part]; tensor != nil {
					projectorTensors = append(projectorTensors, copyTensor(strings.Replace(name, "attn_qkv", "attn_"+part, 1), tensor))
				}
			}
			continue
		}

		fused, err := qwen3VLConcatQKVWeights(name, parts["q"], parts["k"], parts["v"])
		if err != nil {
			return nil, err
		}
		projectorTensors = append(projectorTensors, fused)
	}
	for _, name := range sortedTensorParts(projectorQKVBiases) {
		parts := projectorQKVBiases[name]
		if !tensorPartsComplete(parts) {
			for _, part := range []string{"q", "k", "v"} {
				if tensor := parts[part]; tensor != nil {
					projectorTensors = append(projectorTensors, copyTensor(strings.Replace(name, "attn_qkv", "attn_"+part, 1), tensor))
				}
			}
			continue
		}

		fused, err := concatSourceTensorsDimKind(0, name, gguf.TensorTypeF32, parts["q"], parts["k"], parts["v"])
		if err != nil {
			return nil, err
		}
		projectorTensors = append(projectorTensors, fused)
	}

	modelKV := ggml.KV{}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		key := keyValue.Key
		switch {
		case key == "general.architecture":
			modelKV[key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		case strings.HasPrefix(key, "qwen3vl.vision."), strings.HasPrefix(key, "qwen3vlmoe.vision."):
			continue
		case strings.HasPrefix(key, "qwen3vl."), strings.HasPrefix(key, "qwen3vlmoe."):
			modelKV[key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		case strings.HasPrefix(key, "general."), strings.HasPrefix(key, "tokenizer."):
			modelKV[key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		}
	}

	if sections := qwen3VLMropeSections(src); len(sections) > 0 {
		modelKV["rope.dimension_sections"] = sections
	}
	modelKV["n_deepstack_layers"] = uint32(len(deepstackIndexes))

	visionHidden := uint32(src.GGUF.KeyValue("vision.embedding_length").Uint())
	patchSize := uint32(src.GGUF.KeyValue("vision.patch_size").Uint())
	if patchSize == 0 {
		patchSize = 16
	}
	numPositionEmbeddings := uint32(src.GGUF.KeyValue("vision.num_position_embeddings").Uint())
	if numPositionEmbeddings == 0 {
		numPositionEmbeddings = 2304
	}
	imageSize := uint32(math.Sqrt(float64(numPositionEmbeddings))) * patchSize

	projectorKV := ggml.KV{
		"general.architecture":                     "clip",
		"clip.projector_type":                      "qwen3vl_merger",
		"clip.has_vision_encoder":                  true,
		"clip.vision.block_count":                  uint32(src.GGUF.KeyValue("vision.block_count").Uint()),
		"clip.vision.embedding_length":             visionHidden,
		"clip.vision.feed_forward_length":          visionHidden * 4,
		"clip.vision.attention.head_count":         uint32(src.GGUF.KeyValue("vision.attention.head_count").Uint()),
		"clip.vision.attention.layer_norm_epsilon": float32(src.GGUF.KeyValue("vision.attention.layer_norm_epsilon").Float()),
		"clip.vision.num_channels":                 uint32(src.GGUF.KeyValue("vision.num_channels").Uint()),
		"clip.vision.patch_size":                   patchSize,
		"clip.vision.spatial_merge_size":           uint32(src.GGUF.KeyValue("vision.spatial_merge_size").Uint()),
		"clip.vision.image_size":                   imageSize,
		"clip.vision.projection_dim":               uint32(src.GGUF.KeyValue("embedding_length").Uint()),
		"clip.use_gelu":                            true,
		"clip.vision.temporal_patch_size":          uint32(src.GGUF.KeyValue("vision.temporal_patch_size").Uint()),
		"clip.vision.rope.freq_base":               float32(src.GGUF.KeyValue("vision.rope.freq_base").Float()),
	}
	projectorKV["clip.vision.is_deepstack_layers"] = qwen3VLDeepstackMask(uint32(src.GGUF.KeyValue("vision.block_count").Uint()), deepstackIndexes)
	if ff := src.GGUF.KeyValue("vision.feed_forward_length"); ff.Valid() {
		projectorKV["clip.vision.feed_forward_length"] = uint32(ff.Uint())
	}
	if projection := src.GGUF.KeyValue("vision.out_hidden_size"); projection.Valid() {
		projectorKV["clip.vision.projection_dim"] = uint32(projection.Uint())
	}
	if minPixels := src.GGUF.KeyValue("vision.shortest_edge"); minPixels.Valid() {
		projectorKV["clip.vision.min_pixels"] = uint32(minPixels.Uint())
	}
	if maxPixels := src.GGUF.KeyValue("vision.longest_edge"); maxPixels.Valid() {
		projectorKV["clip.vision.max_pixels"] = uint32(maxPixels.Uint())
	}
	if mean := src.GGUF.KeyValue("vision.image_mean"); mean.Valid() {
		projectorKV["clip.vision.image_mean"] = normalizeGGUFValue(rawGGUFValue(mean.Value))
	}
	if std := src.GGUF.KeyValue("vision.image_std"); std.Valid() {
		projectorKV["clip.vision.image_std"] = normalizeGGUFValue(rawGGUFValue(std.Value))
	}
	if projectorKV["clip.vision.num_channels"] == uint32(0) {
		projectorKV["clip.vision.num_channels"] = uint32(3)
	}
	if projectorKV["clip.vision.temporal_patch_size"] == uint32(0) {
		projectorKV["clip.vision.temporal_patch_size"] = uint32(2)
	}
	if projectorKV["clip.vision.rope.freq_base"] == float32(0) {
		projectorKV["clip.vision.rope.freq_base"] = float32(10000)
	}
	if _, ok := projectorKV["clip.vision.min_pixels"]; !ok {
		projectorKV["clip.vision.min_pixels"] = uint32(65536)
	}
	if _, ok := projectorKV["clip.vision.max_pixels"]; !ok {
		projectorKV["clip.vision.max_pixels"] = uint32(16777216)
	}

	return &Result{
		ModelKV:          modelKV,
		ModelTensors:     modelTensors,
		ProjectorKV:      projectorKV,
		ProjectorTensors: projectorTensors,
	}, nil
}

func qwen3VLVisionTensor(name string) bool {
	return strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.")
}

func qwen3VLPatchTensor(name string, t *sourceTensor) bool {
	return name == "v.patch_embd.weight" && len(t.shape) == 4 && t.shape[2] == 2
}

func qwen3VLSplitLegacyPatchTensor(t *sourceTensor, numChannels uint64) ([]*ggml.Tensor, error) {
	if numChannels == 0 {
		return nil, fmt.Errorf("split qwen3-vl patch tensor %s: num channels is zero", t.name)
	}
	if len(t.shape) != 4 {
		return nil, fmt.Errorf("split qwen3-vl patch tensor %s: expected rank 4, got %v", t.name, t.shape)
	}

	width, height, temporal, flat := t.shape[0], t.shape[1], t.shape[2], t.shape[3]
	if temporal != 2 {
		return nil, fmt.Errorf("split qwen3-vl patch tensor %s: expected temporal size 2, got %d", t.name, temporal)
	}
	if flat%numChannels != 0 {
		return nil, fmt.Errorf("split qwen3-vl patch tensor %s: flattened channel dimension %d is not divisible by %d", t.name, flat, numChannels)
	}

	hidden := flat / numChannels
	names := []string{"v.patch_embd.weight", "v.patch_embd.weight.1"}
	out := make([]*ggml.Tensor, 0, len(names))
	for temporalIndex, name := range names {
		temporalIndex := uint64(temporalIndex)
		shape := []uint64{width, height, numChannels, hidden}
		writer := t.Clone()
		writer.shape = slices.Clone(shape)
		writer.SetRepacker(func(_ string, data []float32, sourceShape []uint64) ([]float32, error) {
			if len(sourceShape) != 4 {
				return nil, fmt.Errorf("split qwen3-vl patch tensor %s: expected source rank 4, got %v", t.name, sourceShape)
			}

			sourceWidth, sourceHeight, sourceTemporal, sourceFlat := sourceShape[0], sourceShape[1], sourceShape[2], sourceShape[3]
			if sourceTemporal <= temporalIndex {
				return nil, fmt.Errorf("split qwen3-vl patch tensor %s: temporal index %d out of range for %v", t.name, temporalIndex, sourceShape)
			}
			if sourceFlat%numChannels != 0 {
				return nil, fmt.Errorf("split qwen3-vl patch tensor %s: flattened channel dimension %d is not divisible by %d", t.name, sourceFlat, numChannels)
			}

			sourceHidden := sourceFlat / numChannels
			result := make([]float32, sourceWidth*sourceHeight*numChannels*sourceHidden)
			var dst uint64
			for outIdx := range sourceHidden {
				for channel := range numChannels {
					flatOC := outIdx*numChannels + channel
					for y := range sourceHeight {
						for x := range sourceWidth {
							src := (((flatOC*sourceTemporal+temporalIndex)*sourceHeight)+y)*sourceWidth + x
							if src >= uint64(len(data)) {
								return nil, fmt.Errorf("split qwen3-vl patch tensor %s: source index %d out of range %d", t.name, src, len(data))
							}
							result[dst] = data[src]
							dst++
						}
					}
				}
			}
			return result, nil
		})

		outType := compatOutputTensorType(name, t.name, t.info.Type)
		if outType != t.info.Type {
			writer.SetOutputType(outType)
		}
		out = append(out, &ggml.Tensor{
			Name:     name,
			Kind:     uint32(outType),
			Shape:    shape,
			WriterTo: writer,
		})
	}

	return out, nil
}

func qwen3VLProjectorTensorName(name string, deepstackIndexes []int32) string {
	name = strings.Replace(name, "v.patch_embed.", "v.patch_embd.", 1)
	name = strings.Replace(name, "v.pos_embed.", "v.position_embd.", 1)
	name = strings.Replace(name, ".norm1.", ".ln1.", 1)
	name = strings.Replace(name, ".norm2.", ".ln2.", 1)
	name = strings.Replace(name, ".mlp.linear_fc1.", ".ffn_up.", 1)
	name = strings.Replace(name, ".mlp.linear_fc2.", ".ffn_down.", 1)

	if strings.HasPrefix(name, "v.merger.") {
		name = strings.Replace(name, "v.merger.linear_fc1", "mm.0", 1)
		name = strings.Replace(name, "v.merger.linear_fc2", "mm.2", 1)
		name = strings.Replace(name, "v.merger.norm", "v.post_ln", 1)
		return name
	}

	if strings.HasPrefix(name, "v.deepstack_merger.") {
		name = strings.Replace(name, "v.deepstack_merger.", "v.deepstack.", 1)
	}
	if !strings.HasPrefix(name, "v.deepstack.") {
		return name
	}

	rest := strings.TrimPrefix(name, "v.deepstack.")
	parts := strings.SplitN(rest, ".", 2)
	if len(parts) != 2 {
		return name
	}

	seqIdx, err := strconv.Atoi(parts[0])
	if err != nil || seqIdx < 0 || seqIdx >= len(deepstackIndexes) {
		return name
	}

	suffix := parts[1]
	suffix = strings.Replace(suffix, "linear_fc1", "fc1", 1)
	suffix = strings.Replace(suffix, "linear_fc2", "fc2", 1)
	return fmt.Sprintf("v.deepstack.%d.%s", deepstackIndexes[seqIdx], suffix)
}

func qwen3VLDeepstackIndexes(src *SourceModel) []int32 {
	if keyValue := src.GGUF.KeyValue("vision.deepstack_visual_indexes"); keyValue.Valid() {
		ints := keyValue.Ints()
		if len(ints) > 0 {
			out := make([]int32, len(ints))
			for i, v := range ints {
				out[i] = int32(v)
			}
			return out
		}
	}

	return []int32{8, 16, 24}
}

func qwen3VLMropeSections(src *SourceModel) []int32 {
	for _, key := range []string{"rope.dimension_sections", "rope.mrope_section"} {
		if keyValue := src.GGUF.KeyValue(key); keyValue.Valid() {
			ints := keyValue.Ints()
			if len(ints) > 0 {
				out := make([]int32, 4)
				for i, v := range ints[:min(len(ints), len(out))] {
					out[i] = int32(v)
				}
				return out
			}
		}
	}

	// The legacy library GGUFs sampled so far omit the HF mrope_section, but
	// llama-server requires rope.dimension_sections for Qwen3-VL M-RoPE.
	return []int32{24, 20, 20, 0}
}

func qwen3VLProjectorQKVTarget(name string) (string, string, bool) {
	for _, part := range []string{"q", "k", "v"} {
		needle := ".attn_" + part + ".weight"
		if strings.Contains(name, needle) {
			return strings.Replace(name, needle, ".attn_qkv.weight", 1), part, true
		}
	}
	return "", "", false
}

func qwen3VLProjectorQKVBiasTarget(name string) (string, string, bool) {
	for _, part := range []string{"q", "k", "v"} {
		needle := ".attn_" + part + ".bias"
		if strings.Contains(name, needle) {
			return strings.Replace(name, needle, ".attn_qkv.bias", 1), part, true
		}
	}
	return "", "", false
}

func qwen3VLConcatQKVWeights(name string, tensors ...*sourceTensor) (*ggml.Tensor, error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("concat qwen3-vl qkv tensor %s: no source tensors", name)
	}
	if len(tensors[0].shape) != 2 {
		return nil, fmt.Errorf("concat qwen3-vl qkv tensor %s: expected rank 2, got %v", name, tensors[0].shape)
	}

	shape := slices.Clone(tensors[0].shape)
	shape[1] = 0
	for _, t := range tensors {
		if len(t.shape) != 2 {
			return nil, fmt.Errorf("concat qwen3-vl qkv tensor %s: expected rank 2, got %v", name, t.shape)
		}
		if t.shape[0] != shape[0] {
			return nil, fmt.Errorf("concat qwen3-vl qkv tensor %s: incompatible shape %v vs %v", name, shape, t.shape)
		}
		shape[1] += t.shape[1]
	}

	return &ggml.Tensor{
		Name:     name,
		Kind:     uint32(gguf.TensorTypeF16),
		Shape:    shape,
		WriterTo: qwen3VLQKVWeightConcat{tensors: tensors},
	}, nil
}

type qwen3VLQKVWeightConcat struct {
	tensors []*sourceTensor
}

func (c qwen3VLQKVWeightConcat) GGUFWriteMemoryEstimate() uint64 {
	var sourceBytes, inputFloatBytes, outputElements uint64
	for _, source := range c.tensors {
		sourceBytes = saturatingAdd(sourceBytes, int64ToUint64(source.info.NumBytes()))
		elements := tensorElementCount(source.shape)
		inputFloatBytes = saturatingAdd(inputFloatBytes, saturatingMul(elements, 4))
		outputElements = saturatingAdd(outputElements, elements)
	}

	return saturatingAdd(sourceBytes, inputFloatBytes, ggufTensorBytes(gguf.TensorTypeF16, outputElements))
}

func (c qwen3VLQKVWeightConcat) WriteTo(w io.Writer) (int64, error) {
	var total int64
	for _, source := range c.tensors {
		data, err := readSourceTensorFloatData(source)
		if err != nil {
			return total, err
		}

		n, err := encodeFloatTensor(w, gguf.TensorTypeF16, data)
		total += int64(n)
		if err != nil {
			return total, err
		}
	}

	return total, nil
}

func ensureTensorParts(parts map[string]*sourceTensor) map[string]*sourceTensor {
	if parts != nil {
		return parts
	}
	return map[string]*sourceTensor{}
}

func tensorPartsComplete(parts map[string]*sourceTensor) bool {
	return parts["q"] != nil && parts["k"] != nil && parts["v"] != nil
}

func sortedTensorParts(parts map[string]map[string]*sourceTensor) []string {
	names := make([]string, 0, len(parts))
	for name := range parts {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}

func qwen3VLDeepstackMask(blockCount uint32, indexes []int32) []bool {
	mask := make([]bool, blockCount)
	for _, idx := range indexes {
		if idx >= 0 && int(idx) < len(mask) {
			mask[idx] = true
		}
	}
	return mask
}

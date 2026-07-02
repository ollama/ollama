package compatmigrate

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/fs/gguf"
)

type qwen35Migrator struct{}

func (qwen35Migrator) NeedsMigration(src *SourceModel) bool {
	arch := src.GGUF.KeyValue("general.architecture").String()
	if arch != "qwen35" && arch != "qwen35moe" {
		return false
	}
	if src.GGUF.KeyValue("vision.block_count").Valid() ||
		src.GGUF.KeyValue("image_token_id").Valid() ||
		src.GGUF.KeyValue("ssm.v_head_reordered").Valid() ||
		src.GGUF.KeyValue("rope.mrope_interleaved").Valid() ||
		sourceTensorHasPrefix(src, "mtp.") ||
		sourceTensorHasPrefix(src, "v.") {
		return true
	}
	return arch == "qwen35moe" && src.GGUF.KeyValue("feed_forward_length").Valid()
}

func (qwen35Migrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	numChannels := src.GGUF.KeyValue("vision.num_channels").Uint()
	if numChannels == 0 {
		numChannels = 3
	}

	var modelTensors []*ggml.Tensor
	var projectorTensors []*ggml.Tensor
	var mtpTensors []*ggml.Tensor
	nextn := qwen35MTPLayerCount(tensors)
	baseBlocks := qwen35BaseBlockCount(src, tensors)
	hasNativeMTP := qwen35HasNativeMTP(tensors)
	isMoE := src.GGUF.KeyValue("general.architecture").String() == "qwen35moe"
	mtpExperts := map[string]map[int]*sourceTensor{}
	projectorQKVWeights := map[string]map[string]*sourceTensor{}
	projectorQKVBiases := map[string]map[string]*sourceTensor{}
	for _, tensor := range tensors {
		if qwen35ProjectorTensor(tensor.name) {
			name := qwen35ProjectorTensorName(tensor.name)
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
			continue
		}
		if strings.HasPrefix(tensor.name, "mtp.") && nextn > 0 && baseBlocks > 0 && !hasNativeMTP {
			names := qwen35MTPTensorNames(tensor.name, baseBlocks, nextn)
			for _, name := range names {
				if isMoE {
					if target, expert, ok := qwen35MTPExpertTensor(name); ok {
						if mtpExperts[target] == nil {
							mtpExperts[target] = map[int]*sourceTensor{}
						}
						mtpExperts[target][expert] = tensor
						continue
					}
				}
				mtpTensors = append(mtpTensors, copyQwen35MTPTensor(name, tensor))
			}
			continue
		}
		if strings.HasPrefix(tensor.name, "mtp.") {
			continue
		}

		modelTensors = append(modelTensors, copyTensor(qwen35ModelTensorName(tensor.name), tensor))
	}
	mergedMTPExperts, err := qwen35MergedMTPExpertTensors(mtpExperts)
	if err != nil {
		return nil, err
	}
	mtpTensors = append(mtpTensors, mergedMTPExperts...)
	modelTensors = append(modelTensors, mtpTensors...)
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
	arch := src.GGUF.KeyValue("general.architecture").String()
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		key := keyValue.Key
		value := normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		switch {
		case key == "general.architecture":
			modelKV[key] = value
		case strings.HasPrefix(key, arch+".vision."), qwen35VisionTokenKey(key, arch):
			continue
		case strings.HasPrefix(key, arch+"."):
			if key == arch+".attention.head_count_kv" {
				value = qwen35MaxHeadCountKV(value, 2)
			}
			if key == arch+".block_count" && nextn > 0 && baseBlocks > 0 && !hasNativeMTP {
				value = baseBlocks + nextn
			}
			if qwen35DropTextKV(key, arch) {
				continue
			}
			if qwen35RopeSectionKey(key) {
				value = padInt32Sections(value)
			}
			modelKV[key] = value
		case strings.HasPrefix(key, "general."), strings.HasPrefix(key, "tokenizer."):
			modelKV[key] = value
		}
	}
	if nextn > 0 && baseBlocks > 0 && !hasNativeMTP {
		modelKV[arch+".nextn_predict_layers"] = nextn
		modelKV[arch+".block_count"] = baseBlocks + nextn
	}

	result := &Result{
		ModelKV:      modelKV,
		ModelTensors: modelTensors,
	}
	if len(projectorTensors) > 0 {
		result.ProjectorKV = qwen35ProjectorKV(src)
		result.ProjectorTensors = projectorTensors
	}
	return result, nil
}

func qwen35ModelTensorName(name string) string {
	if strings.HasSuffix(name, ".ssm_dt") {
		return name + ".bias"
	}
	return name
}

func qwen35BaseBlockCount(src *SourceModel, tensors []*sourceTensor) uint32 {
	maxBlock := -1
	for _, tensor := range tensors {
		if strings.HasPrefix(tensor.name, "mtp.") || qwen35ProjectorTensor(tensor.name) {
			continue
		}
		if block, ok := tensorBlock(tensor.name); ok && block > maxBlock {
			maxBlock = block
		}
	}
	if maxBlock < 0 {
		if blocks := src.GGUF.KeyValue("block_count").Uint(); blocks > 0 {
			return uint32(blocks)
		}
		return 0
	}
	return uint32(maxBlock + 1)
}

func qwen35MTPLayerCount(tensors []*sourceTensor) uint32 {
	maxLayer := -1
	hasMTP := false
	for _, tensor := range tensors {
		if !strings.HasPrefix(tensor.name, "mtp.") {
			continue
		}
		hasMTP = true
		rest := strings.TrimPrefix(tensor.name, "mtp.layers.")
		layer, suffix, ok := strings.Cut(rest, ".")
		if !ok || suffix == "" {
			continue
		}
		n, err := strconv.Atoi(layer)
		if err == nil && n > maxLayer {
			maxLayer = n
		}
	}
	if maxLayer >= 0 {
		return uint32(maxLayer + 1)
	}
	if hasMTP {
		return 1
	}
	return 0
}

func qwen35HasNativeMTP(tensors []*sourceTensor) bool {
	for _, tensor := range tensors {
		if strings.HasPrefix(tensor.name, "blk.") && strings.Contains(tensor.name, ".nextn.") {
			return true
		}
	}
	return false
}

func qwen35MTPTensorNames(name string, base, nextn uint32) []string {
	if rest := strings.TrimPrefix(name, "mtp.layers."); rest != name {
		layer, suffix, ok := strings.Cut(rest, ".")
		if !ok || suffix == "" {
			return nil
		}
		idx, err := strconv.ParseUint(layer, 10, 32)
		if err != nil || uint32(idx) >= nextn {
			return nil
		}
		return []string{fmt.Sprintf("blk.%d.%s", base+uint32(idx), suffix)}
	}

	var suffix string
	switch name {
	case "mtp.fc.weight":
		suffix = "nextn.eh_proj.weight"
	case "mtp.pre_fc_norm_embedding.weight":
		suffix = "nextn.enorm.weight"
	case "mtp.pre_fc_norm_hidden.weight":
		suffix = "nextn.hnorm.weight"
	case "mtp.embed_tokens.weight":
		suffix = "nextn.embed_tokens.weight"
	case "mtp.shared_head.head.weight":
		suffix = "nextn.shared_head_head.weight"
	case "mtp.shared_head.norm.weight", "mtp.norm.weight":
		suffix = "nextn.shared_head_norm.weight"
	default:
		return nil
	}

	names := make([]string, 0, nextn)
	for i := range nextn {
		names = append(names, fmt.Sprintf("blk.%d.%s", base+i, suffix))
	}
	return names
}

func copyQwen35MTPTensor(name string, tensor *sourceTensor) *ggml.Tensor {
	writer := tensor.Clone()
	outType := compatOutputTensorType(name, tensor.name, tensor.info.Type)
	if qwen35ShouldShiftNormAfterMTPRename(name) {
		writer.SetRepacker(qwen35AddOneRepacker)
		outType = gguf.TensorTypeF32
	}
	if outType != tensor.info.Type {
		writer.SetOutputType(outType)
	}
	return &ggml.Tensor{
		Name:     name,
		Kind:     uint32(outType),
		Shape:    slices.Clone(tensor.shape),
		WriterTo: writer,
	}
}

func qwen35AddOneRepacker(_ string, data []float32, _ []uint64) ([]float32, error) {
	out := slices.Clone(data)
	for i := range out {
		out[i] += 1
	}
	return out, nil
}

func qwen35ShouldShiftNormAfterMTPRename(name string) bool {
	if strings.HasSuffix(name, ".ssm_norm.weight") {
		return false
	}
	return strings.HasSuffix(name, "_norm.weight") ||
		strings.HasSuffix(name, ".nextn.enorm.weight") ||
		strings.HasSuffix(name, ".nextn.hnorm.weight")
}

func qwen35MTPExpertTensor(name string) (string, int, bool) {
	if !strings.HasPrefix(name, "blk.") {
		return "", 0, false
	}
	parts := strings.Split(name, ".")
	if len(parts) != 7 || parts[2] != "mlp" || parts[3] != "experts" || parts[6] != "weight" {
		return "", 0, false
	}
	expert, err := strconv.Atoi(parts[4])
	if err != nil || expert < 0 {
		return "", 0, false
	}

	var suffix string
	switch parts[5] + "." + parts[6] {
	case "gate_proj.weight":
		suffix = "ffn_gate_exps.weight"
	case "up_proj.weight":
		suffix = "ffn_up_exps.weight"
	case "down_proj.weight":
		suffix = "ffn_down_exps.weight"
	default:
		return "", 0, false
	}
	return strings.Join([]string{parts[0], parts[1], suffix}, "."), expert, true
}

func qwen35MergedMTPExpertTensors(groups map[string]map[int]*sourceTensor) ([]*ggml.Tensor, error) {
	var out []*ggml.Tensor
	names := make([]string, 0, len(groups))
	for name := range groups {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		experts := groups[name]
		indexes := make([]int, 0, len(experts))
		for index := range experts {
			indexes = append(indexes, index)
		}
		slices.Sort(indexes)
		parts := make([]*sourceTensor, 0, len(indexes))
		for i, index := range indexes {
			if i != index {
				return nil, fmt.Errorf("non-contiguous qwen3.5 MTP experts for %s", name)
			}
			parts = append(parts, experts[index])
		}
		merged, err := stackSourceTensors(name, parts...)
		if err != nil {
			return nil, err
		}
		out = append(out, merged)
	}
	return out, nil
}

func qwen35ProjectorTensor(name string) bool {
	return strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.")
}

func qwen35ProjectorTensorName(name string) string {
	return qwen3VLProjectorTensorName(name, nil)
}

func qwen35ProjectorKV(src *SourceModel) ggml.KV {
	blockCount := uint32(src.GGUF.KeyValue("vision.block_count").Uint())
	projectorKV := ggml.KV{
		"general.architecture":                     "clip",
		"clip.projector_type":                      "qwen3vl_merger",
		"clip.has_vision_encoder":                  true,
		"clip.use_gelu":                            true,
		"clip.vision.block_count":                  blockCount,
		"clip.vision.embedding_length":             uint32(src.GGUF.KeyValue("vision.embedding_length").Uint()),
		"clip.vision.feed_forward_length":          uint32(4304),
		"clip.vision.attention.head_count":         uint32(src.GGUF.KeyValue("vision.attention.head_count").Uint()),
		"clip.vision.attention.layer_norm_epsilon": float32(1e-6),
		"clip.vision.patch_size":                   uint32(src.GGUF.KeyValue("vision.patch_size").Uint()),
		"clip.vision.spatial_merge_size":           uint32(src.GGUF.KeyValue("vision.spatial_merge_size").Uint()),
		"clip.vision.num_channels":                 uint32(src.GGUF.KeyValue("vision.num_channels").Uint()),
		"clip.vision.projection_dim":               uint32(src.GGUF.KeyValue("embedding_length").Uint()),
		"clip.vision.image_size":                   uint32(768),
		"clip.vision.image_mean":                   []float32{0.5, 0.5, 0.5},
		"clip.vision.image_std":                    []float32{0.5, 0.5, 0.5},
		"clip.vision.is_deepstack_layers":          make([]bool, blockCount),
	}
	if ff := src.GGUF.KeyValue("vision.feed_forward_length"); ff.Valid() {
		projectorKV["clip.vision.feed_forward_length"] = uint32(ff.Uint())
	}
	if imageSize := src.GGUF.KeyValue("vision.image_size"); imageSize.Valid() {
		projectorKV["clip.vision.image_size"] = uint32(imageSize.Uint())
	}
	if eps := src.GGUF.KeyValue("vision.attention.layer_norm_epsilon"); eps.Valid() {
		projectorKV["clip.vision.attention.layer_norm_epsilon"] = float32(eps.Float())
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
	return projectorKV
}

func qwen35MaxHeadCountKV(value any, fallback uint32) any {
	switch values := value.(type) {
	case []uint32:
		var maxValue uint32
		for _, value := range values {
			maxValue = max(maxValue, value)
		}
		if maxValue == 0 {
			maxValue = fallback
		}
		return maxValue
	case []int32:
		var maxValue uint32
		for _, value := range values {
			if value > 0 {
				maxValue = max(maxValue, uint32(value))
			}
		}
		if maxValue == 0 {
			maxValue = fallback
		}
		return maxValue
	default:
		return value
	}
}

func qwen35VisionTokenKey(key, arch string) bool {
	switch key {
	case arch + ".image_token_id", arch + ".vision_start_token_id", arch + ".vision_end_token_id":
		return true
	default:
		return false
	}
}

func qwen35DropTextKV(key, arch string) bool {
	if arch == "qwen35moe" && key == arch+".feed_forward_length" {
		return true
	}
	switch key {
	case arch + ".ssm.v_head_reordered", arch + ".rope.mrope_interleaved":
		return true
	default:
		return false
	}
}

func qwen35RopeSectionKey(key string) bool {
	key = strings.TrimPrefix(key, "qwen35.")
	key = strings.TrimPrefix(key, "qwen35moe.")
	switch key {
	case "mrope_sections", "rope.mrope_section", "rope.dimension_sections":
		return true
	default:
		return false
	}
}

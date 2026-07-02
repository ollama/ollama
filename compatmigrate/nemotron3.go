package compatmigrate

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type nemotron3Migrator struct{}

func (nemotron3Migrator) NeedsMigration(src *SourceModel) bool {
	if src.GGUF.KeyValue("general.architecture").String() != "nemotron_h_omni" {
		return false
	}
	return sourceTensorHasPrefix(src, "v.") ||
		sourceTensorHasPrefix(src, "a.") ||
		sourceTensorHasPrefix(src, "mm.")
}

func (nemotron3Migrator) Migrate(src *SourceModel) (*Result, error) {
	if arch := src.GGUF.KeyValue("general.architecture").String(); arch != "nemotron_h_omni" {
		return nil, errUnsupportedFamily
	}

	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	var modelTensors []*ggml.Tensor
	var projectorTensors []*ggml.Tensor
	for _, tensor := range tensors {
		if isNemotron3AudioTensor(tensor.name) {
			continue
		}
		if !isNemotron3ProjectorTensor(tensor.name) {
			modelTensors = append(modelTensors, copyTensor(tensor.name, tensor))
			continue
		}

		projector, err := nemotron3ProjectorTensor(src, tensor)
		if err != nil {
			return nil, err
		}
		projectorTensors = append(projectorTensors, projector...)
	}

	return &Result{
		ModelKV:          nemotron3ModelKV(src),
		ModelTensors:     modelTensors,
		ProjectorKV:      nemotron3ProjectorKV(src),
		ProjectorTensors: projectorTensors,
		Renderer:         "nemotron-3-nano",
		Parser:           "nemotron-3-nano",
	}, nil
}

func nemotron3TextArchitecture(src *SourceModel) string {
	if src.GGUF.KeyValue("expert_count").Uint() > 0 || src.GGUF.KeyValue("expert_used_count").Uint() > 0 {
		return "nemotron_h_moe"
	}
	return "nemotron_h"
}

func nemotron3ModelKV(src *SourceModel) ggml.KV {
	arch := nemotron3TextArchitecture(src)
	out := ggml.KV{"general.architecture": arch}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		key := keyValue.Key
		value := normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		switch {
		case key == "general.architecture":
			continue
		case strings.HasPrefix(key, "general."), strings.HasPrefix(key, "tokenizer."):
			out[key] = value
		case strings.HasPrefix(key, "nemotron_h_omni.vision."), strings.HasPrefix(key, "nemotron_h_omni.audio."):
			continue
		case strings.HasPrefix(key, "nemotron_h_omni."):
			out[arch+"."+strings.TrimPrefix(key, "nemotron_h_omni.")] = value
		}
	}
	return out
}

func nemotron3ProjectorKV(src *SourceModel) ggml.KV {
	imageSize := uint32(src.GGUF.KeyValue("vision.image_size").Uint())
	if imageSize == 0 {
		imageSize = 512
	}
	patchSize := uint32(src.GGUF.KeyValue("vision.patch_size").Uint())
	if patchSize == 0 {
		patchSize = 16
	}
	embeddingLength := uint32(src.GGUF.KeyValue("vision.embedding_length").Uint())
	if embeddingLength == 0 {
		embeddingLength = 1280
	}
	feedForwardLength := uint32(src.GGUF.KeyValue("vision.feed_forward_length").Uint())
	if feedForwardLength == 0 {
		feedForwardLength = embeddingLength * 4
	}
	headCount := uint32(src.GGUF.KeyValue("vision.attention.head_count").Uint())
	if headCount == 0 {
		headCount = 16
	}
	eps := float32(src.GGUF.KeyValue("vision.attention.layer_norm_epsilon").Float())
	if eps == 0 {
		eps = 1e-6
	}
	scaleFactor := uint32(src.GGUF.KeyValue("vision.projector.scale_factor").Uint())
	if scaleFactor == 0 {
		scaleFactor = 2
	}

	out := ggml.KV{
		"general.architecture":                     "clip",
		"clip.has_vision_encoder":                  true,
		"clip.vision.projector_type":               "nemotron_v2_vl",
		"clip.vision.block_count":                  uint32(src.GGUF.KeyValue("vision.block_count").Uint()),
		"clip.vision.embedding_length":             embeddingLength,
		"clip.vision.feed_forward_length":          feedForwardLength,
		"clip.vision.attention.head_count":         headCount,
		"clip.vision.attention.layer_norm_epsilon": eps,
		"clip.vision.patch_size":                   patchSize,
		"clip.vision.image_size":                   imageSize,
		"clip.vision.num_channels":                 uint32(3),
		"clip.vision.image_mean":                   []float32{0.48145466, 0.4578275, 0.40821073},
		"clip.vision.image_std":                    []float32{0.26862954, 0.26130258, 0.27577711},
		"clip.vision.projector.scale_factor":       scaleFactor,
		"clip.vision.projection_dim":               uint32(src.GGUF.KeyValue("embedding_length").Uint()),
		"clip.use_gelu":                            true,
	}

	for _, key := range []string{
		"vision.max_tiles",
		"vision.min_num_patches",
		"vision.max_num_patches",
		"vision.image_token_id",
		"vision.image_start_token_id",
		"vision.image_end_token_id",
	} {
		if value := src.GGUF.KeyValue(key); value.Valid() {
			out["clip."+key] = normalizeGGUFValue(rawGGUFValue(value.Value))
		}
	}
	if mean := src.GGUF.KeyValue("vision.image_mean"); mean.Valid() {
		out["clip.vision.image_mean"] = normalizeGGUFValue(rawGGUFValue(mean.Value))
	}
	if std := src.GGUF.KeyValue("vision.image_std"); std.Valid() {
		out["clip.vision.image_std"] = normalizeGGUFValue(rawGGUFValue(std.Value))
	}

	return out
}

func isNemotron3ProjectorTensor(name string) bool {
	return strings.HasPrefix(name, "v.") || (strings.HasPrefix(name, "mm.") && !strings.HasPrefix(name, "mm.a."))
}

func isNemotron3AudioTensor(name string) bool {
	return strings.HasPrefix(name, "a.") || strings.HasPrefix(name, "mm.a.")
}

func nemotron3ProjectorTensor(src *SourceModel, tensor *sourceTensor) ([]*ggml.Tensor, error) {
	name := tensor.name
	if strings.Contains(name, ".attn_qkv.") {
		return splitSourceTensorDim(tensor, 0,
			strings.Replace(name, "attn_qkv", "attn_q", 1),
			strings.Replace(name, "attn_qkv", "attn_k", 1),
			strings.Replace(name, "attn_qkv", "attn_v", 1),
		)
	}

	name, shape, repacker := nemotron3ProjectorTensorTransform(src, tensor)
	writer := tensor.Clone()
	writer.shape = slices.Clone(shape)
	if repacker != nil {
		writer.SetRepacker(repacker)
	}
	outType := compatOutputTensorType(name, tensor.name, tensor.info.Type)
	if outType != tensor.info.Type || repacker != nil {
		writer.SetOutputType(outType)
	}
	return []*ggml.Tensor{{
		Name:     name,
		Kind:     uint32(outType),
		Shape:    shape,
		WriterTo: writer,
	}}, nil
}

func nemotron3ProjectorTensorTransform(src *SourceModel, tensor *sourceTensor) (string, []uint64, repacker) {
	name := tensor.name
	shape := slices.Clone(tensor.shape)
	visionEmbedding := src.GGUF.KeyValue("vision.embedding_length").Uint()
	if visionEmbedding == 0 {
		visionEmbedding = 1280
	}
	patchSize := src.GGUF.KeyValue("vision.patch_size").Uint()
	if patchSize == 0 {
		patchSize = 16
	}
	imageSize := src.GGUF.KeyValue("vision.image_size").Uint()
	if imageSize == 0 {
		imageSize = 512
	}
	targetSide := imageSize / patchSize

	switch name {
	case "v.cls_embd":
		name = "v.class_embd"
		shape = normalizeNemotron3ClassEmbeddingShape(shape, visionEmbedding)
	case "v.position_embd", "v.position_embd.weight":
		name = "v.position_embd.weight"
		shape = nemotron3PositionEmbeddingOutputShape(shape, visionEmbedding, targetSide)
		return name, shape, func(name string, data []float32, sourceShape []uint64) ([]float32, error) {
			return downsampleNemotron3PositionEmbedding(name, data, sourceShape, visionEmbedding, targetSide)
		}
	case "v.patch_embd.weight":
		shape = nemotron3PatchEmbeddingOutputShape(shape, patchSize, 3, visionEmbedding)
		return name, shape, func(name string, data []float32, sourceShape []uint64) ([]float32, error) {
			return reshapeNemotron3PatchEmbedding(name, data, sourceShape, patchSize, 3, visionEmbedding)
		}
	case "mm.norm.weight":
		name = "mm.model.mlp.0.weight"
	case "mm.1.weight":
		name = "mm.model.mlp.1.weight"
	case "mm.2.weight":
		name = "mm.model.mlp.3.weight"
	}

	return name, shape, nil
}

func normalizeNemotron3ClassEmbeddingShape(shape []uint64, embeddingLength uint64) []uint64 {
	if len(shape) == 2 && embeddingLength > 0 && shape[0] != embeddingLength && shape[1] == embeddingLength {
		return []uint64{shape[1], shape[0]}
	}
	return slices.Clone(shape)
}

func nemotron3PositionEmbeddingOutputShape(shape []uint64, embeddingLength, targetSide uint64) []uint64 {
	if targetSide == 0 {
		return slices.Clone(shape)
	}
	if embeddingLength == 0 {
		if emb, _, ok := nemotron3PositionEmbeddingLayout(shape); ok {
			embeddingLength = emb
		}
	}
	if embeddingLength == 0 {
		return slices.Clone(shape)
	}
	return []uint64{embeddingLength, targetSide * targetSide}
}

func nemotron3PositionEmbeddingLayout(shape []uint64) (embeddingLength, positions uint64, ok bool) {
	switch len(shape) {
	case 2:
		if nemotron3SquareSide(shape[1]) > 0 {
			return shape[0], shape[1], true
		}
		if nemotron3SquareSide(shape[0]) > 0 {
			return shape[1], shape[0], true
		}
	case 3:
		if shape[0] == 1 && nemotron3SquareSide(shape[1]) > 0 {
			return shape[2], shape[1], true
		}
		if shape[2] == 1 && nemotron3SquareSide(shape[1]) > 0 {
			return shape[0], shape[1], true
		}
	}
	return 0, 0, false
}

func downsampleNemotron3PositionEmbedding(name string, data []float32, shape []uint64, embeddingLength, targetSide uint64) ([]float32, error) {
	if targetSide == 0 {
		return data, nil
	}

	detectedEmbedding, positions, ok := nemotron3PositionEmbeddingLayout(shape)
	if !ok {
		return nil, fmt.Errorf("downsample %s: unsupported position embedding shape %v", name, shape)
	}
	if embeddingLength == 0 {
		embeddingLength = detectedEmbedding
	}
	if embeddingLength != detectedEmbedding {
		return nil, fmt.Errorf("downsample %s: embedding length %d does not match source shape %v", name, embeddingLength, shape)
	}
	if uint64(len(data)) != embeddingLength*positions {
		return nil, fmt.Errorf("downsample %s: data length %d does not match shape %v", name, len(data), shape)
	}

	sourceSide := nemotron3SquareSide(positions)
	if sourceSide == 0 {
		return nil, fmt.Errorf("downsample %s: position count %d is not square", name, positions)
	}
	if sourceSide == targetSide {
		return normalizeNemotron3PositionEmbeddingData(data, shape, embeddingLength, positions), nil
	}

	sourceAt := func(pos, emb uint64) float32 {
		switch len(shape) {
		case 2:
			if shape[0] == embeddingLength {
				return data[emb+embeddingLength*pos]
			}
			return data[pos+positions*emb]
		case 3:
			if shape[0] == 1 {
				return data[pos*embeddingLength+emb]
			}
			return data[emb*positions+pos]
		default:
			return 0
		}
	}

	out := make([]float32, embeddingLength*targetSide*targetSide)
	for y := range targetSide {
		sourceY := nemotron3AlignCorners(float64(y), targetSide, sourceSide)
		y0 := uint64(math.Floor(sourceY))
		y1 := min(y0+1, sourceSide-1)
		wy := float32(sourceY - float64(y0))
		for x := range targetSide {
			sourceX := nemotron3AlignCorners(float64(x), targetSide, sourceSide)
			x0 := uint64(math.Floor(sourceX))
			x1 := min(x0+1, sourceSide-1)
			wx := float32(sourceX - float64(x0))

			for emb := range embeddingLength {
				v00 := sourceAt(y0*sourceSide+x0, emb)
				v01 := sourceAt(y0*sourceSide+x1, emb)
				v10 := sourceAt(y1*sourceSide+x0, emb)
				v11 := sourceAt(y1*sourceSide+x1, emb)
				top := v00*(1-wx) + v01*wx
				bottom := v10*(1-wx) + v11*wx
				out[emb+embeddingLength*(y*targetSide+x)] = top*(1-wy) + bottom*wy
			}
		}
	}
	return out, nil
}

func normalizeNemotron3PositionEmbeddingData(data []float32, shape []uint64, embeddingLength, positions uint64) []float32 {
	if len(shape) == 2 && shape[0] == embeddingLength {
		return data
	}
	out := make([]float32, embeddingLength*positions)
	for pos := range positions {
		for emb := range embeddingLength {
			switch len(shape) {
			case 2:
				out[emb+embeddingLength*pos] = data[pos+positions*emb]
			case 3:
				if shape[0] == 1 {
					out[emb+embeddingLength*pos] = data[pos*embeddingLength+emb]
				} else {
					out[emb+embeddingLength*pos] = data[emb*positions+pos]
				}
			}
		}
	}
	return out
}

func nemotron3AlignCorners(target float64, targetSide, sourceSide uint64) float64 {
	if targetSide <= 1 {
		return 0
	}
	return target * float64(sourceSide-1) / float64(targetSide-1)
}

func nemotron3PatchEmbeddingOutputShape(shape []uint64, patchSize, channels, embeddingLength uint64) []uint64 {
	if patchSize == 0 || channels == 0 {
		return slices.Clone(shape)
	}
	flat := patchSize * patchSize * channels
	if embeddingLength == 0 && len(shape) == 2 {
		switch {
		case shape[0] == flat:
			embeddingLength = shape[1]
		case shape[1] == flat:
			embeddingLength = shape[0]
		}
	}
	if embeddingLength == 0 {
		return slices.Clone(shape)
	}
	return []uint64{patchSize, patchSize, channels, embeddingLength}
}

func reshapeNemotron3PatchEmbedding(name string, data []float32, shape []uint64, patchSize, channels, embeddingLength uint64) ([]float32, error) {
	if len(shape) != 2 || patchSize == 0 || channels == 0 {
		return data, nil
	}
	flat := patchSize * patchSize * channels
	if embeddingLength == 0 {
		switch {
		case shape[0] == flat:
			embeddingLength = shape[1]
		case shape[1] == flat:
			embeddingLength = shape[0]
		default:
			return data, nil
		}
	}
	if uint64(len(data)) != flat*embeddingLength {
		return nil, fmt.Errorf("reshape %s: data length %d does not match shape %v", name, len(data), shape)
	}

	out := make([]float32, len(data))
	for emb := range embeddingLength {
		for channel := range channels {
			for y := range patchSize {
				for x := range patchSize {
					flatIndex := x + patchSize*(y+patchSize*channel)
					dst := x + patchSize*(y+patchSize*(channel+channels*emb))
					src := flatIndex + flat*emb
					out[dst] = data[src]
				}
			}
		}
	}
	return out, nil
}

func nemotron3SquareSide(n uint64) uint64 {
	side := uint64(math.Sqrt(float64(n)))
	if side*side == n {
		return side
	}
	if (side+1)*(side+1) == n {
		return side + 1
	}
	return 0
}

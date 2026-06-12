package compatmigrate

import (
	"fmt"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/ollama/ollama/fs/ggml"
)

type glm47FlashMigrator struct{}

func (glm47FlashMigrator) NeedsMigration(src *SourceModel) bool {
	return src.GGUF.KeyValue("general.architecture").String() == "glm4moelite"
}

type glm47FlashMLA struct {
	numHeads   int
	qkNope     int
	qkRope     int
	kvLoraRank int
	vHeadDim   int
}

func (glm47FlashMigrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	mla := glm47FlashInferMLAFromTensors(glm47FlashMLAFromSource(src), tensors)
	modelKV := glm47FlashModelKV(src, mla)

	var modelTensors []*ggml.Tensor
	for _, tensor := range tensors {
		if strings.HasSuffix(tensor.name, ".attn_kv_b.weight") {
			if mla.kvLoraRank == 0 || mla.qkNope == 0 || mla.vHeadDim == 0 {
				return nil, fmt.Errorf("glm-4.7-flash migration cannot split %s without MLA dimensions", tensor.name)
			}
			split, err := glm47FlashSplitKVB(tensor, mla)
			if err != nil {
				return nil, err
			}
			modelTensors = append(modelTensors, split...)
			continue
		}

		modelTensors = append(modelTensors, copyTensor(tensor.name, tensor))
	}

	return &Result{
		ModelKV:      modelKV,
		ModelTensors: modelTensors,
		// The tensor format is a deepseek2-compatible GGUF, but chat/tool prompt
		// formatting still needs the GLM-4.7 renderer/parser. Without this config
		// chat falls back to the plain default template and the model does not
		// naturally terminate on short integration prompts.
		Renderer: "glm-4.7",
		Parser:   "glm-4.7",
	}, nil
}

func glm47FlashInferMLAFromTensors(mla glm47FlashMLA, tensors []*sourceTensor) glm47FlashMLA {
	for _, tensor := range tensors {
		switch {
		case strings.HasSuffix(tensor.name, ".attn_k_b.weight") && len(tensor.shape) == 3:
			if tensor.shape[0] > 0 {
				mla.qkNope = int(tensor.shape[0])
			}
			if tensor.shape[1] > 0 {
				mla.kvLoraRank = int(tensor.shape[1])
			}
			if tensor.shape[2] > 0 {
				mla.numHeads = int(tensor.shape[2])
			}
		case strings.HasSuffix(tensor.name, ".attn_v_b.weight") && len(tensor.shape) == 3:
			if tensor.shape[0] > 0 {
				mla.kvLoraRank = int(tensor.shape[0])
			}
			if tensor.shape[1] > 0 {
				mla.vHeadDim = int(tensor.shape[1])
			}
			if tensor.shape[2] > 0 {
				mla.numHeads = int(tensor.shape[2])
			}
		case strings.HasSuffix(tensor.name, ".attn_q_b.weight") && len(tensor.shape) == 2:
			if mla.numHeads == 0 {
				continue
			}
			switch {
			case int(tensor.shape[0]) > 0 && int(tensor.shape[1])%mla.numHeads == 0:
				if perHead := int(tensor.shape[1]) / mla.numHeads; perHead > mla.qkRope {
					mla.qkNope = perHead - mla.qkRope
				}
			case int(tensor.shape[1]) > 0 && int(tensor.shape[0])%mla.numHeads == 0:
				if perHead := int(tensor.shape[0]) / mla.numHeads; perHead > mla.qkRope {
					mla.qkNope = perHead - mla.qkRope
				}
			}
		case strings.HasSuffix(tensor.name, ".attn_kv_b.weight") && len(tensor.shape) == 2:
			if mla.kvLoraRank == 0 || mla.vHeadDim == 0 || mla.numHeads == 0 || mla.qkNope > 0 {
				continue
			}
			switch {
			case int(tensor.shape[0]) == mla.kvLoraRank && int(tensor.shape[1])%mla.numHeads == 0:
				if perHead := int(tensor.shape[1]) / mla.numHeads; perHead > mla.vHeadDim {
					mla.qkNope = perHead - mla.vHeadDim
				}
			case int(tensor.shape[1]) == mla.kvLoraRank && int(tensor.shape[0])%mla.numHeads == 0:
				if perHead := int(tensor.shape[0]) / mla.numHeads; perHead > mla.vHeadDim {
					mla.qkNope = perHead - mla.vHeadDim
				}
			}
		}
	}

	return mla
}

func glm47FlashModelKV(src *SourceModel, mla glm47FlashMLA) ggml.KV {
	modelKV := ggml.KV{}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		key := keyValue.Key
		switch {
		case key == "general.architecture":
			modelKV[key] = "deepseek2"
		case strings.HasPrefix(key, "glm4moelite."):
			modelKV[strings.Replace(key, "glm4moelite.", "deepseek2.", 1)] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		case strings.HasPrefix(key, "general."), strings.HasPrefix(key, "tokenizer."):
			modelKV[key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		}
	}

	if mla.numHeads > 0 {
		modelKV["deepseek2.attention.head_count"] = uint32(mla.numHeads)
	}
	modelKV["deepseek2.attention.head_count_kv"] = uint32(1)
	if mla.kvLoraRank > 0 {
		modelKV["deepseek2.attention.kv_lora_rank"] = uint32(mla.kvLoraRank)
		modelKV["deepseek2.attention.value_length"] = uint32(mla.kvLoraRank)
	}
	if mla.qkRope > 0 {
		modelKV["deepseek2.rope.dimension_count"] = uint32(mla.qkRope)
	}
	if mla.kvLoraRank > 0 && mla.qkRope > 0 {
		modelKV["deepseek2.attention.key_length"] = uint32(mla.kvLoraRank + mla.qkRope)
	}
	if mla.qkNope > 0 && mla.qkRope > 0 {
		modelKV["deepseek2.attention.key_length_mla"] = uint32(mla.qkNope + mla.qkRope)
	}
	if mla.vHeadDim > 0 {
		modelKV["deepseek2.attention.value_length_mla"] = uint32(mla.vHeadDim)
	}
	modelKV["tokenizer.ggml.pre"] = "glm4"
	glm47FlashSetExtraEOGFromEOSIDs(modelKV)

	return modelKV
}

func glm47FlashSetExtraEOGFromEOSIDs(kv ggml.KV) {
	switch ids := kv["tokenizer.ggml.eos_token_ids"].(type) {
	case []int32:
		if len(ids) >= 2 && ids[1] >= 0 {
			kv["tokenizer.ggml.eot_token_id"] = uint32(ids[1])
		}
		if len(ids) >= 3 && ids[2] >= 0 {
			kv["tokenizer.ggml.eom_token_id"] = uint32(ids[2])
		}
	case []uint32:
		if len(ids) >= 2 {
			kv["tokenizer.ggml.eot_token_id"] = ids[1]
		}
		if len(ids) >= 3 {
			kv["tokenizer.ggml.eom_token_id"] = ids[2]
		}
	}
}

func glm47FlashMLAFromSource(src *SourceModel) glm47FlashMLA {
	mla := glm47FlashMLA{
		numHeads:   int(src.GGUF.KeyValue("attention.head_count").Uint()),
		qkRope:     int(src.GGUF.KeyValue("rope.dimension_count").Uint()),
		kvLoraRank: int(src.GGUF.KeyValue("attention.kv_lora_rank").Uint()),
		vHeadDim:   int(src.GGUF.KeyValue("attention.value_length_mla").Uint()),
	}
	if mla.vHeadDim == 0 {
		mla.vHeadDim = int(src.GGUF.KeyValue("attention.value_length").Uint())
	}
	if keyMLA := int(src.GGUF.KeyValue("attention.key_length_mla").Uint()); keyMLA > mla.qkRope {
		mla.qkNope = keyMLA - mla.qkRope
	} else if key := int(src.GGUF.KeyValue("attention.key_length").Uint()); key > mla.qkRope && key != mla.kvLoraRank+mla.qkRope {
		mla.qkNope = key - mla.qkRope
	}
	return mla
}

func glm47FlashSplitKVB(t *sourceTensor, mla glm47FlashMLA) ([]*ggml.Tensor, error) {
	if len(t.shape) != 2 {
		return nil, fmt.Errorf("glm-4.7-flash migration expected %s to be 2D, got %v", t.name, t.shape)
	}

	kvPerHead := mla.qkNope + mla.vHeadDim
	numHeads := mla.numHeads
	var kvFirst bool
	switch {
	case int(t.shape[0]) == mla.kvLoraRank:
		if kvPerHead > 0 && int(t.shape[1])%kvPerHead == 0 {
			numHeads = int(t.shape[1]) / kvPerHead
		}
		kvFirst = true
	case int(t.shape[1]) == mla.kvLoraRank:
		if kvPerHead > 0 && int(t.shape[0])%kvPerHead == 0 {
			numHeads = int(t.shape[0]) / kvPerHead
		}
		kvFirst = false
	default:
		return nil, fmt.Errorf("glm-4.7-flash migration cannot infer %s layout from shape %v", t.name, t.shape)
	}

	kTensor := t.Clone()
	kTensor.SetRepacker(glm47FlashRepackKVB(mla, true, kvFirst, numHeads))
	vTensor := t.Clone()
	vTensor.SetRepacker(glm47FlashRepackKVB(mla, false, kvFirst, numHeads))

	return []*ggml.Tensor{
		{
			Name:     strings.Replace(t.name, "attn_kv_b", "attn_k_b", 1),
			Kind:     uint32(t.info.Type),
			Shape:    []uint64{uint64(mla.qkNope), uint64(mla.kvLoraRank), uint64(numHeads)},
			WriterTo: kTensor,
		},
		{
			Name:     strings.Replace(t.name, "attn_kv_b", "attn_v_b", 1),
			Kind:     uint32(t.info.Type),
			Shape:    []uint64{uint64(mla.kvLoraRank), uint64(mla.vHeadDim), uint64(numHeads)},
			WriterTo: vTensor,
		},
	}, nil
}

func glm47FlashRepackKVB(mla glm47FlashMLA, extractK bool, kvFirst bool, numHeads int) repacker {
	return func(_ string, data []float32, shape []uint64) ([]float32, error) {
		dims := make([]int, len(shape))
		for i := range shape {
			dims[i] = int(shape[i])
		}

		var tt tensor.Tensor = tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))
		var err error

		if kvFirst {
			tt, err = tensor.Transpose(tt, 1, 0)
			if err != nil {
				return nil, err
			}
			tt = tensor.Materialize(tt)
		}

		if err := tt.Reshape(numHeads, mla.qkNope+mla.vHeadDim, mla.kvLoraRank); err != nil {
			return nil, err
		}

		if extractK {
			tt, err = tt.Slice(nil, tensor.S(0, mla.qkNope), nil)
			if err != nil {
				return nil, err
			}
			tt = tensor.Materialize(tt)
			tt, err = tensor.Transpose(tt, 0, 2, 1)
			if err != nil {
				return nil, err
			}
			tt = tensor.Materialize(tt)
		} else {
			tt, err = tt.Slice(nil, tensor.S(mla.qkNope, mla.qkNope+mla.vHeadDim), nil)
			if err != nil {
				return nil, err
			}
			tt = tensor.Materialize(tt)
		}

		if err := tt.Reshape(tt.Shape().TotalSize()); err != nil {
			return nil, err
		}
		return native.VectorF32(tt.(*tensor.Dense))
	}
}

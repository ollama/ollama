package convert

import (
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"
)

type mllamaModel struct {
	ModelParameters
	TextModel struct {
		llamaModel

		CrossAttentionLayers []int32 `json:"cross_attention_layers"`
	} `json:"text_config"`
	VisionModel struct {
		NumHiddenLayers           uint32  `json:"num_hidden_layers"`
		NumGlobalLayers           uint32  `json:"num_global_layers"`
		IntermediateLayersIndices []int32 `json:"intermediate_layers_indices"`

		HiddenSize       uint32 `json:"hidden_size"`
		IntermediateSize uint32 `json:"intermediate_size"`

		AttentionHeads uint32 `json:"attention_heads"`

		ImageSize   uint32  `json:"image_size"`
		PatchSize   uint32  `json:"patch_size"`
		NumChannels uint32  `json:"num_channels"`
		MaxNumTiles uint32  `json:"max_num_tiles"`
		NormEpsilon float32 `json:"norm_eps"`
		RopeTheta   float32 `json:"rope.freq_base"`
	} `json:"vision_config"`
}

func (m *mllamaModel) KV(t *Tokenizer) KV {
	kv := m.ModelParameters.KV(t)
	kv["general.architecture"] = "mllama"

	for k, v := range m.TextModel.KV(t) {
		if strings.HasPrefix(k, "llama.") {
			kv[strings.ReplaceAll(k, "llama.", "mllama.")] = v
		}
	}

	kv["mllama.attention.cross_attention_layers"] = m.TextModel.CrossAttentionLayers

	kv["mllama.vision.block_count"] = m.VisionModel.NumHiddenLayers
	kv["mllama.vision.global.block_count"] = m.VisionModel.NumGlobalLayers
	kv["mllama.vision.intermediate_layers_indices"] = m.VisionModel.IntermediateLayersIndices

	kv["mllama.vision.embedding_length"] = m.VisionModel.HiddenSize
	kv["mllama.vision.feed_forward_length"] = m.VisionModel.IntermediateSize

	kv["mllama.vision.attention.head_count"] = m.VisionModel.AttentionHeads
	kv["mllama.vision.attention.layer_norm_epsilon"] = m.VisionModel.NormEpsilon

	kv["mllama.vision.image_size"] = m.VisionModel.ImageSize
	kv["mllama.vision.patch_size"] = m.VisionModel.PatchSize
	kv["mllama.vision.max_num_tiles"] = m.VisionModel.MaxNumTiles
	kv["mllama.vision.num_channels"] = m.VisionModel.NumChannels

	return kv
}

func (m *mllamaModel) Replacements() []string {
	return append(
		m.TextModel.Replacements(),
		"language_model.", "",
		"gate_attn", "attn_gate",
		"gate_ffn", "ffn_gate",
		"cross_attn.", "cross_attn_",
		"vision_model", "v",
		"class_embedding", "class_embd",
		"patch_embedding", "patch_embd",
		"gated_positional_embedding.tile_embedding", "tile_position_embd",
		"gated_positional_embedding.embedding", "position_embd.weight",
		"gated_positional_embedding", "position_embd",
		"embedding.weight", "weight",
		"pre_tile_positional_embedding", "pre_tile_position_embd",
		"post_tile_positional_embedding", "post_tile_position_embd",
		"layernorm_pre", "pre_ln",
		"layernorm_post", "post_ln",
		"global_transformer.layers", "global.blk",
		"transformer.layers", "blk",
		"mlp.fc1", "ffn_up",
		"mlp.fc2", "ffn_down",
		"multi_modal_projector", "mm.0",
	)
}

func (m *mllamaModel) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor
	var text []Tensor
	for _, t := range ts {
		if !strings.HasPrefix(t.Name(), "v.") && !strings.HasPrefix(t.Name(), "mm.") {
			text = append(text, t)
		} else if t.Name() == "v.position_embd.gate" {
			for _, name := range []string{"v.position_embd.gate", "v.tile_position_embd.gate"} {
				tt := t.Clone()
				tt.SetRepacker(m.repack(name))
				out = append(out, &ggml.Tensor{
					Name:     name,
					Kind:     t.Kind(),
					Shape:    t.Shape(),
					WriterTo: tt,
				})
			}
		} else {
			if t.Name() == "v.pre_tile_position_embd.gate" || t.Name() == "v.post_tile_position_embd.gate" {
				t.SetRepacker(m.repack(t.Name()))
			} else if strings.HasSuffix(t.Name(), "attn_q.weight") || strings.HasSuffix(t.Name(), "attn_k.weight") {
				t.SetRepacker(m.repack(t.Name()))
			} else if strings.HasSuffix(t.Name(), "attn_gate") || strings.HasSuffix(t.Name(), "ffn_gate") {
				t.SetRepacker(m.repack(t.Name()))
			}

			out = append(out, &ggml.Tensor{
				Name:     t.Name(),
				Kind:     t.Kind(),
				Shape:    t.Shape(),
				WriterTo: t,
			})
		}
	}

	return append(out, m.TextModel.Tensors(text)...)
}

func (m *mllamaModel) repack(name string) Repacker {
	return func(_ string, data []float32, shape []uint64) (_ []float32, err error) {
		dims := make([]int, len(shape))
		for i, dim := range shape {
			dims[i] = int(dim)
		}

		var t tensor.Tensor = tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))

		if strings.HasSuffix(name, "attn_q.weight") || strings.HasSuffix(name, "attn_k.weight") {
			heads := m.VisionModel.AttentionHeads
			if err := t.Reshape(append([]int{int(heads), 2, dims[0] / int(heads) / 2}, dims[1:]...)...); err != nil {
				return nil, err
			}

			if err := t.T(0, 2, 1, 3); err != nil {
				return nil, err
			}

			if err := t.Reshape(dims...); err != nil {
				return nil, err
			}

			if err := t.Transpose(); err != nil {
				return nil, err
			}
		} else {
			t, err = tensor.Tanh(t)
			if err != nil {
				return nil, err
			}

			if name == "v.position_embd.gate" {
				t, err = tensor.Sub(float32(1), t)
				if err != nil {
					return nil, err
				}
			}
		}

		t = tensor.Materialize(t)
		// flatten tensor so it can be return as a vector
		if err := t.Reshape(t.Shape().TotalSize()); err != nil {
			return nil, err
		}

		return native.VectorF32(t.(*tensor.Dense))
	}
}

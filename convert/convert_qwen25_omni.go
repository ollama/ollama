package convert

import (
	"bytes"
	"encoding/binary"
	"io"
	"log/slog"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/x448/float16"

	"github.com/ollama/ollama/fs/ggml"
)

type qwen25OmniModel struct {
	ModelParameters
	TalkerModel struct {
		AudioEndTokenID       uint32  `json:"audio_end_token_id"`
		AudioStartTokenID     uint32  `json:"audio_start_token_id"`
		AudioTokenIndex       uint32  `json:"audio_token_index"`
		HeadDim               uint32  `json:"head_dim"`
		HiddenSize            uint32  `json:"hidden_size"`
		ImageTokenIndex       uint32  `json:"image_token_index"`
		IntermediateSize      uint32  `json:"intermediate_size"`
		MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
		MaxWindowLayers       uint32  `json:"max_window_layers"`
		NumAttentionHeads     uint32  `json:"num_attention_heads"`
		HiddenLayers          uint32  `json:"num_hidden_layers"`
		NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
		RMSNormEPS            float32 `json:"rms_norm_eps"`
		RopeTheta             float32 `json:"rope_theta"`
		VideoTokenIndex       uint32  `json:"video_token_index"`
		VisionEndTokenID      uint32  `json:"vision_end_token_id"`
		VisionStartTokenID    uint32  `json:"vision_start_token_id"`
	} `json:"talker_config"`

	ThinkerModel struct {
		TextModel struct {
			HiddenSize            uint32  `json:"hidden_size"`
			IntermediateSize      uint32  `json:"intermediate_size"`
			MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
			NumAttentionHeads     uint32  `json:"num_attention_heads"`
			HiddenLayers          uint32  `json:"num_hidden_layers"`
			RopeTheta             float32 `json:"rope_theta"`
			NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
			RMSNormEPS            float32 `json:"rms_norm_eps"`
		} `json:"text_config"`
	} `json:"thinker_config"`

	VisionModel struct {
	} `json:"vision_config"`

	Token2WavModel struct {
	} `json:"token2wav_config"`
}

var _ ModelConverter = (*qwen25OmniModel)(nil)

func (q *qwen25OmniModel) KV(t *Tokenizer) ggml.KV {
	kv := q.ModelParameters.KV(t)
	kv["general.architecture"] = "qwen25omni"
	kv["qwen25omni.block_count"] = q.ThinkerModel.TextModel.HiddenLayers
	kv["qwen25omni.context_length"] = q.ThinkerModel.TextModel.MaxPositionEmbeddings
	kv["qwen25omni.embedding_length"] = q.ThinkerModel.TextModel.HiddenSize
	kv["qwen25omni.feed_forward_length"] = q.ThinkerModel.TextModel.IntermediateSize
	kv["qwen25omni.attention.head_count"] = q.ThinkerModel.TextModel.NumAttentionHeads
	kv["qwen25omni.attention.head_count_kv"] = q.ThinkerModel.TextModel.NumKeyValueHeads
	kv["qwen25omni.rope.freq_base"] = q.ThinkerModel.TextModel.RopeTheta
	kv["qwen25omni.attention.layer_norm_rms_epsilon"] = q.ThinkerModel.TextModel.RMSNormEPS

	return kv
}

func (q *qwen25OmniModel) Tensors(ts []Tensor) []ggml.Tensor {
	var out []ggml.Tensor

	for _, t := range ts {
		if strings.HasSuffix(t.Name(), "patch_embed.proj.weight") {
			var buf bytes.Buffer
			t.WriteTo(&buf)
			newTensors := splitPatchEmbed(buf, t.Kind(), t.Shape())
			out = append(out, newTensors...)
		} else {
			out = append(out, ggml.Tensor{
				Name:     t.Name(),
				Kind:     t.Kind(),
				Shape:    t.Shape(),
				WriterTo: t,
			})
		}
	}

	return out
}

func splitPatchEmbed(buf bytes.Buffer, kind uint32, shape []uint64) []ggml.Tensor {
	slog.Debug("patch stuff", "kind", kind, "shape", shape)

	if kind != tensorKindF16 {
		panic("tensor is of wrong type")
	}

	if len(shape) != 5 || (len(shape) == 5 && shape[2] != 2) {
		panic("wrong sized tensor")
	}

	// determine the size of the tensor based on its shape
	shapeToSize := func(s []int) int {
		r := 1
		for _, n := range s {
			r *= int(n)
		}
		return r
	}

	// tensor.WithShape() wants []int
	intShape := make([]int, len(shape))
	for i, v := range shape {
		intShape[i] = int(v)
	}

	u16s := make([]uint16, shapeToSize(intShape))
	if err := binary.Read(&buf, binary.LittleEndian, u16s); err != nil {
		panic("bad read")
	}

	f32s := make([]float32, len(u16s))
	for i := range u16s {
		f32s[i] = float16.Frombits(u16s[i]).Float32()
	}

	newTensors := []ggml.Tensor{}

	getDataFromSlice := func(f32s []float32, shape []int, s []tensor.Slice) patchEmbed {
		slog.Debug("getDataFromSlice", "num f32s", len(f32s), "shape", shape)
		n := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(f32s))
		t, err := n.Slice(s...)
		if err != nil {
			panic(err)
		}

		ts, err := native.SelectF32(t.Materialize().(*tensor.Dense), 0)
		if err != nil {
			panic(err)
		}

		slog.Debug("first vals", "val 1", ts[0][0], "val 2", ts[0][1], "val 3", ts[0][2])

		f16s := make(patchEmbed, shapeToSize(shape))
		for r, row := range ts {
			for c, col := range row {
				f16s[r+c] = float16.Fromfloat32(col).Bits()
			}
		}

		return f16s
	}

	p := getDataFromSlice(f32s, intShape, []tensor.Slice{nil, nil, tensor.S(0, 1, 1), nil, nil})
	newTensors = append(newTensors, ggml.Tensor{
		Name:     "patch_embed.proj.0.weight",
		Kind:     kind,
		Shape:    append(shape[:2], shape[3:]...),
		WriterTo: p,
	})

	p = getDataFromSlice(f32s, intShape, []tensor.Slice{nil, nil, tensor.S(1, 2, 1), nil, nil})
	newTensors = append(newTensors, ggml.Tensor{
		Name:     "patch_embed.proj.1.weight",
		Kind:     kind,
		Shape:    append(shape[:2], shape[3:]...),
		WriterTo: p,
	})

	return newTensors
}

type patchEmbed []uint16

func (t patchEmbed) WriteTo(w io.Writer) (int64, error) {
	err := binary.Write(w, binary.LittleEndian, t)
	return 0, err
}

func (p *qwen25OmniModel) Replacements() []string {
	return []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"thinker.audio_tower.layers", "a.blk",
		"thinker.visual.blocks", "v.blk",
		"thinker.model.layers", "blk",
		"talker.model.layers", "tlk.blk",
		"token2wav.code2wav_bigvgan_model", "t2w.b",
		"token2wav.code2wav_dit_model", "t2w.d",
		"input_layernorm", "attn_norm",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.q_proj", "attn_q",
		"self_attn.o_proj", "attn_output",
		"mlp.down_proj", "ffn_down",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
		"post_attention_layernorm", "ffn_norm",
		"model.norm", "output_norm",
	}
}

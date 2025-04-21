package convert

import (
	"bytes"
	"encoding/binary"
	"io"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"
	"github.com/x448/float16"
)

type qwen25VLModel struct {
	ModelParameters
	HiddenSize            uint32  `json:"hidden_size"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	HiddenLayers          uint32  `json:"num_hidden_layers"`
	RopeTheta             float32 `json:"rope_theta"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	RMSNormEPS            float32 `json:"rms_norm_eps"`

	VisionModel struct {
	} `json:"vision_config"`
}

var _ ModelConverter = (*qwen25VLModel)(nil)

func (q *qwen25VLModel) KV(t *Tokenizer) ggml.KV {
	kv := q.ModelParameters.KV(t)
	kv["general.architecture"] = "qwen25vl"
	kv["qwen25vl.block_count"] = q.HiddenLayers
	kv["qwen25vl.context_length"] = q.MaxPositionEmbeddings
	kv["qwen25vl.embedding_length"] = q.HiddenSize
	kv["qwen25vl.feed_forward_length"] = q.IntermediateSize
	kv["qwen25vl.attention.head_count"] = q.NumAttentionHeads
	kv["qwen25vl.attention.head_count_kv"] = q.NumKeyValueHeads
	kv["qwen25vl.rope.freq_base"] = q.RopeTheta
	kv["qwen25vl.attention.layer_norm_rms_epsilon"] = q.RMSNormEPS

	return kv
}

func (q *qwen25VLModel) Tensors(ts []Tensor) []ggml.Tensor {
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

func (p *qwen25VLModel) Replacements() []string {
	return []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.layers", "blk",
		"visual.blocks", "v.blk",
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
		Name:     "v.patch_embed_0.weight",
		Kind:     kind,
		Shape:    append(shape[:2], shape[3:]...),
		WriterTo: p,
	})

	p = getDataFromSlice(f32s, intShape, []tensor.Slice{nil, nil, tensor.S(1, 2, 1), nil, nil})
	newTensors = append(newTensors, ggml.Tensor{
		Name:     "v.patch_embed_1.weight",
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

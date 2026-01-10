package convert

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"io"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"
)

type gptossModel struct {
	ModelParameters
	HiddenLayers          uint32  `json:"num_hidden_layers"`
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	HiddenSize            uint32  `json:"hidden_size"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	AttentionHeads        uint32  `json:"num_attention_heads"`
	KeyValueHeads         uint32  `json:"num_key_value_heads"`
	HeadDim               uint32  `json:"head_dim"`
	Experts               uint32  `json:"num_experts"`
	LocalExperts          uint32  `json:"num_local_experts"`
	ExpertsPerToken       uint32  `json:"experts_per_token"`
	RMSNormEpsilon        float32 `json:"rms_norm_eps"`
	InitialContextLength  uint32  `json:"initial_context_length"`
	RopeTheta             float32 `json:"rope_theta"`
	RopeScalingFactor     float32 `json:"rope_scaling_factor"`
	RopeScaling           struct {
		Factor float32 `json:"factor"`
	} `json:"rope_scaling"`
	SlidingWindow uint32 `json:"sliding_window"`
}

var _ ModelConverter = (*gptossModel)(nil)

func (m *gptossModel) KV(t *Tokenizer) KV {
	kv := m.ModelParameters.KV(t)
	kv["general.architecture"] = "gptoss"
	kv["general.file_type"] = uint32(4)
	kv["gptoss.context_length"] = cmp.Or(m.MaxPositionEmbeddings, uint32(m.RopeScalingFactor*float32(m.InitialContextLength)))
	kv["gptoss.block_count"] = m.HiddenLayers
	kv["gptoss.embedding_length"] = m.HiddenSize
	kv["gptoss.feed_forward_length"] = m.IntermediateSize
	kv["gptoss.expert_count"] = cmp.Or(m.Experts, m.LocalExperts)
	kv["gptoss.expert_used_count"] = m.ExpertsPerToken
	kv["gptoss.attention.head_count"] = m.AttentionHeads
	kv["gptoss.attention.head_count_kv"] = m.KeyValueHeads
	kv["gptoss.attention.key_length"] = m.HeadDim
	kv["gptoss.attention.value_length"] = m.HeadDim
	kv["gptoss.attention.layer_norm_rms_epsilon"] = cmp.Or(m.RMSNormEpsilon, 1e-5)
	kv["gptoss.attention.sliding_window"] = m.SlidingWindow
	kv["gptoss.rope.freq_base"] = m.RopeTheta
	kv["gptoss.rope.scaling.factor"] = cmp.Or(m.RopeScalingFactor, m.RopeScaling.Factor)
	kv["gptoss.rope.scaling.original_context_length"] = m.InitialContextLength
	kv["tokenizer.ggml.bos_token_id"] = uint32(199998) // <|startoftext|>
	kv["tokenizer.ggml.add_bos_token"] = false
	kv["tokenizer.ggml.eos_token_id"] = uint32(199999) // <|endoftext|>
	kv["tokenizer.ggml.eos_token_ids"] = []int32{
		199999, /* <|endoftext|> */
		200002, /* <|return|> */
		200012, /* <|call|> */
	}
	kv["tokenizer.ggml.add_eos_token"] = false
	return kv
}

func (m *gptossModel) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor
	mxfp4s := make(map[string]*mxfp4)
	for _, t := range ts {
		if strings.HasSuffix(t.Name(), ".blocks") || strings.HasSuffix(t.Name(), ".scales") {
			dot := strings.LastIndex(t.Name(), ".")
			name, suffix := t.Name()[:dot], t.Name()[dot+1:]
			if _, ok := mxfp4s[name]; !ok {
				mxfp4s[name] = &mxfp4{}
			}

			switch suffix {
			case "blocks":
				mxfp4s[name].blocks = t
			case "scales":
				mxfp4s[name].scales = t
			}
		} else if strings.HasSuffix(t.Name(), "gate_up_exps.bias") {
			// gate_up_exps is interleaved, need to split into gate_exps and up_exps
			// e.g. gate_exps, up_exps = gate_up_exps[:, 0::2, ...], gate_up_exps[:, 1::2, ...]
			out = append(out, slices.Collect(splitDim(t, 1,
				split{
					Replacer: strings.NewReplacer("gate_up_exps", "gate_exps"),
					slices:   []tensor.Slice{nil, tensor.S(0, int(t.Shape()[1]), 2)},
				},
				split{
					Replacer: strings.NewReplacer("gate_up_exps", "up_exps"),
					slices:   []tensor.Slice{nil, tensor.S(1, int(t.Shape()[1]), 2)},
				},
			))...)
		} else {
			out = append(out, &ggml.Tensor{
				Name:     t.Name(),
				Kind:     t.Kind(),
				Shape:    t.Shape(),
				WriterTo: t,
			})
		}
	}

	for name, mxfp4 := range mxfp4s {
		dims := mxfp4.blocks.Shape()
		if !strings.HasSuffix(name, ".weight") {
			name = name + ".weight"
		}
		if strings.Contains(name, "ffn_down_exps") {
			out = append(out, &ggml.Tensor{
				Name:     name,
				Kind:     uint32(ggml.TensorTypeMXFP4),
				Shape:    []uint64{dims[0], dims[1], dims[2] * dims[3] * 2},
				WriterTo: mxfp4,
			})
		} else if strings.Contains(name, "ffn_gate_up_exps") {
			// gate_up_exps is interleaved, need to split into gate_exps and up_exps
			// e.g. gate_exps, up_exps = gate_up_exps[:, 0::2, ...], gate_up_exps[:, 1::2, ...]
			out = append(out, &ggml.Tensor{
				Name:     strings.Replace(name, "gate_up", "gate", 1),
				Kind:     uint32(ggml.TensorTypeMXFP4),
				Shape:    []uint64{dims[0], dims[1] / 2, dims[2] * dims[3] * 2},
				WriterTo: mxfp4.slice(1, 0, int(dims[1]), 2),
			}, &ggml.Tensor{
				Name:     strings.Replace(name, "gate_up", "up", 1),
				Kind:     uint32(ggml.TensorTypeMXFP4),
				Shape:    []uint64{dims[0], dims[1] / 2, dims[2] * dims[3] * 2},
				WriterTo: mxfp4.slice(1, 1, int(dims[1]), 2),
			})
		}
	}

	return out
}

func (m *gptossModel) Replacements() []string {
	var replacements []string
	if m.MaxPositionEmbeddings > 0 {
		// hf flavored model
		replacements = []string{
			"lm_head", "output",
			"model.embed_tokens", "token_embd",
			"model.layers", "blk",
			"input_layernorm", "attn_norm",
			"self_attn.q_proj", "attn_q",
			"self_attn.k_proj", "attn_k",
			"self_attn.v_proj", "attn_v",
			"self_attn.o_proj", "attn_out",
			"self_attn.sinks", "attn_sinks",
			"post_attention_layernorm", "ffn_norm",
			"mlp.router", "ffn_gate_inp",
			"mlp.experts.gate_up_proj_", "ffn_gate_up_exps.",
			"mlp.experts.down_proj_", "ffn_down_exps.",
			"model.norm", "output_norm",
		}
	} else {
		replacements = []string{
			// noop replacements so other replacements will not be applied
			".blocks", ".blocks",
			".scales", ".scales",
			// real replacements
			"block", "blk",
			"attn.norm", "attn_norm",
			"attn.qkv", "attn_qkv",
			"attn.sinks", "attn_sinks",
			"attn.out", "attn_out",
			"mlp.norm", "ffn_norm",
			"mlp.gate", "ffn_gate_inp",
			"mlp.mlp1_", "ffn_gate_up_exps.",
			"mlp.mlp2_", "ffn_down_exps.",
			"embedding", "token_embd",
			"norm", "output_norm",
			"unembedding", "output",
			"scale", "weight",
		}
	}
	return replacements
}

type mxfp4 struct {
	slices []tensor.Slice

	blocks, scales Tensor
}

func (m *mxfp4) slice(dim, start, end, step int) *mxfp4 {
	slice := slices.Repeat([]tensor.Slice{nil}, len(m.blocks.Shape()))
	slice[dim] = tensor.S(start, end, step)
	return &mxfp4{
		slices: slice,
		blocks: m.blocks,
		scales: m.scales,
	}
}

func (m *mxfp4) WriteTo(w io.Writer) (int64, error) {
	var b bytes.Buffer
	if _, err := m.blocks.WriteTo(&b); err != nil {
		return 0, err
	}

	blocksDims := make([]int, len(m.blocks.Shape()))
	for i, d := range m.blocks.Shape() {
		blocksDims[i] = int(d)
	}

	bts := b.Bytes()
	var tmp [16]byte
	for i := 0; i < b.Len(); i += 16 {
		for j := range 8 {
			// transform a1b2c3 ... x7y8z9 -> 71xa82yb93zc
			a, b := bts[i+j], bts[i+j+8]
			tmp[2*j+0] = (a & 0x0F) | (b << 4)
			tmp[2*j+1] = (a >> 4) | (b & 0xF0)
		}

		copy(bts[i:i+16], tmp[:])
	}

	var blocks tensor.Tensor = tensor.New(tensor.WithShape(blocksDims...), tensor.WithBacking(bts))

	var s bytes.Buffer
	if _, err := m.scales.WriteTo(&s); err != nil {
		return 0, err
	}

	scalesDims := slices.Repeat([]int{1}, len(m.blocks.Shape()))
	for i, d := range m.scales.Shape() {
		scalesDims[i] = int(d)
	}

	var scales tensor.Tensor = tensor.New(tensor.WithShape(scalesDims...), tensor.WithBacking(s.Bytes()))

	out, err := tensor.Concat(3, scales, blocks)
	if err != nil {
		return 0, err
	}

	if len(m.slices) > 0 {
		out, err = out.Slice(m.slices...)
		if err != nil {
			return 0, err
		}
	}

	out = tensor.Materialize(out)

	if err := out.Reshape(out.Shape().TotalSize()); err != nil {
		return 0, err
	}

	u8s, err := native.VectorU8(out.(*tensor.Dense))
	if err != nil {
		return 0, err
	}

	if err := binary.Write(w, binary.LittleEndian, u8s); err != nil {
		return 0, err
	}

	return int64(len(u8s)), nil
}

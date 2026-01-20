package convert

import (
	"encoding/binary"
	"io"
	"slices"
	"strings"

	"github.com/d4l3k/go-bfloat16"
	"github.com/ollama/ollama/fs/ggml"
)

// bf16ToF32 wraps a tensor and converts BF16 data to F32 on write
type bf16ToF32 struct {
	Tensor
}

func (c bf16ToF32) WriteTo(w io.Writer) (int64, error) {
	// Read BF16 data from original tensor
	var buf strings.Builder
	if _, err := c.Tensor.WriteTo(&buf); err != nil {
		return 0, err
	}
	bf16Data := []byte(buf.String())

	// Convert BF16 to F32
	f32s := bfloat16.DecodeFloat32(bf16Data)

	// Write F32 data
	if err := binary.Write(w, binary.LittleEndian, f32s); err != nil {
		return 0, err
	}
	return int64(len(f32s) * 4), nil
}

type lfm2Model struct {
	ModelParameters
	HiddenSize            uint32   `json:"hidden_size"`
	NumHiddenLayers       uint32   `json:"num_hidden_layers"`
	MaxPositionEmbeddings uint32   `json:"max_position_embeddings"`
	IntermediateSize      uint32   `json:"intermediate_size"`
	NumAttentionHeads     uint32   `json:"num_attention_heads"`
	NumKeyValueHeads      uint32   `json:"num_key_value_heads"`
	RopeTheta             float32  `json:"rope_theta"`
	NormEps               float32  `json:"norm_eps"`
	ConvLCache            uint32   `json:"conv_L_cache"`
	LayerTypes            []string `json:"layer_types"`
	TieEmbedding          bool     `json:"tie_embedding"`
}

var _ ModelConverter = (*lfm2Model)(nil)

func (p *lfm2Model) KV(t *Tokenizer) KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "lfm2"
	kv["lfm2.vocab_size"] = p.VocabSize
	kv["lfm2.block_count"] = p.NumHiddenLayers
	kv["lfm2.embedding_length"] = p.HiddenSize
	kv["lfm2.feed_forward_length"] = p.IntermediateSize
	kv["lfm2.context_length"] = p.MaxPositionEmbeddings

	// Build per-layer KV head count array based on layer_types
	// (0 = shortconv layer, non-zero = attention layer with that many KV heads)
	kvHeadCounts := make([]uint32, p.NumHiddenLayers)
	for i := range p.NumHiddenLayers {
		if int(i) < len(p.LayerTypes) && p.LayerTypes[i] == "full_attention" {
			kvHeadCounts[i] = p.NumKeyValueHeads
		}
	}

	kv["lfm2.attention.head_count"] = p.NumAttentionHeads
	kv["lfm2.attention.head_count_kv"] = kvHeadCounts
	kv["lfm2.attention.key_length"] = p.HiddenSize / p.NumAttentionHeads
	kv["lfm2.attention.value_length"] = p.HiddenSize / p.NumAttentionHeads
	kv["lfm2.attention.layer_norm_rms_epsilon"] = p.NormEps
	kv["lfm2.rope.freq_base"] = p.RopeTheta
	kv["lfm2.shortconv.l_cache"] = p.ConvLCache

	return kv
}

func (p *lfm2Model) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	for _, t := range ts {
		shape := t.Shape()
		kind := t.Kind()
		var writer io.WriterTo = t

		// Squeeze conv weights: [D, 1, K] -> [D, K] and convert to F32
		if strings.HasSuffix(t.Name(), "shortconv.conv.weight") {
			if len(shape) == 3 && shape[1] == 1 {
				shape = []uint64{shape[0], shape[2]}
			}
			// Convert BF16 to F32 for accuracy (small kernel, runtime casts to F32 anyway)
			if kind == tensorKindBF16 {
				kind = tensorKindFP32
				writer = bf16ToF32{t}
			}
		}

		out = append(out, &ggml.Tensor{
			Name:     t.Name(),
			Kind:     kind,
			Shape:    slices.Clone(shape),
			WriterTo: writer,
		})
	}

	return out
}

func (p *lfm2Model) Replacements() []string {
	return []string{
		"model.embed_tokens", "token_embd",
		"model.embedding_norm", "output_norm",
		"model.layers", "blk",
		"operator_norm", "attn_norm",
		"self_attn.q_proj", "attn_q",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.out_proj", "attn_output",
		"self_attn.q_layernorm", "attn_q_norm",
		"self_attn.k_layernorm", "attn_k_norm",
		"conv.conv", "shortconv.conv",
		"conv.in_proj", "shortconv.in_proj",
		"conv.out_proj", "shortconv.out_proj",
		"feed_forward.w1", "ffn_gate",
		"feed_forward.w2", "ffn_down",
		"feed_forward.w3", "ffn_up",
		"ffn_norm", "ffn_norm",
	}
}

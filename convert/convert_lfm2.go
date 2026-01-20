package convert

import (
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

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

	// Build per-layer head count arrays based on layer_types
	headCounts := make([]uint32, p.NumHiddenLayers)
	kvHeadCounts := make([]uint32, p.NumHiddenLayers)
	for i := uint32(0); i < p.NumHiddenLayers; i++ {
		if i < uint32(len(p.LayerTypes)) && p.LayerTypes[i] == "full_attention" {
			headCounts[i] = p.NumAttentionHeads
			kvHeadCounts[i] = p.NumKeyValueHeads
		} else {
			// Conv layers have 0 head counts
			headCounts[i] = 0
			kvHeadCounts[i] = 0
		}
	}

	kv["lfm2.attention.head_count"] = headCounts
	kv["lfm2.attention.head_count_kv"] = kvHeadCounts
	kv["lfm2.attention.layer_norm_rms_epsilon"] = p.NormEps
	kv["lfm2.rope.freq_base"] = p.RopeTheta
	kv["lfm2.shortconv.l_cache"] = p.ConvLCache

	// Renderer and parser config for thinking model
	kv["tokenizer.chat_template.renderer"] = "lfm2-thinking"
	kv["tokenizer.chat_template.parser"] = "lfm2-thinking"

	return kv
}

func (p *lfm2Model) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	for _, t := range ts {
		shape := t.Shape()

		// Squeeze conv weights: [D, 1, K] -> [D, K]
		if strings.HasSuffix(t.Name(), "shortconv.conv.weight") && len(shape) == 3 && shape[1] == 1 {
			shape = []uint64{shape[0], shape[2]}

			// No repacker needed - data layout is already correct since the middle dim is 1
			t.SetRepacker(func(_ string, data []float32, _ []uint64) ([]float32, error) {
				return data, nil
			})
		}

		out = append(out, &ggml.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    slices.Clone(shape),
			WriterTo: t,
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

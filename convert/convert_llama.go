package convert

import (
	"cmp"
	"fmt"
	"math"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/ollama/ollama/fs/ggml"
)

type llamaModel struct {
	ModelParameters
	NLayers               uint32  `json:"n_layers"`
	NumHiddenLayers       uint32  `json:"num_hidden_layers"`
	NLayer                uint32  `json:"n_layer"`
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	NCtx                  uint32  `json:"n_ctx"`
	HiddenSize            uint32  `json:"hidden_size"`
	NEmbd                 uint32  `json:"n_embd"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	NInner                uint32  `json:"n_inner"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	NHead                 uint32  `json:"n_head"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	RopeTheta             float32 `json:"rope_theta"`
	RopeScaling           struct {
		Type                          string  `json:"type"`
		RopeType                      string  `json:"rope_type"`
		Factor                        float32 `json:"factor"`
		LowFrequencyFactor            float32 `json:"low_freq_factor"`
		HighFrequencyFactor           float32 `json:"high_freq_factor"`
		OriginalMaxPositionEmbeddings uint32  `json:"original_max_position_embeddings"`

		factors ropeFactor
	} `json:"rope_scaling"`
	RMSNormEPS       float32 `json:"rms_norm_eps"`
	LayerNormEPS     float32 `json:"layer_norm_eps"`
	LayerNormEpsilon float32 `json:"layer_norm_epsilon"`
	NormEpsilon      float32 `json:"norm_epsilon"`
	HeadDim          uint32  `json:"head_dim"`

	skipRepack bool
}

var _ ModelConverter = (*llamaModel)(nil)

func (p *llamaModel) KV(t *Tokenizer) KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "llama"
	kv["llama.vocab_size"] = p.VocabSize

	kv["llama.block_count"] = cmp.Or(p.NLayers, p.NumHiddenLayers, p.NLayer)

	if contextLength := cmp.Or(p.MaxPositionEmbeddings, p.NCtx); contextLength > 0 {
		kv["llama.context_length"] = contextLength
	}

	if embeddingLength := cmp.Or(p.HiddenSize, p.NEmbd); embeddingLength > 0 {
		kv["llama.embedding_length"] = cmp.Or(p.HiddenSize, p.NEmbd)
	}

	if feedForwardLength := cmp.Or(p.IntermediateSize, p.NInner); feedForwardLength > 0 {
		kv["llama.feed_forward_length"] = cmp.Or(p.IntermediateSize, p.NInner)
	}

	if headCount := cmp.Or(p.NumAttentionHeads, p.NHead); headCount > 0 {
		kv["llama.attention.head_count"] = cmp.Or(p.NumAttentionHeads, p.NHead)
		kv["llama.rope.dimension_count"] = p.HiddenSize / headCount
	}

	if p.HeadDim > 0 {
		kv["llama.attention.head_dim"] = p.HeadDim
	}

	if p.RopeTheta > 0 {
		kv["llama.rope.freq_base"] = p.RopeTheta
	}

	if p.RopeScaling.Type == "linear" {
		kv["llama.rope.scaling.type"] = p.RopeScaling.Type
		kv["llama.rope.scaling.factor"] = p.RopeScaling.Factor
	} else if p.RopeScaling.RopeType == "llama3" {
		dim := p.HiddenSize / p.NumAttentionHeads
		for i := uint32(0); i < dim; i += 2 {
			factor := cmp.Or(p.RopeScaling.Factor, 8.0)
			factorLow := cmp.Or(p.RopeScaling.LowFrequencyFactor, 1.0)
			factorHigh := cmp.Or(p.RopeScaling.HighFrequencyFactor, 4.0)

			original := cmp.Or(p.RopeScaling.OriginalMaxPositionEmbeddings, 8192)
			lambdaLow := float32(original) / factorLow
			lambdaHigh := float32(original) / factorHigh

			lambda := 2 * math.Pi * math.Pow(float64(p.RopeTheta), float64(i)/float64(dim))
			if lambda < float64(lambdaHigh) {
				p.RopeScaling.factors = append(p.RopeScaling.factors, 1.0)
			} else if lambda > float64(lambdaLow) {
				p.RopeScaling.factors = append(p.RopeScaling.factors, factor)
			} else {
				smooth := (float32(original)/float32(lambda) - factorLow) / (factorHigh - factorLow)
				p.RopeScaling.factors = append(p.RopeScaling.factors, 1.0/((1-smooth)/factor+smooth))
			}
		}
	}

	if p.NumKeyValueHeads > 0 {
		kv["llama.attention.head_count_kv"] = p.NumKeyValueHeads
	}

	if p.RMSNormEPS > 0 {
		kv["llama.attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
	}

	if layerNormEpsilon := cmp.Or(p.LayerNormEPS, p.LayerNormEpsilon, p.NormEpsilon); layerNormEpsilon > 0 {
		kv["llama.attention.layer_norm_epsilon"] = layerNormEpsilon
	}

	if p.HeadDim > 0 {
		kv["llama.attention.key_length"] = p.HeadDim
		kv["llama.attention.value_length"] = p.HeadDim
	}

	return kv
}

func (p *llamaModel) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	if p.RopeScaling.factors != nil {
		out = append(out, &ggml.Tensor{
			Name:     "rope_freqs.weight",
			Kind:     0,
			Shape:    []uint64{uint64(len(p.RopeScaling.factors))},
			WriterTo: p.RopeScaling.factors,
		})
	}

	for _, t := range ts {
		if strings.HasSuffix(t.Name(), "attn_q.weight") || strings.HasSuffix(t.Name(), "attn_k.weight") ||
			strings.HasSuffix(t.Name(), "attn_q_proj.weight") || strings.HasSuffix(t.Name(), "attn_k_proj.weight") {
			if !p.skipRepack {
				t.SetRepacker(p.repack)
			}
		}

		out = append(out, &ggml.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}

	return out
}

func (p *llamaModel) Replacements() []string {
	return []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.norm", "output_norm",
		"model.layers", "blk",
		"input_layernorm", "attn_norm",
		"self_attn.q_proj", "attn_q",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",
		"mlp.gate_proj", "ffn_gate",
		"mlp.down_proj", "ffn_down",
		"mlp.up_proj", "ffn_up",
		"post_attention_layernorm", "ffn_norm",
	}
}

func (p *llamaModel) repack(name string, data []float32, shape []uint64) ([]float32, error) {
	var dims []int
	for _, dim := range shape {
		dims = append(dims, int(dim))
	}

	var heads uint32
	if strings.HasSuffix(name, "attn_q.weight") || strings.HasSuffix(name, "attn_q_proj.weight") {
		heads = p.NumAttentionHeads
	} else if strings.HasSuffix(name, "attn_k.weight") || strings.HasSuffix(name, "attn_k_proj.weight") {
		heads = cmp.Or(p.NumKeyValueHeads, p.NumAttentionHeads)
	} else {
		return nil, fmt.Errorf("unknown tensor for repack: %s", name)
	}

	n := tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))
	if err := n.Reshape(append([]int{int(heads), 2, dims[0] / int(heads) / 2}, dims[1:]...)...); err != nil {
		return nil, err
	}

	if err := n.T(0, 2, 1, 3); err != nil {
		return nil, err
	}

	if err := n.Reshape(dims...); err != nil {
		return nil, err
	}

	if err := n.Transpose(); err != nil {
		return nil, err
	}

	ts, err := native.SelectF32(n, 1)
	if err != nil {
		return nil, err
	}

	var f32s []float32
	for _, t := range ts {
		f32s = append(f32s, t...)
	}

	return f32s, nil
}

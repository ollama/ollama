package convert

import (
	"cmp"
	"fmt"
	"log/slog"
	"math"
	"strings"
	"unsafe"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	cllama "github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/llm"
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
		Type                            string  `json:"type"`
		RopeType                        string  `json:"rope_type"`
		Factor                          float32 `json:"factor"`
		LowFrequencyFactor              float32 `json:"low_freq_factor"`
		HighFrequencyFactor             float32 `json:"high_freq_factor"`
		OriginalMaxPositionalEmbeddings uint32  `json:"original_max_positional_embeddings"`

		factors ropeFactor
	} `json:"rope_scaling"`
	RMSNormEPS       float32 `json:"rms_norm_eps"`
	LayerNormEPS     float32 `json:"layer_norm_eps"`
	LayerNormEpsilon float32 `json:"layer_norm_epsilon"`
	NormEpsilon      float32 `json:"norm_epsilon"`
	HeadDim          uint32  `json:"head_dim"`
}

var _ ModelConverter = (*llamaModel)(nil)

func (p *llamaModel) KV(t *Tokenizer) llm.KV {
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

			original := cmp.Or(p.RopeScaling.OriginalMaxPositionalEmbeddings, 8192)
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

func (p *llamaModel) Tensors(ts []Tensor) []llm.Tensor {
	var out []llm.Tensor

	if p.RopeScaling.factors != nil {
		out = append(out, llm.Tensor{
			Name:     "rope_freqs.weight",
			Kind:     0,
			Shape:    []uint64{uint64(len(p.RopeScaling.factors))},
			WriterTo: p.RopeScaling.factors,
		})
	}

	for _, t := range ts {
		if strings.HasSuffix(t.Name(), "attn_q.weight") ||
			strings.HasSuffix(t.Name(), "attn_k.weight") {
			t.SetRepacker(p.repack)
		}

		out = append(out, llm.Tensor{
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
	if strings.HasSuffix(name, "attn_q.weight") {
		heads = p.NumAttentionHeads
	} else if strings.HasSuffix(name, "attn_k.weight") {
		heads = cmp.Or(p.NumKeyValueHeads, p.NumAttentionHeads)
	} else {
		return nil, fmt.Errorf("unknown tensor for repack: %s", name)
	}

	// name = "model.layers.6.self_attn.k_proj.weight"
	// data = [41943004]float32
	// dims = [2]int{1024,4096}
	// heads = 0x8

	n := tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))

	overhead := cllama.GGMLTensorOverhead()
	tensorCount := (uintptr)(5)
	tensorSize := unsafe.Sizeof(float32(0)) * (uintptr)(len(data))
	memSize := overhead + tensorSize*tensorCount
	params := cllama.NewGGMLInitParams(memSize)
	ctx := cllama.GGMLInit(params)
	// TODO error handling
	defer cllama.GGMLFree(ctx)
	dims64 := make([]int64, len(dims))
	for i, d := range dims {
		dims64[i] = (int64)(d)
	}
	a := cllama.GGMLNewTensor(ctx, cllama.GGML_TYPE_F32, len(dims64), dims64)
	cllama.LoadData(*a, unsafe.Pointer(&data[0]), unsafe.Sizeof(data))

	if err := n.Reshape(append([]int{int(heads), 2, dims[0] / int(heads) / 2}, dims[1:]...)...); err != nil {
		return nil, err
	}

	b := cllama.GGMLReshape4d(ctx, *a, (int64)(heads), 2, dims64[0]/(int64)(heads)/2, dims64[1])

	if err := n.T(0, 2, 1, 3); err != nil {
		return nil, err
	}

	c := cllama.GGMLCont(ctx, *cllama.GGMLPermute(ctx, *b, 0, 2, 1, 3))

	if err := n.Reshape(dims...); err != nil {
		return nil, err
	}

	// TODO - this crashes with ggml_is_contiguous false assert
	d := cllama.GGMLReshape2d(ctx, *c, dims64[0], dims64[1])

	if err := n.Transpose(); err != nil {
		return nil, err
	}

	e := cllama.GGMLTranspose(ctx, *d)
	g := cllama.GGMLNewGraph(ctx)
	cllama.GGMLBuildForwardExpand(*g, *e)
	cllama.GGMLGraphComputeWithCtx(ctx, *g, 4)

	ts, err := native.SelectF32(n, 1)
	if err != nil {
		return nil, err
	}

	ts2 := cllama.GGMLGetDataF32(*e)

	var f32s []float32
	for _, t := range ts {
		f32s = append(f32s, t...)
	}

	slog.Info("first 3 items:",
		slog.Group(
			"old",
			"0", f32s[0], "1", f32s[1], "2", f32s[2],
		),
		slog.Group(
			"new",
			"0", ts2[0], "1", ts2[1], "2", ts2[2],
		),
	)

	return f32s, nil
}

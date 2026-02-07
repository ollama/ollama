package gemma

import (
	"math"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

type TextOptions struct {
	HiddenSize           int     `json:"hidden_size"`
	NumHiddenLayers      int     `json:"num_hidden_layers"`
	IntermediateSize     int     `json:"intermediate_size"`
	NumAttentionHeads    int     `json:"num_attention_heads"`
	NumKeyValueHeads     int     `json:"num_key_value_heads"`
	HeadDim              int     `json:"head_dim"`
	RMSNormEps           float32 `json:"rms_norm_eps"`
	SlidingWindow        int     `json:"sliding_window"`
	SlidingWindowPattern int     `json:"sliding_window_pattern"`

	RoPE map[bool]mlx.RoPE
}

type TextModel struct {
	EmbedTokens mlx.Embedding      `weight:"model.embed_tokens"`
	Layers      []TextDecoderLayer `weight:"model.layers"`
	Norm        RMSNorm            `weight:"model.norm"`

	Options TextOptions
}

func (m TextModel) Forward(inputs *mlx.Array, caches []cache.Cache) *mlx.Array {
	B, L := inputs.Dim(0), inputs.Dim(1)
	hiddenStates := m.EmbedTokens.Forward(inputs)

	hiddenSize := mlx.FromValue(m.Options.HiddenSize).AsType(hiddenStates.DType())
	hiddenStates = hiddenStates.Multiply(hiddenSize.Sqrt())

	for i, layer := range m.Layers {
		hiddenStates = layer.Forward(hiddenStates, caches[i], B, L, m.Options.RoPE[(i+1)%m.Options.SlidingWindowPattern == 0], m.Options)
	}

	hiddenStates = m.Norm.Forward(hiddenStates, m.Options.RMSNormEps)
	return hiddenStates
}

type TextDecoderLayer struct {
	InputNorm    RMSNorm       `weight:"input_layernorm"`
	Attention    TextAttention `weight:"self_attn"`
	PostAttnNorm RMSNorm       `weight:"post_attention_layernorm"`
	PreFFNorm    RMSNorm       `weight:"pre_feedforward_layernorm"`
	MLP          TextMLP       `weight:"mlp"`
	PostFFNorm   RMSNorm       `weight:"post_feedforward_layernorm"`
}

func (m TextDecoderLayer) Forward(hiddenStates *mlx.Array, cache cache.Cache, B, L int, rope mlx.RoPE, opts TextOptions) *mlx.Array {
	residual := hiddenStates
	hiddenStates = m.InputNorm.Forward(hiddenStates, opts.RMSNormEps)
	hiddenStates = m.Attention.Forward(hiddenStates, cache, B, L, rope, opts)
	hiddenStates = m.PostAttnNorm.Forward(hiddenStates, opts.RMSNormEps)
	hiddenStates = hiddenStates.Add(residual)

	residual = hiddenStates
	hiddenStates = m.PreFFNorm.Forward(hiddenStates, opts.RMSNormEps)
	hiddenStates = m.MLP.Forward(hiddenStates, opts)
	hiddenStates = m.PostFFNorm.Forward(hiddenStates, opts.RMSNormEps)
	hiddenStates = hiddenStates.Add(residual)
	return hiddenStates
}

type TextAttention struct {
	QProj mlx.Linear `weight:"q_proj"`
	QNorm RMSNorm    `weight:"q_norm"`
	KProj mlx.Linear `weight:"k_proj"`
	KNorm RMSNorm    `weight:"k_norm"`
	VProj mlx.Linear `weight:"v_proj"`
	OProj mlx.Linear `weight:"o_proj"`
}

func (m TextAttention) Forward(hiddenStates *mlx.Array, cache cache.Cache, B, L int, rope mlx.RoPE, opts TextOptions) *mlx.Array {
	query := m.QProj.Forward(hiddenStates)
	key := m.KProj.Forward(hiddenStates)
	value := m.VProj.Forward(hiddenStates)

	query = query.AsStrided(
		[]int{B, opts.NumAttentionHeads, L, opts.HeadDim},
		[]int{L * opts.NumAttentionHeads * opts.HeadDim, opts.HeadDim, opts.NumAttentionHeads * opts.HeadDim, 1},
		0)
	key = key.AsStrided(
		[]int{B, opts.NumKeyValueHeads, L, opts.HeadDim},
		[]int{L * opts.NumKeyValueHeads * opts.HeadDim, opts.HeadDim, opts.NumKeyValueHeads * opts.HeadDim, 1},
		0)
	value = value.AsStrided(
		[]int{B, opts.NumKeyValueHeads, L, opts.HeadDim},
		[]int{L * opts.NumKeyValueHeads * opts.HeadDim, opts.HeadDim, opts.NumKeyValueHeads * opts.HeadDim, 1},
		0)

	query = m.QNorm.Forward(query, opts.RMSNormEps)
	key = m.KNorm.Forward(key, opts.RMSNormEps)

	query = rope.Forward(query, cache.Offset())
	key = rope.Forward(key, cache.Offset())
	key, value = cache.Update(key, value)

	attention := mlx.ScaledDotProductAttention(query, key, value, nil, 1.0/float32(math.Sqrt(float64(opts.HeadDim))))
	attention = attention.Transpose(0, 2, 1, 3).Reshape(B, L, -1)
	return m.OProj.Forward(attention)
}

type TextMLP struct {
	GateProj mlx.Linear `weight:"gate_proj"`
	UpProj   mlx.Linear `weight:"up_proj"`
	DownProj mlx.Linear `weight:"down_proj"`
}

func (m TextMLP) Forward(h *mlx.Array, opts TextOptions) *mlx.Array {
	return m.DownProj.Forward(mlx.GELUApprox(m.GateProj.Forward(h)).Multiply(m.UpProj.Forward(h)))
}

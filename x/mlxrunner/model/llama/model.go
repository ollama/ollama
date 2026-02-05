package llama

import (
	"encoding/json"
	"log/slog"
	"math"

	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

type Options struct {
	HiddenAct         string  `json:"hidden_act"`
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	RMSNormEps        float32 `json:"rms_norm_eps"`

	mlx.RoPE
}

type Model struct {
	EmbedTokens mlx.Embedding `weight:"model.embed_tokens"`
	Layers      []Layer       `weight:"model.layers"`
	Norm        mlx.RMSNorm   `weight:"model.norm"`
	Output      mlx.Linear    `weight:"lm_head"`

	Options
}

func (m Model) NumLayers() int {
	return len(m.Layers)
}

func (m Model) Forward(inputs *mlx.Tensor, caches []cache.Cache) *mlx.Tensor {
	slog.Debug("Model.forward", "input shape", inputs.Dims(), "m.EmbedTokens", m.EmbedTokens.Weight.Dims())
	B, L := inputs.Dim(0), inputs.Dim(1)
	hiddenStates := m.EmbedTokens.Forward(inputs)
	for i, layer := range m.Layers {
		hiddenStates = layer.Forward(hiddenStates, caches[i], B, L, m.Options)
	}
	hiddenStates = m.Norm.Forward(hiddenStates, m.RMSNormEps)
	hiddenStates = m.Output.Forward(hiddenStates)
	slog.Debug("Model.forward", "output shape", hiddenStates.Dims(), "m.Output", m.Output.Weight.Dims())
	return hiddenStates
}

type Layer struct {
	AttentionNorm mlx.RMSNorm `weight:"input_layernorm"`
	Attention     Attention   `weight:"self_attn"`
	MLPNorm       mlx.RMSNorm `weight:"post_attention_layernorm"`
	MLP           MLP         `weight:"mlp"`
}

func (m Layer) Forward(hiddenStates *mlx.Tensor, c cache.Cache, B, L int, opts Options) *mlx.Tensor {
	residual := hiddenStates
	hiddenStates = m.AttentionNorm.Forward(hiddenStates, opts.RMSNormEps)
	hiddenStates = m.Attention.Forward(hiddenStates, c, B, L, opts)
	hiddenStates = hiddenStates.Add(residual)

	residual = hiddenStates
	hiddenStates = m.MLPNorm.Forward(hiddenStates, opts.RMSNormEps)
	hiddenStates = m.MLP.Forward(hiddenStates)
	hiddenStates = hiddenStates.Add(residual)
	return hiddenStates
}

type Attention struct {
	QueryProj  mlx.Linear `weight:"q_proj"`
	KeyProj    mlx.Linear `weight:"k_proj"`
	ValueProj  mlx.Linear `weight:"v_proj"`
	OutputProj mlx.Linear `weight:"o_proj"`
}

func (m Attention) Forward(hiddenStates *mlx.Tensor, cache cache.Cache, B, L int, opts Options) *mlx.Tensor {
	query := m.QueryProj.Forward(hiddenStates)
	query = query.Reshape(B, L, opts.NumAttentionHeads, -1).Transpose(0, 2, 1, 3)

	key := m.KeyProj.Forward(hiddenStates)
	key = key.Reshape(B, L, opts.NumKeyValueHeads, -1).Transpose(0, 2, 1, 3)

	value := m.ValueProj.Forward(hiddenStates)
	value = value.Reshape(B, L, opts.NumKeyValueHeads, -1).Transpose(0, 2, 1, 3)

	query = opts.RoPE.Forward(query, cache.Offset())
	key = opts.RoPE.Forward(key, cache.Offset())
	key, value = cache.Update(key, value)

	attention := mlx.ScaledDotProductAttention(query, key, value, nil, 1.0/float32(math.Sqrt(float64(key.Dim(-1)))))
	attention = attention.Transpose(0, 2, 1, 3).Reshape(B, L, -1)
	return m.OutputProj.Forward(attention)
}

type MLP struct {
	Gate mlx.Linear `weight:"gate_proj"`
	Up   mlx.Linear `weight:"up_proj"`
	Down mlx.Linear `weight:"down_proj"`
}

func (m MLP) Forward(h *mlx.Tensor) *mlx.Tensor {
	return m.Down.Forward(mlx.SILU(m.Gate.Forward(h)).Multiply(m.Up.Forward(h)))
}

func init() {
	base.Register("MistralForCausalLM", func(root *model.Root) (base.Model, error) {
		bts, err := root.ReadFile("config.json")
		if err != nil {
			return nil, err
		}

		var opts Options
		// TODO: implement json.Unmarshal for Options
		if err := json.Unmarshal(bts, &opts); err != nil {
			return nil, err
		}

		if err := json.Unmarshal(bts, &opts.RoPE); err != nil {
			return nil, err
		}

		return &Model{
			Layers:  make([]Layer, opts.NumHiddenLayers),
			Options: opts,
		}, nil
	})
}

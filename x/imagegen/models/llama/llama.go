//go:build mlx

package llama

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
	"github.com/ollama/ollama/x/imagegen/safetensors"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
)

type Config struct {
	HiddenSize            int32   `json:"hidden_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	IntermediateSize      int32   `json:"intermediate_size"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads"`
	VocabSize             int32   `json:"vocab_size"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`
	HeadDim               int32   `json:"-"`
	Scale                 float32 `json:"-"`
}

type Model struct {
	EmbedTokens *nn.Embedding `weight:"model.embed_tokens"`
	Layers      []*Layer      `weight:"model.layers"`
	Norm        *nn.RMSNorm   `weight:"model.norm"`
	Output      *nn.Linear    `weight:"lm_head,optional"`

	tok *tokenizer.Tokenizer
	*Config
}

type Layer struct {
	Attention     *Attention
	MLP           *MLP
	AttentionNorm *nn.RMSNorm `weight:"input_layernorm"`
	MLPNorm       *nn.RMSNorm `weight:"post_attention_layernorm"`
}

type Attention struct {
	QProj *nn.Linear `weight:"self_attn.q_proj"`
	KProj *nn.Linear `weight:"self_attn.k_proj"`
	VProj *nn.Linear `weight:"self_attn.v_proj"`
	OProj *nn.Linear `weight:"self_attn.o_proj"`
}

type MLP struct {
	GateProj *nn.Linear `weight:"mlp.gate_proj"`
	UpProj   *nn.Linear `weight:"mlp.up_proj"`
	DownProj *nn.Linear `weight:"mlp.down_proj"`
}

func Load(modelPath string) (*Model, error) {
	data, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))

	weights, err := safetensors.LoadModelWeights(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load weights: %w", err)
	}

	tok, err := tokenizer.Load(filepath.Join(modelPath, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	m := &Model{
		Layers: make([]*Layer, cfg.NumHiddenLayers),
		Config: &cfg,
		tok:    tok,
	}
	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return nil, err
	}
	m.Output = nn.NewLinear(m.EmbedTokens.Weight, nil)

	mlx.Eval(mlx.Collect(m)...)
	weights.ReleaseAll()

	return m, nil
}

func (m *Model) Forward(tokens *mlx.Array, caches []cache.Cache) *mlx.Array {
	B, L := tokens.Shape()[0], tokens.Shape()[1]
	h := m.EmbedTokens.Forward(tokens)
	for i, layer := range m.Layers {
		h = layer.Forward(h, caches[i], B, L, m.Config)
	}
	return m.Output.Forward(m.Norm.Forward(h, m.RMSNormEps))
}

func (l *Layer) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	h := mlx.Add(x, l.Attention.Forward(l.AttentionNorm.Forward(x, cfg.RMSNormEps), c, B, L, cfg))
	return mlx.Add(h, l.MLP.Forward(l.MLPNorm.Forward(h, cfg.RMSNormEps)))
}

func (a *Attention) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	q = mlx.AsStrided(q, []int32{B, cfg.NumAttentionHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumAttentionHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumAttentionHeads * cfg.HeadDim), 1}, 0)
	k = mlx.AsStrided(k, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)
	v = mlx.AsStrided(v, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)

	q = mlx.RoPE(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, c.Offset())
	k = mlx.RoPE(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, c.Offset())

	k, v = c.Update(k, v, int(L))
	out := mlx.ScaledDotProductAttention(q, k, v, cfg.Scale, L > 1)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	return m.DownProj.Forward(mlx.Mul(mlx.SiLU(m.GateProj.Forward(x)), m.UpProj.Forward(x)))
}

// Interface methods
func (m *Model) NumLayers() int                     { return len(m.Layers) }
func (m *Model) MaxContextLength() int32            { return m.MaxPositionEmbeddings }
func (m *Model) VocabSize() int32                   { return m.Config.VocabSize }
func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }

func (m *Model) NewCache(maxSeqLen int32) []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i := range caches {
		caches[i] = cache.NewKVCache()
	}
	return caches
}

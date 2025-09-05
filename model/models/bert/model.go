package bert

import (
	"cmp"
	"errors"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Model struct {
	model.Base
	model.WordPiece

	TokenEmbedding     *nn.Embedding `gguf:"token_embd"`
	TypeEmbedding      *nn.Embedding `gguf:"token_types"`
	PositionEmbedding  *nn.Embedding `gguf:"position_embd"`
	TokenEmbeddingNorm *nn.LayerNorm `gguf:"token_embd_norm"`

	Layers []EncoderLayer `gguf:"blk"`

	Options
}

// Forward implements model.Model.
func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)
	hiddenStates = hiddenStates.Add(ctx, m.TypeEmbedding.Weight.View(ctx, 0, m.hiddenSize))
	hiddenStates = hiddenStates.Add(ctx, m.PositionEmbedding.Forward(ctx, ctx.Input().FromIntSlice(batch.Positions, len(batch.Positions))))
	hiddenStates = m.TokenEmbeddingNorm.Forward(ctx, hiddenStates, m.eps)

	for _, layer := range m.Layers {
		hiddenStates = layer.Forward(ctx, hiddenStates, &m.Options)
	}

	switch m.poolingType {
	case 0: // None
	case 1: // Mean
		hiddenStates = hiddenStates.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx).Mean(ctx)
		hiddenStates = hiddenStates.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	default:
		return nil, errors.New("unsupported pooling type")
	}

	return hiddenStates, nil
}

type EncoderLayer struct {
	*Attention
	AttentionNorm *nn.LayerNorm `gguf:"attn_output_norm"`

	*MLP
	MLPNorm *nn.LayerNorm `gguf:"layer_output_norm"`
}

func (e *EncoderLayer) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	// Attention
	residual := hiddenStates
	hiddenStates = e.Attention.Forward(ctx, hiddenStates, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)
	hiddenStates = e.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)

	// MLP
	residual = hiddenStates
	hiddenStates = e.MLP.Forward(ctx, hiddenStates, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)
	hiddenStates = e.MLPNorm.Forward(ctx, hiddenStates, opts.eps)

	return hiddenStates
}

type Attention struct {
	Query     *nn.Linear    `gguf:"attn_q"`
	QueryNorm *nn.LayerNorm `gguf:"attn_q_norm"`

	Key     *nn.Linear    `gguf:"attn_k"`
	KeyNorm *nn.LayerNorm `gguf:"attn_k_norm"`

	Value *nn.Linear `gguf:"attn_v"`

	Output *nn.Linear `gguf:"attn_output"`
}

func (a *Attention) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	batchSize := hiddenStates.Dim(1)

	query := a.Query.Forward(ctx, hiddenStates)
	if a.QueryNorm != nil {
		query = a.QueryNorm.Forward(ctx, query, opts.eps)
	}
	query = query.Reshape(ctx, opts.headDim(), opts.numHeads, batchSize)

	key := a.Key.Forward(ctx, hiddenStates)
	if a.KeyNorm != nil {
		key = a.KeyNorm.Forward(ctx, key, opts.eps)
	}
	key = key.Reshape(ctx, opts.headDim(), cmp.Or(opts.numKVHeads, opts.numHeads), batchSize)

	value := a.Value.Forward(ctx, hiddenStates)
	value = value.Reshape(ctx, opts.headDim(), cmp.Or(opts.numKVHeads, opts.numHeads), batchSize)

	attention := nn.Attention(ctx, query, key, value, 1/math.Sqrt(float64(opts.headDim())), nil)
	attention = attention.Reshape(ctx, opts.hiddenSize, batchSize)
	return a.Output.Forward(ctx, attention)
}

type MLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (m *MLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	return m.Down.Forward(ctx, m.Up.Forward(ctx, hiddenStates).GELU(ctx))
}

type Options struct {
	hiddenSize,
	numHeads,
	numKVHeads,
	keyLength,
	valueLength int
	poolingType int
	eps         float32
}

func (o Options) headDim() int {
	return cmp.Or(o.keyLength, o.valueLength, o.hiddenSize/o.numHeads)
}

func New(c fs.Config) (model.Model, error) {
	return &Model{
		Layers: make([]EncoderLayer, c.Uint("block_count")),
		Options: Options{
			hiddenSize:  int(c.Uint("embedding_length")),
			numHeads:    int(c.Uint("attention.head_count")),
			numKVHeads:  int(c.Uint("attention.head_count_kv")),
			eps:         c.Float("attention.layer_norm_epsilon"),
			poolingType: int(c.Uint("pooling_type")),
		},
	}, nil
}

func init() {
	model.Register("bert", New)
	model.Register("bert_embed", New)
}

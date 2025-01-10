package bert

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
)

func init() {
	model.Register("bert", New)
}

type Options struct {
	hiddenSize, numHeads int64
	eps                  float32
}

type Model struct {
	model.Base
	model.BytePairEncoding

	TokenEmbedding     *nn.Embedding `ggml:"token_embd"`
	TypeEmbedding      *nn.Embedding `ggml:"type_embd,alt:token_types"`
	PositionEmbedding  *nn.Embedding `ggml:"position_embd"`
	TokenEmbeddingNorm *nn.LayerNorm `ggml:"token_embd_norm"`

	Layers []EncoderLayer `ggml:"blk"`

	*Options
}

// Forward implements model.Model.
func (m *Model) Forward(ctx ml.Context, opts model.Options) (ml.Tensor, error) {
	inputs, err := ctx.FromIntSlice(opts.Inputs(), len(opts.Inputs()))
	if err != nil {
		return nil, err
	}
	fmt.Println("inputs", inputs.Shape(), ml.Dump(inputs))

	types, err := ctx.FromIntSlice([]int32{0}, 1)
	if err != nil {
		return nil, err
	}
	fmt.Println("types", types.Shape(), ml.Dump(types))

	positions, err := ctx.FromIntSlice(opts.Positions(), len(opts.Positions()))
	if err != nil {
		return nil, err
	}
	fmt.Println("positions", positions.Shape(), ml.Dump(positions))

	hiddenState := m.TokenEmbedding.Forward(ctx, inputs)
	fmt.Println("TokenEmbedding.Forward", hiddenState.Shape(), ml.Dump(hiddenState))
	return hiddenState, nil
	hiddenState = hiddenState.Add(ctx, m.TypeEmbedding.Forward(ctx, types))
	fmt.Println("TypeEmbedding.Forward", hiddenState.Shape(), ml.Dump(hiddenState))
	hiddenState = hiddenState.Add(ctx, m.PositionEmbedding.Forward(ctx, positions))
	fmt.Println("PositionEmbedding.Forward", hiddenState.Shape(), ml.Dump(hiddenState))
	hiddenState = m.TokenEmbeddingNorm.Forward(ctx, hiddenState, m.eps)
	fmt.Println("TokenEmbeddingNorm.Forward", hiddenState.Shape(), ml.Dump(hiddenState))

	for i, layer := range m.Layers {
		hiddenState = layer.Forward(ctx, hiddenState, positions, opts.Cache.Sub(i), m.Options)
		fmt.Println("EncoderLayer.Forward", i, hiddenState.Shape(), ml.Dump(hiddenState))
	}

	return hiddenState, nil
}

type EncoderLayer struct {
	*SelfAttention
	MLPNorm *nn.LayerNorm `ggml:"attn_output_norm"`
	*MLP
	LayerOutputNorm *nn.LayerNorm `ggml:"ffn_output_norm"`
}

func (e *EncoderLayer) Forward(ctx ml.Context, hiddenState, positionIDs ml.Tensor, cache model.Cache, opts *Options) ml.Tensor {
	residual := hiddenState

	hiddenState = e.SelfAttention.Forward(ctx, hiddenState, positionIDs, cache, opts)
	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	hiddenState = e.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = e.MLP.Forward(ctx, hiddenState, opts)
	hiddenState = hiddenState.Add(ctx, residual)
	return e.LayerOutputNorm.Forward(ctx, hiddenState, opts.eps)
}

type SelfAttention struct {
	Query  *nn.Linear `ggml:"attn_q"`
	Key    *nn.Linear `ggml:"attn_k"`
	Value  *nn.Linear `ggml:"attn_v"`
	Output *nn.Linear `ggml:"attn_output"`
}

func (sa *SelfAttention) Forward(ctx ml.Context, hiddenState, positionIDs ml.Tensor, cache model.Cache, opts *Options) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	headDim := opts.hiddenSize / opts.numHeads

	query := sa.Query.Forward(ctx, hiddenState)
	query = query.Reshape(ctx, headDim, opts.numHeads, batchSize)

	key := sa.Key.Forward(ctx, hiddenState)
	key = key.Reshape(ctx, opts.numHeads, headDim, batchSize)

	value := sa.Value.Forward(ctx, hiddenState)
	value = value.Reshape(ctx, headDim, opts.numHeads, batchSize)

	key, value = cache.Put(ctx, key, value, cache.Options)

	query = query.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	key = key.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	scores := key.Mulmat(ctx, query)
	scores = scores.Scale(ctx, 1.0/math.Sqrt(float64(headDim)))
	scores = scores.Softmax(ctx)

	attention := value.Mulmat(ctx, scores)
	attention = attention.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	attention = attention.Reshape(ctx, opts.hiddenSize, batchSize)

	return sa.Output.Forward(ctx, attention)
}

type MLP struct {
	Up   *nn.Linear `ggml:"ffn_up"`
	Down *nn.Linear `ggml:"ffn_down"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
	return mlp.Down.Forward(ctx, mlp.Up.Forward(ctx, hiddenState).GELU(ctx))
}

func New(c ml.Config) (model.Model, error) {
	return &Model{
		BytePairEncoding: model.NewBytePairEncoding(
			c.String("tokenizer.ggml.pretokenizer", `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`),
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Uints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				BOS:    c.Uint("tokenizer.ggml.bos_token_id"),
				EOS:    c.Uint("tokenizer.ggml.eos_token_id"),
			},
		),
		Options: &Options{
			hiddenSize: int64(c.Uint("embedding_length")),
			numHeads:   int64(c.Uint("attention.head_count")),
			eps:        c.Float("attention.layer_norm_epsilon"),
		},
	}, nil
}

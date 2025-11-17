package nomicbert

import (
	"cmp"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/fast"
	"github.com/ollama/ollama/ml/nn/pooling"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Model struct {
	model.Base
	model.TextProcessor

	TokenEmbedding     *nn.Embedding `gguf:"token_embd"`
	TypeEmbedding      *nn.Embedding `gguf:"token_types"`
	TokenEmbeddingNorm *nn.LayerNorm `gguf:"token_embd_norm"`

	Layers []EncoderLayer `gguf:"blk"`

	Options
}

type Options struct {
	hiddenSize   int
	numHeads     int
	headDim      int
	eps          float32
	poolingType  pooling.Type
	normalize    bool
	ropeFreqBase float32
}

// Single Encoder Layer
type EncoderLayer struct {
	*Attention

	AttentionNorm *nn.LayerNorm `gguf:"attn_output_norm"`

	*MLP

	MLPNorm *nn.LayerNorm `gguf:"layer_output_norm"`
}

type Attention struct {
	QKV    *nn.Linear `gguf:"attn_qkv"`
	Output *nn.Linear `gguf:"attn_output"`
}

type MLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	typeEmbed := m.TypeEmbedding.Weight.Slice(ctx, 1, 0, 1, 1)
	hiddenStates = hiddenStates.Add(ctx, typeEmbed)

	hiddenStates = m.TokenEmbeddingNorm.Forward(ctx, hiddenStates, m.eps)

	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	for _, layer := range m.Layers {
		hiddenStates = layer.Forward(ctx, hiddenStates, positions, &m.Options)
	}

	hiddenStates = m.poolingType.Forward(ctx, hiddenStates)

	if m.normalize {
		hiddenStates = hiddenStates.L2Norm(ctx, 1e-12)
	}

	return hiddenStates, nil
}

func (e *EncoderLayer) Forward(ctx ml.Context, hiddenStates ml.Tensor, positions ml.Tensor, opts *Options) ml.Tensor {
	residual := hiddenStates
	hiddenStates = e.Attention.Forward(ctx, hiddenStates, positions, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)
	hiddenStates = e.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)

	residual = hiddenStates
	hiddenStates = e.MLP.Forward(ctx, hiddenStates)
	hiddenStates = hiddenStates.Add(ctx, residual)
	hiddenStates = e.MLPNorm.Forward(ctx, hiddenStates, opts.eps)

	return hiddenStates
}

func (a *Attention) Forward(ctx ml.Context, hiddenStates ml.Tensor, positions ml.Tensor, opts *Options) ml.Tensor {
	batchSize := hiddenStates.Dim(1)

	qkv := a.QKV.Forward(ctx, hiddenStates)

	qkv = qkv.Reshape(ctx, opts.headDim, opts.numHeads*3, batchSize)
	chunks := qkv.Chunk(ctx, 1, opts.numHeads)
	query, key, value := chunks[0], chunks[1], chunks[2]

	query = fast.RoPE(ctx, query, positions, opts.headDim, opts.ropeFreqBase, 1.0, rope.WithTypeNeoX())
	key = fast.RoPE(ctx, key, positions, opts.headDim, opts.ropeFreqBase, 1.0, rope.WithTypeNeoX())

	attention := nn.Attention(ctx, query, key, value, 1.0/math.Sqrt(float64(opts.headDim)), nil)

	attention = attention.Reshape(ctx, opts.hiddenSize, batchSize)

	return a.Output.Forward(ctx, attention)
}

func (m *MLP) Forward(ctx ml.Context, hiddenStates ml.Tensor) ml.Tensor {
	hidden := m.Gate.Forward(ctx, hiddenStates).SILU(ctx, m.Up.Forward(ctx, hiddenStates))

	return m.Down.Forward(ctx, hidden)
}

func New(c fs.Config) (model.Model, error) {
	hiddenSize := int(c.Uint("embedding_length"))
	numHeads := int(c.Uint("attention.head_count"))
	headDim := hiddenSize / numHeads

	processor := model.NewWordPiece(
		&model.Vocabulary{
			Values: c.Strings("tokenizer.ggml.tokens"),
			Scores: c.Floats("tokenizer.ggml.scores"),
			Types:  c.Ints("tokenizer.ggml.token_type"),
			AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
			BOS: []int32{
				int32(cmp.Or(
					c.Uint("tokenizer.ggml.cls_token_id"),
					c.Uint("tokenizer.ggml.bos_token_id"),
				)),
			},
			AddEOS: c.Bool("tokenizer.ggml.add_eos_token", true),
			EOS: []int32{
				int32(cmp.Or(
					c.Uint("tokenizer.ggml.separator_token_id"),
					c.Uint("tokenizer.ggml.eos_token_id"),
				)),
			},
		},
		false,
	)

	return &Model{
		TextProcessor: processor,
		Layers:        make([]EncoderLayer, c.Uint("block_count")),
		Options: Options{
			hiddenSize:   hiddenSize,
			numHeads:     numHeads,
			headDim:      headDim,
			eps:          c.Float("attention.layer_norm_epsilon"),
			poolingType:  pooling.Type(c.Uint("pooling_type")),
			normalize:    c.Bool("normalize_embeddings", false),
			ropeFreqBase: c.Float("rope.freq_base", 1000.0),
		},
	}, nil
}

func init() {
	model.Register("nomic-bert", New)
	model.Register("nomic-bert_embed", New)
}

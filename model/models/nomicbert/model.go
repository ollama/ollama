package nomicbert

import (
	"cmp"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
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

	// MoE specific options (used by v2 / MoE models only)
	numExperts      int
	numExpertsUsed  int
	moeEveryNLayers int
}

func (o Options) applyRotaryPositionEmbeddings(ctx ml.Context, states, positions ml.Tensor) ml.Tensor {
	return nn.RoPE(ctx, states, positions, o.headDim, o.ropeFreqBase, 1.0, rope.WithTypeNeoX())
}

type EncoderLayer struct {
	*Attention

	AttentionNorm *nn.LayerNorm `gguf:"attn_output_norm"`

	FeedForward FeedForward

	MLPNorm *nn.LayerNorm `gguf:"layer_output_norm"`
}

type Attention struct {
	QKV    *nn.Linear `gguf:"attn_qkv"`
	Output *nn.Linear `gguf:"attn_output"`
}

type FeedForward interface {
	Forward(ml.Context, ml.Tensor, *Options) ml.Tensor
}

type dense struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *dense) Forward(ctx ml.Context, hiddenStates ml.Tensor, _ *Options) ml.Tensor {
	hidden := mlp.Gate.Forward(ctx, hiddenStates).SILU(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, hidden)
}

// denseGELU implements MLP with GELU activation for v2 MoE dense layers
type denseGELU struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *denseGELU) Forward(ctx ml.Context, hiddenStates ml.Tensor, _ *Options) ml.Tensor {
	return mlp.Down.Forward(ctx, mlp.Up.Forward(ctx, hiddenStates).GELU(ctx))
}

// sparse implements MoE with expert routing
type sparse struct {
	Router *nn.Linear      `gguf:"ffn_gate_inp"`
	Up     *nn.LinearBatch `gguf:"ffn_up_exps"`
	Down   *nn.LinearBatch `gguf:"ffn_down_exps"`
}

func (moe *sparse) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	hiddenDim, sequenceLength, batchSize := hiddenStates.Dim(0), hiddenStates.Dim(1), hiddenStates.Dim(2)
	hiddenStates = hiddenStates.Reshape(ctx, hiddenDim, sequenceLength*batchSize)

	routerLogits := moe.Router.Forward(ctx, hiddenStates)
	routingWeights := routerLogits.Softmax(ctx)
	selectedExperts := routingWeights.TopK(ctx, opts.numExpertsUsed)

	routingWeights = routingWeights.Reshape(ctx, 1, opts.numExperts, hiddenStates.Dim(1)).Rows(ctx, selectedExperts)

	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), 1, hiddenStates.Dim(1))

	hiddenStates = moe.Up.Forward(ctx, hiddenStates, selectedExperts).GELU(ctx)
	experts := moe.Down.Forward(ctx, hiddenStates, selectedExperts)

	experts = experts.Mul(ctx, routingWeights)

	nextStates := experts.View(ctx, 0, experts.Dim(0), experts.Stride(2), experts.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		nextStates = nextStates.Add(ctx, experts.View(ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2)))
	}

	return nextStates
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
	hiddenStates = e.FeedForward.Forward(ctx, hiddenStates, opts)
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

	query = opts.applyRotaryPositionEmbeddings(ctx, query, positions)
	key = opts.applyRotaryPositionEmbeddings(ctx, key, positions)

	attention := nn.Attention(ctx, query, key, value, 1.0/math.Sqrt(float64(opts.headDim)), nil)

	attention = attention.Reshape(ctx, opts.hiddenSize, batchSize)

	return a.Output.Forward(ctx, attention)
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

	blockCount := int(c.Uint("block_count"))
	moeEveryNLayers := int(c.Uint("moe_every_n_layers", 0))
	layers := make([]EncoderLayer, blockCount)

	for i := range layers {
		if moeEveryNLayers > 0 {
			// Layer uses MoE if (i+1) % moe_every_n_layers == 0
			if (i+1)%moeEveryNLayers == 0 {
				layers[i].FeedForward = &sparse{}
			} else {
				layers[i].FeedForward = &denseGELU{}
			}
		} else {
			layers[i].FeedForward = &dense{}
		}
	}

	return &Model{
		TextProcessor: processor,
		Layers:        layers,
		Options: Options{
			hiddenSize:      hiddenSize,
			numHeads:        numHeads,
			headDim:         headDim,
			eps:             c.Float("attention.layer_norm_epsilon"),
			poolingType:     pooling.Type(c.Uint("pooling_type")),
			normalize:       c.Bool("normalize_embeddings", false),
			ropeFreqBase:    c.Float("rope.freq_base", 1000.0),
			numExperts:      int(c.Uint("expert_count")),
			numExpertsUsed:  int(c.Uint("expert_used_count")),
			moeEveryNLayers: moeEveryNLayers,
		},
	}, nil
}

func init() {
	model.Register("nomic-bert", New)
	model.Register("nomic-bert_embed", New)
	model.Register("nomic-bert-moe", New)
	model.Register("nomic-bert-moe_embed", New)
}

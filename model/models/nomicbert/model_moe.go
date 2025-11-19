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

type MoEModel struct {
	model.Base
	model.TextProcessor

	TokenEmbedding     *nn.Embedding `gguf:"token_embd"`
	TypeEmbedding      *nn.Embedding `gguf:"token_types"`
	TokenEmbeddingNorm *nn.LayerNorm `gguf:"token_embd_norm"`

	Layers []MoEEncoderLayer `gguf:"blk"`

	Options
}

type MoEEncoderLayer struct {
	*MoEAttention

	AttentionNorm *nn.LayerNorm `gguf:"attn_output_norm"`

	FeedForward MoEFeedForward

	MLPNorm *nn.LayerNorm `gguf:"layer_output_norm"`
}

type MoEAttention struct {
	QKV    *nn.Linear `gguf:"attn_qkv"`
	Output *nn.Linear `gguf:"attn_output"`
}

// MoEFeedForward interface for both dense and sparse (MoE) implementations
type MoEFeedForward interface {
	Forward(ml.Context, ml.Tensor, *Options) ml.Tensor
}

type denseMLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *denseMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, _ *Options) ml.Tensor {
	return mlp.Down.Forward(ctx, mlp.Up.Forward(ctx, hiddenStates).GELU(ctx))
}

type sparseMoE struct {
	Router *nn.Linear      `gguf:"ffn_gate_inp"`
	Up     *nn.LinearBatch `gguf:"ffn_up_exps"`
	Down   *nn.LinearBatch `gguf:"ffn_down_exps"`
}

func (moe *sparseMoE) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	hiddenDim, sequenceLength, batchSize := hiddenStates.Dim(0), hiddenStates.Dim(1), hiddenStates.Dim(2)
	hiddenStates = hiddenStates.Reshape(ctx, hiddenDim, sequenceLength*batchSize)

	// multiply context vector from previous layer with experts, make into probabilities and select top-k (8 in this case)
	routerLogits := moe.Router.Forward(ctx, hiddenStates)
	routingWeights := routerLogits.Softmax(ctx)
	selectedExperts := routingWeights.TopK(ctx, opts.numExpertsUsed)

	// extract weights for the 8 selected experts
	routingWeights = routingWeights.Reshape(ctx, 1, opts.numExperts, hiddenStates.Dim(1)).Rows(ctx, selectedExperts)

	//  sum reshaping for batch processing
	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), 1, hiddenStates.Dim(1))

	// apply GELU activation function through selected experts (BERT-style)
	hiddenStates = moe.Up.Forward(ctx, hiddenStates, selectedExperts).GELU(ctx)
	experts := moe.Down.Forward(ctx, hiddenStates, selectedExperts)

	// apply routing weights to each expert's output
	experts = experts.Mul(ctx, routingWeights)

	// combine all expert outputs into an updated context vector
	nextStates := experts.View(ctx, 0, experts.Dim(0), experts.Stride(2), experts.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		nextStates = nextStates.Add(ctx, experts.View(ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2)))
	}

	return nextStates
}

func (m *MoEModel) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
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

func (e *MoEEncoderLayer) Forward(ctx ml.Context, hiddenStates ml.Tensor, positions ml.Tensor, opts *Options) ml.Tensor {
	residual := hiddenStates
	hiddenStates = e.MoEAttention.Forward(ctx, hiddenStates, positions, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)
	hiddenStates = e.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)

	residual = hiddenStates
	hiddenStates = e.FeedForward.Forward(ctx, hiddenStates, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)
	hiddenStates = e.MLPNorm.Forward(ctx, hiddenStates, opts.eps)

	return hiddenStates
}

func (a *MoEAttention) Forward(ctx ml.Context, hiddenStates ml.Tensor, positions ml.Tensor, opts *Options) ml.Tensor {
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

func NewMoE(c fs.Config) (model.Model, error) {
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

	// Create layers with appropriate FeedForward type based on MoE configuration
	blockCount := int(c.Uint("block_count"))
	moeEveryNLayers := int(c.Uint("moe_every_n_layers", 0))
	layers := make([]MoEEncoderLayer, blockCount)

	for i := range layers {
		// determine if this layer should use MoE
		// if moe_every_n_layers > 0, layer i uses MoE if (i+1) % moe_every_n_layers == 0
		// l0: dense, l1: MoE, l2: dense, l3: MoE, etc.
		useMoE := moeEveryNLayers > 0 && (i+1)%moeEveryNLayers == 0

		if useMoE {
			layers[i].FeedForward = &sparseMoE{}
		} else {
			layers[i].FeedForward = &denseMLP{}
		}
	}

	return &MoEModel{
		TextProcessor: processor,
		Layers:        layers,
		Options: Options{
			hiddenSize:      hiddenSize,
			numHeads:        numHeads,
			headDim:         headDim,
			eps:             c.Float("attention.layer_norm_epsilon"),
			poolingType:     pooling.Type(c.Uint("pooling_type")),
			normalize:       c.Bool("normalize_embeddings", false),
			ropeFreqBase:    c.Float("rope.freq_base", 10000.0),
			numExperts:      int(c.Uint("expert_count")),
			numExpertsUsed:  int(c.Uint("expert_used_count")),
			moeEveryNLayers: moeEveryNLayers,
		},
	}, nil
}

func init() {
	model.Register("nomic-bert-moe", NewMoE)
	model.Register("nomic-bert-moe_embed", NewMoE)
}

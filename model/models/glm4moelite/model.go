package glm4moelite

import (
	"errors"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/tokenizer"
)

var ErrOldModelFormat = errors.New("this model uses a weight format that is no longer supported; please re-download it")

type Options struct {
	numExpertsUsed      int
	numExperts          int
	normTopKProb        bool
	routedScalingFactor float32

	kvLoraRank,
	qkNopeHeadDim,
	qkRopeHeadDim,
	kqNopeHeadDim,
	qkHeadDim int
	qLoraRank int
	vHeadDim  int

	hiddenSize,
	numHeads,
	numKVHeads int

	eps,
	ropeBase float32
	kqScale float64
}

func (o Options) applyRotaryPositionEmbeddings(ctx ml.Context, t, p ml.Tensor) ml.Tensor {
	return nn.RoPE(ctx, t, p, o.qkRopeHeadDim, o.ropeBase, 1.0)
}

type Attention struct {
	Q *nn.Linear `gguf:"attn_q"`

	QA     *nn.Linear  `gguf:"attn_q_a"`
	QANorm *nn.RMSNorm `gguf:"attn_q_a_norm"`
	QB     *nn.Linear  `gguf:"attn_q_b"`

	KVA     *nn.Linear  `gguf:"attn_kv_a_mqa"`
	KVANorm *nn.RMSNorm `gguf:"attn_kv_a_norm"`

	KB *nn.Linear `gguf:"attn_k_b"`
	VB *nn.Linear `gguf:"attn_v_b"`

	Output *nn.Linear `gguf:"attn_out,alt:attn_output"`
}

func (attn *Attention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	seqLength := hiddenStates.Dim(1)

	var query ml.Tensor
	if opts.qLoraRank == 0 {
		query = attn.Q.Forward(ctx, hiddenStates)
	} else {
		query = attn.QA.Forward(ctx, hiddenStates)
		query = attn.QANorm.Forward(ctx, query, opts.eps)
		query = attn.QB.Forward(ctx, query)
	}

	query = query.Reshape(ctx, query.Dim(0)/opts.numHeads, opts.numHeads, seqLength)
	queryChunks := query.ChunkSections(ctx, 0, opts.qkNopeHeadDim, opts.qkRopeHeadDim)

	compressedKV := attn.KVA.Forward(ctx, hiddenStates)
	kPass := compressedKV.Slice(ctx, 0, 0, opts.kvLoraRank, 1)
	kRot := compressedKV.View(ctx,
		opts.kvLoraRank*compressedKV.Stride(0), opts.qkRopeHeadDim,
		compressedKV.Stride(1), 1,
		compressedKV.Stride(1), compressedKV.Dim(1),
	)

	qRot := opts.applyRotaryPositionEmbeddings(ctx, queryChunks[1], positions)
	kRot = opts.applyRotaryPositionEmbeddings(ctx, kRot, positions)
	kPass = attn.KVANorm.Forward(ctx, kPass, opts.eps)

	// MLA absorption: absorb K projection into query
	qPass := queryChunks[0].Permute(ctx, 0, 2, 1, 3)
	qPassAbsorb := attn.KB.Forward(ctx, qPass).Permute(ctx, 0, 2, 1, 3)
	query = qRot.Concat(ctx, qPassAbsorb, 0)

	kPass = kPass.Reshape(ctx, opts.kvLoraRank, 1, seqLength)
	key := kRot.Concat(ctx, kPass, 0)

	attention := nn.AttentionWithVMLA(ctx, query, key, kPass, nil, attn.VB.Weight, opts.kqScale, cache)

	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), seqLength)
	return attn.Output.Forward(ctx, attention)
}

type MLP interface {
	Forward(ml.Context, ml.Tensor, *Options) ml.Tensor
}

type sparse struct {
	Router       *nn.Linear `gguf:"ffn_gate_inp"`
	Gate         *nn.Linear `gguf:"ffn_gate_exps"`
	Up           *nn.Linear `gguf:"ffn_up_exps"`
	Down         *nn.Linear `gguf:"ffn_down_exps"`
	SharedExpert *dense     `gguf:",suf:_shexp"`
	ExpProbsBias ml.Tensor  `gguf:"exp_probs_b.bias,alt:exp_probs_b"`
}

func (moe *sparse) Moe(ctx ml.Context, hiddenStates, topKIndices, topKWeights ml.Tensor, opts *Options) ml.Tensor {
	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), 1, hiddenStates.Dim(1))

	upStates := moe.Up.Weight.MulmatID(ctx, hiddenStates, topKIndices)
	hiddenStates = moe.Gate.Weight.MulmatID(ctx, hiddenStates, topKIndices)
	hiddenStates = hiddenStates.SILU(ctx, upStates)

	experts := moe.Down.Weight.MulmatID(ctx, hiddenStates, topKIndices)
	experts = experts.Mul(ctx, topKWeights)

	nextStates := experts.View(ctx, 0, experts.Dim(0), experts.Stride(2), experts.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		nextStates = nextStates.Add(ctx, experts.View(ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2)))
	}
	return nextStates
}

func (moe *sparse) topKIndices(ctx ml.Context, scores ml.Tensor, opts *Options) ml.Tensor {
	if moe.ExpProbsBias != nil {
		scores = scores.Add(ctx, moe.ExpProbsBias)
	}
	topKIndices := scores.TopK(ctx, opts.numExpertsUsed)
	return topKIndices
}

func (moe *sparse) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	residuals := hiddenStates

	routerLogits := moe.Router.Forward(ctx, hiddenStates)
	scores := routerLogits.Sigmoid(ctx)
	topKIndices := moe.topKIndices(ctx, scores, opts)
	topKWeights := scores.Reshape(ctx, 1, opts.numExperts, hiddenStates.Dim(1)).Rows(ctx, topKIndices)

	if opts.normTopKProb {
		topKWeights = topKWeights.Reshape(ctx, opts.numExpertsUsed, hiddenStates.Dim(1))
		topKWeights = topKWeights.Div(ctx, topKWeights.SumRows(ctx))
		topKWeights = topKWeights.Reshape(ctx, 1, opts.numExpertsUsed, hiddenStates.Dim(1))
	}

	topKWeights = topKWeights.Scale(ctx, float64(opts.routedScalingFactor))
	hiddenStates = moe.Moe(ctx, hiddenStates, topKIndices, topKWeights, opts)
	sharedExpertResult := moe.SharedExpert.Forward(ctx, residuals, opts)

	hiddenStates = hiddenStates.Add(ctx, sharedExpertResult)
	return hiddenStates
}

type dense struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *dense) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates).SILU(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, hiddenStates)
}

type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	Attention     *Attention

	MLPNorm *nn.RMSNorm `gguf:"ffn_norm"`
	MLP     MLP
}

func (t *Layer) Forward(ctx ml.Context, hiddenStates, positions, outputs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	residual := hiddenStates
	hiddenStates = t.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = t.Attention.Forward(ctx, hiddenStates, positions, cache, opts)

	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenStates = hiddenStates.Add(ctx, residual)
	residual = hiddenStates

	hiddenStates = t.MLPNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = t.MLP.Forward(ctx, hiddenStates, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)
	return hiddenStates
}

type Model struct {
	model.Base
	tokenizer.Tokenizer

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`

	OutputNorm *nn.RMSNorm `gguf:"output_norm"`
	Output     *nn.Linear  `gguf:"output,alt:token_embd"`

	*Options
}

func New(c fs.Config) (model.Model, error) {
	layers := make([]Layer, c.Uint("block_count"))

	firstDenseLayerIndex := int(c.Uint("leading_dense_block_count"))
	for i := range layers {
		if i < firstDenseLayerIndex {
			layers[i].MLP = &dense{}
		} else {
			layers[i].MLP = &sparse{}
		}
	}

	keyLength := int(c.Uint("attention.key_length"))
	valueLength := int(c.Uint("attention.value_length"))
	kqScale := 1.0 / math.Sqrt(float64(keyLength))

	var pre []string
	switch c.String("tokenizer.ggml.pre") {
	case "glm4":
		pre = []string{
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		}
	default:
		return nil, model.ErrUnsupportedTokenizer
	}

	m := Model{
		Tokenizer: tokenizer.NewBytePairEncoding(
			&tokenizer.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", false),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
			pre...,
		),
		Layers: layers,
		Options: &Options{
			hiddenSize:     int(c.Uint("embedding_length")),
			numHeads:       int(c.Uint("attention.head_count")),
			numKVHeads:     int(c.Uint("attention.head_count_kv")),
			eps:            c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:       c.Float("rope.freq_base"),
			numExperts:     int(c.Uint("expert_count")),
			numExpertsUsed: int(c.Uint("expert_used_count")),
			normTopKProb:   c.Bool("expert_weights_norm", true),

			qLoraRank:     int(c.Uint("attention.q_lora_rank")),
			kvLoraRank:    int(c.Uint("attention.kv_lora_rank")),
			qkHeadDim:     keyLength,
			vHeadDim:      valueLength,
			qkRopeHeadDim: int(c.Uint("rope.dimension_count")),
			qkNopeHeadDim: keyLength - int(c.Uint("rope.dimension_count")),
			kqNopeHeadDim: keyLength - int(c.Uint("rope.dimension_count")),

			routedScalingFactor: c.Float("expert_weights_scale"),

			kqScale: kqScale,
		},
	}

	m.Cache = kvcache.NewCausalCache(m.Shift)
	return &m, nil
}

func (m Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return m.applyRotaryPositionEmbeddings(ctx, key, shift), nil
}

func (m *Model) Validate() error {
	for _, layer := range m.Layers {
		if layer.Attention != nil && (layer.Attention.KB == nil || layer.Attention.VB == nil) {
			return ErrOldModelFormat
		}
	}
	return nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	for i, layer := range m.Layers {
		m.Cache.SetLayer(i)

		var outputs ml.Tensor
		if i == len(m.Layers)-1 {
			outputs = batch.Outputs
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, outputs, m.Cache, m.Options)
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func init() {
	model.Register("glm4moelite", New)
}

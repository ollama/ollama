package longcatflash

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Transformer struct {
	model.Base
	model.BytePairEncoding

	TokenEmbedding    *nn.Embedding      `gguf:"token_embd"`
	TransformerBlocks []TransformerBlock `gguf:"blk"`
	OutputNorm        *nn.RMSNorm        `gguf:"output_norm"`
	Output            *nn.Linear         `gguf:"output,alt:token_embd"`

	Options
}

type Options struct {
	hiddenSize,
	numHeads,
	numKVHeads int

	keyLength,
	valueLength int

	qLoraRank,
	kvLoraRank int

	qkRopeHeadDim,
	qkNopeHeadDim int

	numExperts,
	numExpertsUsed,
	numRoutedExperts int

	routedScalingFactor float32

	eps,
	ropeBase,
	ropeScale float32

	ropeType              string
	originalContextLength int

	kqScale        float64
	mlaScaleQLoRA  float32
	mlaScaleKVLoRA float32
}

func (o Options) applyRotaryPositionEmbeddings(ctx ml.Context, states, positions ml.Tensor) ml.Tensor {
	opts := []func(*rope.Options){rope.WithTypeNeoX()}
	if o.originalContextLength > 0 {
		opts = append(opts,
			rope.WithOriginalContextLength(o.originalContextLength),
			rope.WithExtrapolationFactor(1.),
		)
	}
	if o.ropeType == "yarn" {
		attnFactor := float32(1.0 / (1.0 + 0.1*math.Log(float64(o.ropeScale))))
		opts = append(opts, rope.WithAttentionFactor(attnFactor))
	}

	return nn.RoPE(ctx, states, positions, o.qkRopeHeadDim, o.ropeBase, 1./o.ropeScale, opts...)
}

func New(c fs.Config) (model.Model, error) {
	blocks := make([]TransformerBlock, c.Uint("block_count"))
	for i := range blocks {
		blocks[i].LayerIdx = i
	}

	numExperts := int(c.Uint("expert_count"))
	numRoutedExperts := int(c.Uint("expert_routed_count"))
	if numRoutedExperts <= 0 {
		numRoutedExperts = numExperts
	}
	numExpertsUsed := int(c.Uint("expert_used_count"))
	if numExpertsUsed > numExperts {
		numExpertsUsed = numExperts
	}

	qkHeadDim := int(c.Uint("attention.key_length"))
	numHeads := int(c.Uint("attention.head_count"))
	if qkHeadDim == 0 && numHeads > 0 {
		qkHeadDim = int(c.Uint("embedding_length")) / numHeads
	}
	qkRopeHeadDim := int(c.Uint("rope.dimension_count"))
	qkNopeHeadDim := qkHeadDim - qkRopeHeadDim
	if qkNopeHeadDim < 0 {
		qkNopeHeadDim = 0
	}

	ropeScale := c.Float("rope.scaling.factor", 1)
	ropeType := c.String("rope.scaling.type")
	mScale := float32(1.0)
	if ropeType == "yarn" && ropeScale > 1 {
		mScale = float32(1.0 + float64(c.Float("rope.scaling.yarn_log_multiplier"))*math.Log(float64(ropeScale)))
	}
	kqScale := float64(mScale*mScale) / math.Sqrt(float64(qkHeadDim))
	embedLen := int(c.Uint("embedding_length"))
	qLoraRank := int(c.Uint("attention.q_lora_rank"))
	kvLoraRank := int(c.Uint("attention.kv_lora_rank"))
	mlaScaleQLoRA := float32(1.0)
	if qLoraRank > 0 {
		mlaScaleQLoRA = float32(math.Sqrt(float64(embedLen / qLoraRank)))
	}
	mlaScaleKVLoRA := float32(1.0)
	if kvLoraRank > 0 {
		mlaScaleKVLoRA = float32(math.Sqrt(float64(embedLen / kvLoraRank)))
	}

	m := Transformer{
		TransformerBlocks: blocks,
		BytePairEncoding: model.NewBytePairEncoding(
			&model.Vocabulary{
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
		),
		Options: Options{
			hiddenSize:            embedLen,
			numHeads:              numHeads,
			numKVHeads:            int(c.Uint("attention.head_count_kv")),
			keyLength:             int(c.Uint("attention.key_length")),
			valueLength:           int(c.Uint("attention.value_length")),
			qLoraRank:             qLoraRank,
			kvLoraRank:            kvLoraRank,
			qkRopeHeadDim:         qkRopeHeadDim,
			qkNopeHeadDim:         qkNopeHeadDim,
			numExperts:            numExperts,
			numExpertsUsed:        numExpertsUsed,
			numRoutedExperts:      numRoutedExperts,
			routedScalingFactor:   c.Float("expert_weights_scale"),
			eps:                   c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:              c.Float("rope.freq_base"),
			ropeScale:             ropeScale,
			ropeType:              ropeType,
			originalContextLength: int(c.Uint("rope.scaling.original_context_length")),
			kqScale:               kqScale,
			mlaScaleQLoRA:         mlaScaleQLoRA,
			mlaScaleKVLoRA:        mlaScaleKVLoRA,
		},
	}

	m.Cache = kvcache.NewWrapperCache(
		kvcache.NewSWAMemCache(int32(c.Uint("attention.sliding_window")), 4096, m.Shift),
		kvcache.NewCausalCache(m.Shift),
	)
	return &m, nil
}

func (m *Transformer) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return m.applyRotaryPositionEmbeddings(ctx, key, shift), nil
}

func (m *Transformer) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))
	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	for i, block := range m.TransformerBlocks {
		if m.Cache != nil {
			m.Cache.SetLayer(i)
			if c, ok := m.Cache.(*kvcache.WrapperCache); ok {
				c.SetLayerType(i % 2)
			}
		}

		var outputs ml.Tensor
		if i == len(m.TransformerBlocks)-1 {
			outputs = batch.Outputs
		}

		hiddenStates = block.Forward(ctx, hiddenStates, positions, outputs, m.Cache, &m.Options)
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

type MLA struct {
	Q *nn.Linear `gguf:"attn_q"`

	QA     *nn.Linear  `gguf:"attn_q_a"`
	QANorm *nn.RMSNorm `gguf:"attn_q_a_norm"`
	QB     *nn.Linear  `gguf:"attn_q_b"`

	KVA     *nn.Linear  `gguf:"attn_kv_a_mqa"`
	KVANorm *nn.RMSNorm `gguf:"attn_kv_a_norm"`
	KVB     *nn.Linear  `gguf:"attn_kv_b"`

	Output *nn.Linear `gguf:"attn_out,alt:attn_output"`
}

func (attn *MLA) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	seqLen := hiddenStates.Dim(1)

	var query ml.Tensor
	if opts.qLoraRank == 0 {
		query = attn.Q.Forward(ctx, hiddenStates)
	} else {
		query = attn.QA.Forward(ctx, hiddenStates)
		query = attn.QANorm.Forward(ctx, query, opts.eps)
		query = attn.QB.Forward(ctx, query)
	}

	query = query.Reshape(ctx, opts.keyLength, opts.numHeads, seqLen)
	queryChunks := query.ChunkSections(ctx, 0, opts.qkNopeHeadDim, opts.qkRopeHeadDim)
	qPass := queryChunks[0]
	qRot := queryChunks[1]

	compressedKV := attn.KVA.Forward(ctx, hiddenStates)
	kvChunks := compressedKV.ChunkSections(ctx, 0, opts.kvLoraRank, opts.qkRopeHeadDim)
	kPass := kvChunks[0]
	kRot := kvChunks[1]
	kPass = attn.KVANorm.Forward(ctx, kPass, opts.eps)

	qPass = qPass.Contiguous(ctx).Scale(ctx, float64(opts.mlaScaleQLoRA))
	qRot = qRot.Contiguous(ctx).Scale(ctx, float64(opts.mlaScaleQLoRA))
	kPass = kPass.Contiguous(ctx).Scale(ctx, float64(opts.mlaScaleKVLoRA))

	kPass = attn.KVB.Forward(ctx, kPass)
	kPass = kPass.Reshape(ctx, opts.qkNopeHeadDim+opts.valueLength, opts.numHeads, seqLen)
	kvPass := kPass.ChunkSections(ctx, 0, opts.qkNopeHeadDim, opts.valueLength)
	kPass = kvPass[0]
	valueStates := kvPass[1]

	kRot = kRot.Reshape(ctx, opts.qkRopeHeadDim, 1, seqLen)
	qRot = opts.applyRotaryPositionEmbeddings(ctx, qRot, positions)
	kRot = opts.applyRotaryPositionEmbeddings(ctx, kRot, positions)

	if opts.numHeads > 1 {
		kRot = kRot.Repeat(ctx, 1, opts.numHeads)
	}

	query = qPass.Concat(ctx, qRot, 0)
	key := kPass.Concat(ctx, kRot, 0)

	attention := nn.Attention(ctx, query, key, valueStates, opts.kqScale, cache)
	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), seqLen)
	return attn.Output.Forward(ctx, attention)
}

type MLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
	gate := mlp.Gate.Forward(ctx, hiddenState)
	up := mlp.Up.Forward(ctx, hiddenState)
	return mlp.Down.Forward(ctx, gate.SILU(ctx, up))
}

type TopKRouter struct {
	Classifier          *nn.Linear `gguf:"ffn_gate_inp"`
	ScoreCorrectionBias ml.Tensor  `gguf:"exp_probs_b.bias,alt:exp_probs_b"`
}

func (router *TopKRouter) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) (ml.Tensor, ml.Tensor) {
	routerLogits := router.Classifier.Forward(ctx, hiddenStates)
	scores := routerLogits.Softmax(ctx)
	scoresForChoice := scores
	if router.ScoreCorrectionBias != nil {
		scoresForChoice = scores.Add(ctx, router.ScoreCorrectionBias)
	}
	topKIndices := scoresForChoice.TopK(ctx, opts.numExpertsUsed)
	topKWeights := scores.Reshape(ctx, 1, opts.numExperts, hiddenStates.Dim(1)).Rows(ctx, topKIndices)
	topKWeights = topKWeights.Contiguous(ctx).Scale(ctx, float64(opts.routedScalingFactor))
	return topKWeights, topKIndices
}

type Experts struct {
	GateUp *nn.LinearBatch `gguf:"ffn_gate_up_exps"`
	Down   *nn.LinearBatch `gguf:"ffn_down_exps"`
}

func (experts *Experts) Forward(ctx ml.Context, hiddenStates, topKIndices, topKWeights ml.Tensor, opts *Options) ml.Tensor {
	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), 1, hiddenStates.Dim(1))

	var downIndices ml.Tensor = topKIndices
	var routedMask, zeroMask ml.Tensor
	if opts.numRoutedExperts > 0 && opts.numRoutedExperts < opts.numExperts {
		topKIndicesF32 := topKIndices.Cast(ctx, ml.DTypeF32)
		numRoutedConst := ctx.Input().FromFloats([]float32{float32(opts.numRoutedExperts)}, 1)
		routedMask = numRoutedConst.Sub(ctx, topKIndicesF32).Step(ctx)
		zeroMask = topKIndicesF32.Sub(ctx, numRoutedConst).Step(ctx)
		downIndices = topKIndicesF32.Mul(ctx, routedMask).Cast(ctx, ml.DTypeI32)
	}

	gateUp := experts.GateUp.Forward(ctx, hiddenStates, topKIndices)
	gateUpChunks := gateUp.ChunkSections(ctx, 0, gateUp.Dim(0)/2, gateUp.Dim(0)/2)
	gate := gateUpChunks[0]
	up := gateUpChunks[1]

	hiddenStates = gate.SILU(ctx, up)
	expertsOut := experts.Down.Forward(ctx, hiddenStates, downIndices)
	expertsOut = expertsOut.Mul(ctx, topKWeights)

	if routedMask != nil {
		routedMask = routedMask.Reshape(ctx, routedMask.Dim(0), 1, routedMask.Dim(1))
		expertsOut = expertsOut.Mul(ctx, routedMask)
	}

	nextStates := expertsOut.View(ctx, 0, expertsOut.Dim(0), expertsOut.Stride(2), expertsOut.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		nextStates = nextStates.Add(ctx, expertsOut.View(ctx, i*expertsOut.Stride(1), expertsOut.Dim(0), expertsOut.Stride(2), expertsOut.Dim(2)))
	}

	if zeroMask != nil {
		zeroWeights := topKWeights.Mul(ctx, zeroMask.Reshape(ctx, 1, zeroMask.Dim(0), zeroMask.Dim(1)))
		zeroWeightsPerToken := zeroWeights.View(ctx, 0, zeroWeights.Dim(0), zeroWeights.Stride(1), zeroWeights.Dim(2))
		for i := 1; i < opts.numExpertsUsed; i++ {
			zeroWeightsPerToken = zeroWeightsPerToken.Add(ctx, zeroWeights.View(ctx, i*zeroWeights.Stride(1), zeroWeights.Dim(0), zeroWeights.Stride(1), zeroWeights.Dim(2)))
		}
		zeroWeightsPerToken = zeroWeightsPerToken.Reshape(ctx, zeroWeightsPerToken.Dim(1), 1)
		hiddenOrig := hiddenStates.Reshape(ctx, hiddenStates.Dim(0), hiddenStates.Dim(2))
		zeroContrib := hiddenOrig.Mul(ctx, zeroWeightsPerToken)
		nextStates = nextStates.Add(ctx, zeroContrib)
	}

	return nextStates
}

type MoE struct {
	Router  *TopKRouter
	Experts *Experts
}

func (moe *MoE) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	if moe == nil || moe.Router == nil || moe.Experts == nil || opts.numExpertsUsed == 0 || opts.numExperts == 0 {
		return hiddenStates.Contiguous(ctx).Scale(ctx, 0)
	}

	topKWeights, topKIndices := moe.Router.Forward(ctx, hiddenStates, opts)
	return moe.Experts.Forward(ctx, hiddenStates, topKIndices, topKWeights, opts)
}

type TransformerBlock struct {
	InputLayerNorm0         *nn.RMSNorm `gguf:"attn_norm_0"`
	InputLayerNorm1         *nn.RMSNorm `gguf:"attn_norm_1"`
	PostAttentionLayerNorm0 *nn.RMSNorm `gguf:"ffn_norm_0"`
	PostAttentionLayerNorm1 *nn.RMSNorm `gguf:"ffn_norm_1"`

	Attention0 *MLA `gguf:",suf:_0"`
	Attention1 *MLA `gguf:",suf:_1"`
	MLP0       *MLP `gguf:",suf:_0"`
	MLP1       *MLP `gguf:",suf:_1"`

	ShortcutMoE *MoE
	LayerIdx    int
}

func (layer *TransformerBlock) Forward(ctx ml.Context, hiddenStates, positions, outputs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	residual := hiddenStates
	hiddenStates = layer.InputLayerNorm0.Forward(ctx, hiddenStates, opts.eps)

	if cache != nil {
		cache.SetLayer(layer.LayerIdx * 2)
	}
	hiddenStates = layer.Attention0.Forward(ctx, hiddenStates, positions, cache, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
	hiddenStates = layer.PostAttentionLayerNorm0.Forward(ctx, hiddenStates, opts.eps)

	shortcutOutput := layer.ShortcutMoE.Forward(ctx, hiddenStates, opts)
	hiddenStates = layer.MLP0.Forward(ctx, hiddenStates, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
	hiddenStates = layer.InputLayerNorm1.Forward(ctx, hiddenStates, opts.eps)

	if cache != nil {
		cache.SetLayer(layer.LayerIdx*2 + 1)
	}
	hiddenStates = layer.Attention1.Forward(ctx, hiddenStates, positions, cache, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)

	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
		shortcutOutput = shortcutOutput.Rows(ctx, outputs)
	}

	residual = hiddenStates
	hiddenStates = layer.PostAttentionLayerNorm1.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = layer.MLP1.Forward(ctx, hiddenStates, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)
	hiddenStates = hiddenStates.Add(ctx, shortcutOutput)

	return hiddenStates
}

func init() {
	model.Register("longcatflash", New)
}

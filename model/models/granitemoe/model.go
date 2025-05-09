package granitemoe

import (
	"fmt"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Options struct {
	hiddenSize, numHeads, numKVHeads, expertCount, expertUsedCount int
	eps, ropeBase, ropeScale                                       float32
	residualMultiplier,
	embeddingMultiplier,
	attentionMultiplier,
	logitsScaling float64
	ropeDim uint32
}

type Model struct {
	model.Base
	model.BytePairEncoding

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	*Options
}

func New(c fs.Config) (model.Model, error) {
	if !strings.EqualFold(c.String("tokenizer.ggml.model"), "gpt2") {
		return nil, fmt.Errorf("tokenizer %s not yet supported", c.String("tokenizer.ggml.model"))
	}
	var tokenizerPreExprs []string
	if tokenizerPre := c.String("tokenizer.ggml.pretokenizer", ""); tokenizerPre == "" {
		tokenizerPreName := c.String("tokenizer.ggml.pre", "")
		var tokenizerPreErr error
		if tokenizerPreExprs, tokenizerPreErr = model.GetKnownPretokenizerExpressions(tokenizerPreName); tokenizerPreErr != nil {
			return nil, tokenizerPreErr
		}
	} else {
		tokenizerPreExprs = []string{tokenizerPre}
	}
	m := Model{
		BytePairEncoding: model.NewMultiRegexBytePairEncoding(
			tokenizerPreExprs,
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				BOS:    int32(c.Uint("tokenizer.ggml.bos_token_id")),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", false),
				EOS:    int32(c.Uint("tokenizer.ggml.eos_token_id")),
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
			},
		),
		Layers: make([]Layer, c.Uint("block_count")),
		Options: &Options{
			hiddenSize:          int(c.Uint("embedding_length")),
			numHeads:            int(c.Uint("attention.head_count")),
			numKVHeads:          int(c.Uint("attention.head_count_kv")),
			expertCount:         int(c.Uint("expert_count")),
			expertUsedCount:     int(c.Uint("expert_used_count")),
			eps:                 c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:            c.Float("rope.freq_base"),
			ropeScale:           c.Float("rope.freq_scale", 1),
			ropeDim:             c.Uint("rope.dimension_count"),
			residualMultiplier:  float64(c.Float("residual_scale")),
			attentionMultiplier: float64(c.Float("attention.scale")),
			embeddingMultiplier: float64(c.Float("embedding_scale")),
			logitsScaling:       float64(c.Float("logit_scale")),
		},
	}
	slog.Debug("Parsed Options", "", m.Options)

	m.Cache = kvcache.NewCausalCache(m.Shift)

	return &m, nil
}

type SelfAttention struct {
	Query       *nn.Linear `gguf:"attn_q"`
	Key         *nn.Linear `gguf:"attn_k"`
	Value       *nn.Linear `gguf:"attn_v"`
	Output      *nn.Linear `gguf:"attn_output"`
	RopeFactors ml.Tensor  `gguf:"rope_freqs.weight"`
}

func (sa *SelfAttention) Forward(ctx ml.Context, hiddenState, positionIDs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	headDim := opts.hiddenSize / opts.numHeads
	ropeType := uint32(0)

	q := sa.Query.Forward(ctx, hiddenState)
	q = q.Reshape(ctx, headDim, opts.numHeads, batchSize)
	q = q.RoPE(ctx, positionIDs, sa.RopeFactors, opts.ropeDim, ropeType, opts.ropeBase, opts.ropeScale)

	k := sa.Key.Forward(ctx, hiddenState)
	k = k.Reshape(ctx, headDim, opts.numKVHeads, batchSize)
	k = k.RoPE(ctx, positionIDs, sa.RopeFactors, opts.ropeDim, ropeType, opts.ropeBase, opts.ropeScale)

	v := sa.Value.Forward(ctx, hiddenState)
	v = v.Reshape(ctx, headDim, opts.numKVHeads, batchSize)

	kqv := nn.Attention(ctx, q, k, v, opts.attentionMultiplier, cache)
	kqv = kqv.Reshape(ctx, opts.hiddenSize, batchSize)

	return sa.Output.Forward(ctx, kqv)
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return key.RoPE(ctx, shift, m.Layers[layer].SelfAttention.RopeFactors, uint32(0), m.Options.ropeDim, m.Options.ropeBase, m.Options.ropeScale), nil
}

type MoE struct {
	UpExps   *nn.Linear `gguf:"ffn_up_exps"`
	DownExps *nn.Linear `gguf:"ffn_down_exps"`
	GateInp  *nn.Linear `gguf:"ffn_gate_inp"`
	GateExps *nn.Linear `gguf:"ffn_gate_exps"`
}

func (moe *MoE) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
	shape := hiddenState.Shape()
	nEmbed := shape[0]
	nTokens := 1
	if len(shape) > 1 {
		nTokens = shape[1]
	}

	// GateInp * hiddenState => "logits"
	logits := moe.GateInp.Forward(ctx, hiddenState)
	// softmax(logits) => "probs"
	probs := logits.Softmax(ctx)
	// topk(probs, n_expert_used) => "selected_experts"
	selectedExperts := probs.TopK(ctx, opts.expertUsedCount)
	// Rows (probs, selected_experts) => "weights"
	weights := probs.Reshape(ctx, 1, opts.expertCount, nTokens).Rows(ctx, selectedExperts)
	// norm weights (weights / sum weights)
	weights = weights.Reshape(ctx, opts.expertUsedCount, nTokens)
	weightSum := weights.SumRows(ctx)
	weights = weights.Div(ctx, weightSum)
	weights = weights.Reshape(ctx, 1, opts.expertUsedCount, nTokens)
	// (?) reshape hiddenState => hiddenState
	hiddenState = hiddenState.Reshape(ctx, nEmbed, 1, nTokens)
	// UpExps * hiddenState (ids = selected_experts) => "up"
	up := moe.UpExps.Weight.MulmatID(ctx, hiddenState, selectedExperts)
	// GateExps * hiddenState (ids = selected_experts) => "gate"
	gate := moe.GateExps.Weight.MulmatID(ctx, hiddenState, selectedExperts)
	// SILU (gate) => "gate"
	gate = gate.SILU(ctx)
	// up * gate => "par"
	par := up.Mul(ctx, gate)
	// DownExps * par (ids = selected_experts) => "experts"
	experts := moe.DownExps.Weight.MulmatID(ctx, par, selectedExperts)
	// experts * weights => experts
	experts = experts.Mul(ctx, weights)
	expertStride1 := experts.Stride(1)
	expertStride2 := experts.Stride(2)
	// sum experts
	var moeOut ml.Tensor
	for i := range opts.expertUsedCount {
		curExpert := experts.View(ctx, i*expertStride1, nEmbed, expertStride2, nTokens)
		if i == 0 {
			moeOut = curExpert
		} else {
			moeOut = moeOut.Add(ctx, curExpert)
		}
	}
	return moeOut
}

type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention *SelfAttention
	MoENorm       *nn.RMSNorm `gguf:"ffn_norm"`
	MoE           *MoE
}

func (l *Layer) Forward(ctx ml.Context, hiddenState, positionIDs, outputs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	residual := hiddenState

	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, hiddenState, positionIDs, cache, opts)

	// In the final layer (outputs != nil), optimize by pruning to just the token positions
	// we need logits for.
	if outputs != nil {
		hiddenState = hiddenState.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenState = hiddenState.Scale(ctx, opts.residualMultiplier)
	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	hiddenState = l.MoENorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MoE.Forward(ctx, hiddenState, opts)
	hiddenState = hiddenState.Scale(ctx, opts.residualMultiplier)
	return hiddenState.Add(ctx, residual)
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions, err := ctx.Input().FromIntSlice(batch.Positions, len(batch.Positions))
	if err != nil {
		return nil, err
	}

	outputs, err := ctx.Input().FromIntSlice(batch.Outputs, len(batch.Outputs))
	if err != nil {
		return nil, err
	}

	hiddenState := m.TokenEmbedding.Forward(ctx, batch.Inputs)
	hiddenState = hiddenState.Scale(ctx, m.Options.embeddingMultiplier)

	for i, layer := range m.Layers {
		m.Cache.SetLayer(i)

		var lastLayerOutputs ml.Tensor
		if i == len(m.Layers)-1 {
			lastLayerOutputs = outputs
		}

		hiddenState = layer.Forward(ctx, hiddenState, positions, lastLayerOutputs, m.Cache, m.Options)
	}

	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	hiddenState = m.Output.Forward(ctx, hiddenState)
	hiddenState = hiddenState.Scale(ctx, 1.0/m.Options.logitsScaling)
	return hiddenState, nil
}

func init() {
	model.Register("granitemoe", New)
}

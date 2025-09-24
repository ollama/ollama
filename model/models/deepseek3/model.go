// gpt oss:
// - layernorm
// decoder layer = transformer block
// - attention
// - residual + hiddenStates
// - post attention
// - mlp

// the decorder layer is the same

package deepseek3

import (
	"cmp"
	"fmt"
	"log/slog"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/fast"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

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
	qLoraRank          *int
	attnImplementation string
	vHeadDim           int

	hiddenSize,
	numHeads,
	numKVHeads,
	keyLength,
	valueLength,
	originalContextLength int

	eps,
	ropeBase,
	ropeScale float32
	ropeType string
}

func (o Options) headDim() int {
	return cmp.Or(o.keyLength, o.valueLength, o.hiddenSize/o.numHeads)
}

func (o Options) RoPEOptions() []func(*rope.Options) {
	return []func(*rope.Options){
		rope.WithTypeNeoX(),
		rope.WithOriginalContextLength(o.originalContextLength),
		rope.WithExtrapolationFactor(1.),
	}
}

// -------------------------------------------------------------------------------------------------------------------
// tested

type AttentionBlock struct {
	Norm *nn.RMSNorm `gguf:"attn_norm"`

	Q *nn.Linear `gguf:"attn_q"`

	QA     *nn.Linear  `gguf:"attn_q_a"`
	QANorm *nn.RMSNorm `gguf:"attn_q_a_norm"`
	QB     *nn.Linear  `gguf:"attn_q_b"`

	KVA     *nn.Linear  `gguf:"attn_kv_a_mqa"`
	KVANorm *nn.RMSNorm `gguf:"attn_kv_a_norm"`
	KVB     *nn.Linear  `gguf:"attn_kv_b"`

	Output *nn.Linear `gguf:"attn_out,alt:attn_output"`
}

func (attn *AttentionBlock) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	hiddenStates = attn.Norm.Forward(ctx, hiddenStates, opts.eps)

	seqLength := hiddenStates.Dim(1)
	residual := hiddenStates

	var query ml.Tensor
	if opts.qLoraRank == nil {
		query = attn.Q.Forward(ctx, hiddenStates)
	} else {
		query = attn.QA.Forward(ctx, hiddenStates)
		query = attn.QANorm.Forward(ctx, query, opts.eps)
		query = attn.QB.Forward(ctx, query)
	}

	query = query.Reshape(ctx, query.Dim(0)/opts.numHeads, opts.numHeads, seqLength)

	qPass := query.View(ctx, 0,
		opts.qkNopeHeadDim, query.Stride(1),
		query.Dim(1), query.Stride(2),
		query.Dim(2))

	qRot := query.View(ctx, opts.qkNopeHeadDim*query.Stride(0),
		opts.qkRopeHeadDim, query.Stride(1),
		query.Dim(1), query.Stride(2),
		query.Dim(2))

	compressedKV := attn.KVA.Forward(ctx, hiddenStates)

	kPass := compressedKV.View(ctx, 0, opts.kvLoraRank, compressedKV.Stride(1), compressedKV.Dim(1))

	kRot := compressedKV.View(ctx, opts.kvLoraRank*compressedKV.Stride(0),
		opts.qkRopeHeadDim, compressedKV.Stride(1),
		1, compressedKV.Stride(1),
		compressedKV.Dim(1))

	kPass = attn.KVANorm.Forward(ctx, kPass, opts.eps)
	kPass = attn.KVB.Forward(ctx, kPass)

	kPass = kPass.Reshape(ctx, kPass.Dim(0)/opts.numKVHeads, opts.numKVHeads, seqLength)

	kPass = kPass.View(ctx, 0, opts.kqNopeHeadDim, kPass.Stride(1), kPass.Dim(1), kPass.Stride(2), kPass.Dim(2))

	value := kPass.View(ctx, opts.kqNopeHeadDim*kPass.Stride(0),
		opts.vHeadDim, kPass.Stride(1),
		kPass.Dim(1), kPass.Stride(2),
		kPass.Dim(2)).Contiguous(ctx)

	slog.Info("", "hello", "world")
	slog.Info("DEBUG: qRot", "shape", qRot.Shape())
	slog.Info("DEBUG: positions", "shape", positions.Shape())
	slog.Info("DEBUG: opts.qkRopeHeadDim", "value", opts.qkRopeHeadDim)
	slog.Info("DEBUG: opts.ropeBase", "value", opts.ropeBase)
	slog.Info("DEBUG: opts.ropeScale", "value", opts.ropeScale)
	slog.Info("DEBUG: opts.RoPEOptions()", "value", opts.RoPEOptions())

	qRot = qRot.Contiguous(ctx)

	// this is all new
	// qRot = fast.RoPE(ctx, qRot, positions, opts.qkRopeHeadDim, opts.ropeBase, opts.ropeScale) //, opts.RoPEOptions()...)
	// kRot = fast.RoPE(ctx, kRot, positions, opts.qkRopeHeadDim, opts.ropeBase, 1./opts.ropeScale, opts.RoPEOptions()...)

	// qRot = qRot.Contiguous(ctx)
	// return qRot

	kRot = kRot.Repeat(ctx, 1, qPass.Dim(1))

	query = qPass.Concat(ctx, qRot, 0)
	key := kPass.Concat(ctx, kRot, 0)

	if opts.attnImplementation == "flash_attention_2" && opts.qkHeadDim != opts.vHeadDim {

		print("not implemented")
	}

	scaling := scalingFactor(opts)

	// attention := nn.Attention(ctx, query, key, value, 1, nil)
	attention := nn.Attention(ctx, query, key, value, scaling, nil)

	if opts.attnImplementation == "flash_attention_2" && opts.qkHeadDim != opts.vHeadDim {
		// attention = attention[:, :, :, : self.vHeadDim]
		print("not implemented")
	}

	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), seqLength)
	return attn.Output.Forward(ctx, attention).Add(ctx, residual)
}

type SharedExpert struct {
	Norm *nn.RMSNorm `gguf:"ffn_norm"`
	Gate *nn.Linear  `gguf:"ffn_gate_shexp"`
	Up   *nn.Linear  `gguf:"ffn_up_shexp"`
	Down *nn.Linear  `gguf:"ffn_down_shexp"`
}

func (se *SharedExpert) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	hiddenStates = se.Norm.Forward(ctx, hiddenStates, opts.eps) // this is newly added
	hiddenStates = se.Gate.Forward(ctx, hiddenStates).SILU(ctx).Mul(ctx, se.Up.Forward(ctx, hiddenStates))
	return se.Down.Forward(ctx, hiddenStates)
}

// is there some way to share the norm methods?
type MLP interface {
	Forward(ml.Context, ml.Tensor, *Options) ml.Tensor
}

type MoEBlock struct {
	Norm         *nn.RMSNorm `gguf:"ffn_norm"`
	Router       *nn.Linear  `gguf:"ffn_gate_inp"`
	Gate         *nn.Linear  `gguf:"ffn_gate_exps"`
	Up           *nn.Linear  `gguf:"ffn_up_exps"`
	Down         *nn.Linear  `gguf:"ffn_down_exps"`
	SharedExpert *SharedExpert
	ExpProbsBias ml.Tensor `gguf:"exp_probs_b.bias,alt:exp_probs_b"`
}

func (moe *MoEBlock) Moe(ctx ml.Context, hiddenStates ml.Tensor, topKIndices ml.Tensor, topKWeights ml.Tensor, opts *Options) ml.Tensor {
	hiddenStates = moe.Norm.Forward(ctx, hiddenStates, opts.eps) // this is newly added
	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), 1, hiddenStates.Dim(1))

	upStates := moe.Up.Weight.MulmatID(ctx, hiddenStates, topKIndices)
	hiddenStates = moe.Gate.Weight.MulmatID(ctx, hiddenStates, topKIndices)
	hiddenStates = hiddenStates.SILU(ctx)
	hiddenStates = hiddenStates.Mul(ctx, upStates)
	experts := moe.Down.Weight.MulmatID(ctx, hiddenStates, topKIndices)
	experts = experts.Mul(ctx, topKWeights)
	nextStates := experts.View(ctx, 0, experts.Dim(0), experts.Stride(2), experts.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		nextStates = nextStates.Add(ctx, experts.View(ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2)))
	}
	return nextStates
}

func (moe *MoEBlock) getTopKIndices(ctx ml.Context, scores ml.Tensor, opts *Options) ml.Tensor {
	scores = scores.Add(ctx, moe.ExpProbsBias)
	topKIndices := scores.TopK(ctx, opts.numExpertsUsed)
	return topKIndices
}

// sparse block = Moe block
func (moe *MoEBlock) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	// hiddenDim, sequenceLength, batchSize := hiddenStates.Dim(0), hiddenStates.Dim(1), hiddenStates.Dim(2)
	residuals := hiddenStates

	routerLogits := moe.Router.Forward(ctx, hiddenStates)
	scores := routerLogits.Sigmoid(ctx)
	topKIndices := moe.getTopKIndices(ctx, scores, opts)
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

type MLPBlock struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *MLPBlock) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates).SILU(ctx).Mul(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, hiddenStates)
}

type TransformerBlock struct { // Similar to the decoder layer
	// AttentionNorm *nn.RMSNorm `gguf:"attn_norm"` // input_layernorm
	Attention *AttentionBlock

	MLPNorm *nn.RMSNorm `gguf:"ffn_norm"` // called the post attention norm
	MLP                 // will be an MLP or MoE block
	// TODO:
	// - postattentionnorm (mlp norm) can be moved into the mlp block
	// - pre attention norm can be moved into the attention block
}

func (t *TransformerBlock) Forward(ctx ml.Context, hiddenStates, positions, outputs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	residual := hiddenStates
	// hiddenStates = t.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = t.Attention.Forward(ctx, hiddenStates, positions, cache, opts)

	if outputs != nil { // not sure what this is for
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenStates = hiddenStates.Add(ctx, residual)
	residual = hiddenStates

	hiddenStates = t.MLPNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = t.MLP.Forward(ctx, hiddenStates, opts)
	return hiddenStates.Add(ctx, residual)
}

type Transformer struct {
	model.Base
	model.BytePairEncoding

	TokenEmbedding    *nn.Embedding      `gguf:"token_embd"`
	TransformerBlocks []TransformerBlock `gguf:"blk"`

	OutputNorm *nn.RMSNorm `gguf:"output_norm"`
	Output     *nn.Linear  `gguf:"output,alt:token_embd"`

	*Options
}

// const float mscale = attn_factor * (1.0f + hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
// const float kq_scale = 1.0f*mscale*mscale/sqrtf(float(n_embd_head_k));
// const float attn_factor = 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));

// self.scaling = self.qk_head_dim ** (-0.5)
// if self.config.rope_scaling is not None:
// 	mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
// 	scaling_factor = self.config.rope_scaling["factor"]
// 	if mscale_all_dim:
// 		mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
// 		self.scaling = self.scaling * mscale * mscale

func yarn_get_mscale(scaling_factor float64, mscale float64) float64 {
	if scaling_factor <= 1.0 {
		return 1.0
	}
	return 0.1*mscale*math.Log(scaling_factor) + 1.0 // there has to be a better way to do this
}

func scalingFactor(opts *Options) float64 {
	scaling := math.Pow(float64(opts.qkHeadDim), -0.5)
	mscaleAllDim := 1.0
	if opts.ropeType == "yarn" {
		mscale := yarn_get_mscale(float64(opts.ropeScale), mscaleAllDim)
		scaling = scaling * mscale * mscale
	}
	return scaling
}

func New(c fs.Config) (model.Model, error) {
	fmt.Printf("DEBUG: the total number of layers: %v", c.Uint("block_count"))
	transformerBlocks := make([]TransformerBlock, 1)
	// transformerBlocks := make([]TransformerBlock, c.Uint("block_count"))

	firstDenseLayerIndex := int(c.Uint("leading_dense_block_count")) // or whatever key your gguf uses
	fmt.Printf("first dense: %v", firstDenseLayerIndex)
	for i := range transformerBlocks {
		if i < firstDenseLayerIndex {
			transformerBlocks[i].MLP = &MLPBlock{} // gguf tags on MLPBlock fields
		} else {
			transformerBlocks[i].MLP = &MoEBlock{} // gguf tags on Router/Experts fields
		}
	}

	qLoraRankVal := int(c.Uint("q_lora_rank"))

	m := Transformer{
		TransformerBlocks: transformerBlocks,
		BytePairEncoding: model.NewBytePairEncoding(
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
		),
		Options: &Options{
			hiddenSize:     int(c.Uint("embedding_length")),
			numHeads:       int(c.Uint("attention.head_count")),
			numKVHeads:     int(c.Uint("attention.head_count_kv")),
			keyLength:      int(c.Uint("attention.key_length")),
			valueLength:    int(c.Uint("attention.value_length")),
			eps:            c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:       c.Float("rope.freq_base"),
			ropeScale:      c.Float("rope.scaling.factor"),
			numExperts:     int(c.Uint("expert_count")),
			numExpertsUsed: int(c.Uint("expert_used_count")),
			normTopKProb:   c.Bool("norm_top_k_prob", true),

			qLoraRank:     &qLoraRankVal,
			kvLoraRank:    int(c.Uint("kv_lora_rank")),
			qkHeadDim:     int(c.Uint("attention.key_length")),
			vHeadDim:      int(c.Uint("attention.value_length")),
			qkRopeHeadDim: int(c.Uint("rope.dimension_count")),
			qkNopeHeadDim: int(c.Uint("attention.key_length")) - int(c.Uint("rope.dimension_count")),
			kqNopeHeadDim: int(c.Uint("attention.key_length")) - int(c.Uint("rope.dimension_count")),

			routedScalingFactor:   c.Float("routed_scaling_factor"),
			originalContextLength: int(c.Uint("rope.scaling.original_context_length")),
			ropeType:              c.String("rope.scaling.type"),
		},
	}
	// m.Cache = kvcache.NewCausalCache(nil) // TODO: add correct cache
	m.Cache = kvcache.NewCausalCache(m.Shift)

	return &m, nil
}

func (m Transformer) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return fast.RoPE(ctx, key, shift, m.headDim(), m.ropeBase, 1./m.ropeScale, m.RoPEOptions()...), nil // this is from gpt oss, also uses yarn
	// ropeDim := cmp.Or(m.ropeDim, m.hiddenSize/m.numHeads)
	// return fast.RoPE(ctx, key, shift, ropeDim, m.ropeBase, m.ropeScale, rope.WithTypeNeoX()), nil // this was qwen?
}

// Todo:
// - implement the shift
// add the cache
// do the rope function

func (m *Transformer) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromIntSlice(batch.Positions, len(batch.Positions))

	// token embedding
	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	for i, layer := range m.TransformerBlocks {
		// m.Cache.SetLayer(i)

		var outputs ml.Tensor
		if i == len(m.TransformerBlocks)-1 {
			outputs = ctx.Input().FromIntSlice(batch.Outputs, len(batch.Outputs))
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, outputs, m.Cache, m.Options)
	}

	// output norm
	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	// output projection
	return m.Output.Forward(ctx, hiddenStates), nil
}

func init() {
	model.Register("deepseek2", New)
}

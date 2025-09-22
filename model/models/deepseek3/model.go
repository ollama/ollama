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

	// yarn/rope tuning parameters (computed in New and used during forward)
	mScale              float32
	kqScale             float64
	attnFactor          float32
	yarn_log_multiplier float32
}

func (o Options) DebugPrint() {
	var qlr any
	if o.qLoraRank != nil {
		qlr = *o.qLoraRank
	} else {
		qlr = nil
	}
	fmt.Printf("DEBUG: Options:\n")
	fmt.Printf("  numExperts=%d\n", o.numExperts)
	fmt.Printf("  numExpertsUsed=%d\n", o.numExpertsUsed)
	fmt.Printf("  normTopKProb=%t\n", o.normTopKProb)
	fmt.Printf("  routedScalingFactor=%f\n", o.routedScalingFactor)
	fmt.Printf("  kvLoraRank=%d\n", o.kvLoraRank)
	fmt.Printf("  qkNopeHeadDim=%d\n", o.qkNopeHeadDim)
	fmt.Printf("  qkRopeHeadDim=%d\n", o.qkRopeHeadDim)
	fmt.Printf("  kqNopeHeadDim=%d\n", o.kqNopeHeadDim)
	fmt.Printf("  qkHeadDim=%d\n", o.qkHeadDim)
	fmt.Printf("  qLoraRank=%v\n", qlr)
	fmt.Printf("  attnImplementation=%s\n", o.attnImplementation)
	fmt.Printf("  vHeadDim=%d\n", o.vHeadDim)
	fmt.Printf("  hiddenSize=%d\n", o.hiddenSize)
	fmt.Printf("  numHeads=%d\n", o.numHeads)
	fmt.Printf("  numKVHeads=%d\n", o.numKVHeads)
	fmt.Printf("  keyLength=%d\n", o.keyLength)
	fmt.Printf("  valueLength=%d\n", o.valueLength)
	fmt.Printf("  originalContextLength=%d\n", o.originalContextLength)
	fmt.Printf("  eps=%g\n", o.eps)
	fmt.Printf("  ropeBase=%g\n", o.ropeBase)
	fmt.Printf("  ropeScale=%g\n", o.ropeScale)
	fmt.Printf("  ropeType=%s\n", o.ropeType)
	fmt.Printf("  mscale=%g\n", o.mScale)
	fmt.Printf("  kqScale=%g\n", o.kqScale)
	fmt.Printf("  attnFactor=%g\n", o.attnFactor)
}

func (o Options) headDim() int {
	return cmp.Or(o.keyLength, o.valueLength, o.hiddenSize/o.numHeads)
}

func (o Options) RoPEOptions() []func(*rope.Options) {
	attnFactor := float32(1.0 / (1.0 + 0.1*math.Log(float64(o.ropeScale))))
	fmt.Printf("DEBUG: originalContextLength: %v\n", o.originalContextLength)
	fmt.Printf("DEBUG: attnFactor: %v\n", attnFactor)
	return []func(*rope.Options){
		// rope.WithTypeNeoX(),
		rope.WithOriginalContextLength(o.originalContextLength),
		rope.WithExtrapolationFactor(1.),
		// rope.WithAttentionFactor(o.attnFactor),
		rope.WithAttentionFactor(attnFactor),
	}
}

// -------------------------------------------------------------------------------------------------------------------
// tested

type AttentionBlock struct {
	// Norm *nn.RMSNorm `gguf:"attn_norm"`

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

	mScale := float32(1.0 + float64(opts.yarn_log_multiplier)*math.Log(float64(opts.ropeScale)))
	fmt.Printf("DEBUG: ropeScale: %v\n", opts.ropeScale)
	fmt.Printf("DEBUG: keyLength: %v\n", opts.keyLength)
	kqScale := float64(mScale * mScale / float32(math.Sqrt(float64(opts.qkHeadDim)))) // check what n_embd_head_k -- this goes into the attention
	fmt.Printf("DEBUG: mScale: %v\n", mScale)
	fmt.Printf("DEBUG: kqScale: %v\n", kqScale)

	// hiddenStates = attn.Norm.Forward(ctx, hiddenStates, opts.eps)

	// return hiddenStates

	seqLength := hiddenStates.Dim(1)
	// residual := hiddenStates

	var query ml.Tensor
	fmt.Printf("DEBUG: opts.qLoraRank: %v\n", opts.qLoraRank)
	if opts.qLoraRank == nil {
		fmt.Printf("DEBUG: qLoraRank is nil\n")
		query = attn.Q.Forward(ctx, hiddenStates)
	} else {
		fmt.Printf("DEBUG: qLoraRank is not nil\n")
		query = attn.QA.Forward(ctx, hiddenStates)
		query = attn.QANorm.Forward(ctx, query, opts.eps)
		query = attn.QB.Forward(ctx, query)
	}

	// return query

	fmt.Printf("DEBUG: query: %v\n", query.Shape())

	query = query.Reshape(ctx, query.Dim(0)/opts.numHeads, opts.numHeads, seqLength)

	fmt.Printf("DEBUG: query after reshape: %v\n", query.Shape())
	fmt.Printf("DEBUG: query dtype: %v\n", query.DType())

	// query = query.Contiguous(ctx)
	// return query

	qPass := query.View(ctx, 0,
		opts.qkNopeHeadDim, query.Stride(1),
		query.Dim(1), query.Stride(2),
		query.Dim(2))

	qRot := query.View(ctx, opts.qkNopeHeadDim*query.Stride(0),
		opts.qkRopeHeadDim, query.Stride(1),
		query.Dim(1), query.Stride(2),
		query.Dim(2))

	fmt.Printf("DEBUG: qRot shape: %v\n", qRot.Shape())
	fmt.Printf("DEBUG: qPass shape: %v\n", qPass.Shape())

	// fmt.Printf("DEBUG: qRot dtype: %v\n", qRot.Shape())
	// qRot = qRot.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	// return qRot

	compressedKV := attn.KVA.Forward(ctx, hiddenStates)
	fmt.Printf("DEBUG: compressedKV: %v\n", compressedKV.Shape())

	kPass := compressedKV.View(ctx, 0, opts.kvLoraRank, compressedKV.Stride(1), compressedKV.Dim(1))
	fmt.Printf("DEBUG: kPass: %v\n", kPass.Shape())

	kRot := compressedKV.View(ctx, opts.kvLoraRank*compressedKV.Stride(0),
		opts.qkRopeHeadDim, compressedKV.Stride(1),
		1, compressedKV.Stride(1),
		compressedKV.Dim(1))
	fmt.Printf("DEBUG: kRot: %v\n", kRot.Shape())

	kPass = attn.KVANorm.Forward(ctx, kPass, opts.eps)
	fmt.Printf("DEBUG: kPass after norm: %v\n", kPass.Shape())

	kPass = attn.KVB.Forward(ctx, kPass)
	fmt.Printf("DEBUG: kPass after linear KVB: %v\n", kPass.Shape())

	// kPass = kPass.Reshape(ctx, kPass.Dim(0)/opts.numKVHeads, opts.numKVHeads, seqLength)
	kv := kPass.Reshape(ctx, kPass.Dim(0)/opts.numKVHeads, opts.numKVHeads, seqLength)


	fmt.Printf("DEBUG: kPass after reshape: %v\n", kPass.Shape())

	// kPass = kPass.View(ctx, 0, opts.kqNopeHeadDim, kPass.Stride(1), kPass.Dim(1), kPass.Stride(2), kPass.Dim(2))
	// kPass = kv.View(ctx, 0, opts.kqNopeHeadDim, kPass.Stride(1), kPass.Dim(1), kPass.Stride(2), kPass.Dim(2))

	kPass = kv.View(ctx, 0, opts.kqNopeHeadDim, kv.Stride(1), kv.Dim(1), kv.Stride(2), kv.Dim(2))

	fmt.Printf("DEBUG: kPass after view: %v\n", kPass.Shape())

	// value := kPass.View(ctx, opts.kqNopeHeadDim*kPass.Stride(0),
	// 	opts.vHeadDim, kPass.Stride(1),
	// 	kPass.Dim(1), kPass.Stride(2),
	// 	kPass.Dim(2)).Contiguous(ctx)

	value := kv.View(ctx, opts.kqNopeHeadDim*kv.Stride(0),
		opts.vHeadDim, kv.Stride(1),
		kv.Dim(1), kv.Stride(2),
		kv.Dim(2)).Contiguous(ctx)

	ml.SetName(value, "value_before_rope (never does rope)")

	fmt.Printf("DEBUG: value: %v\n", value.Shape())

	slog.Info("", "hello", "world")
	// slog.Info("DEBUG: qRot", "shape", qRot.Shape())
	// slog.Info("DEBUG: positions", "shape", positions.Shape())
	// slog.Info("DEBUG: opts.qkRopeHeadDim", "value", opts.qkRopeHeadDim)
	// slog.Info("DEBUG: opts.ropeBase", "value", opts.ropeBase)
	// slog.Info("DEBUG: opts.ropeScale", "value", opts.ropeScale)
	// slog.Info("DEBUG: opts.RoPEOptions()", "value", opts.RoPEOptions())

	fmt.Printf("DEBUG: BRUHHHHHHH: %v\n", qRot.Shape())

	// qRot = qRot.Contiguous(ctx)
	fmt.Printf("Attention factor: %v\n", opts.attnFactor)
	fmt.Printf("DEBUG: qRot before rope: %v\n", qRot.Shape())
	fmt.Printf("DEBUG: qRot dtype: %v\n", qRot.DType())
	// qRot = qRot.Contiguous(ctx)
	// return qRot

	fmt.Printf("DEBUG: opts.ropeBase: %v\n", opts.ropeBase)
	fmt.Printf("DEBUG: 1 / opts.ropeScale: %v\n", 1./opts.ropeScale)
	fmt.Printf("DEBUG: opts.qkRopeHeadDim: %v\n", opts.qkRopeHeadDim)

	// qRot = qRot.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)

	fmt.Printf("DEBUG: qkRopeHeadDim: %v\n", opts.qkRopeHeadDim)

	// qRot = qRot.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	// fmt.Printf("DEBUG: qRot after permute: %v\n", qRot.Shape())

	// a couple questions:
	// - is the input to the pytorch correct?

	// so right before it, its correct

	// this is all new
	ml.SetName(qRot, "qRot_before_rope")

	fmt.Printf("DEBUG: qRot before rope shape: %v\n", qRot.Shape())
	fmt.Printf("DEBUG: positions: %v\n", positions.Shape())

	qRot = fast.RoPE(ctx, qRot, positions, opts.qkRopeHeadDim, opts.ropeBase, 1./opts.ropeScale, opts.RoPEOptions()...)

	ml.SetName(qRot, "qRot_after_rope")
	// but right after it, its wrong

	// fmt.Printf("DEBUG: qRot after rope (after change): %v\n", qRot.Shape())

	// return qRot

	// qRot = qRot.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	// return qRot

	ml.SetName(kRot, "kRot_before_rope")



	kRot = fast.RoPE(ctx, kRot, positions, opts.qkRopeHeadDim, opts.ropeBase, 1./opts.ropeScale, opts.RoPEOptions()...)

	fmt.Printf("DEBUG: kRot after rope shape: %v\n", kRot.Shape())

	ml.SetName(kRot, "kRot_after_rope")
	// qRot = fast.RoPE(ctx, qRot, positions, opts.qkRopeHeadDim, opts.ropeBase, opts.attnFactor, opts.RoPEOptions()...)
	// kRot = fast.RoPE(ctx, kRot, positions, opts.qkRopeHeadDim, opts.ropeBase, opts.attnFactor, opts.RoPEOptions()...)

	kRot = kRot.Repeat(ctx, 1, qPass.Dim(1))

	fmt.Printf("DEBUG: kRot repeat: %v\n", kRot.Shape())

	// query = qPass.Concat(ctx, qRot, 0)
	query = qRot.Concat(ctx, qPass, 0)
	ml.SetName(query, "query_concat")
	// key := kPass.Concat(ctx, kRot, 0)
	key := kRot.Concat(ctx, kPass, 0)
	ml.SetName(key, "key_concat")

	// so in llamacpp, they do
	// ggml_tensor * Qcur = ggml_concat(ctx0, q_pe, q_nope, 0);
	// cb(Qcur, "Qcur", il);

	// ggml_tensor * Kcur = ggml_concat(ctx0, ggml_repeat(ctx0, k_pe, q_pe), k_nope, 0);
	// cb(Kcur, "Kcur", il);


	// attention := nn.Attention(ctx, query, key, value, 1, nil)
	// attention := nn.Attention(ctx, query, key, value, scaling, nil)
	// attention := nn.Attention(ctx, query, key, value, opts.kqScale, nil)
	attention := nn.Attention(ctx, query, key, value, opts.kqScale, cache)
	ml.SetName(attention, "attention")

	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), seqLength)
	ml.SetName(attention, "attention_reshape")

	return attn.Output.Forward(ctx, attention) //.Add(ctx, residual) // here i add residual
}

type SharedExpert struct {
	// Norm *nn.RMSNorm `gguf:"ffn_norm"`
	Gate *nn.Linear  `gguf:"ffn_gate_shexp"`
	Up   *nn.Linear  `gguf:"ffn_up_shexp"`
	Down *nn.Linear  `gguf:"ffn_down_shexp"`
}

func (se *SharedExpert) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	// hiddenStates = se.Norm.Forward(ctx, hiddenStates, opts.eps) // this is newly added
	hiddenStates = se.Gate.Forward(ctx, hiddenStates).SILU(ctx).Mul(ctx, se.Up.Forward(ctx, hiddenStates))
	return se.Down.Forward(ctx, hiddenStates)
}

// is there some way to share the norm methods?
type MLP interface {
	Forward(ml.Context, ml.Tensor, *Options) ml.Tensor
}

type MoEBlock struct {
	// Norm         *nn.RMSNorm `gguf:"ffn_norm"`
	Router       *nn.Linear  `gguf:"ffn_gate_inp"`
	Gate         *nn.Linear  `gguf:"ffn_gate_exps"`
	Up           *nn.Linear  `gguf:"ffn_up_exps"`
	Down         *nn.Linear  `gguf:"ffn_down_exps"`
	SharedExpert *SharedExpert
	ExpProbsBias ml.Tensor `gguf:"exp_probs_b.bias,alt:exp_probs_b"`
}

func (moe *MoEBlock) Moe(ctx ml.Context, hiddenStates ml.Tensor, topKIndices ml.Tensor, topKWeights ml.Tensor, opts *Options) ml.Tensor {
	// hiddenStates = moe.Norm.Forward(ctx, hiddenStates, opts.eps) // this is newly added
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
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"` // input_layernorm
	Attention *AttentionBlock

	MLPNorm *nn.RMSNorm `gguf:"ffn_norm"` // called the post attention norm
	MLP                 // will be an MLP or MoE block
	// TODO:
	// - postattentionnorm (mlp norm) can be moved into the mlp block
	// - pre attention norm can be moved into the attention block
}

func (t *TransformerBlock) Forward(ctx ml.Context, hiddenStates, positions, outputs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {

	fmt.Printf("DEBUG: m.mscale: %f, m.kqScale: %f, m.attnFactor: %f\n", opts.mScale, opts.kqScale, opts.attnFactor)

	residual := hiddenStates
	fmt.Printf("DEBUG: residuals: %v\n", residual.Shape())
	hiddenStates = t.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = t.Attention.Forward(ctx, hiddenStates, positions, cache, opts)
	fmt.Printf("DEBUG: hiddenStates after attention: %v\n", hiddenStates.Shape())

	if outputs != nil { // not sure what this is for
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenStates = hiddenStates.Add(ctx, residual)
	residual = hiddenStates

	hiddenStates = t.MLPNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = t.MLP.Forward(ctx, hiddenStates, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)
	// ml.SetName("l_out", "hiddenStates")

	// temp := hiddenStates.Duplicate(ctx)
	// ml.SetName(temp, "l_out")

	return hiddenStates// hiddenStates.Add(ctx, residual)
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

func New(c fs.Config) (model.Model, error) {
	// fmt.Printf("DEBUG: the total number of layers: %v", c.Uint("block_count"))
	// transformerBlocks := make([]TransformerBlock, 3)
	fmt.Printf("DEBUG: the total number of layers: %v", c.Uint("block_count"))
	transformerBlocks := make([]TransformerBlock, c.Uint("block_count"))
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

	fmt.Printf("DEBUG: HELLO c.Float(\"rope.scaling.yarn_log_multiplier\"): %f, c.Float(\"rope.scaling.factor\"): %f\n", c.Float("rope.scaling.yarn_log_multiplier"), c.Float("rope.scaling.factor"))
	fmt.Printf("DEBUG: HELLO c.Uint(\"attention.key_length\"): %d\n", c.Uint("attention.key_length"))

	// qLoraRankVal := int(c.Uint("q_lora_rank"))
	mScale := float32(1.0 + float64(c.Float("rope.scaling.yarn_log_multiplier"))*math.Log(float64(c.Float("rope.scaling.factor"))))
	kqScale := float64(mScale * mScale / float32(math.Sqrt(float64(c.Uint("attention.key_length"))))) // check what n_embd_head_k -- this goes into the attention
	attnFactor := float32(1.0 / (1.0 + 0.1*math.Log(float64(c.Float("rope.scaling.factor")))))        // this goes into the rope

	fmt.Printf("DEBUG: HELLO mScale: %f, kqScale: %f, attnFactor: %f\n", mScale, kqScale, attnFactor)

	// fmt.Printf("DEBUG: mscale: %f, kqScale: %f, attnFactor: %f\n", mscale, kqScale, attnFactor)

	// fmt.Printf("DEBUG: qLoraRankVal: %v\n", qLoraRankVal

	qLoraRankVal := int(c.Uint("attention.q_lora_rank"))

	m := Transformer{
		TransformerBlocks: transformerBlocks,
		BytePairEncoding: model.NewBytePairEncoding(
			// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
			// `\p{N}{1,3}|[一-龥぀-ゟ゠-ヿ]+|[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+`,
			`[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`+"`"+`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
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
			ropeScale:      c.Float("rope.scaling.factor", 1),
			numExperts:     int(c.Uint("expert_count")),
			numExpertsUsed: int(c.Uint("expert_used_count")),
			normTopKProb:   c.Bool("norm_top_k_prob", true),

			qLoraRank:     &qLoraRankVal,
			kvLoraRank:    int(c.Uint("attention.kv_lora_rank")),
			qkHeadDim:     int(c.Uint("attention.key_length")),
			vHeadDim:      int(c.Uint("attention.value_length")),
			qkRopeHeadDim: int(c.Uint("rope.dimension_count")),
			qkNopeHeadDim: int(c.Uint("attention.key_length")) - int(c.Uint("rope.dimension_count")),
			kqNopeHeadDim: int(c.Uint("attention.key_length")) - int(c.Uint("rope.dimension_count")),

			routedScalingFactor:   c.Float("expert_weights_scale"),
			originalContextLength: int(c.Uint("rope.scaling.original_context_length")),
			ropeType:              c.String("rope.scaling.type"),

			mScale:              mScale,
			kqScale:             kqScale,
			attnFactor:          attnFactor,
			yarn_log_multiplier: c.Float("rope.scaling.yarn_log_multiplier"),
		},
	}

	// mscale: 1.368888, kq_scale: 0.135234, attn_factor: 1.000000
	// mscale: 1.368888, kq_scale: 0.135234, attn_factor: 1.000000
	// mscale: 1.368888, kq_scale: 0.135234, attn_factor: 1.000000

	// mscale: 0.631112, kqScale: 0.028745, attnFactor: 1.584505

	// mscale := 1.0 + float64(c.Float("rope.scaling.yarn_log_multiplier")) * math.Log(float64(c.Float("rope.scaling.factor")))
	// kqScale := 1.0 * mscale * mscale / math.Sqrt(float64(c.Uint("attention.key_length"))) // check what n_embd_head_k -- this goes into the attention
	// attnFactor := 1.0 / mscale // this goes into the rope

	// m.Options.mscale = mscale
	// m.Options.kqScale = kqScale
	// m.Options.attnFactor = attnFactor

	fmt.Printf("DEBUG: mscale: %f, kqScale: %f, attnFactor: %f\n", mScale, kqScale, attnFactor)

	// fmt.Printf("DEBUG: m.mscale: %f, m.kqScale: %f, m.attnFactor: %f\n", m.mscale, m.kqScale, m.attnFactor)

	fmt.Printf("DEBUG: qLoraRankVal: %v\n", qLoraRankVal)

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
	// Print all options once per forward
	if m.Options != nil {
		m.Options.DebugPrint()
	}
	positions := ctx.Input().FromIntSlice(batch.Positions, len(batch.Positions))

	fmt.Printf("DEBUG: positions: %v\n", positions.Shape())

	// token embedding
	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	fmt.Printf("DEBUG: hiddenStates after token embedding: %v\n", hiddenStates.Shape())

	for i, layer := range m.TransformerBlocks {
		m.Cache.SetLayer(i)
		fmt.Printf("DEBUG: layer: %v\n", i)

		var outputs ml.Tensor
		if i == len(m.TransformerBlocks)-1 {
			outputs = ctx.Input().FromIntSlice(batch.Outputs, len(batch.Outputs))
			fmt.Printf("DEBUG: outputs from last layer: %v\n", outputs.Shape())
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, outputs, m.Cache, m.Options)
		fmt.Printf("DEBUG: hiddenStates after layer: %v\n", hiddenStates.Shape())
	}

	// output norm
	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	fmt.Printf("DEBUG: hiddenStates after output norm: %v\n", hiddenStates.Shape())
	// output projection
	return m.Output.Forward(ctx, hiddenStates), nil
}

func init() {
	model.Register("deepseek2", New)
}

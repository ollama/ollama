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
	// model.BytePairEncoding

	// TokenEmbedding    *nn.Embedding      `gguf:"token_embd"`
	TransformerBlocks []TransformerBlock `gguf:"blk"`
	// OutputNorm        *nn.RMSNorm        `gguf:"output_norm"`
	// Output            *nn.Linear         `gguf:"output,alt:token_embd"`

	Options
}

type TransformerBlock struct {
	Attention *AttentionBlock
	MLP       *MLPBlock
	// MoE       *MoEBlock
	// the only diff is its MLP or MoE
}

type Options struct {
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
}

func (o Options) headDim() int {
	return cmp.Or(o.keyLength, o.valueLength, o.hiddenSize/o.numHeads)
}

func (o Options) RoPEOptions() []func(*rope.Options) {
	return []func(*rope.Options){
		rope.WithTypeNeoX(),
		rope.WithOriginalContextLength(o.originalContextLength),
		rope.WithExtrapolationFactor(1.),
		// NOTE: ggml sets this implicitly so there's no need to set it here
		// rope.WithAttentionFactor(0.1*float32(math.Log(float64(o.ropeScale))) + 1.0),
	}
}

type MLPBlock struct {
	Norm *nn.RMSNorm `gguf:"ffn_norm"`
	Up   *nn.Linear  `gguf:"ffn_up"`
	Down *nn.Linear  `gguf:"ffn_down"`
	Gate *nn.Linear  `gguf:"ffn_gate"`
}

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

	kRot = kRot.Repeat(ctx, 1, qPass.Dim(1))

	query = qPass.Concat(ctx, qRot, 0)
	key := kPass.Concat(ctx, kRot, 0)

	if opts.attnImplementation == "flash_attention_2" && opts.qkHeadDim != opts.vHeadDim {

		print("not implemented")
	}

	attention := nn.Attention(ctx, query, key, value, 1, nil)

	if opts.attnImplementation == "flash_attention_2" && opts.qkHeadDim != opts.vHeadDim {
		// attention = attention[:, :, :, : self.vHeadDim]
		print("not implemented")
	}

	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), seqLength)
	return attn.Output.Forward(ctx, attention).Add(ctx, residual)
}

func New(c fs.Config) (model.Model, error) {
	m := Transformer{
		TransformerBlocks: make([]TransformerBlock, 1),
		// BytePairEncoding: model.NewBytePairEncoding(
		// 	c.String("tokenizer.ggml.pretokenizer", `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`),
		// ),
	}
	// m.Cache = kvcache.NewCausalCache(nil)

	m.Cache = kvcache.NewCausalCache(nil)

	return &m, nil
}

func (m *Transformer) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	return batch.Inputs, nil
}

func init() {
	model.Register("deepseek2", New)
}

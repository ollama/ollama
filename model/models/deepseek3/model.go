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
	"math"

	// "github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/fast"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	// "github.com/ollama/ollama/model"
	// "github.com/ollama/ollama/model/input"
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
	// MLP       *MLPBlock
	// MoE       *MoEBlock
	// the only diff is its MLP or MoE
}

type Options struct {
	kvLoraRank,
	qkNopeHeadDim,
	qkRopeHeadDim,
	kqNopeHeadDim,
	qkHeadDim int
	qLoraRank          *int // can we specify this as a int or nil?
	attnImplementation string
	vHeadDim           int
	// headDim int

	hiddenSize,
	numHeads,
	numKVHeads,
	keyLength,
	valueLength,
	// numExperts,
	// numExpertsUsed,
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

type AttentionBlock struct {
	Norm *nn.RMSNorm `gguf:"attn_norm"`

	// if q_lora_rank is None
	Q *nn.Linear `gguf:"attn_q"`
	// else
	QA     *nn.Linear  `gguf:"attn_q_a"`
	QANorm *nn.RMSNorm `gguf:"attn_q_a_norm"`
	QB     *nn.Linear  `gguf:"attn_q_b"`

	KVA     *nn.Linear  `gguf:"attn_kv_a_mqa"`
	KVANorm *nn.RMSNorm `gguf:"attn_kv_a_norm"`
	KVB     *nn.Linear  `gguf:"attn_kv_b"`

	Output *nn.Linear `gguf:"attn_out,alt:attn_output"` // where does attn_out come from?
}

// func (attn *AttentionBlock) Forward(ctx ml.Context, ...) {
func (attn *AttentionBlock) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	what, seqLength, batchSize := hiddenStates.Dim(0), hiddenStates.Dim(1), hiddenStates.Dim(2)
	fmt.Printf("DEBUG: what: %v\n", what)
	fmt.Printf("DEBUG: seqLength: %v\n", seqLength)
	fmt.Printf("DEBUG: batchSize: %v\n", batchSize)

	fmt.Printf("DEBUG: hiddenStates: %v\n", ml.Dump(ctx, hiddenStates, ml.DumpWithPrecision(10)))
	residual := hiddenStates

	var query ml.Tensor
	// idk how to set up the beginning
	if opts.qLoraRank == nil {
		query = attn.Q.Forward(ctx, hiddenStates)
	} else {
		// q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
		fmt.Printf("DEBUG: hiddenStates: %v\n", hiddenStates.Shape())
		query = attn.QA.Forward(ctx, hiddenStates)
		fmt.Printf("DEBUG: query: %v\n", query.Shape())

		fmt.Printf("opts.eps: %v\n", opts.eps)

		query = attn.QANorm.Forward(ctx, query, opts.eps)
		fmt.Printf("DEBUG: query: %v\n", query.Shape())

		query = attn.QB.Forward(ctx, query)

	}

	return query

	// Debug: Print tensor shapes before reshape
	fmt.Printf("DEBUG: query shape before reshape: %v\n", query.Shape())
	fmt.Printf("DEBUG: trying to reshape to: [%d, %d, %d]\n", opts.headDim(), opts.numHeads, batchSize)

	// q_states = q_states.view(query_shape).transpose(1, 2)
	// PyTorch: [1, 4, 24576] -> [1, 128, 4, 192]
	// Ollama:  [24576, 4] -> [192, 128, 4] (head_dim, num_heads, seq_len)
	actualHeadDim := query.Dim(0) / opts.numHeads // 24576 / 128 = 192
	fmt.Printf("DEBUG: reshaping [%d, %d] -> [%d, %d, %d]\n", query.Dim(0), query.Dim(1), actualHeadDim, opts.numHeads, seqLength)
	query = query.Reshape(ctx, actualHeadDim, opts.numHeads, seqLength)
	// Split query into no-position and rotary parts along head dimension
	// query shape: [192, 128, 4] -> qPass: [128, 128, 4], qRot: [64, 128, 4]
	fmt.Printf("DEBUG: query after reshape: %v\n", query.Shape())

	qPass := query.View(ctx, 0, opts.qkNopeHeadDim, query.Stride(1), query.Dim(1), query.Stride(2), query.Dim(2))
	fmt.Printf("DEBUG: qPass: %v\n", qPass.Shape())

	qRot := query.View(ctx, opts.qkNopeHeadDim*query.Stride(0), opts.qkRopeHeadDim, query.Stride(1), query.Dim(1), query.Stride(2), query.Dim(2))
	fmt.Printf("DEBUG: qRot: %v\n", qRot.Shape())

	compressedKV := attn.KVA.Forward(ctx, hiddenStates)
	fmt.Printf("DEBUG: compressedKV: %v\n", compressedKV.Shape())
	// kPass, kRot := compressedKV.Split(ctx, opts.kvLoraRank, opts.qkRopeHeadDim, -1)

	kPass := compressedKV.View(ctx, 0, opts.kvLoraRank, compressedKV.Stride(1), compressedKV.Dim(1), compressedKV.Stride(2), compressedKV.Dim(2))
	fmt.Printf("DEBUG: kPass: %v\n", kPass.Shape())

	kRot := compressedKV.View(ctx, opts.kvLoraRank*compressedKV.Stride(0), opts.qkRopeHeadDim, compressedKV.Stride(1), compressedKV.Dim(1), compressedKV.Stride(2), compressedKV.Dim(2))
	fmt.Printf("DEBUG: kRot: %v\n", kRot.Shape())

	// kPass := compressedKV.View(ctx, 0,
	// 	compressedKV.Dim(0), compressedKV.Dim(1),
	// 	compressedKV.Dim(2), opts.kvLoraRank) // shape: [B,H,S,kvRank]
	// fmt.Printf("DEBUG: kPass: %v\n", kPass.Shape())

	// kRot := compressedKV.View(ctx, opts.kvLoraRank,
	// 	compressedKV.Dim(0), compressedKV.Dim(1),
	// 	compressedKV.Dim(2), opts.qkRopeHeadDim) // shape: [B,H,S,ropeDim]
	// fmt.Printf("DEBUG: kRot: %v\n", kRot.Shape())

	// k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
	kPass = attn.KVANorm.Forward(ctx, kPass, opts.eps)
	fmt.Printf("DEBUG: kPass: %v\n", kPass.Shape())

	kPass = attn.KVB.Forward(ctx, kPass)
	fmt.Printf("DEBUG: kPass: %v\n", kPass.Shape())

	// Calculate actual dimensions for kPass reshape
	actualKVHeadDim := kPass.Dim(0) / opts.numKVHeads // 32768 / 128 = 256
	fmt.Printf("DEBUG: reshaping kPass [%d, %d] -> [%d, %d, %d]\n", kPass.Dim(0), kPass.Dim(1), actualKVHeadDim, opts.numKVHeads, seqLength)

	kPass = kPass.Reshape(ctx, actualKVHeadDim, opts.numKVHeads, seqLength)
	fmt.Printf("DEBUG: kPass after reshape: %v\n", kPass.Shape())
	// Split kPass into key and value parts along the head dimension
	// kPass shape: [256, 128, 4] -> split into key part and value part
	// Note: opts.kqNopeHeadDim should be the key part size, opts.vHeadDim should be value part size

	// For now, let's assume we want the first part as key and second part as value
	// We need to determine the correct split dimensions based on your config
	// keyPartDim := opts.kqNopeHeadDim // This should be configured correctly
	// valuePartDim := opts.vHeadDim    // This should be configured correctly

	fmt.Printf("DEBUG: Splitting kPass [%d, %d, %d] into key[%d] and value[%d]\n",
		kPass.Dim(0), kPass.Dim(1), kPass.Dim(2), opts.kqNopeHeadDim, opts.vHeadDim)

	// Extract key part (first keyPartDim dimensions)
	kPass = kPass.View(ctx, 0, opts.kqNopeHeadDim, kPass.Stride(1), kPass.Dim(1), kPass.Stride(2), kPass.Dim(2))
	fmt.Printf("DEBUG: keyPass: %v\n", kPass.Shape())

	// Extract value part (remaining dimensions)
	value := kPass.View(ctx, opts.kqNopeHeadDim*kPass.Stride(0), opts.vHeadDim, kPass.Stride(1), kPass.Dim(1), kPass.Stride(2), kPass.Dim(2))
	fmt.Printf("DEBUG: value: %v\n", value.Shape())

	// Use keyPart as kPass for the rest of the computation
	// kPass = keyPart

	// Make kRot contiguous before reshaping (View operations create non-contiguous tensors)
	kRot = kRot.Contiguous(ctx)
	fmt.Printf("DEBUG: turn kRot contiguous\n")

	// k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
	// kRot = kRot.Reshape(ctx, opts.qkRopeHeadDim, opts.numKVHeads, seqLength)
	// kRot = kRot.Reshape(ctx, opts.qkRopeHeadDim, seqLength, batchSize)

	// kRot = kRot.Reshape(ctx, opts.qkRopeHeadDim, 1, seqLength, batchSize)
	fmt.Printf("DEBUG: opts.qkRopeHeadDim: %v\n", opts.qkRopeHeadDim)
	fmt.Printf("DEBUG: kRot.Dim(0): %v\n", kRot.Dim(0))
	kRot = kRot.Reshape(ctx, opts.qkRopeHeadDim, 1, seqLength)

	fmt.Printf("DEBUG: kRot after reshape: %v\n", kRot.Shape())
	// fmt.Printf("DEBUG: qRot check shape: %v\n", qRot.Shape())

	// interleave is just a memory optimization, so we might be able to skip
	// gpt-oss and deepseek do the same thing for this
	fmt.Printf("DEBUG: positions: %v\n", positions.Shape())

	fmt.Printf("DEBUG: qRot before rope: %v\n", qRot.Shape())
	fmt.Printf("DEBUG: positions: %v\n", positions.Shape())
	fmt.Printf("DEBUG: opts.qkRopeHeadDim: %v\n", opts.qkRopeHeadDim)
	fmt.Printf("DEBUG: opts.ropeBase: %v\n", opts.ropeBase)
	fmt.Printf("DEBUG: opts.ropeScale: %v\n", opts.ropeScale)
	fmt.Printf("DEBUG: opts.RoPEOptions(): %v\n", opts.RoPEOptions())
	// qRot = fast.RoPE(ctx, qRot, positions, opts.headDim(), opts.ropeBase, 1./opts.ropeScale, opts.RoPEOptions()...)
	qRot = fast.RoPE(ctx, qRot, positions, opts.qkRopeHeadDim, opts.ropeBase, 1./opts.ropeScale, opts.RoPEOptions()...)

	fmt.Printf("DEBUG: qRot after RoPE: %v\n", qRot.Shape())

	fmt.Printf("DEBUG: intermission\n")
	fmt.Printf("DEBUG: kRot before rope: %v\n", kRot.Shape())
	fmt.Printf("DEBUG: positions: %v\n", positions.Shape())
	fmt.Printf("DEBUG: opts.qkRopeHeadDim: %v\n", opts.qkRopeHeadDim)
	fmt.Printf("DEBUG: opts.ropeBase: %v\n", opts.ropeBase)
	fmt.Printf("DEBUG: opts.ropeScale: %v\n", opts.ropeScale)
	fmt.Printf("DEBUG: opts.RoPEOptions(): %v\n", opts.RoPEOptions())
	// kRot = fast.RoPE(ctx, kRot, positions, opts.headDim(), opts.ropeBase, 1./opts.ropeScale, opts.RoPEOptions()...)
	// kRot = fast.RoPE(ctx, kRot, positions, opts.kvLoraRank, opts.ropeBase, 1./opts.ropeScale, opts.RoPEOptions()...)
	kRot = fast.RoPE(ctx, kRot, positions, opts.qkRopeHeadDim, opts.ropeBase, 1./opts.ropeScale, opts.RoPEOptions()...)
	fmt.Printf("DEBUG: kRot after RoPE: %v\n", kRot.Shape())
	fmt.Printf("DEBUG: kPass shape: %v\n", kPass.Shape())

	kRot = kRot.Repeat(ctx, 1, kPass.Dim(1))

	// here is where we die
	fmt.Printf("DEBUG: kRot: %v\n", kRot.Shape())

	query = qPass.Concat(ctx, qRot, 0)
	fmt.Printf("DEBUG: query: %v\n", query.Shape())
	key := kPass.Concat(ctx, kRot, 0)
	fmt.Printf("DEBUG: key: %v\n", key.Shape())

	fmt.Printf("DEBUG: attnImplementation: %v\n", opts.attnImplementation)

	if opts.attnImplementation == "flash_attention_2" && opts.qkHeadDim != opts.vHeadDim {
		// value = value.Pad(ctx, ???)
		print("not implemented")
	}

	// ripped this from llama4/model.go
	// what is the format of the attention? is 1/math.Sqrt(headDim) the scale?
	attention := nn.Attention(ctx, query, key, value, 1./math.Sqrt(float64(opts.headDim())), cache)
	fmt.Printf("DEBUG: attention: %v\n", attention.Shape())

	fmt.Printf("DEBUG: attnImplementation: %v\n", opts.attnImplementation)

	if opts.attnImplementation == "flash_attention_2" && opts.qkHeadDim != opts.vHeadDim {
		// attention = attention[:, :, :, : self.vHeadDim]
		print("not implemented")
	}
	// this is for attn_output.reshape(*input_shape, -1).contiguous() but we need it for
	// attn_output.reshape(batch_size, seq_length, -1).contiguous()
	fmt.Printf("DEBUG: batchSize: %v\n", batchSize)
	fmt.Printf("DEBUG: attention.Dim(0)*attention.Dim(1): %v\n", attention.Dim(0)*attention.Dim(1))
	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), seqLength)

	fmt.Printf("DEBUG: attention: %v\n", attention.Shape())
	return attn.Output.Forward(ctx, attention).Add(ctx, residual)
}

func New(c fs.Config) (model.Model, error) {
	m := Transformer{
		TransformerBlocks: make([]TransformerBlock, 1),
		// BytePairEncoding: model.NewBytePairEncoding(
		// 	c.String("tokenizer.ggml.pretokenizer", `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`),
		// ),
	}
	return &m, nil
}

func (m *Transformer) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	return batch.Inputs, nil
}

func init() {
	model.Register("deepseek2", New)
}

// decoder layer = transformer block

// gpt oss:
// - layernorm
// - attention
// - residual + hiddenStates
// - post attention
// - mlp

// the decorder layer is the same

package deepseek3

import (
	"cmp"
	"math"
	"strings"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/fast"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type TransformerBlock struct {
	Attention *AttentionBlock
	MLP       *MLPBlock
	MoE       *MoEBlock
	// the only diff is its MLP or MoE
}

type Options struct {
	qLoraRank,
	kvLoraRank,
	qkNopeHeadDim,
	qkRopeHeadDim,
	attnImplementation,

	hiddenSize,
	numHeads,
	numKVHeads,
	keyLength,
	valueLength,
	numExperts,
	numExpertsUsed,
	originalContextLength int

	eps,
	ropeBase,
	ropeScale float32
}

func (o Options) headDim() int {
	return cmp.Or(o.keyLength, o.valueLength, o.hiddenSize/o.numHeads)
}

type AttentionBlock struct {
	Norm *nn.RMSNorm `gguf:"attn_norm"`

	// if q_lora_rank is None
	Q *nn.Linear `gguf:attn_q`
	// else
	QA *nn.Linear `gguf:attn_q_a`
	QANorm *nn.RMSNorm `gguf:attn_q_a_norm`
	QB *nn.Linear `gguf:attn_q_b`

	KVA *nn.Linear `gguf:attn_kv_a_mqa`
	KVANorm *nn.RMSNorm `gguf:attn_kv_a_norm`
	KVB *nn.Linear `gguf:attn_kv_b`

	Output *nn.Linear `gguf:"attn_out,alt:attn_output"` // where does attn_out come from?
}

// func (attn *AttentionBlock) Forward(ctx ml.Context, ...) {
func (attn *AttentionBlock) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	sequenceLength, batchSize := hiddenStates.Dim(1), hiddenStates.Dim(2)
	// idk how to set up the beginning
	if opts.qLoraRank == nil {
		query = attn.Q.Forward(ctx, hiddenStates)
	} else {
		// q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
		query = attn.QA.Forward(ctx, hiddenStates)
		query = attn.QANorm.Forward(ctx, query, opts.eps)
		query = attn.QB.Forward(ctx, query)
	}
	// q_states = q_states.view(query_shape).transpose(1, 2)
	query = query.Reshape(ctx, opts.headDim(), opts.numHeads, batchSize)
	qPass, qRot := query.Split(ctx, opts.qkNopeHeadDim, opts.qkRopeHeadDim, -1)

	compressedKV := attn.KVA.Forward(ctx, hiddenStates)
	kPass, kRot := compressedKV.Split(ctx, opts.kvLoraRank, opts.qkRopeHeadDim, -1)

	// k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
	kPass = attn.KVANorm.Forward(ctx, kPass, opts.eps)
	kPass = attn.KVB.Forward(ctx, kPass)
	kPass = kPass.Reshape(ctx, opts.headDim, opts.numKVHeads, batchSize)
	kPass, value := kPass.Split(ctx, opts.kqNopeHeadDim, opts.vHeadDim, -1)

	kRot = kRot.Reshape(ctx, batchSize, 1, seqLength, opts.qkRopeHeadDim)

	// interleave is just a memory optimization, so we might be able to skip
	// gpt-oss and deepseek do the same thing for this
	qRot =fast.RoPE(ctx, query, positions, opts.headDim(), opts.ropeBase, 1./opts.ropeScale, opts.RoPEOptions()...)
	kRot = fast.RoPE(ctx, key, positions, opts.headDim(), opts.ropeBase, 1./opts.ropeScale, opts.RoPEOptions()...)
	// }

	kRot = Repeat(ctx, kRot, 1, 1, opts.numHeads, 1)

	query = qPass.Concat(ctx, qRot, 0)
	key := kPass.Concat(ctx, kRot, 0)

	if opts.attnImplementation == "flash_attention_2" and opts.qkHeadDim != opts.vHeadDim {
		// value = value.Pad(ctx, ???)
		print("not implemented")
	}

	// ripped this from llama4/model.go
	// what is the format of the attention? is 1/math.Sqrt(headDim) the scale?
	attention := nn.Attention(ctx, query, key, value, 1./math.Sqrt(float64(headDim)), cache)

	if opts.attnImplementation == "flash_attention_2" and opts.qkHeadDim != opts.vHeadDim {
		// attention = attention[:, :, :, : self.vHeadDim]
		print("not implemented")
	}
	// this is for attn_output.reshape(*input_shape, -1).contiguous() but we need it for 
	// attn_output.reshape(batch_size, seq_length, -1).contiguous()
	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), batchSize)
	return attn.Output.Forward(ctx, attention).Add(ctx, residual)
}


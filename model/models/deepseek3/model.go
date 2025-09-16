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

	// TokenEmbedding    *nn.Embedding      `gguf:"token_embd"`
	TransformerBlocks []TransformerBlock `gguf:"blk"`
	// OutputNorm        *nn.RMSNorm        `gguf:"output_norm"`
	// Output            *nn.Linear         `gguf:"output,alt:token_embd"`

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	*Options
}

type TransformerBlock struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	Attention *AttentionBlock

	MLPNorm *nn.RMSNorm `gguf:"ffn_norm"`
	MLP
	// MoEBlock *MoEBlock
	// the only diff is its MLP or MoE
}

func (t *TransformerBlock) Forward(ctx ml.Context, hiddenStates, positions, outputs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	hiddenStates = t.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = t.Attention.Forward(ctx, hiddenStates, positions, cache, opts)
	hiddenStates = t.MLPNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = t.MLP.Forward(ctx, hiddenStates, opts)
	return hiddenStates
}

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

// TODO:
// - double check the annotations for gguf
// - make sure the intermediate size is correct

// should we add a norm to the mlp block
// type MLPBlock struct {
// 	Gate *nn.Linear `gguf:"ffn_gate"`
// 	Up   *nn.Linear `gguf:"ffn_up"`
// 	Down *nn.Linear `gguf:"ffn_down"`
// }

// func (mlp *MLPBlock) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
// 	hiddenState = mlp.Gate.Forward(ctx, hiddenState).SILU(ctx).Mul(ctx, mlp.Up.Forward(ctx, hiddenState))
// 	return mlp.Down.Forward(ctx, hiddenState)
// }

// // nn.ModuleList([DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.n_routed_experts)])

type SharedExpert struct {
	Gate *nn.Linear `gguf:"ffn_gate_shexp"`
	Up   *nn.Linear `gguf:"ffn_up_shexp"`
	Down *nn.Linear `gguf:"ffn_down_shexp"`
}

func (se *SharedExpert) Forward(ctx ml.Context, hiddenStates ml.Tensor) ml.Tensor {
	hiddenStates = se.Gate.Forward(ctx, hiddenStates).SILU(ctx).Mul(ctx, se.Up.Forward(ctx, hiddenStates))
	return se.Down.Forward(ctx, hiddenStates)
}

type MoEBlock struct {
	// Router *Router
	Router *nn.Linear `gguf:"ffn_gate_inp"`
	Gate   *nn.Linear `gguf:"ffn_gate_exps"`
	Up     *nn.Linear `gguf:"ffn_up_exps"`
	Down   *nn.Linear `gguf:"ffn_down_exps"`
	// Experts *Experts   `gguf:"blk"` // since this is nn.ModuleList, we need a slice?
	SharedExpert *SharedExpert
	ExpProbsBias ml.Tensor `gguf:"exp_probs_b.bias,alt:exp_probs_b"`
}

func (moe *MoEBlock) Moe(ctx ml.Context, hiddenStates ml.Tensor, topKIndices ml.Tensor, topKWeights ml.Tensor, opts *Options) ml.Tensor {
	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), 1, hiddenStates.Dim(1))
	fmt.Printf("DEBUG: hiddenStates: %v\n", hiddenStates.Shape())

	upStates := moe.Up.Weight.MulmatID(ctx, hiddenStates, topKIndices)
	fmt.Printf("DEBUG: upStates: %v\n", upStates.Shape())
	hiddenStates = moe.Gate.Weight.MulmatID(ctx, hiddenStates, topKIndices)
	fmt.Printf("DEBUG: hiddenStates: %v\n", hiddenStates.Shape())
	hiddenStates = hiddenStates.SILU(ctx)
	fmt.Printf("DEBUG: hiddenStates: %v\n", hiddenStates.Shape())
	hiddenStates = hiddenStates.Mul(ctx, upStates)
	fmt.Printf("DEBUG: hiddenStates: %v\n", hiddenStates.Shape())
	experts := moe.Down.Weight.MulmatID(ctx, hiddenStates, topKIndices)
	fmt.Printf("DEBUG: experts: %v\n", experts.Shape())
	experts = experts.Mul(ctx, topKWeights)
	fmt.Printf("DEBUG: experts: %v\n", experts.Shape())
	nextStates := experts.View(ctx, 0, experts.Dim(0), experts.Stride(2), experts.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		nextStates = nextStates.Add(ctx, experts.View(ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2)))
		fmt.Printf("DEBUG: nextStates: %v\n", nextStates.Shape())
	}
	fmt.Printf("DEBUG: nextStates: %v\n", nextStates.Shape())
	return nextStates
}

func (moe *MoEBlock) getTopKIndices(ctx ml.Context, scores ml.Tensor, opts *Options) ml.Tensor {
	scores = scores.Add(ctx, moe.ExpProbsBias)
	fmt.Printf("DEBUG: scores: %v\n", scores.Shape())
	topKIndices := scores.TopK(ctx, opts.numExpertsUsed)
	fmt.Printf("DEBUG: topKIndices: %v\n", topKIndices.Shape())
	return topKIndices
}

// sparse block = Moe block
func (moe *MoEBlock) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	hiddenDim, sequenceLength, batchSize := hiddenStates.Dim(0), hiddenStates.Dim(1), hiddenStates.Dim(2)
	fmt.Printf("DEBUG: hiddenDim: %d, sequenceLength: %d, batchSize: %d\n", hiddenDim, sequenceLength, batchSize)
	residuals := hiddenStates
	fmt.Printf("DEBUG: residuals: %v\n", residuals.Shape())

	// topKIndices, topKWeights := moe.Router.Forward(ctx, hiddenStates, opts)

	fmt.Printf("DEBUG: hello, we're in the ROUTER!\n")
	// routerLogits := r.Gate.Forward(ctx, hiddenStates)
	routerLogits := moe.Router.Forward(ctx, hiddenStates)

	fmt.Printf("DEBUG: routerLogits: %v\n", routerLogits.Shape())
	scores := routerLogits.Sigmoid(ctx)

	//
	topKIndices := moe.getTopKIndices(ctx, scores, opts)
	//

	fmt.Printf("DEBUG: scores ORIG shape ***: %v\n", scores.Shape())

	topKWeights := scores.Reshape(ctx, 1, opts.numExperts, hiddenStates.Dim(1)).Rows(ctx, topKIndices)
	fmt.Printf("DEBUG: topKWeights: %v\n", topKWeights.Shape())

	fmt.Printf("DEBUG: topKWeights shape ***: %v\n", topKWeights.Shape())

	// so here, topKWeights is not the same, however that is because its not in sorted order?
	// I believe... check with Mike
	// return topKWeights

	if opts.normTopKProb {
		topKWeights = topKWeights.Reshape(ctx, opts.numExpertsUsed, hiddenStates.Dim(1))
		fmt.Printf("DEBUG: topKWeights (1): %v\n", topKWeights.Shape())
		topKWeights = topKWeights.Div(ctx, topKWeights.SumRows(ctx))
		fmt.Printf("DEBUG: topKWeights (2): %v\n", topKWeights.Shape())
		topKWeights = topKWeights.Reshape(ctx, 1, opts.numExpertsUsed, hiddenStates.Dim(1))
		fmt.Printf("DEBUG: topKWeights (3): %v\n", topKWeights.Shape())
	}

	fmt.Printf("DEBUG: topKIndices: %v\n", topKIndices.Shape())
	fmt.Printf("DEBUG: topKWeights: %v\n", topKWeights.Shape())

	// topk_weights = topk_weights * self.routed_scaling_factor
	topKWeights = topKWeights.Scale(ctx, float64(opts.routedScalingFactor))
	fmt.Printf("DEBUG: topKWeights (4): %v\n", topKWeights.Shape())

	// hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
	// hiddenStates = hiddenStates.Reshape(ctx, hiddenDim, sequenceLength*batchSize)
	// fmt.Printf("DEBUG: hiddenStates: %v\n", hiddenStates.Shape())
	// MOE stuff
	hiddenStates = moe.Moe(ctx, hiddenStates, topKIndices, topKWeights, opts)

	fmt.Printf("DEBUG: post MOE ++++++++: %v\n", hiddenStates.Shape())

	// check here

	return hiddenStates

	sharedExpertResult := moe.SharedExpert.Forward(ctx, residuals)
	fmt.Printf("DEBUG: sharedExpertResult: %v\n", sharedExpertResult.Shape())

	hiddenStates = hiddenStates.Add(ctx, sharedExpertResult)
	return hiddenStates
}

// -------------------------------------------------------------------------------------------------------------------
// tested

type MLP interface {
	Forward(ml.Context, ml.Tensor, *Options) ml.Tensor
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
	fmt.Printf("DEBUG: the total number of layers: %v", c.Uint("block_count"))
	transformerBlocks := make([]TransformerBlock, c.Uint("block_count"))

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
			ropeScale:      c.Float("rope.freq_scale", 1),
			numExperts:     int(c.Uint("expert_count")),
			numExpertsUsed: int(c.Uint("expert_used_count")),
			normTopKProb:   c.Bool("norm_top_k_prob", true),

			qLoraRank:          &qLoraRankVal,
			kvLoraRank:          int(c.Uint("kv_lora_rank")),
			qkHeadDim:           int(c.Uint("attention.key_length")),
			vHeadDim:    		int(c.Uint("attention.value_length")),
			qkRopeHeadDim:       int(c.Uint("rope.dimension_count")),
			qkNopeHeadDim:       int(c.Uint("attention.key_length")) - int(c.Uint("rope.dimension_count")),
			kqNopeHeadDim:       int(c.Uint("attention.key_length")) - int(c.Uint("rope.dimension_count")),

			routedScalingFactor: c.Float("routed_scaling_factor"),
		},
	}
	m.Cache = kvcache.NewCausalCache(nil) // TODO: add correct cache

	return &m, nil
}

// residual = hidden_states
// hidden_states = self.input_layernorm(hidden_states)
// # Self Attention
// hidden_states, _ = self.self_attn(
// 	hidden_states=hidden_states,
// 	attention_mask=attention_mask,
// 	position_ids=position_ids,
// 	past_key_values=past_key_values,
// 	use_cache=use_cache,
// 	cache_position=cache_position,
// 	position_embeddings=position_embeddings,
// 	**kwargs,
// )
// hidden_states = residual + hidden_states

// # Fully Connected
// residual = hidden_states
// hidden_states = self.post_attention_layernorm(hidden_states)
// hidden_states = self.mlp(hidden_states)
// hidden_states = residual + hidden_states
// return hidden_states

// func (m *Transformer) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
// 	return batch.Inputs, nil

// 	hiddenStates := m.InputLayerNorm.Forward(ctx, batch.Inputs)

// }

func (m *Transformer) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromIntSlice(batch.Positions, len(batch.Positions))

	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	for i, layer := range m.TransformerBlocks {
		// m.Cache.SetLayer(i)

		var outputs ml.Tensor
		if i == len(m.TransformerBlocks)-1 {
			outputs = ctx.Input().FromIntSlice(batch.Outputs, len(batch.Outputs))
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, outputs, m.Cache, m.Options)
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func init() {
	model.Register("deepseek2", New)
}

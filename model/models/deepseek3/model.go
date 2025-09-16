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
	// model.BytePairEncoding

	// TokenEmbedding    *nn.Embedding      `gguf:"token_embd"`
	TransformerBlocks []TransformerBlock `gguf:"blk"`
	// OutputNorm        *nn.RMSNorm        `gguf:"output_norm"`
	// Output            *nn.Linear         `gguf:"output,alt:token_embd"`

	Options
}

type TransformerBlock struct {
	Attention *AttentionBlock
	MLP
	// MoEBlock *MoEBlock
	// the only diff is its MLP or MoE
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

// type Experts struct {
// 	Gate *nn.Linear `gguf:"ffn_gate_exps"`
// 	Up   *nn.Linear `gguf:"ffn_up_exps"`
// 	Down *nn.Linear `gguf:"ffn_down_exps"`
// }

// func (e *Experts) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
// 	// hiddenDim, sequenceLength, batchSize := hiddenStates.Dim(0), hiddenStates.Dim(1), hiddenStates.Dim(2)
// 	// hiddenStates = hiddenStates.Reshape(ctx, hiddenDim, sequenceLength*batchSize)
// 	// routerLogits := mlp.Router.Forward(ctx, hiddenStates)

// 	// routingWeights := routerLogits.Softmax(ctx)
// 	// selectedExperts := routingWeights.TopK(ctx, opts.numExpertsUsed)
// 	// routingWeights = routingWeights.Reshape(ctx, 1, opts.numExperts, hiddenStates.Dim(1)).Rows(ctx, selectedExperts)
// 	// if opts.normTopKProb {
// 	// 	routingWeights = routingWeights.Reshape(ctx, opts.numExpertsUsed, hiddenStates.Dim(1))
// 	// 	routingWeights = routingWeights.Div(ctx, routingWeights.SumRows(ctx))
// 	// 	routingWeights = routingWeights.Reshape(ctx, 1, opts.numExpertsUsed, hiddenStates.Dim(1))
// 	// }

// 	// where we need to figureo ut how to implement the router

// 	// --------------------------------------------------------------------------------------

// 	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), 1, hiddenStates.Dim(1))

// 	upStates := mlp.Up.Weight.MulmatID(ctx, hiddenStates, selectedExperts)

// 	hiddenStates = mlp.Gate.Weight.MulmatID(ctx, hiddenStates, selectedExperts)
// 	hiddenStates = hiddenStates.SILU(ctx)
// 	hiddenStates = hiddenStates.Mul(ctx, upStates)

// 	experts := mlp.Down.Weight.MulmatID(ctx, hiddenStates, selectedExperts)
// 	experts = experts.Mul(ctx, routingWeights)

// 	nextStates := experts.View(ctx, 0, experts.Dim(0), experts.Stride(2), experts.Dim(2))
// 	for i := 1; i < opts.numExpertsUsed; i++ {
// 		nextStates = nextStates.Add(ctx, experts.View(ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2)))
// 	}

// 	return nextStates
// }

// pass in the topk weights
// func (e *Experts) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
// 	// finalHiddenStates := ctx.Zeros(topkWeights.DType(), hiddenState.Shape()...)
// 	// selectedExperts = passed in?

// 	// cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens)

// 	// eLen = nExperts
// 	nEmbedding, nTokens := hiddenState.Dim(0), hiddenState.Dim(1)
// 	nExperts := e.Len()
// 	weights := prod.Reshape(ctx, 1, nExperts, nTokens).Rows(ctx, selectedExperts)

// 	// do we need softmax, normalization, or scale

// 	hiddenState = hiddenState.Reshape(ctx, nEmbedding, 1, nTokens)

// 	// ggml_tensor * up = build_lora_mm_id(up_exps, cur, selected_experts)
// 	upExps := e.Experts[0].Up.Weight // can we implement it with forward?
// 	up := upExps.MulmatID(ctx, hiddenState, selectedExperts)
// 	gateExps := e.Experts[0].Gate.Weight
// 	gate := gateExps.MulmatID(ctx, hiddenState, selectedExperts)

// 	gate = gate.SILU(ctx).Mul(ctx, up)

// 	downExps := e.Experts[0].Down.Weight

// 	experts := downExps.MulmatID(ctx, hiddenState, selectedExperts) // bmm
// 	experts = experts.Mul(ctx, weights)

// 	nExpertsUsed := experts.Dim(1)

// 	// ordering the views before the adds
// 	hiddenExperts := make([]ml.Tensor, nExpertsUsed)
// 	for i := 1; i < nExpertsUsed; i++ {
// 		hiddenExperts[i] = experts.View(ctx)
// 	}

// }

// type Router struct {
// 	Gate *nn.Linear `gguf:"ffn_gate_inp"` // nn.Parameter vs nn.Linear
// }

// note that TopK returns us the values

// topKIndices implements the MoE routing logic for DeepSeek3
// This is a simplified version for testing purposes
// func topKIndices(ctx ml.Context, scores ml.Tensor, bias ml.Tensor, nGroups, topKGroup, topK int) ml.Tensor {
// 	fmt.Printf("DEBUG: scores shape: %v\n", scores.Shape())
// 	fmt.Printf("DEBUG: bias shape: %v\n", bias.Shape())
// 	fmt.Printf("DEBUG: nGroups: %d\n", nGroups)
// 	fmt.Printf("DEBUG: topKGroup: %d\n", topKGroup)
// 	fmt.Printf("DEBUG: topK: %d\n", topK)

// 	nExperts, nTokens := scores.Dim(0), scores.Dim(1)
// 	expertsPerGroup := nExperts / nGroups
// 	fmt.Printf("DEBUG: expertsPerGroup: %d\n", expertsPerGroup)

// 	scoresForChoice := scores.Add(ctx, bias)
// 	fmt.Printf("DEBUG: scoresForChoice shape: %v\n", scoresForChoice.Shape())
// 	scoresG := scoresForChoice.Reshape(ctx, nGroups, expertsPerGroup, nTokens)
// 	fmt.Printf("DEBUG: scoresG shape: %v\n", scoresG.Shape())

// 	// ----

// 	// top2_vals, _ = scores_g.topk(k=2, dim=1)

// 	// To do TopK on dimension 1, we need to permute or reshape first

// 	// nGroups, expertsPerGroup, nTokens --> expertsPerGroup, nGroups, nTokens

// 	// nGroups, expertsPerGroup, nTokens (4, 2, 10) --> expertsPerGroup, nGroups, nTokens (2, 4, 10)

// 	// bruh this whole thing is topK indices
// 	scoresGTransposed := scoresG.Reshape(ctx, nGroups, expertsPerGroup, nTokens, 1) // in prep for permute
// 	fmt.Printf("DEBUG: scoresGTransposed Reshape shape: %v\n", scoresGTransposed.Shape())
// 	// 4, 2, 10, 1
// 	scoresGTransposed = scoresGTransposed.Permute(ctx, 1, 0, 2, 3)
// 	// 2, 4, 10, 1
// 	fmt.Printf("DEBUG: scoresGTransposed Permute shape: %v\n", scoresGTransposed.Shape())

// 	top2Indices := scoresGTransposed.TopK(ctx, 2)
// 	fmt.Printf("DEBUG: top2Indices Ktop shape: %v\n", top2Indices.Shape())

// 	top2Indices = top2Indices.Permute(ctx, 1, 0, 2, 3) // 4, 2, 10, 1
// 	fmt.Printf("DEBUG: top2Indices unPermute shape: %v\n", top2Indices.Shape())
// 	top2Indices = top2Indices.Contiguous(ctx)
// 	top2Indices = top2Indices.Reshape(ctx, nGroups, expertsPerGroup, nTokens)
// 	fmt.Printf("DEBUG: top2Indices unReshape shape: %v\n", top2Indices.Shape())

// 	// topK values
// 	fmt.Printf("DEBUG: **********************:\n")
// 	fmt.Printf("DEBUG: top2Indices shape: %v\n", top2Indices.Shape())
// 	fmt.Printf("DEBUG: scoresG shape: %v\n", scoresG.Shape())

// 	scoresFlat := scoresG.Reshape(ctx, 1, expertsPerGroup, nGroups*nTokens)
// 	idxFlat := top2Indices.Reshape(ctx, nGroups, topK, nTokens, 1)
// 	idxFlat = idxFlat.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
// 	idxFlat = idxFlat.Reshape(ctx, topK, nGroups*nTokens) // int32
// 	fmt.Printf("DEBUG: idxFlat shape: %v\n", idxFlat.Shape())

// 	valsFlat := scoresFlat.Rows(ctx, idxFlat)
// 	fmt.Printf("DEBUG: valsFlat shape: %v\n", valsFlat.Shape())

// 	top2Vals := valsFlat.Reshape(ctx, topK, nGroups, nTokens).Permute(ctx, 1, 0, 2, 3)
// 	fmt.Printf("DEBUG: top2Vals shape: %v\n", top2Vals.Shape())

// 	// we should check here to make sure everything is correct up to this point **!!**

// 	// top2Vals := scoresG.Rows(ctx, top2Indices)
// 	// fmt.Printf("DEBUG: top2Vals shape: %v\n", top2Vals.Shape())
// 	top2Vals = top2Vals.Contiguous(ctx)
// 	top2Vals = top2Vals.Reshape(ctx, top2Vals.Dim(0), top2Vals.Dim(1), top2Vals.Dim(2), 1)
// 	fmt.Printf("DEBUG: top2Vals Reshape shape: %v\n", top2Vals.Shape())
// 	top2Vals = top2Vals.Permute(ctx, 1, 0, 2, 3) // 2, 4, 10, 1
// 	fmt.Printf("DEBUG: top2Vals Permute shape: %v\n", top2Vals.Shape())

// 	groupScores := top2Vals.SumRows(ctx) // 1, 4, 10, 1
// 	fmt.Printf("DEBUG: groupScores shape: %v\n", groupScores.Shape())
// 	groupScores = groupScores.Reshape(ctx, 4, 10) // 4, 10
// 	fmt.Printf("DEBUG: groupScores Reshape shape: %v\n", groupScores.Shape())

// 	fmt.Printf("DEBUG: **********************:\n")

// 	// fmt.Printf("DEBUG: groupScores shape: %v\n", groupScores.Shape())
// 	groupIdx := groupScores.TopK(ctx, topKGroup)
// 	fmt.Printf("DEBUG: groupIdx shape: %v\n", groupIdx.Shape())

// 	return groupIdx

// 	// group idx generates the

// 	// fmt.Printf("DEBUG: groupIdx shape: %v\n", groupIdx.Shape())

// 	// // baseIdx := groupIdx.Scale(ctx, float64(expertsPerGroup))
// 	// // fmt.Printf("DEBUG: baseIdx shape: %v\n", baseIdx.Shape())
// 	// // all this to create eLocal
// 	// baseIdx := groupIdx.
// 	// 	Scale(ctx, float64(expertsPerGroup)). // group_id * expertsPerGroup
// 	// 	Reshape(ctx, 1, topKGroup, 1, nTokens)
// 	// fmt.Printf("DEBUG: baseIdx shape: %v\n", baseIdx.Shape())

// 	// // eLocal seed: [1, 1, E, 1]
// 	// eLocal := ctx.Arange(0, float32(expertsPerGroup), 1, ml.DTypeF32).
// 	// 	Reshape(ctx, 1, 1, expertsPerGroup, 1)
// 	// fmt.Printf("DEBUG: eLocal shape: %v\n", eLocal.Shape())

// 	// // --- Expand to a common shape [1, K, E, T] ---

// 	// // baseIdx [1, K, 1, T] -> repeat along experts axis (E)

// 	// baseIdx2d := baseIdx.Reshape(ctx, topKGroup, nTokens) // [K, T]
// 	// fmt.Printf("DEBUG: baseIdx2d shape: %v\n", baseIdx2d.Shape())
// 	// baseIdxRep := baseIdx2d.Repeat(ctx, 1, expertsPerGroup)
// 	// fmt.Printf("DEBUG: baseIdxRep shape: %v\n", baseIdxRep.Shape())
// 	// baseIdxExpanded := baseIdxRep.Reshape(ctx, 1, topKGroup, expertsPerGroup, nTokens)
// 	// fmt.Printf("DEBUG: baseIdxExpanded shape: %v\n", baseIdxExpanded.Shape())

// 	// // eLocal [1, 1, E, 1] -> repeat along group-K and tokens-T
// 	// // eLocal2d := eLocal.Reshape(ctx, expertsPerGroup, 1) // [E, 1]
// 	// // fmt.Printf("DEBUG: eLocal2d shape: %v\n", eLocal2d.Shape())
// 	// // eLocalRep := eLocal2d.Repeat(ctx, topKGroup, nTokens) // [E*K, T]
// 	// // fmt.Printf("DEBUG: eLocalRep shape: %v\n", eLocalRep.Shape())
// 	// // eLocalExpanded := eLocalRep.Reshape(ctx, 1, topKGroup, expertsPerGroup, nTokens)
// 	// // fmt.Printf("DEBUG: eLocalExpanded shape: %v\n", eLocalExpanded.Shape())
// 	// eLocal = eLocal.Reshape(ctx, 1, 1, expertsPerGroup, 1) // [1, 1, E, 1]
// 	// fmt.Printf("DEBUG: eLocal shape: %v\n", eLocal.Shape())
// 	// // eLocalRep := eLocal.Repeat(ctx, 1, topKGroup)
// 	// // fmt.Printf("DEBUG: eLocalRep shape: %v\n", eLocalRep.Shape())
// 	// // eLocalExpanded := eLocalRep.Reshape(ctx, 1, topKGroup, expertsPerGroup, 1)
// 	// // fmt.Printf("DEBUG: eLocalExpanded shape: %v\n", eLocalExpanded.Shape())

// 	// // allowed := baseIdx.Add(ctx, eLocal)
// 	// // allowed := eLocal.Add(ctx, baseIdx)
// 	// allowed := eLocal.Add(ctx, baseIdxExpanded)

// 	// // allowed := baseIdxExpanded.Add(ctx, eLocalExpanded)
// 	// fmt.Printf("DEBUG: allowed shape: %v\n", allowed.Shape())
// 	// allowedFlat := allowed.Reshape(ctx, nExperts*topKGroup, nTokens)
// 	// fmt.Printf("DEBUG: allowedFlat shape: %v\n", allowedFlat.Shape())
// 	// allowedScores := scoresForChoice.Rows(ctx, allowedFlat) // might need to reshape
// 	// fmt.Printf("DEBUG: allowedScores shape: %v\n", allowedScores.Shape())

// 	// fullMasked := ctx.Zeros(scoresForChoice.DType(), scoresForChoice.Shape()...)
// 	// fmt.Printf("DEBUG: fullMasked shape: %v\n", fullMasked.Shape())
// 	// // 2. Scatter allowed_scores into full_masked at allowed_flat positions
// 	// allowedScoresData := allowedScores.Floats()
// 	// allowedFlatData := allowedFlat.Floats()

// 	// for i := 0; i < len(allowedFlatData); i++ {
// 	// 	index := int(allowedFlatData[i])
// 	// 	value := allowedScoresData[i]
// 	// 	valueTensor := ctx.FromFloatSlice([]float32{value}, 1)
// 	// 	fullMasked = fullMasked.Set(ctx, valueTensor, index)
// 	// }

// 	// topKIndices := fullMasked.TopK(ctx, topK)
// 	// fmt.Printf("DEBUG: topKIndices shape: %v\n", topKIndices.Shape())
// 	// return topKIndices // do we need to permutate

// }

// func (r *Router) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
// }

// type Router struct {
// 	Gate *nn.Linear `gguf:"ffn_gate_inp"`
// }

// func (r *Router) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) (ml.Tensor, ml.Tensor) {
// 	// not sure how to do this view
// 	// hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), 1, hiddenStates.Dim(1))
// 	// fmt.Printf("DEBUG: hiddenStates: %v\n", hiddenStates.Shape())
// 	// so the logits are even derived the same way as in the qwen3moe model
// 	fmt.Printf("DEBUG: hello, we're in the ROUTER!\n")
// 	routerLogits := r.Gate.Forward(ctx, hiddenStates)
// 	fmt.Printf("DEBUG: routerLogits: %v\n", routerLogits.Shape())
// 	scores := routerLogits.Sigmoid(ctx)
// 	fmt.Printf("DEBUG: scores: %v\n", scores.Shape())
// 	topKIndices := scores.TopK(ctx, opts.numExpertsUsed)
// 	fmt.Printf("DEBUG: topKIndices: %v\n", topKIndices.Shape())
// 	topKWeights := scores.Reshape(ctx, 1, opts.numExperts, hiddenStates.Dim(1)).Rows(ctx, topKIndices)
// 	fmt.Printf("DEBUG: topKWeights: %v\n", topKWeights.Shape())

// 	// if self.norm_topK_prob
// 	if opts.normTopKProb {
// 		topKWeights = topKWeights.Reshape(ctx, opts.numExpertsUsed, hiddenStates.Dim(1))
// 		fmt.Printf("DEBUG: topKWeights: %v\n", topKWeights.Shape())
// 		topKWeights = topKWeights.Div(ctx, topKWeights.SumRows(ctx))
// 		fmt.Printf("DEBUG: topKWeights: %v\n", topKWeights.Shape())
// 		topKWeights = topKWeights.Reshape(ctx, 1, opts.numExpertsUsed, hiddenStates.Dim(1))
// 		fmt.Printf("DEBUG: topKWeights: %v\n", topKWeights.Shape())
// 	}
// 	return topKIndices, topKWeights
// }

// type Experts struct {
// 	Gate *nn.Linear `gguf:"ffn_gate_exps"`
// 	Up   *nn.Linear `gguf:"ffn_up_exps"`
// 	Down *nn.Linear `gguf:"ffn_down_exps"`
// }

// func (e *Experts) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
// 	// hiddenDim, sequenceLength, batchSize := hiddenStates.Dim(0), hiddenStates.Dim(1), hiddenStates.Dim(2)
// 	// hiddenStates = hiddenStates.Reshape(ctx, hiddenDim, sequenceLength*batchSize)
// 	// routerLogits := mlp.Router.Forward(ctx, hiddenStates)

// // routingWeights := routerLogits.Softmax(ctx)
// // selectedExperts := routingWeights.TopK(ctx, opts.numExpertsUsed)
// // routingWeights = routingWeights.Reshape(ctx, 1, opts.numExperts, hiddenStates.Dim(1)).Rows(ctx, selectedExperts)
// // if opts.normTopKProb {
// // 	routingWeights = routingWeights.Reshape(ctx, opts.numExpertsUsed, hiddenStates.Dim(1))
// // 	routingWeights = routingWeights.Div(ctx, routingWeights.SumRows(ctx))
// // 	routingWeights = routingWeights.Reshape(ctx, 1, opts.numExpertsUsed, hiddenStates.Dim(1))
// // }

// nn.ModuleList([DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.n_routed_experts)])

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
	transformerBlocks := make([]TransformerBlock, 4)

	firstDenseLayerIndex := int(c.Uint("leading_dense_block_count")) // or whatever key your gguf uses
	fmt.Printf("first dense: %v", firstDenseLayerIndex)
	for i := range transformerBlocks {
		if i < firstDenseLayerIndex {
			transformerBlocks[i].MLP = &MLPBlock{} // gguf tags on MLPBlock fields
		} else {
			transformerBlocks[i].MLP = &MoEBlock{} // gguf tags on Router/Experts fields
		}
	}
	m := Transformer{
		TransformerBlocks: transformerBlocks,
		// BytePairEncoding: model.NewBytePairEncoding(
		// 	c.String("tokenizer.ggml.pretokenizer", `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`),
		// ),
	}
	m.Cache = kvcache.NewCausalCache(nil)

	return &m, nil
}

func (m *Transformer) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	return batch.Inputs, nil
}

func init() {
	model.Register("deepseek2", New)
}

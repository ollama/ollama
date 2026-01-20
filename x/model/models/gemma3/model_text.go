//go:build mlx

package gemma3

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/x/kvcache"
	"github.com/ollama/ollama/x/ml"
	"github.com/ollama/ollama/x/ml/nn"
	"github.com/ollama/ollama/x/model/input"
)

type TextConfig struct {
	hiddenSize, numHeads, numKVHeads int
	attnKeyLen                       int
	eps, ropeScale                   float32
	ropeLocalBase, ropeGlobalBase    float32
	largeModelScaling                bool
}

type TextModel struct {
	TokenEmbedding *nn.Embedding `gguf:"embed_tokens"`
	Layers         []TextLayer   `gguf:"layers"`
	OutputNorm     *nn.RMSNorm   `gguf:"norm"`
	Output         *nn.Linear    `gguf:"embed_tokens"`

	*TextConfig
}

const (
	gemmaGlobalCacheCount = 6
	gemma27BLayerCount    = 62
)

// const (
// 	cacheTypeSWA = iota
// 	cacheTypeCausal
// )

func newTextModel(c fs.Config) *TextModel {
	numBlocks := int(c.Uint("block_count"))

	m := TextModel{
		Layers: make([]TextLayer, numBlocks),
		TextConfig: &TextConfig{
			hiddenSize:     int(c.Uint("embedding_length")),                    // 2560 -- config.json: text_config.hidden_size
			numHeads:       int(c.Uint("attention.head_count")),                // 8 -- hard coded in python implementation for the model, 4 in some places, then overridden as 8
			numKVHeads:     int(c.Uint("attention.head_count_kv")),             // 4 -- same as above
			attnKeyLen:     int(c.Uint("attention.key_length", 256)),           //256 -- rope settings, hardcoded in model definition python
			eps:            c.Float("attention.layer_norm_rms_epsilon", 1e-06), // 1e-06 - hardcoded in model definition python
			ropeLocalBase:  c.Float("rope.local.freq_base", 10000.0),           // 10000 - hardcoded in python
			ropeGlobalBase: c.Float("rope.global.freq_base", 1000000.0),        // 1e+06 - hardcoded in python
			ropeScale:      1,                                                  // 1 - default is 1, implied in python code
			// vocabSize:      vocabSize,                                          // 262144
			// attnValLen:     int(c.Uint("attention.value_length", 256)),         //256
			// NOTE: the rope.scaling.factor is set incorrectly in the official QAT weights
			//       (8 instead of 1)
			// ropeScale:      c.Float("rope.scaling.factor", 1.0),
		},
	}
	if numBlocks == gemma27BLayerCount {
		m.largeModelScaling = true
	}

	return &m
}

type TextSelfAttention struct {
	Query     *nn.Linear  `gguf:"q_proj"`
	QueryNorm *nn.RMSNorm `gguf:"q_norm"`
	Key       *nn.Linear  `gguf:"k_proj"`
	KeyNorm   *nn.RMSNorm `gguf:"k_norm"`
	Value     *nn.Linear  `gguf:"v_proj"`
	Output    *nn.Linear  `gguf:"o_proj"`
}

func (sa *TextSelfAttention) Forward(ctx ml.Context, layer int, hiddenState ml.Tensor, offset int, cache kvcache.Cache, opts *TextConfig) ml.Tensor {
	B := hiddenState.Dim(0)
	L := hiddenState.Dim(1)
	ropeBase := opts.ropeLocalBase
	if (layer+1)%gemmaGlobalCacheCount == 0 {
		ropeBase = opts.ropeGlobalBase
	}

	q := sa.Query.Forward(ctx, hiddenState)
	k := sa.Key.Forward(ctx, hiddenState)
	v := sa.Value.Forward(ctx, hiddenState)
	q = q.Reshape(ctx, B, L, opts.numHeads, -1).Transpose(ctx, 0, 2, 1, 3)
	k = k.Reshape(ctx, B, L, opts.numKVHeads, -1).Transpose(ctx, 0, 2, 1, 3)
	v = v.Reshape(ctx, B, L, opts.numKVHeads, -1).Transpose(ctx, 0, 2, 1, 3).Contiguous(ctx, false)
	q = sa.QueryNorm.Forward(ctx, q, opts.eps)
	k = sa.KeyNorm.Forward(ctx, k, opts.eps)
	traditional := false
	q = q.RoPE(ctx, opts.attnKeyLen, traditional, opts.ropeScale, offset, ml.WithRoPEBase(ropeBase))
	k = k.RoPE(ctx, opts.attnKeyLen, traditional, opts.ropeScale, offset, ml.WithRoPEBase(ropeBase))

	// TODO - this is wrong somehow so commenting out
	// if opts.largeModelScaling {
	// 	q = q.Scale(ctx, 1.0/math.Sqrt(float64(opts.hiddenSize/opts.numHeads)))
	// } else {
	// 	q = q.Scale(ctx, 1.0/math.Sqrt(float64(opts.attnKeyLen)))
	// }

	scaleFactor := math.Pow(256, -0.5)

	kqv := nn.Attention(ctx, q, k, v, scaleFactor, cache)
	kqv = kqv.Transpose(ctx, 0, 2, 1, 3).Reshape(ctx, B, L, -1)
	return sa.Output.Forward(ctx, kqv)
}

func (m *TextModel) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	// ropeBase := m.TextConfig.ropeLocalBase
	// if (layer+1)%gemmaGlobalCacheCount == 0 {
	// 	ropeBase = m.TextConfig.ropeGlobalBase
	// }
	// 	q = q.RoPE(ctx, opts.attnKeyLen, traditional, opts.ropeScale, offset, ml.WithRoPEBase(ropeBase))
	panic("not yet implemented")
	// return key.RoPE(ctx, shift, m.TextConfig.attnKeyLen, ropeBase, 1/m.TextConfig.ropeScale, rope.WithTypeNeoX()), nil
}

type TextMLP struct {
	Up   *nn.Linear `gguf:"up_proj"`
	Down *nn.Linear `gguf:"down_proj"`
	Gate *nn.Linear `gguf:"gate_proj"`
}

func (mlp *TextMLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *TextConfig) ml.Tensor {
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).GELU(ctx, mlp.Up.Forward(ctx, hiddenState))
	return mlp.Down.Forward(ctx, hiddenState)
}

type TextLayer struct {
	AttentionNorm     *nn.RMSNorm        `gguf:"input_layernorm"`
	SelfAttention     *TextSelfAttention `gguf:"self_attn"`
	PostAttentionNorm *nn.RMSNorm        `gguf:"post_attention_layernorm"`
	MLPNorm           *nn.RMSNorm        `gguf:"pre_feedforward_layernorm"`
	MLP               *TextMLP           `gguf:"mlp"`
	PostMLPNorm       *nn.RMSNorm        `gguf:"post_feedforward_layernorm"`
}

func (l *TextLayer) Forward(ctx ml.Context, layer int, hiddenState, outputs ml.Tensor, offset int, cache kvcache.Cache, opts *TextConfig) ml.Tensor {
	residual := hiddenState
	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, layer, hiddenState, offset, cache, opts)
	hiddenState = l.PostAttentionNorm.Forward(ctx, hiddenState, opts.eps)

	// In the final layer (outputs != nil), optimize by pruning to just the token positions
	// we need logits for.
	if outputs != nil {
		hiddenState = hiddenState.TakeAxes(ctx, outputs, 1)
		residual = residual.TakeAxes(ctx, outputs, 1)
	}

	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState
	hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MLP.Forward(ctx, hiddenState, opts) // TODO this is where it goes bad most likely...
	hiddenState = l.PostMLPNorm.Forward(ctx, hiddenState, opts.eps)
	return hiddenState.Add(ctx, residual)
}

func (m *TextModel) Forward(ctx ml.Context, batch input.Batch, cache kvcache.Cache) ml.Tensor {
	hiddenState := m.TokenEmbedding.Forward(ctx, batch.Inputs)
	hiddenState = hiddenState.Scale(ctx, math.Sqrt(float64(m.TextConfig.hiddenSize)))

	// set image embeddings
	// var except []int
	// for _, image := range batch.Multimodal {
	// 	visionOutputs := image.Multimodal[0].Tensor
	// 	ctx.Forward(visionOutputs.Copy(ctx, hiddenState.AsStrided(ctx,
	// 		[]int{visionOutputs.Dim(0) * visionOutputs.Dim(1)},
	// 		[]int{image.Index * hiddenState.Stride(1)}, 0)))

	// 	for i := range visionOutputs.Dim(1) {
	// 		except = append(except, image.Index+i)
	// 	}
	// }

	for i, layer := range m.Layers {
		// gemma alternates between the sliding window (local) and causal (global)
		// kv cache every 6 layers
		if cache != nil {
			// cacheType := cacheTypeSWA
			// if (i+1)%gemmaGlobalCacheCount == 0 {
			// 	cacheType = cacheTypeCausal
			// }
			cache.SetLayer(i)

			// TODO this needs to come back
			// wc := cache.(*kvcache.WrapperCache)
			// wc.SetLayerType(cacheType)

			// if causal, ok := wc.UnderlyingCache().(*kvcache.Causal); ok {
			// 	causal.SetCausal(ctx, kvcache.CausalOptions{Except: except})
			// }
		}

		var offset int
		var lastLayerOutputs ml.Tensor
		if i == len(m.Layers)-1 {
			offset = batch.Offset
			lastLayerOutputs = batch.Outputs
		}

		hiddenState = layer.Forward(ctx, i, hiddenState, lastLayerOutputs, offset, cache, m.TextConfig)
	}
	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	return hiddenState
}

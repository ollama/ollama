package inkling2

import (
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

const draftArchitecture = "Inkling2ForConditionalGenerationMTP"

func init() { model.RegisterDraft(draftArchitecture, newDraft) }

type draftConfig struct {
	Width       int     `json:"hidden_size"`
	NumLayers   int     `json:"num_hidden_layers"`
	Heads       int     `json:"num_attention_heads"`
	KVHeads     int     `json:"num_key_value_heads"`
	HeadDim     int     `json:"head_dim"`
	FFWidth     int     `json:"intermediate_size"`
	LocalLayers []int   `json:"local_layer_ids"`
	Window      int     `json:"sliding_window_size"`
	RopeTheta   float32 `json:"rope_theta"`
	Epsilon     float32 `json:"rms_norm_eps"`
	Block       int     `json:"block_size"`
	StateWidth  int     `json:"markov_rank"`
	HeadType    string  `json:"markov_head_type"`
	Taps        []int   `json:"target_layer_ids"`
	QueryBias   bool    `json:"q_bias"`
	OutputBias  bool    `json:"o_bias"`
}

type draft struct {
	config draftConfig
	target *targetModel
	prefix string

	Layers        []draftLayer
	Fuse          *mlx.Array
	Mask          *mlx.Array
	Norm          *mlx.Array
	RecurrentNorm *mlx.Array
	Previous      *nn.Embedding
	Recurrent     *nn.Linear
	Folded        *nn.Linear
}

type draftLayer struct {
	local           bool
	InputNorm       *mlx.Array
	FfNorm          *mlx.Array
	ContextNorm     *mlx.Array
	QueryNorm       *mlx.Array
	KeyNorm         *mlx.Array
	ContextKeyNorm  *mlx.Array
	Query           *nn.Linear
	Key, Value      *nn.Linear
	ContextKey      *nn.Linear
	ContextValue    *nn.Linear
	AttentionOutput *nn.Linear
	Gate, Up, Down  *nn.Linear
}

var _ model.BlockSelector = (*draft)(nil)

func newDraft(root *model.Root, target model.Model) (model.DraftModel, error) {
	m, ok := target.(*targetModel)
	if !ok {
		return nil, fmt.Errorf("Inkling2 MTP requires an Inkling2 target, got %T", target)
	}
	path := root.Draft.Config
	if path == "" {
		path = "draft/config.json"
	}
	var cfg draftConfig
	if err := root.Manifest.ReadConfigJSON(path, &cfg); err != nil {
		return nil, fmt.Errorf("load Inkling2 MTP config: %w", err)
	}
	const maxDraftBlockSize = 1024
	if cfg.NumLayers > maxLayerCount || cfg.Block > maxDraftBlockSize {
		return nil, fmt.Errorf("Inkling2 MTP layer count %d or block size %d exceeds its limit", cfg.NumLayers, cfg.Block)
	}
	if cfg.Width != m.config.HiddenSize || cfg.NumLayers <= 0 || cfg.Heads <= 0 || cfg.KVHeads <= 0 ||
		cfg.Heads%cfg.KVHeads != 0 || cfg.HeadDim <= 0 || cfg.HeadDim%2 != 0 || cfg.FFWidth <= 0 ||
		cfg.Window <= 0 || cfg.RopeTheta <= 0 || cfg.Epsilon <= 0 || cfg.Block < 2 ||
		cfg.StateWidth <= 0 || cfg.HeadType != "rnn_hidden" || cfg.QueryBias || cfg.OutputBias {
		return nil, fmt.Errorf("unsupported Inkling2 MTP configuration")
	}
	if len(cfg.Taps) == 0 || !slices.IsSorted(cfg.Taps) || cfg.Taps[0] < 0 ||
		cfg.Taps[len(cfg.Taps)-1] != len(m.plan) || len(slices.Compact(slices.Clone(cfg.Taps))) != len(cfg.Taps) {
		return nil, fmt.Errorf("invalid Inkling2 MTP target visits %v", cfg.Taps)
	}
	for _, layer := range cfg.LocalLayers {
		if layer < 0 || layer >= cfg.NumLayers {
			return nil, fmt.Errorf("invalid Inkling2 MTP local layer %d", layer)
		}
	}
	prefix := root.Draft.TensorPrefix
	if prefix == "" {
		prefix = "draft."
	}
	return &draft{config: cfg, target: m, prefix: prefix}, nil
}

func (d *draft) LoadWeights(tensors map[string]*mlx.Array) error {
	c := d.config
	r := weightReader{tensors: tensors}
	d.Fuse = r.tensor(d.prefix+"fuse_weight", c.NumLayers+1, len(c.Taps))
	d.Mask = r.tensor(d.prefix+"mask_embed", c.Width)
	d.Norm = r.tensor(d.prefix+"norm.weight", c.Width)

	p := d.prefix + "markov_head."
	d.RecurrentNorm = r.tensor(p+"dflare_norm.weight", c.Width)
	d.Previous = nn.NewEmbedding(r.tensor(p+"markov_w1.weight", d.target.config.VocabSize, c.StateWidth))
	d.Recurrent = r.projection(p+"rnn_proj", 3*c.StateWidth, 2*(c.StateWidth+c.Width), true)
	latent := r.projection(p+"hidden_out", c.Width, c.StateWidth, true)

	d.Layers = make([]draftLayer, c.NumLayers)
	for i := range d.Layers {
		layer := fmt.Sprintf("%slayers.%d.", d.prefix, i)
		a := layer + "self_attn."
		kvWidth := c.KVHeads * c.HeadDim
		d.Layers[i] = draftLayer{
			local:           slices.Contains(c.LocalLayers, i),
			InputNorm:       r.tensor(layer+"input_layernorm.weight", c.Width),
			FfNorm:          r.tensor(layer+"post_attention_layernorm.weight", c.Width),
			ContextNorm:     r.tensor(layer+"dflare_norm.weight", c.Width),
			QueryNorm:       r.tensor(a+"q_norm.weight", c.HeadDim),
			KeyNorm:         r.tensor(a+"k_norm.weight", c.HeadDim),
			ContextKeyNorm:  r.tensor(a+"k_norm_ctx.weight", c.HeadDim),
			Query:           r.linear(a+"q_proj.weight", c.Heads*c.HeadDim, c.Width),
			Key:             r.linear(a+"k_proj.weight", kvWidth, c.Width),
			Value:           r.linear(a+"v_proj.weight", kvWidth, c.Width),
			ContextKey:      r.linear(a+"k_proj_ctx.weight", kvWidth, c.Width),
			ContextValue:    r.linear(a+"v_proj_ctx.weight", kvWidth, c.Width),
			AttentionOutput: r.linear(a+"o_proj.weight", c.Width, c.Heads*c.HeadDim),
			Gate:            r.linear(layer+"mlp.gate_proj.weight", c.FFWidth, c.Width),
			Up:              r.linear(layer+"mlp.up_proj.weight", c.FFWidth, c.Width),
			Down:            r.linear(layer+"mlp.down_proj.weight", c.Width, c.FFWidth),
		}
	}
	if r.err != nil {
		return r.err
	}
	d.Fuse = mlx.SoftmaxAxis(d.Fuse.AsType(mlx.DTypeFloat32), -1, true).AsType(mlx.DTypeBFloat16)

	// Fold the latent projection into the shared target head once. Both source
	// matrices retain their original BF16 values, with FP32 accumulation.
	output := d.target.Output.Weight.AsType(mlx.DTypeFloat32)
	foldWeight := output.Matmul(latent.Weight.AsType(mlx.DTypeFloat32)).AsType(mlx.DTypeBFloat16)
	foldBias := output.Matmul(latent.Bias.AsType(mlx.DTypeFloat32)).AsType(mlx.DTypeBFloat16)
	d.Folded = nn.NewLinear(foldWeight, foldBias)
	d.target.draftTaps = slices.Clone(c.Taps)
	return nil
}

func (d *draft) BlockParams() (int, int32) {
	// The learned mask embeddings do not use a token ID from the vocabulary.
	return d.config.Block, 0
}

func (d *draft) AdaptiveDraftDepth() bool {
	// The published head was trained with an anchor and seven masks.
	return d.config.Block == 8
}

func (d *draft) NewCaches() []cache.Cache {
	states := make([]cache.Cache, 0, len(d.Layers)+1)
	for _, layer := range d.Layers {
		if layer.local {
			// The rotating cache reserves one slot for the incoming block.
			// Keep the whole preceding window available to its first row.
			states = append(states, cache.NewRotatingKVCache(d.config.Window+1))
		} else {
			states = append(states, cache.NewKVCache())
		}
	}
	// The proposal head conditions on the most recent committed target row.
	return append(states, cache.NewRotatingKVCache(1))
}

func (d *draft) mix(features *mlx.Array, row int, norm *mlx.Array) *mlx.Array {
	taps := len(d.config.Taps)
	weights := d.Fuse.Slice(mlx.Slice(row, row+1), mlx.Slice()).Reshape(1, 1, taps, 1).AsType(mlx.DTypeFloat32)
	x := features.Reshape(features.Dim(0), features.Dim(1), taps, d.config.Width).AsType(mlx.DTypeFloat32)
	return modelRMS(mlx.Sum(mlx.Mul(x, weights), 2, false), norm, d.config.Epsilon)
}

func (d *draft) Forward(b *batch.Batch, _, states []cache.Layer) (hidden, auxHidden *mlx.Array) {
	c := d.config
	featureCache := states[len(d.Layers)].(cache.Attention)
	var contextLength int
	if b.Hidden != nil {
		contextLength = b.Hidden.Dim(1)
		f := d.mix(b.Hidden, len(d.Layers), d.RecurrentNorm).ExpandDims(1)
		featureCache.Update(b, f, f)
	}
	var x *mlx.Array
	var block *batch.Batch
	contextStart := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))
	if b.InputIDs != nil {
		batchSize, length := b.InputIDs.Dim(0), b.InputIDs.Dim(1)
		anchor := d.target.embedTokens(b.InputIDs.Slice(mlx.Slice(), mlx.Slice(0, 1)))
		masks := mlx.BroadcastTo(d.Mask, int32(batchSize), int32(length-1), int32(c.Width))
		x = anchor.Concatenate(1, masks)
		offsets := make([]int32, len(b.SeqOffsets))
		for i, pos := range b.SeqOffsets {
			offsets[i] = pos + int32(contextLength)
		}
		block = &batch.Batch{InputIDs: b.InputIDs, SeqOffsets: offsets, SeqQueryLens: b.SeqQueryLens}
	}
	for i, layer := range d.Layers {
		var k, v *mlx.Array
		if b.Hidden != nil {
			ctx := d.mix(b.Hidden, i, layer.ContextNorm)
			k, v = layer.kv(ctx, layer.ContextKey, layer.ContextValue, layer.ContextKeyNorm, contextStart, c)
		}
		store := states[i].(cache.Attention)
		if x == nil {
			if k != nil {
				store.Update(b, k, v)
			}
			continue
		}
		x = layer.forward(x, k, v, block, store, c)
	}
	if x == nil {
		return nil, nil
	}
	x = modelRMS(x, d.Norm, c.Epsilon)
	f := featureCache.View(block).V().Squeeze(1)
	f = mlx.BroadcastTo(f, int32(x.Dim(0)), int32(x.Dim(1)), int32(c.Width))
	// Keep the committed conditioning with each row so proposal selection can
	// run without storing mutable request state on the model.
	return x.Concatenate(-1, f), nil
}

func (l draftLayer) kv(x *mlx.Array, key, value *nn.Linear, keyNorm, offsets *mlx.Array, c draftConfig) (k, v *mlx.Array) {
	b, n := x.Dim(0), x.Dim(1)
	k = modelRMS(key.Forward(x).Reshape(b, n, c.KVHeads, c.HeadDim), keyNorm, c.Epsilon).Transpose(0, 2, 1, 3)
	v = value.Forward(x).Reshape(b, n, c.KVHeads, c.HeadDim).Transpose(0, 2, 1, 3)
	return mlx.RoPEWithBase(k, c.HeadDim, false, c.RopeTheta, 1, offsets), v
}

func (l draftLayer) forward(x, contextKey, contextValue *mlx.Array, b *batch.Batch, store cache.Attention, c draftConfig) *mlx.Array {
	rows, n := x.Dim(0), x.Dim(1)
	offsets := mlx.FromValues(b.SeqOffsets, rows)
	xNorm := modelRMS(x, l.InputNorm, c.Epsilon)
	q := modelRMS(l.Query.Forward(xNorm).Reshape(rows, n, c.Heads, c.HeadDim), l.QueryNorm, c.Epsilon).Transpose(0, 2, 1, 3)
	q = mlx.RoPEWithBase(q, c.HeadDim, false, c.RopeTheta, 1, offsets)
	k, v := l.kv(xNorm, l.Key, l.Value, l.KeyNorm, offsets, c)
	if contextKey != nil {
		k = contextKey.Concatenate(2, k)
		v = contextValue.Concatenate(2, v)
	}
	history := store.Update(b, k, v)
	k, v = history.K(), history.V()
	if l.local {
		// The local draft window is measured from the first block position.
		// The whole proposed block attends bidirectionally inside that window.
		end := k.Dim(2)
		start := max(0, end-n-c.Window)
		k = k.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice())
		v = v.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice())
	}
	scale := float32(1 / math.Sqrt(float64(c.HeadDim)))
	out := nn.ScaledDotProductAttention(b, q, scale, nn.WithKV(k, v, nil))
	attention := l.AttentionOutput.Forward(out.Transpose(0, 2, 1, 3).Reshape(rows, n, c.Heads*c.HeadDim))
	residual := draftResidualSum(x, attention)
	ff := modelRMS(residual, l.FfNorm, c.Epsilon)
	return draftResidualSum(residual, l.Down.Forward(mlx.SwiGLU(l.Gate.Forward(ff), l.Up.Forward(ff))))
}

// SGLang's draft normalizes the FP32 sum and stores a BF16 residual for
// the next addition. Keep the sum for normalization and round it on reuse.
func draftResidualSum(residual, update *mlx.Array) *mlx.Array {
	return residual.AsType(mlx.DTypeBFloat16).AsType(mlx.DTypeFloat32).Add(update.AsType(mlx.DTypeFloat32))
}

func (d *draft) Unembed(h *mlx.Array) *mlx.Array {
	hidden := h.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, d.config.Width))
	if hidden.Dim(1) == 7 {
		if logits := d.target.shortVocabularyProjection(hidden); logits != nil {
			return logits.AsType(mlx.DTypeBFloat16)
		}
	}
	return hidden.AsType(mlx.DTypeFloat32).Matmul(d.target.Output.Weight.AsType(mlx.DTypeFloat32).Transpose(1, 0)).AsType(mlx.DTypeBFloat16)
}

func (d *draft) Select(h, logits, anchor *mlx.Array, choose func(ids, scores *mlx.Array) (tokens, probs *mlx.Array)) (tokens, ids, probs *mlx.Array) {
	c := d.config
	b, length, vocab := logits.Dim(0), logits.Dim(1), logits.Dim(2)
	candidates := mlx.BroadcastTo(mlx.Arange(0, float64(vocab), 1, mlx.DTypeInt32), int32(b), int32(vocab))
	state := mlx.Zeros(mlx.DTypeBFloat16, b, c.StateWidth)
	last := anchor
	feature := h.Slice(mlx.Slice(), mlx.Slice(0, 1), mlx.Slice(c.Width, 2*c.Width)).Squeeze(1)
	sequence := make([]*mlx.Array, length)
	dist := make([]*mlx.Array, length)
	for i := range length {
		prev := d.Previous.Forward(last)
		hidden := h.Slice(mlx.Slice(), mlx.Slice(i, i+1), mlx.Slice(0, c.Width)).Squeeze(1)
		gates := d.Recurrent.Forward(mlx.Concatenate([]*mlx.Array{state, prev, hidden, feature}, -1))
		width := c.StateWidth
		keep := gates.Slice(mlx.Slice(), mlx.Slice(0, width)).Sigmoid()
		update := gates.Slice(mlx.Slice(), mlx.Slice(width, 2*width)).Tanh()
		latent := gates.Slice(mlx.Slice(), mlx.Slice(2*width, 3*width)).Tanh()
		state = mlx.Add(mlx.Mul(keep, state), mlx.Mul(mlx.AddScalar(mlx.MulScalar(keep, -1), 1), update))
		base := logits.Slice(mlx.Slice(), mlx.Slice(i, i+1), mlx.Slice()).Squeeze(1)
		// The BF16 head rounds the sum before softcapping, then writes the
		// capped scores back to BF16 before the sampler sees them.
		corrected := base.Add(d.Folded.Forward(latent))
		scores := softcap(corrected.AsType(mlx.DTypeFloat32), d.target.config.LogitCap).AsType(corrected.DType())
		last, dist[i] = choose(candidates, scores)
		sequence[i] = last
	}
	return mlx.Stack(sequence, 1), nil, mlx.Stack(dist, 1)
}

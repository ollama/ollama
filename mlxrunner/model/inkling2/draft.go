package inkling2

import (
	"fmt"
	"math"
	"slices"
	"sync"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

const draftArchitecture = "Inkling2ForConditionalGenerationMTP"

func init() { model.RegisterDraft(draftArchitecture, newDraft) }

// draftBits and draftVocabularyBits are the affine weight widths of the
// draft's projections and folded head, and of its copy of the vocabulary head,
// on Metal. Proposals only steer verification, so the draft trades weight
// precision for bytes read per round. Zero keeps the published BF16 weights.
const draftBits, draftVocabularyBits = 8, 4

// draftGroupSize is MLX's default affine quantization group.
const draftGroupSize = 64

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
	Fuse          *mlx.Array // FP32 values of the BF16 softmax of the tap weights, [layers+1, taps]
	Mask          *mlx.Array
	Norm          *mlx.Array
	RecurrentNorm *mlx.Array
	Previous      *nn.Embedding
	Recurrent     *nn.Linear
	Folded        *draftProjection
	FoldedBias    *mlx.Array
	// Vocabulary is the draft's own copy of the shared head, or nil to read
	// the target's.
	Vocabulary *draftProjection

	// correct is the compiled score correction.
	correct func(base, folded, bias *mlx.Array) *mlx.Array
}

type draftLayer struct {
	local          bool
	InputNorm      *mlx.Array
	FfNorm         *mlx.Array
	ContextNorm    *mlx.Array
	QueryNorm      *mlx.Array
	KeyNorm        *mlx.Array
	ContextKeyNorm *mlx.Array
	QKV            *draftProjection
	ContextKV      *draftProjection
	Output         *draftProjection
	GateUp         *draftProjection
	Down           *draftProjection
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
	if mlx.MetalIsAvailable() {
		return d.loadWeights(tensors, draftBits, draftVocabularyBits)
	}
	return d.loadWeights(tensors, 0, 0)
}

// loadWeights quantizes the projections and folded head to bits and the copy
// of the vocabulary head to vocabularyBits. Zero keeps the BF16 weights, and
// a zero vocabularyBits reads the target's head.
func (d *draft) loadWeights(tensors map[string]*mlx.Array, bits, vocabularyBits int) error {
	c := d.config
	r := weightReader{tensors: tensors}
	fuse := r.tensor(d.prefix+"fuse_weight", c.NumLayers+1, len(c.Taps))
	d.Mask = r.tensor(d.prefix+"mask_embed", c.Width)
	d.Norm = r.float32(d.prefix+"norm.weight", c.Width)

	p := d.prefix + "markov_head."
	d.RecurrentNorm = r.float32(p+"dflare_norm.weight", c.Width)
	d.Previous = nn.NewEmbedding(r.tensor(p+"markov_w1.weight", d.target.config.VocabSize, c.StateWidth))
	d.Recurrent = r.projection(p+"rnn_proj", 3*c.StateWidth, 2*(c.StateWidth+c.Width), true)
	latent := r.projection(p+"hidden_out", c.Width, c.StateWidth, true)

	type layerWeights struct{ q, k, v, ck, cv, o, gate, up, down *mlx.Array }
	raw := make([]layerWeights, c.NumLayers)
	d.Layers = make([]draftLayer, c.NumLayers)
	for i := range d.Layers {
		layer := fmt.Sprintf("%slayers.%d.", d.prefix, i)
		a := layer + "self_attn."
		kvWidth := c.KVHeads * c.HeadDim
		d.Layers[i] = draftLayer{
			local:          slices.Contains(c.LocalLayers, i),
			InputNorm:      r.float32(layer+"input_layernorm.weight", c.Width),
			FfNorm:         r.float32(layer+"post_attention_layernorm.weight", c.Width),
			ContextNorm:    r.float32(layer+"dflare_norm.weight", c.Width),
			QueryNorm:      r.float32(a+"q_norm.weight", c.HeadDim),
			KeyNorm:        r.float32(a+"k_norm.weight", c.HeadDim),
			ContextKeyNorm: r.float32(a+"k_norm_ctx.weight", c.HeadDim),
		}
		raw[i] = layerWeights{
			q:    r.tensor(a+"q_proj.weight", c.Heads*c.HeadDim, c.Width),
			k:    r.tensor(a+"k_proj.weight", kvWidth, c.Width),
			v:    r.tensor(a+"v_proj.weight", kvWidth, c.Width),
			ck:   r.tensor(a+"k_proj_ctx.weight", kvWidth, c.Width),
			cv:   r.tensor(a+"v_proj_ctx.weight", kvWidth, c.Width),
			o:    r.tensor(a+"o_proj.weight", c.Width, c.Heads*c.HeadDim),
			gate: r.tensor(layer+"mlp.gate_proj.weight", c.FFWidth, c.Width),
			up:   r.tensor(layer+"mlp.up_proj.weight", c.FFWidth, c.Width),
			down: r.tensor(layer+"mlp.down_proj.weight", c.Width, c.FFWidth),
		}
	}
	if r.err != nil {
		return r.err
	}
	for i, w := range raw {
		l := &d.Layers[i]
		l.QKV = newDraftProjection(bits, w.q, w.k, w.v)
		l.ContextKV = newDraftProjection(bits, w.ck, w.cv)
		l.Output = newDraftProjection(bits, w.o)
		l.GateUp = newDraftProjection(bits, w.gate, w.up)
		l.Down = newDraftProjection(bits, w.down)
	}
	// The mixes consume the BF16 softmax of the tap weights in FP32.
	d.Fuse = mlx.SoftmaxAxis(fuse.AsType(mlx.DTypeFloat32), -1, true).AsType(mlx.DTypeBFloat16).AsType(mlx.DTypeFloat32)

	// Fold the latent projection into the shared target head once. Both source
	// matrices retain their original BF16 values, with FP32 accumulation.
	output := d.target.Output.Weight.AsType(mlx.DTypeFloat32)
	d.Folded = newDraftProjection(bits, output.Matmul(latent.Weight.AsType(mlx.DTypeFloat32)).AsType(mlx.DTypeBFloat16))
	d.FoldedBias = output.Matmul(latent.Bias.AsType(mlx.DTypeFloat32)).AsType(mlx.DTypeBFloat16)
	if vocabularyBits > 0 {
		head := d.target.Output.Weight
		if d.target.ShortOutput != nil {
			head = d.target.ShortOutput.Weight
		}
		d.Vocabulary = newDraftProjection(vocabularyBits, head.AsType(mlx.DTypeBFloat16))
	}
	d.correct = compiledCorrection(d.target.config.LogitCap)
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

// mixes returns the normalized tap mixture of the context rows for every draft
// layer and, last, for the proposal head. One FP32 product with the tap
// weights forms all of them.
func (d *draft) mixes(features *mlx.Array) []*mlx.Array {
	c := d.config
	b, n := features.Dim(0), features.Dim(1)
	taps := features.Reshape(b, n, len(c.Taps), c.Width).AsType(mlx.DTypeFloat32)
	mixed := d.Fuse.Matmul(taps)
	out := make([]*mlx.Array, len(d.Layers)+1)
	for i := range out {
		norm := d.RecurrentNorm
		if i < len(d.Layers) {
			norm = d.Layers[i].ContextNorm
		}
		mix := mixed.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(i, i+1), mlx.Slice()).Reshape(b, n, c.Width)
		out[i] = modelRMS(mix, norm, c.Epsilon)
	}
	return out
}

func (d *draft) Forward(b *batch.Batch, _, states []cache.Layer) (hidden, auxHidden *mlx.Array) {
	c := d.config
	featureCache := states[len(d.Layers)].(cache.Attention)
	var contextLength int
	var mixes []*mlx.Array
	if b.Hidden != nil {
		contextLength = b.Hidden.Dim(1)
		mixes = d.mixes(b.Hidden)
		f := mixes[len(d.Layers)].ExpandDims(1)
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
	// Submit a proposal block layer by layer so Metal starts on it while the
	// host builds the remaining layers and the proposal head. The guard reads
	// the incoming batch, before its offsets move past the context, so the
	// load-time shape check at offset zero stays lazy.
	submit := x != nil && submitsEarly(b)
	for i, layer := range d.Layers {
		var k, v *mlx.Array
		if mixes != nil {
			kv := layer.ContextKV.forward(mixes[i])
			k, v = layer.rotate(kv[0], kv[1], layer.ContextKeyNorm, contextStart, c)
		}
		store := states[i].(cache.Attention)
		if x == nil {
			if k != nil {
				store.Update(b, k, v)
			}
			continue
		}
		x = layer.forward(x, k, v, block, store, c)
		if submit && i+1 < len(d.Layers) {
			mlx.AsyncEval(x)
		}
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

// rotate normalizes and rotates projected keys and arranges keys and values
// by head.
func (l draftLayer) rotate(key, value, keyNorm, offsets *mlx.Array, c draftConfig) (k, v *mlx.Array) {
	b, n := key.Dim(0), key.Dim(1)
	k = modelRMS(key.Reshape(b, n, c.KVHeads, c.HeadDim), keyNorm, c.Epsilon).Transpose(0, 2, 1, 3)
	v = value.Reshape(b, n, c.KVHeads, c.HeadDim).Transpose(0, 2, 1, 3)
	return mlx.RoPEWithBase(k, c.HeadDim, false, c.RopeTheta, 1, offsets), v
}

func (l draftLayer) forward(x, contextKey, contextValue *mlx.Array, b *batch.Batch, store cache.Attention, c draftConfig) *mlx.Array {
	rows, n := x.Dim(0), x.Dim(1)
	offsets := mlx.FromValues(b.SeqOffsets, rows)
	xNorm := modelRMS(x, l.InputNorm, c.Epsilon)
	qkv := l.QKV.forward(xNorm)
	q := modelRMS(qkv[0].Reshape(rows, n, c.Heads, c.HeadDim), l.QueryNorm, c.Epsilon).Transpose(0, 2, 1, 3)
	q = mlx.RoPEWithBase(q, c.HeadDim, false, c.RopeTheta, 1, offsets)
	k, v := l.rotate(qkv[1], qkv[2], l.KeyNorm, offsets, c)
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
	attention := l.Output.forward(out.Transpose(0, 2, 1, 3).Reshape(rows, n, c.Heads*c.HeadDim))[0]
	residual := draftResidualSum(x, attention)
	ff := modelRMS(residual, l.FfNorm, c.Epsilon)
	gateUp := l.GateUp.forward(ff)
	return draftResidualSum(residual, l.Down.forward(mlx.SwiGLU(gateUp[0], gateUp[1]))[0])
}

// SGLang's draft normalizes the FP32 sum and stores a BF16 residual for
// the next addition. Keep the sum for normalization and round it on reuse.
var draftResidualSum = mlx.Compile2("InklingDraftResidualSum", func(residual, update *mlx.Array) *mlx.Array {
	return residual.AsType(mlx.DTypeBFloat16).AsType(mlx.DTypeFloat32).Add(update.AsType(mlx.DTypeFloat32))
}, mlx.Shapeless())

// draftProjection multiplies one input by the weights of several projections
// stacked as the rows of one matrix, so the input is read and the weights are
// launched once. Quantized weights are affine groups of draftGroupSize.
type draftProjection struct {
	Weight, Scales, Biases *mlx.Array
	widths                 []int
	bits                   int
}

func newDraftProjection(bits int, weights ...*mlx.Array) *draftProjection {
	p := &draftProjection{}
	for _, w := range weights {
		p.widths = append(p.widths, w.Dim(0))
	}
	w := weights[0]
	if len(weights) > 1 {
		w = mlx.Concatenate(weights, 0)
	}
	// Affine groups span whole input columns; keep a width they cannot tile.
	if bits > 0 && w.Dim(1)%draftGroupSize == 0 {
		p.bits = bits
		p.Weight, p.Scales, p.Biases = mlx.Quantize(w, draftGroupSize, bits, "affine")
	} else {
		p.Weight = w
	}
	return p
}

// forward returns one product per stacked projection, shaped like x with the
// projection's width as the last axis.
func (p *draftProjection) forward(x *mlx.Array) []*mlx.Array {
	shape := x.Dims()
	flat := x.Reshape(-1, shape[len(shape)-1])
	var y *mlx.Array
	if p.Scales != nil {
		y = mlx.QuantizedMatmul(flat, p.Weight, p.Scales, p.Biases, true, draftGroupSize, p.bits, "affine", nil)
	} else {
		y = flat.Matmul(p.Weight.Transpose(1, 0))
	}
	outs := make([]*mlx.Array, len(p.widths))
	start := 0
	for i, width := range p.widths {
		part := y
		if len(p.widths) > 1 {
			part = y.Slice(mlx.Slice(), mlx.Slice(start, start+width))
		}
		shape[len(shape)-1] = width
		outs[i] = part.Reshape(shape...)
		start += width
	}
	return outs
}

func (d *draft) Unembed(h *mlx.Array) *mlx.Array {
	hidden := h.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, d.config.Width))
	if d.Vocabulary != nil {
		return d.Vocabulary.forward(hidden)[0]
	}
	if logits := d.target.shortVocabularyProjection(hidden); logits != nil {
		return logits.AsType(mlx.DTypeBFloat16)
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
		step := compiledGatedState(gates, state)
		state = step[0]
		folded := d.Folded.forward(step[1])[0]
		base := logits.Slice(mlx.Slice(), mlx.Slice(i, i+1), mlx.Slice()).Squeeze(1)
		last, dist[i] = choose(candidates, d.correct(base, folded, d.FoldedBias))
		sequence[i] = last
	}
	return mlx.Stack(sequence, 1), nil, mlx.Stack(dist, 1)
}

// gatedState advances the recurrent proposal state in BF16 and returns the
// latent that feeds the folded head.
func gatedState(gates, state *mlx.Array) (next, latent *mlx.Array) {
	width := state.Dim(state.NumDims() - 1)
	keep := gates.Slice(mlx.Slice(), mlx.Slice(0, width)).Sigmoid()
	update := gates.Slice(mlx.Slice(), mlx.Slice(width, 2*width)).Tanh()
	latent = gates.Slice(mlx.Slice(), mlx.Slice(2*width, 3*width)).Tanh()
	next = mlx.Add(mlx.Mul(keep, state), mlx.Mul(mlx.AddScalar(mlx.MulScalar(keep, -1), 1), update))
	return next, latent
}

var compiledGatedState = mlx.Compile("InklingDraftGatedState", func(in ...*mlx.Array) []*mlx.Array {
	next, latent := gatedState(in[0], in[1])
	return []*mlx.Array{next, latent}
})

// correctScores adds the folded head and its bias to the base scores. The BF16
// head rounds the sum before softcapping, then writes the capped scores back
// to BF16 before the sampler sees them.
func correctScores(base, folded, bias *mlx.Array, logitCap float32) *mlx.Array {
	corrected := base.Add(folded.Add(bias))
	return softcap(corrected.AsType(mlx.DTypeFloat32), logitCap).AsType(corrected.DType())
}

// draftCorrections holds one compiled correction per logit cap. A compiled
// function is never released, so building one per model load would leak.
var draftCorrections sync.Map // math.Float32bits(logitCap) -> func(base, folded, bias *mlx.Array) *mlx.Array

func compiledCorrection(logitCap float32) func(base, folded, bias *mlx.Array) *mlx.Array {
	f, _ := draftCorrections.LoadOrStore(math.Float32bits(logitCap), mlx.Compile3("InklingDraftCorrection", func(base, folded, bias *mlx.Array) *mlx.Array {
		return correctScores(base, folded, bias, logitCap)
	}))
	return f.(func(base, folded, bias *mlx.Array) *mlx.Array)
}

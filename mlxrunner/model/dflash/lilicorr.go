package dflash

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func init() {
	model.RegisterDraft("LiLiCorrDraftModel", newLiLiCorr)
}

type lilicorrConfig struct {
	TopK       int     `json:"lilicorr_candidate_topk"`
	Width      int     `json:"lilicorr_hidden_size"`
	Layers     int     `json:"lilicorr_num_layers"`
	Heads      int     `json:"lilicorr_num_heads"`
	MLPRatio   float64 `json:"lilicorr_mlp_ratio"`
	FactorDim  int     `json:"lilicorr_factor_dim"`
	VectorEps  float32 `json:"lilicorr_vector_eps"`
	LogitScale float32 `json:"lilicorr_logit_scale"`
	ConvKernel int     `json:"conv_kernel_size"`
	ConvGroup  int     `json:"conv_group_size"`
}

type lilicorrModel struct {
	*Model
	config lilicorrConfig
	Head   *lilicorrHead
}

func newLiLiCorr(root *model.Root, target model.Model) (model.DraftModel, error) {
	draft, err := newModel(root, target, false)
	if err != nil {
		return nil, err
	}
	base := draft.(*Model)
	path := root.Draft.Config
	if path == "" {
		path = "draft/config.json"
	}
	data, err := root.Manifest.ReadConfig(path)
	if err != nil {
		return nil, err
	}
	var raw struct {
		DFlash lilicorrConfig `json:"dflash_config"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	cfg := raw.DFlash
	if cfg.Width == 0 {
		cfg.Width = int(base.HiddenSize)
	}
	if cfg.Width <= 0 || cfg.Heads <= 0 || cfg.Width%cfg.Heads != 0 || cfg.Layers <= 0 || cfg.MLPRatio <= 0 || cfg.FactorDim <= 0 || cfg.VectorEps <= 0 {
		return nil, fmt.Errorf("invalid LiLiCorr hidden, attention, or factor dimensions")
	}
	if int(float64(cfg.Width)*cfg.MLPRatio) <= 0 {
		return nil, fmt.Errorf("invalid LiLiCorr MLP intermediate size")
	}
	if cfg.TopK <= 0 || cfg.TopK > 16 || cfg.TopK&(cfg.TopK-1) != 0 || cfg.TopK > int(base.VocabSize) {
		return nil, fmt.Errorf("LiLiCorr candidate_topk must be a power of two <= 16 and fit the vocabulary")
	}
	if cfg.ConvKernel <= 0 || cfg.ConvKernel > base.BlockSize || cfg.ConvGroup <= 0 || int(base.HiddenSize)%cfg.ConvGroup != 0 {
		return nil, fmt.Errorf("invalid LiLiCorr convolution geometry")
	}
	base.contextNormFP32 = true
	return &lilicorrModel{Model: base, config: cfg}, nil
}

func (m *lilicorrModel) LoadWeights(tensors map[string]*mlx.Array) error {
	if err := m.Model.LoadWeights(tensors); err != nil {
		return err
	}
	loader := lilicorrLoader{
		tensors: tensors,
		linears: model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant),
		eps:     m.RMSNormEps,
	}
	width := int(m.HiddenSize)
	for i, layer := range m.Layers {
		prefix := fmt.Sprintf("%slayers.%d.", m.tensorPrefix, i)
		layer.Attention.Sinks = loader.tensor(prefix+"self_attn.attention_sink_bias", int(m.NumAttentionHeads))
		for name, dst := range map[string]**convolution{"attention_conv": &layer.AttentionConv, "mlp_conv": &layer.MLPConv} {
			*dst = &convolution{
				Base:       loader.tensor(prefix+name+".base_kernel", 2, m.config.ConvKernel, width),
				Projection: loader.linear(prefix+name+".kernel_projection", width, 2*m.config.ConvKernel*width/m.config.ConvGroup),
				groupSize:  m.config.ConvGroup,
			}
		}
	}
	m.Head = loadLiLiCorrHead(&loader, m.tensorPrefix+"lilicorr.", m.config, width, m.BlockSize)
	return loader.err
}

func (m *lilicorrModel) NewCaches() []cache.Cache {
	// The final slot keeps the projected context anchor in the same snapshot
	// lifecycle as the draft KV, including prefix reuse and rejected blocks.
	return append(m.Model.NewCaches(), cache.NewRotatingKVCache(1))
}

func (m *lilicorrModel) Forward(b *batch.Batch, _, caches []cache.Cache) (hidden, anchor *mlx.Array) {
	hctx := m.projectContext(b.Hidden)
	context := caches[len(m.Layers)].(*cache.RotatingKVCache)
	if hctx != nil {
		rows := hctx.ExpandDims(1)
		context.Update(b, rows, rows)
	}
	if b.InputIDs != nil {
		anchor = context.View(b).V().Reshape(1, int(m.HiddenSize))
		// Keep cache offsets aligned with the trunk's temporary block rows.
		// Normal speculative rollback removes these placeholders.
		rows := mlx.Zeros(anchor.DType(), 1, 1, b.InputIDs.Dim(1), int(m.HiddenSize))
		context.Update(b, rows, rows)
	}
	hidden, _ = m.Model.forward(b, caches[:len(m.Layers)], hctx)
	return hidden, anchor
}

// LiLiCorr greedily follows its learned candidate lattice, even for a sampled
// target. Point-mass proposal probabilities keep target verification exact.
func (m *lilicorrModel) SelectBlock(hidden, logits, anchor *mlx.Array) (tokens, ids, probs *mlx.Array) {
	b, length := logits.Dim(0), logits.Dim(1)
	k := m.config.TopK
	ids = logits.Negative().ArgpartitionAxis(k-1, -1).Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, k))
	order := logits.TakeAlongAxis(ids, -1).Negative().ArgsortAxis(-1)
	ids = ids.TakeAlongAxis(order, -1).AsType(mlx.DTypeInt32)
	logits = logits.AsType(mlx.DTypeFloat32)
	logProbs := logits.TakeAlongAxis(ids, -1).Subtract(logits.LogsumexpAxis(-1, true))
	start, transitions := m.Head.scores(m.target.TokenEmbeddings(ids), logProbs, hidden, anchor)
	path, distributions := make([]*mlx.Array, length), make([]*mlx.Array, length)
	var previous *mlx.Array
	for pos := range length {
		candidates := ids.Slice(mlx.Slice(), mlx.Slice(pos, pos+1), mlx.Slice()).Reshape(b, k)
		row := start
		if pos > 0 {
			edges := transitions.Slice(mlx.Slice(), mlx.Slice(pos-1, pos), mlx.Slice(), mlx.Slice()).Reshape(b, k, k)
			index := mlx.BroadcastTo(previous.Reshape(b, 1, 1), int32(b), 1, int32(k))
			row = edges.TakeAlongAxis(index, 1).Squeeze(1)
		}
		previous = row.Argmax(-1, true)
		token := candidates.TakeAlongAxis(previous, -1).AsType(mlx.DTypeInt32)
		probability := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, b, 1), 1)
		distribution := mlx.Zeros(mlx.DTypeFloat32, b, k).PutAlongAxis(previous, probability, -1)
		path[pos], distributions[pos] = token, distribution.ExpandDims(1)
	}
	return mlx.Concatenate(path, 1), ids, mlx.Concatenate(distributions, 1)
}

type lilicorrHead struct {
	config                             lilicorrConfig
	Token, Pass, Context               nn.LinearLayer
	FeatureNorm                        *nn.LayerNorm
	FeatureUp, FeatureDown             nn.LinearLayer
	Slot, Rank, AttentionBias          *mlx.Array
	Layers                             []lilicorrLayer
	Norm, AnchorNorm                   *nn.RMSNorm
	Factor                             []*nn.Linear
	Outgoing, Incoming, AnchorOutgoing nn.LinearLayer
}

type lilicorrLayer struct {
	AttnNorm, MLPNorm     *nn.RMSNorm
	QKV, Output, Up, Down nn.LinearLayer
}

func loadLiLiCorrHead(l *lilicorrLoader, prefix string, c lilicorrConfig, input, block int) *lilicorrHead {
	w := c.Width
	h := &lilicorrHead{
		config:         c,
		Factor:         make([]*nn.Linear, 3),
		Token:          l.linear(prefix+"token_proj", input, w),
		Pass:           l.linear(prefix+"pass_hidden_proj", input, w),
		Context:        l.linear(prefix+"context_proj", input, w),
		FeatureNorm:    &nn.LayerNorm{Weight: l.tensor(prefix+"feature_norm.weight", 5), Bias: l.tensor(prefix+"feature_norm.bias", 5), Eps: 1e-5},
		FeatureUp:      l.linear(prefix+"feature_mlp.up_proj", 5, w),
		FeatureDown:    l.linear(prefix+"feature_mlp.down_proj", w, w),
		Slot:           l.tensor(prefix+"slot_embedding", 1, 1, block-1, 1, w),
		Rank:           l.tensor(prefix+"rank_embedding", 1, 1, 1, c.TopK, w),
		Norm:           l.norm(prefix+"output_norm", w),
		AnchorNorm:     l.norm(prefix+"anchor_norm", w),
		Outgoing:       l.linear(prefix+"out_head", w, c.FactorDim),
		Incoming:       l.linear(prefix+"in_head", w, c.FactorDim),
		AnchorOutgoing: l.linear(prefix+"anchor_out_head", w, c.FactorDim),
	}

	factorWeight := l.tensor(prefix+"factor_input_proj.weight", w, 3*w)
	factorBias := l.tensor(prefix+"factor_input_proj.bias", w)
	if l.err == nil {
		for i := range h.Factor {
			h.Factor[i] = &nn.Linear{Weight: factorWeight.Slice(mlx.Slice(), mlx.Slice(i*w, (i+1)*w))}
		}
		h.Factor[0].Bias = factorBias
	}
	for i := range c.Layers {
		p := fmt.Sprintf("%slayers.%d.", prefix, i)
		h.Layers = append(h.Layers, lilicorrLayer{
			AttnNorm: l.norm(p+"attn_norm", w),
			MLPNorm:  l.norm(p+"mlp_norm", w),
			QKV:      &nn.Linear{Weight: l.tensor(p+"attn.in_proj_weight", 3*w, w), Bias: l.tensor(p+"attn.in_proj_bias", 3*w)},
			Output:   l.linear(p+"attn.out_proj", w, w),
			Up:       l.linear(p+"mlp.up_proj", w, int(float64(w)*c.MLPRatio)),
			Down:     l.linear(p+"mlp.down_proj", int(float64(w)*c.MLPRatio), w),
		})
	}
	relative := l.tensor(prefix+"relative_slot_bias", c.Heads, 2*block-1)
	same := l.tensor(prefix+"same_slot_bias", c.Heads)
	if l.err != nil {
		return nil
	}
	positions := mlx.FloorDivideScalar(mlx.Arange(0, float64((block-1)*c.TopK), 1, mlx.DTypeInt32), int32(c.TopK))
	distance := positions.ExpandDims(1).Subtract(positions.ExpandDims(0))
	indices := mlx.AddScalar(distance, float32(block-1)).AsType(mlx.DTypeInt32)
	h.AttentionBias = relative.TakeAxis(indices, 1).Add(same.Reshape(c.Heads, 1, 1).Multiply(distance.Equal(mlx.FromValue(0)).AsType(same.DType())))
	return h
}

func (h *lilicorrHead) scores(embeddings, logProbs, pass, anchor *mlx.Array) (start, pairs *mlx.Array) {
	b, slots, k := logProbs.Dim(0), logProbs.Dim(1), logProbs.Dim(2)
	width, heads := h.config.Width, h.config.Heads
	dtype := h.Slot.DType()
	ranks := mlx.Arange(0, float64(k), 1, mlx.DTypeFloat32).Reshape(1, 1, k)
	rankFraction := mlx.BroadcastTo(mlx.DivScalar(ranks, float32(max(1, k-1))), int32(b), int32(slots), int32(k))
	first := mlx.BroadcastTo(ranks.Equal(mlx.FromValue(float32(0))).AsType(mlx.DTypeFloat32), int32(b), int32(slots), int32(k))
	features := mlx.Stack([]*mlx.Array{logProbs, mlx.Exp(logProbs), logProbs.Subtract(logProbs.MaxAxis(-1, true)), rankFraction, first}, -1).AsType(dtype)
	features = mlx.LayerNormFn(features.AsType(mlx.DTypeFloat32), h.FeatureNorm.Weight.AsType(mlx.DTypeFloat32), h.FeatureNorm.Bias.AsType(mlx.DTypeFloat32), h.FeatureNorm.Eps).AsType(dtype)
	features = h.FeatureDown.Forward(mlx.SiLU(h.FeatureUp.Forward(features)))
	x := h.Token.Forward(embeddings.AsType(dtype)).Add(h.Pass.Forward(pass.AsType(dtype)).ExpandDims(2)).Add(features)
	x = x.Add(h.Slot.Reshape(1, slots, 1, width)).Add(h.Rank.Reshape(1, 1, k, width)).Reshape(b, slots*k, width)
	anchor = h.Context.Forward(anchor.AsType(dtype))
	for _, layer := range h.Layers {
		qkv := layer.QKV.Forward(rmsNormFP32(x, layer.AttnNorm)).Reshape(b, slots*k, 3, heads, width/heads)
		parts := make([]*mlx.Array, 3)
		for i := range parts {
			parts[i] = qkv.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice()).Squeeze(2).Transpose(0, 2, 1, 3)
		}
		attn := mlx.FastScaledDotProductAttention(parts[0], parts[1], parts[2], float32(1/math.Sqrt(float64(width/heads))), "array", h.AttentionBias.ExpandDims(0))
		x = x.Add(layer.Output.Forward(attn.Transpose(0, 2, 1, 3).Reshape(b, slots*k, width)))
		x = x.Add(layer.Down.Forward(mlx.SiLU(layer.Up.Forward(rmsNormFP32(x, layer.MLPNorm)))))
	}
	x = rmsNormFP32(x, h.Norm).Reshape(b, slots, k, width)
	anchor = rmsNormFP32(anchor, h.AnchorNorm)
	anchorRows := mlx.BroadcastTo(anchor.Reshape(b, 1, 1, width), int32(b), int32(slots), int32(k), int32(width))
	factors := mlx.SiLU(h.Factor[0].Forward(x).Add(h.Factor[1].Forward(anchorRows)).Add(h.Factor[2].Forward(x.Multiply(anchorRows))))
	outgoing := normalizedVector(h.Outgoing.Forward(factors), h.config.VectorEps)
	incoming := normalizedVector(h.Incoming.Forward(factors), h.config.VectorEps)
	anchorOut := normalizedVector(h.AnchorOutgoing.Forward(anchor), h.config.VectorEps)
	start = incoming.Slice(mlx.Slice(), mlx.Slice(0, 1), mlx.Slice(), mlx.Slice()).Squeeze(1).Multiply(anchorOut.ExpandDims(1)).SumAxis(-1, false)
	previous := outgoing.Slice(mlx.Slice(), mlx.Slice(0, slots-1), mlx.Slice(), mlx.Slice())
	next := incoming.Slice(mlx.Slice(), mlx.Slice(1, slots), mlx.Slice(), mlx.Slice()).Transpose(0, 1, 3, 2)
	pairs = previous.Matmul(next)
	return mlx.MulScalar(start.AsType(mlx.DTypeFloat32), h.config.LogitScale), mlx.MulScalar(pairs.AsType(mlx.DTypeFloat32), h.config.LogitScale)
}

// LiLiCorr context and correlator norms scale before rounding back to BF16.
func rmsNormFP32(x *mlx.Array, norm *nn.RMSNorm) *mlx.Array {
	return mlx.RMSNormFn(x.AsType(mlx.DTypeFloat32), norm.Weight.AsType(mlx.DTypeFloat32), norm.Eps).AsType(x.DType())
}

func normalizedVector(x *mlx.Array, eps float32) *mlx.Array {
	xf := x.AsType(mlx.DTypeFloat32)
	norm := xf.Multiply(xf).SumAxis(-1, true).Sqrt().AsType(x.DType())
	return x.Divide(mlx.Maximum(norm, mlx.FromValue(eps).AsType(x.DType())))
}

type lilicorrLoader struct {
	tensors map[string]*mlx.Array
	linears model.LinearFactory
	eps     float32
	err     error
}

func (l *lilicorrLoader) tensor(name string, shape ...int) *mlx.Array {
	t := l.tensors[name]
	if l.err == nil && (t == nil || !slices.Equal(t.Dims(), shape)) {
		l.err = fmt.Errorf("LiLiCorr %s: expected shape %v", name, shape)
	}
	return t
}

func (l *lilicorrLoader) norm(name string, width int) *nn.RMSNorm {
	return nn.NewRMSNorm(l.tensor(name+".weight", width), l.eps)
}

func (l *lilicorrLoader) linear(name string, input, output int) nn.LinearLayer {
	if l.err != nil {
		return nil
	}
	layer := l.linears.Make(name)
	width := 0
	switch q := layer.(type) {
	case *nn.Linear:
		if q.Weight.NumDims() == 2 {
			width = q.Weight.Dim(1)
		}
	case *nn.QuantizedLinear:
		if q.Weight.NumDims() == 2 && q.Bits > 0 {
			width = q.Weight.Dim(1) * 32 / q.Bits
		}
	}
	if layer == nil || width != input || int(layer.OutputDim()) != output {
		l.err = fmt.Errorf("LiLiCorr %s: expected projection [%d, %d]", name, output, input)
	}
	return layer
}

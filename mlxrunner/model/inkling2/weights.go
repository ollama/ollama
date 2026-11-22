package inkling2

import (
	"fmt"
	"slices"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

const textPrefix = "model.llm."

func (m *targetModel) LoadWeights(tensors map[string]*mlx.Array) error {
	reader := weightReader{tensors: tensors}
	c := m.config
	embedding := reader.tensor(textPrefix+"embed.weight", c.VocabSize, c.HiddenSize)
	output := reader.tensor(textPrefix+"unembed.weight", c.VocabSize, c.HiddenSize)
	m.EmbedNorm = reader.float32(textPrefix+"embed_norm.weight", c.HiddenSize)
	m.OutputNorm = reader.float32(textPrefix+"norm.weight", c.HiddenSize)
	m.OutputQuery = reader.tensor(textPrefix+"attn_res.output_query", c.HiddenSize)
	if reader.err != nil {
		return reader.err
	}
	m.Embed = nn.NewEmbedding(embedding)
	output = output.Slice(mlx.Slice(0, c.UnpaddedVocab), mlx.Slice())
	if m.fp32Output {
		// Short products read the BF16 head and accumulate in FP32.
		if mlx.MetalIsAvailable() {
			m.ShortOutput = nn.NewLinear(output, nil)
		}
		output = output.AsType(mlx.DTypeFloat32)
	}
	m.Output = nn.NewLinear(output, nil)

	var queries, kvQueries *mlx.Array
	if c.Loop != nil {
		rows := 2 * (c.Loop.End - c.Loop.Start)
		queries = reader.tensor(textPrefix+"middle_cycle_cross_pass_queries", rows, c.HiddenSize)
		kvQueries = reader.tensor(textPrefix+"middle_cycle_cross_pass_kv_queries", rows, c.HiddenSize)
		if reader.err != nil {
			return reader.err
		}
	}
	m.Layers = make([]decoderLayer, c.NumLayers)
	for _, visit := range m.plan {
		// One set of weights is shared by all visits of a physical layer.
		if visit.pass == 0 {
			m.Layers[visit.layer] = reader.layer(c, visit.layer, visit.local, queries, kvQueries)
		}
	}
	if m.Vision != nil && reader.err == nil {
		reader.vision(m.Vision, c.HiddenSize)
	}
	return reader.err
}

type weightReader struct {
	tensors map[string]*mlx.Array
	err     error
}

func (r *weightReader) tensor(name string, shape ...int) *mlx.Array {
	tensor := r.tensors[name]
	if r.err == nil {
		switch {
		case tensor == nil:
			r.err = fmt.Errorf("Inkling2 tensor %q is missing", name)
		case !slices.Equal(tensor.Dims(), shape):
			r.err = fmt.Errorf("Inkling2 tensor %q has shape %v, want %v", name, tensor.Dims(), shape)
		}
	}
	return tensor
}

// float32 returns a tensor that is used in FP32: norms, the router and its bias.
func (r *weightReader) float32(name string, shape ...int) *mlx.Array {
	weight := r.tensor(name, shape...)
	if r.err != nil {
		return weight
	}
	return weight.AsType(mlx.DTypeFloat32)
}

func (r *weightReader) linear(name string, out, in int) *nn.Linear {
	return nn.NewLinear(r.tensor(name, out, in), nil)
}

func (r *weightReader) layer(c textConfig, index int, local bool, queries, kvQueries *mlx.Array) decoderLayer {
	prefix := fmt.Sprintf("%slayers.%d.", textPrefix, index)
	attention := r.attention(c, prefix+"attn.", local)
	layer := decoderLayer{
		Attention:         attention,
		AttentionOutput:   r.shortConv(prefix+"attn_sconv", c.ConvTaps, c.HiddenSize, 0),
		FeedForwardNorm:   r.float32(prefix+"mlp_norm.weight", c.HiddenSize),
		FeedForwardQuery:  r.tensor(fmt.Sprintf("%sattn_res.mlp_queries.%d", textPrefix, index), c.HiddenSize),
		FeedForwardOutput: r.shortConv(prefix+"mlp_sconv", c.ConvTaps, c.HiddenSize, 0),
		FeedForward: routedExperts{
			Router:     r.float32(prefix+"mlp.gate.weight", c.Experts, c.HiddenSize),
			GateUp:     r.expertBank(prefix+"mlp.experts.w13_weight", c.Experts, 2*c.ExpertWidth, c.HiddenSize),
			Down:       r.expertBank(prefix+"mlp.experts.w2_weight", c.Experts, c.HiddenSize, c.ExpertWidth),
			topK:       c.TopK,
			routeScale: c.RouteScale,
			gateCap:    c.GateCap,
			upCap:      c.UpCap,
		},
	}
	if c.GateBias {
		layer.FeedForward.Bias = r.float32(prefix+"mlp.gate.bias", c.Experts)
	}
	if local {
		norm := r.float32(prefix+"attn_norm.weight", c.HiddenSize)
		layer.AttentionNorm = []*mlx.Array{norm, norm, norm}
	} else {
		layer.AttentionNorm = make([]*mlx.Array, 3)
		for i, projection := range []string{"q", "k", "v"} {
			layer.AttentionNorm[i] = r.float32(prefix+projection+"_input_norm.weight", c.HiddenSize)
		}
	}
	if r.err == nil {
		layer.AttentionNorms = mlx.Stack(layer.AttentionNorm, 0)
	}
	if index > 0 {
		shape := []int{3, c.HiddenSize}
		if local {
			shape = []int{c.HiddenSize}
		}
		layer.AttentionQuery = r.tensor(fmt.Sprintf("%sattn_res.attn_queries.%d", textPrefix, index-1), shape...)
	}
	if c.Loop != nil && index >= c.Loop.Start && index < c.Loop.End {
		layer.MessageProjections = make([]messageGate, 4)
		row := 2 * (index - c.Loop.Start)
		q := queries.Slice(mlx.Slice(row, row+1), mlx.Slice())
		kv := kvQueries.Slice(mlx.Slice(row, row+2), mlx.Slice())
		ffn := queries.Slice(mlx.Slice(row+1, row+2), mlx.Slice())
		layer.MessageQueries = mlx.Concatenate([]*mlx.Array{q, kv, ffn}, 0)
		var downs, gates []*mlx.Array
		for _, name := range []string{"attn", "k", "v", "mlp"} {
			stem := prefix + "controllers." + name + "."
			gain := r.float32(stem+"gain_weight", c.HiddenSize, c.Loop.GateRank)
			forget := r.float32(stem+"forget_weight", c.HiddenSize, c.Loop.GateRank)
			down := r.tensor(stem+"down_weight", c.Loop.GateRank, 2*c.HiddenSize)
			if r.err == nil {
				downs = append(downs, down)
				gates = append(gates, mlx.Concatenate([]*mlx.Array{gain, forget}, 0).Transpose(1, 0))
			}
		}
		if r.err == nil {
			layer.Controllers = newLoopControllers(mlx.Stack(downs, 0), mlx.Stack(gates, 0))
			for i := range layer.MessageProjections {
				layer.MessageProjections[i] = layer.Controllers.gate(i)
			}
		}
	}
	return layer
}

func (r *weightReader) attention(c textConfig, prefix string, local bool) selfAttention {
	dim, heads, kvHeads := c.HeadDim, c.Heads, c.KVHeads
	relativeRange, window, dynamicHeads := c.RelativeRange, 0, kvHeads
	if local {
		dim, heads, kvHeads = c.LocalHeadDim, c.LocalHeads, c.LocalKVHeads
		relativeRange, window, dynamicHeads = c.Window, c.Window, 0
	}
	width, kvWidth := dim*heads, dim*kvHeads
	query := r.tensor(prefix+"wq_du.weight", width, c.HiddenSize)
	relative := r.tensor(prefix+"wr_du.weight", heads*c.RelativeDim, c.HiddenSize)
	gate := r.tensor(prefix+"gate_attn.weight", width, c.HiddenSize)
	sink := r.tensor(prefix+"attn_sink_proj.weight", heads, c.HiddenSize)
	key := r.tensor(prefix+"wk_dv.weight", kvWidth, c.HiddenSize)
	value := r.tensor(prefix+"wv_dv.weight", kvWidth, c.HiddenSize)
	var keyTaps, valueTaps *mlx.Array
	if dynamicHeads > 0 {
		keyTaps = r.tensor(prefix+"k_sconv.weight_proj", dynamicHeads*c.ConvTaps, c.HiddenSize)
		valueTaps = r.tensor(prefix+"v_sconv.weight_proj", dynamicHeads*c.ConvTaps, c.HiddenSize)
	}
	a := selfAttention{
		Output:             r.linear(prefix+"wo_ud.weight", c.HiddenSize, width),
		PositionProjection: r.tensor(prefix+"rel_logits_proj.proj", c.RelativeDim, relativeRange),
		// The manifest loader maps the publisher's .scale suffix to _scale.
		QueryScale:    r.tensor(prefix+"q_l2_clip_scale", heads, dim),
		RelativeScale: r.tensor(prefix+"rel_l2_clip_scale", heads, c.RelativeDim),
		KeyConv:       r.shortConv(prefix+"k_sconv", c.ConvTaps, kvWidth, dynamicHeads),
		ValueConv:     r.shortConv(prefix+"v_sconv", c.ConvTaps, kvWidth, dynamicHeads),
		headDim:       dim, heads: heads, kvHeads: kvHeads, window: window,
		rmsEpsilon: c.NormEps, qkClip: c.QKClip, relativeClip: c.RelativeClip,
		logFloor: c.LogFloor, logAlpha: c.LogAlpha,
	}
	if r.err != nil {
		return a
	}
	groups := [][]*mlx.Array{{query, relative, gate, sink}, {key, keyTaps}, {value, valueTaps}}
	var stacked []*mlx.Array
	var ends [3]int
	for i, group := range groups {
		for _, w := range group {
			if w != nil {
				stacked = append(stacked, w)
				ends[i] += w.Dim(0)
			}
		}
		if i > 0 {
			ends[i] += ends[i-1]
		}
	}
	projection := mlx.Concatenate(stacked, 0)
	rows := func(start, end int) *nn.Linear {
		return nn.NewLinear(projection.Slice(mlx.Slice(start, end), mlx.Slice()), nil)
	}
	a.Projection = nn.NewLinear(projection, nil)
	a.QueryRows, a.KeyRows, a.ValueRows = rows(0, ends[0]), rows(ends[0], ends[1]), rows(ends[1], ends[2])
	return a
}

func (r *weightReader) shortConv(prefix string, taps, channels, heads int) shortConv {
	if heads > 0 {
		return shortConv{Weight: r.tensor(prefix+".static_w", taps, channels), heads: heads}
	}
	weight := r.tensor(prefix+".weight", channels, 1, taps)
	if r.err != nil {
		return shortConv{}
	}
	return shortConv{Weight: mlx.Contiguous(weight.Squeeze(1).Transpose(1, 0), false)}
}

func (r *weightReader) expertBank(name string, experts, rows, cols int) expertBank {
	weight := r.tensors[name]
	if weight == nil || weight.DType() != mlx.DTypeUint32 {
		return expertBank{Weight: r.tensor(name, experts, rows, cols)}
	}
	result := expertBank{}
	// The publisher's NVFP4 banks store packed E2M1 weights with an E4M3 scale per 16 columns in each row.
	// Each expert's outer weight scale is its amax divided by the bank-wide denominator.
	// MXFP8 banks have block scales and omit the per-expert maxima.
	if r.tensors[name+"_amax"] != nil {
		result.Weight = r.tensor(name, experts, rows, cols/8)
		result.BlockScales = r.tensor(name+"_scale", experts, rows, cols/16)
		result.Maxima = r.tensor(name+"_amax", experts)
		result.Denominator = r.tensor(name+"_outer_scale_denominator", 1)
	} else {
		result.Weight = r.tensor(name, experts, rows, cols/4)
		result.BlockScales = r.tensor(name+"_scale", experts, rows, cols/32)
	}
	return result
}

func (r *weightReader) vision(v *visionEncoder, textWidth int) {
	const prefix = "model.visual."
	c := v.config
	embedding := prefix + "backbone.embeddings."
	v.Patch = r.linear(embedding+"patch_embedding.weight", c.Width, c.Patch*c.Patch*3)
	v.Class = r.tensor(embedding+"class_embedding", c.Width)
	v.ClassOffset = r.tensor(embedding+"cls_position_embedding", c.Width)
	v.Positions = r.tensor(embedding+"position_embedding", c.Grid*c.Grid, c.Width)
	v.PreNorm = r.visionNorm(prefix+"backbone.ln_pre", c)
	v.Project = r.projection(prefix+"proj_in", c.Projector, c.Merge*c.Merge*c.Width, c.ProjBias)
	v.Out = r.projection(prefix+"proj_out", textWidth, c.Projector, c.ProjBias)
	v.Layers = make([]visionLayer, c.Layers)
	intermediate := int(float64(c.Width) * c.MLPRatio)
	for i := range v.Layers {
		layer := fmt.Sprintf("%sbackbone.layers.%d.", prefix, i)
		attn := layer + "self_attn."
		v.Layers[i] = visionLayer{
			Norm:      r.visionNorm(layer+"ln_1", c),
			FfNorm:    r.visionNorm(layer+"ln_2", c),
			Q:         r.projection(attn+"q_proj", c.Width, c.Width, true),
			K:         r.projection(attn+"k_proj", c.Width, c.Width, true),
			V:         r.projection(attn+"v_proj", c.Width, c.Width, true),
			O:         r.projection(attn+"out_proj", c.Width, c.Width, true),
			Up:        r.projection(layer+"mlp.c_fc", intermediate, c.Width, true),
			Down:      r.projection(layer+"mlp.c_proj", c.Width, intermediate, true),
			AttnScale: r.tensor(layer+"ls_1.gamma", c.Width),
			FfScale:   r.tensor(layer+"ls_2.gamma", c.Width),
		}
	}
}

func (r *weightReader) visionNorm(prefix string, config visionConfig) *nn.LayerNorm {
	return &nn.LayerNorm{
		Weight: r.tensor(prefix+".weight", config.Width),
		Bias:   r.tensor(prefix+".bias", config.Width),
		Eps:    config.Epsilon,
	}
}

func (r *weightReader) projection(prefix string, out, in int, withBias bool) *nn.Linear {
	weight := r.tensor(prefix+".weight", out, in)
	var bias *mlx.Array
	if withBias {
		bias = r.tensor(prefix+".bias", out)
	}
	if r.err != nil {
		return nil
	}
	return nn.NewLinear(weight, bias)
}

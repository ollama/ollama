package inkling2

import (
	"fmt"
	"slices"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

const textPrefix = "model.llm."

// normalizeMlxCommunityTensors detects and normalizes the mlx-community
// Inkling-2 tensor layout (model.language_model.* prefix, int4 affine
// projections, mlx-community NVFP4 expert banks) into the names and shapes
// that LoadWeights expects. Returns the normalized map and true when the
// source is an mlx-community model; returns nil, false otherwise.
func normalizeMlxCommunityTensors(tensors map[string]*mlx.Array) (map[string]*mlx.Array, bool) {
	if tensors["model.language_model.embed_tokens.weight"] == nil {
		return nil, false
	}
	out := make(map[string]*mlx.Array, len(tensors)*2)

	// dequantMaybe dequantizes an int4 affine tensor if its companions are
	// present. The manifest loader renames .scale → _scale and .bias → _qbias
	// (when a _scale sibling exists), so we look up those remapped names.
	dequantMaybe := func(name string) *mlx.Array {
		w := tensors[name]
		if w == nil {
			return nil
		}
		scale, bias := tensors[name+"_scale"], tensors[name+"_qbias"]
		if scale != nil && bias != nil {
			return mlx.Dequantize(w, scale, bias, 64, 4, "affine", nil).AsType(mlx.DTypeBFloat16)
		}
		return w
	}

	// Global tensors: embed/unembed use mlx-community int4 format.
	// The manifest loader renames .weight.scale → weight_scale and .weight.bias → weight_qbias,
	// so we look up those post-rename keys. Pass through as int4 so LoadWeights can create
	// QuantizedEmbedding/QuantizedLinear, saving ~1.56 GiB GPU vs dequantizing at load time.
	out[textPrefix+"embed.weight"] = tensors["model.language_model.embed_tokens.weight"]
	out[textPrefix+"embed.weight_scale"] = tensors["model.language_model.embed_tokens.weight_scale"]
	out[textPrefix+"embed.weight_qbias"] = tensors["model.language_model.embed_tokens.weight_qbias"]
	out[textPrefix+"unembed.weight"] = tensors["lm_head.weight"]
	out[textPrefix+"unembed.weight_scale"] = tensors["lm_head.weight_scale"]
	out[textPrefix+"unembed.weight_qbias"] = tensors["lm_head.weight_qbias"]
	out[textPrefix+"embed_norm.weight"] = tensors["model.language_model.embed_norm.weight"]
	out[textPrefix+"norm.weight"] = tensors["model.language_model.norm.weight"]
	out[textPrefix+"attn_res.output_query"] = tensors["model.language_model.final_input.query"]

	// Cross-pass loop queries from message_input.query [numLoopLayers, 4, hiddenSize].
	// Columns: 0=q_even, 1=kv1_even, 2=kv2_odd, 3=q_odd.
	if miq := tensors["model.language_model.message_input.query"]; miq != nil {
		c0 := miq.Slice(mlx.Slice(), mlx.Slice(0, 1), mlx.Slice())
		c1 := miq.Slice(mlx.Slice(), mlx.Slice(1, 2), mlx.Slice())
		c2 := miq.Slice(mlx.Slice(), mlx.Slice(2, 3), mlx.Slice())
		c3 := miq.Slice(mlx.Slice(), mlx.Slice(3, 4), mlx.Slice())
		h := int32(miq.Dim(2))
		// Interleave by concatenating on dim1 then reshaping: [N,1,H]+[N,1,H]→[N,2,H]→[2N,H]
		out[textPrefix+"middle_cycle_cross_pass_queries"] = mlx.Reshape(mlx.Concatenate([]*mlx.Array{c0, c3}, 1), -1, h)
		out[textPrefix+"middle_cycle_cross_pass_kv_queries"] = mlx.Reshape(mlx.Concatenate([]*mlx.Array{c1, c2}, 1), -1, h)
	}

	// Per-layer text tensors
	for n := 0; ; n++ {
		src := fmt.Sprintf("model.language_model.layers.%d.", n)
		if tensors[src+"post_attention_layernorm.weight"] == nil && tensors[src+"mlp.gate.weight"] == nil {
			break
		}
		dst := fmt.Sprintf("%slayers.%d.", textPrefix, n)
		attnSrc := src + "self_attn."
		attnDst := dst + "attn."

		// Layer norms
		if w := tensors[src+"input_layernorm.weight"]; w != nil {
			out[dst+"attn_norm.weight"] = w
		}
		out[dst+"mlp_norm.weight"] = tensors[src+"post_attention_layernorm.weight"]
		for _, kv := range []struct{ s, d string }{
			{"q_input_layernorm.weight", "q_input_norm.weight"},
			{"k_input_layernorm.weight", "k_input_norm.weight"},
			{"v_input_layernorm.weight", "v_input_norm.weight"},
		} {
			if w := tensors[src+kv.s]; w != nil {
				out[dst+kv.d] = w
			}
		}

		// Post-conv: [H, 4, 1] → [H, 1, 4] (safe byte-level relabel)
		for _, kv := range []struct{ s, d string }{
			{"post_attention_convolution.conv1d.weight", "attn_sconv.weight"},
			{"post_mlp_convolution.conv1d.weight", "mlp_sconv.weight"},
		} {
			if w := tensors[src+kv.s]; w != nil {
				out[dst+kv.d] = w.Reshape(w.Dim(0), 1, 4)
			}
		}

		// K/V sconv: local layers use conv1d [H,4,1]→[H,1,4]; global layers
		// use static_weight passthrough + weight_proj dequantize.
		for _, ch := range []string{"k", "v"} {
			convSrc := attnSrc + ch + "_sconv."
			convDst := attnDst + ch + "_sconv."
			if w := tensors[convSrc+"conv1d.weight"]; w != nil {
				out[convDst+"weight"] = w.Reshape(w.Dim(0), 1, 4)
			}
			if w := tensors[convSrc+"static_weight"]; w != nil {
				out[convDst+"static_w"] = w
			}
			if w := dequantMaybe(convSrc + "weight_proj.weight"); w != nil {
				out[convDst+"weight_proj"] = w
			}
		}

		// MLP router
		out[dst+"mlp.gate.weight"] = tensors[src+"mlp.gate.weight"]
		if b := tensors[src+"mlp.gate.e_score_correction_bias"]; b != nil {
			out[dst+"mlp.gate.bias"] = b
		}

		// Expert banks: weight (U32) + weight_scale (U8) + global_scale (F32).
		// The manifest loader renames .weight.scale → .weight_scale.
		for _, bank := range []struct{ s, d string }{
			{"gate_up_proj", "w13_weight"},
			{"down_proj", "w2_weight"},
		} {
			bSrc := src + "mlp.experts." + bank.s + "."
			bDst := dst + "mlp.experts." + bank.d
			if w := tensors[bSrc+"weight"]; w != nil {
				out[bDst] = w
			}
			if s := tensors[bSrc+"weight_scale"]; s != nil {
				out[bDst+"_scale"] = s
			}
			if g := tensors[bSrc+"global_scale"]; g != nil {
				out[bDst+"_global_scale"] = g
			}
		}

		// Attention projections: int4 affine → dequantize to BF16
		for _, kv := range []struct{ s, d string }{
			{"q_proj.weight", "wq_du.weight"},
			{"k_proj.weight", "wk_dv.weight"},
			{"v_proj.weight", "wv_dv.weight"},
			{"o_proj.weight", "wo_ud.weight"},
			{"r_proj.weight", "wr_du.weight"},
			{"gate_proj.weight", "gate_attn.weight"},
			{"sink_proj.weight", "attn_sink_proj.weight"},
		} {
			if w := dequantMaybe(attnSrc + kv.s); w != nil {
				out[attnDst+kv.d] = w
			}
		}

		// Clip scales: manifest loader renames .q_l2_clip.scale → .q_l2_clip_scale.
		for _, kv := range []struct{ s, d string }{
			{"q_l2_clip_scale", "q_l2_clip_scale"},
			{"rel_l2_clip_scale", "rel_l2_clip_scale"},
			{"rel_logits_proj.proj", "rel_logits_proj.proj"},
		} {
			if w := tensors[attnSrc+kv.s]; w != nil {
				out[attnDst+kv.d] = w
			}
		}

		// Per-layer queries
		if q := tensors[src+"mlp_input.query"]; q != nil {
			out[fmt.Sprintf("%sattn_res.mlp_queries.%d", textPrefix, n)] = q
		}
		if q := tensors[src+"attention_input.query"]; q != nil {
			out[fmt.Sprintf("%sattn_res.attn_queries.%d", textPrefix, n-1)] = q
		}

		// Loop controllers (present on loop layers only)
		for _, ctrl := range []struct{ s, d string }{
			{"q_message_gate", "attn"},
			{"k_message_gate", "k"},
			{"v_message_gate", "v"},
			{"mlp_message_gate", "mlp"},
		} {
			ctrlSrc := src + ctrl.s + "."
			ctrlDst := dst + "controllers." + ctrl.d + "."
			for _, gate := range []struct{ s, d string }{
				{"message_gate_proj.weight", "gain_weight"},
				{"forget_gate_proj.weight", "forget_weight"},
				{"down_proj.weight", "down_weight"},
			} {
				if w := tensors[ctrlSrc+gate.s]; w != nil {
					out[ctrlDst+gate.d] = w
				}
			}
		}
	}

	// Vision tensors
	vt := "model.vision_tower."
	vb := "model.visual.backbone."

	for _, kv := range []struct{ s, d string }{
		{vt + "embeddings.class_embedding", vb + "embeddings.class_embedding"},
		{vt + "embeddings.cls_position_embedding", vb + "embeddings.cls_position_embedding"},
		{vt + "embeddings.patch_embedding.weight", vb + "embeddings.patch_embedding.weight"},
		{vt + "embeddings.position_embedding.weight", vb + "embeddings.position_embedding"},
		{vt + "pre_layernorm.weight", vb + "ln_pre.weight"},
		{vt + "pre_layernorm.bias", vb + "ln_pre.bias"},
		{vt + "merger.linear_1.weight", "model.visual.proj_in.weight"},
		{vt + "merger.linear_2.weight", "model.visual.proj_out.weight"},
	} {
		if v := tensors[kv.s]; v != nil {
			out[kv.d] = v
		}
	}

	for n := 0; ; n++ {
		vsrc := fmt.Sprintf("%slayers.%d.", vt, n)
		if tensors[vsrc+"layernorm_before.weight"] == nil {
			break
		}
		vdst := fmt.Sprintf("%slayers.%d.", vb, n)
		vattn := vsrc + "self_attn."
		vdstAttn := vdst + "self_attn."

		for _, kv := range []struct{ s, d string }{
			{"layernorm_before.weight", "ln_1.weight"},
			{"layernorm_before.bias", "ln_1.bias"},
			{"layernorm_after.weight", "ln_2.weight"},
			{"layernorm_after.bias", "ln_2.bias"},
			{"lambda_1", "ls_1.gamma"},
			{"lambda_2", "ls_2.gamma"},
		} {
			if v := tensors[vsrc+kv.s]; v != nil {
				out[vdst+kv.d] = v
			}
		}

		for _, t := range []string{
			"q_proj.weight", "q_proj.bias",
			"k_proj.weight", "k_proj.bias",
			"v_proj.weight", "v_proj.bias",
			"out_proj.weight", "out_proj.bias",
		} {
			if v := tensors[vattn+t]; v != nil {
				out[vdstAttn+t] = v
			}
		}

		for _, kv := range []struct{ s, d string }{
			{"mlp.fc1.weight", "mlp.c_fc.weight"},
			{"mlp.fc1.bias", "mlp.c_fc.bias"},
			{"mlp.fc2.weight", "mlp.c_proj.weight"},
			{"mlp.fc2.bias", "mlp.c_proj.bias"},
		} {
			if v := tensors[vsrc+kv.s]; v != nil {
				out[vdst+kv.d] = v
			}
		}
	}

	return out, true
}

func (m *targetModel) LoadWeights(tensors map[string]*mlx.Array) error {
	if normalized, ok := normalizeMlxCommunityTensors(tensors); ok {
		tensors = normalized
	}
	reader := weightReader{tensors: tensors}
	c := m.config
	m.EmbedNorm = reader.float32(textPrefix+"embed_norm.weight", c.HiddenSize)
	m.OutputNorm = reader.float32(textPrefix+"norm.weight", c.HiddenSize)
	m.OutputQuery = reader.tensor(textPrefix+"attn_res.output_query", c.HiddenSize)
	if reader.err != nil {
		return reader.err
	}
	embW := tensors[textPrefix+"embed.weight"]
	if embW != nil && embW.DType() == mlx.DTypeUint32 {
		// mlx-community int4 affine: keep quantized to save ~1.56 GiB GPU memory
		embS := tensors[textPrefix+"embed.weight_scale"]
		embB := tensors[textPrefix+"embed.weight_qbias"]
		m.Embed = &nn.QuantizedEmbedding{Weight: embW, Scales: embS, QBiases: embB, GroupSize: 64, Bits: 4, Mode: "affine"}
		outW := tensors[textPrefix+"unembed.weight"]
		outS := tensors[textPrefix+"unembed.weight_scale"]
		outB := tensors[textPrefix+"unembed.weight_qbias"]
		if outW == nil || outS == nil || outB == nil {
			return fmt.Errorf("Inkling2 mlx-community unembed tensors incomplete (w=%v s=%v b=%v)", outW != nil, outS != nil, outB != nil)
		}
		outW = outW.Slice(mlx.Slice(0, c.UnpaddedVocab), mlx.Slice())
		outS = outS.Slice(mlx.Slice(0, c.UnpaddedVocab), mlx.Slice())
		outB = outB.Slice(mlx.Slice(0, c.UnpaddedVocab), mlx.Slice())
		m.Output = &nn.QuantizedLinear{Weight: outW, Scales: outS, QBiases: outB, GroupSize: 64, Bits: 4, Mode: "affine"}
	} else {
		embedding := reader.tensor(textPrefix+"embed.weight", c.VocabSize, c.HiddenSize)
		output := reader.tensor(textPrefix+"unembed.weight", c.VocabSize, c.HiddenSize)
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
	}

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
		// Publisher NVFP4: packed U8 weights, E4M3 block scale, per-expert amax + denominator.
		result.Weight = r.tensor(name, experts, rows, cols/8)
		result.BlockScales = r.tensor(name+"_scale", experts, rows, cols/16)
		result.Maxima = r.tensor(name+"_amax", experts)
		result.Denominator = r.tensor(name+"_outer_scale_denominator", 1)
	} else if r.tensors[name+"_global_scale"] != nil {
		// mlx-community NVFP4: packed U32 weights, U8 block scale, per-expert global_scale.
		// Store global_scale as Maxima and [Nvfp4MaxProduct] as synthetic Denominator so that
		// nvfp4OuterScale produces activationMax * global_scale / (Nvfp4MaxProduct * nvfp4ActivationDenominator).
		result.Weight = r.tensor(name, experts, rows, cols/8)
		result.BlockScales = r.tensor(name+"_scale", experts, rows, cols/16)
		result.Maxima = r.tensor(name+"_global_scale", experts)
		result.Denominator = mlx.FromValue(float32(mlx.Nvfp4MaxProduct)).Reshape(1)
	} else {
		// MXFP8
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

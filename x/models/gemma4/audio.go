package gemma4

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
)

type AudioConfig struct {
	ModelType         string  `json:"model_type"`
	HiddenSize        int32   `json:"hidden_size"`
	NumHiddenLayers   int32   `json:"num_hidden_layers"`
	NumAttentionHeads int32   `json:"num_attention_heads"`
	ConvKernelSize    int32   `json:"conv_kernel_size"`
	ResidualWeight    float32 `json:"residual_weight"`
	ChunkSize         int32   `json:"attention_chunk_size"`
	ContextLeft       int32   `json:"attention_context_left"`
	ContextRight      int32   `json:"attention_context_right"`
	LogitCap          float32 `json:"attention_logit_cap"`
	InvalidLogit      float32 `json:"attention_invalid_logits_value"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	GradientClipping  float32 `json:"gradient_clipping"`

	// Unified-embedder field.
	SamplesPerToken int32 `json:"audio_samples_per_token"`
}

func (a *AudioConfig) unified() bool {
	return a.ModelType == "gemma4_unified_audio"
}

// applyAudioDefaults fills absent fields with the reference config defaults.
func applyAudioDefaults(a *AudioConfig) {
	if a.HiddenSize == 0 {
		a.HiddenSize = 1024
	}
	if a.NumHiddenLayers == 0 {
		a.NumHiddenLayers = 12
	}
	if a.NumAttentionHeads == 0 {
		a.NumAttentionHeads = 8
	}
	if a.ConvKernelSize == 0 {
		a.ConvKernelSize = 5
	}
	if a.ResidualWeight == 0 {
		a.ResidualWeight = 0.5
	}
	if a.ChunkSize == 0 {
		a.ChunkSize = 12
	}
	if a.ContextLeft == 0 {
		a.ContextLeft = 13
	}
	if a.LogitCap == 0 {
		a.LogitCap = 50
	}
	if a.InvalidLogit == 0 {
		a.InvalidLogit = -1e9
	}
	if a.RMSNormEps == 0 {
		a.RMSNormEps = 1e-6
	}
	if a.GradientClipping == 0 {
		a.GradientClipping = 1e10
	}
}

// preparedAudio carries one chunk's soft-token count to softRun.
type preparedAudio struct {
	numTokens int32
}

// AudioTower is the Gemma 4 audio encoder: a USM conformer over log-mel
// frames, subsampled 4x in time by two stride-2 convs.
type AudioTower struct {
	SubsampleConv []*AudioSubsampleConv
	InputProj     nn.LinearLayer
	Layers        []*AudioLayer
	OutputProj    nn.LinearLayer

	// Gradient-clipping bounds in the working dtype.
	ClipMin, ClipMax *mlx.Array
}

type AudioSubsampleConv struct {
	Conv *mlx.Array // [out, 3, 3, in]
	Norm *mlx.Array // LayerNorm weight, no bias
}

type AudioLayer struct {
	FFW1, FFW2   *AudioFeedForward
	Attention    *AudioAttention
	LConv        *AudioLightConv
	PreAttnNorm  *mlx.Array
	PostAttnNorm *mlx.Array
	OutNorm      *mlx.Array
}

type AudioFeedForward struct {
	PreNorm  *mlx.Array
	Up, Down nn.LinearLayer
	PostNorm *mlx.Array
}

type AudioAttention struct {
	QProj, KProj, VProj, Post nn.LinearLayer

	// QScale is softplus(per_dim_scale) in f32; RelK is the sinusoidal
	// relative-position table through this layer's relative_k_proj,
	// [1, heads, headDim, positions] in f32.
	QScale *mlx.Array
	RelK   *mlx.Array
}

type AudioLightConv struct {
	PreNorm  *mlx.Array
	Start    nn.LinearLayer
	DWConv   *mlx.Array // [hidden, kernel, 1]
	ConvNorm *mlx.Array
	End      nn.LinearLayer
}

// audioRelPosTable builds the [positions, hidden] sinusoidal table over
// relative positions contextSize/2 down to 0, each row [sin..., cos...].
func audioRelPosTable(a *AudioConfig) *mlx.Array {
	positions := int(a.ChunkSize+a.ContextLeft-1+a.ContextRight)/2 + 1
	hidden := int(a.HiddenSize)
	half := hidden / 2
	increment := math.Log(10000) / float64(half-1)

	table := make([]float32, positions*hidden)
	for i := range positions {
		p := float64(positions - 1 - i)
		for d := range half {
			angle := p * math.Exp(-float64(d)*increment)
			table[i*hidden+d] = float32(math.Sin(angle))
			table[i*hidden+half+d] = float32(math.Cos(angle))
		}
	}
	return mlx.FromValues(table, positions, hidden)
}

// audioWeightRoot locates the tower's tensor prefix by probing for the
// subsample conv projection.
func audioWeightRoot(tensors map[string]*mlx.Array) (string, error) {
	for _, root := range []string{"model.", ""} {
		if tensors[root+"audio_tower.subsample_conv_projection.layer0.conv.weight"] != nil {
			return root, nil
		}
	}
	return "", fmt.Errorf("config declares a gemma4_audio tower but audio_tower weights are missing from the manifest")
}

func (m *Model) loadAudioWeights(tensors map[string]*mlx.Array, linears model.LinearFactory) error {
	if m.Audio.unified() {
		return m.loadUnifiedAudioWeights(tensors, linears)
	}
	a := m.Audio
	root, err := audioWeightRoot(tensors)
	if err != nil {
		return err
	}
	at := root + "audio_tower."

	requiredNorm := func(name string) (*mlx.Array, error) {
		w := tensors[name]
		if w == nil {
			return nil, fmt.Errorf("missing audio weight: %s", name)
		}
		return w, nil
	}

	tower := &AudioTower{
		SubsampleConv: make([]*AudioSubsampleConv, 2),
		Layers:        make([]*AudioLayer, a.NumHiddenLayers),
	}
	for i := range tower.SubsampleConv {
		sp := fmt.Sprintf("%ssubsample_conv_projection.layer%d.", at, i)
		conv := tensors[sp+"conv.weight"]
		if conv == nil {
			return fmt.Errorf("missing audio weight: %sconv.weight", sp)
		}
		norm, err := requiredNorm(sp + "norm.weight")
		if err != nil {
			return err
		}
		// Conv weights arrive [out, in, kH, kW]; mlx wants channels last.
		tower.SubsampleConv[i] = &AudioSubsampleConv{Conv: mlx.Transpose(conv, 0, 2, 3, 1), Norm: norm}
	}
	if tower.InputProj, err = makeClippableLinear(linears, tensors, at+"subsample_conv_projection.input_proj_linear"); err != nil {
		return err
	}

	table := audioRelPosTable(a)
	headDim := a.HiddenSize / a.NumAttentionHeads
	positions := int32(table.Dim(0))
	for i := range tower.Layers {
		lp := fmt.Sprintf("%slayers.%d.", at, i)
		layer := &AudioLayer{
			FFW1:      &AudioFeedForward{},
			FFW2:      &AudioFeedForward{},
			Attention: &AudioAttention{},
			LConv:     &AudioLightConv{},
		}

		for _, ffw := range []struct {
			f    *AudioFeedForward
			name string
		}{{layer.FFW1, "feed_forward1."}, {layer.FFW2, "feed_forward2."}} {
			fp := lp + ffw.name
			if ffw.f.PreNorm, err = requiredNorm(fp + "pre_layer_norm.weight"); err != nil {
				return err
			}
			if ffw.f.Up, err = makeClippableLinear(linears, tensors, fp+"ffw_layer_1"); err != nil {
				return err
			}
			if ffw.f.Down, err = makeClippableLinear(linears, tensors, fp+"ffw_layer_2"); err != nil {
				return err
			}
			if ffw.f.PostNorm, err = requiredNorm(fp + "post_layer_norm.weight"); err != nil {
				return err
			}
		}

		attn := layer.Attention
		if attn.QProj, err = makeClippableLinear(linears, tensors, lp+"self_attn.q_proj"); err != nil {
			return err
		}
		if attn.KProj, err = makeClippableLinear(linears, tensors, lp+"self_attn.k_proj"); err != nil {
			return err
		}
		if attn.VProj, err = makeClippableLinear(linears, tensors, lp+"self_attn.v_proj"); err != nil {
			return err
		}
		if attn.Post, err = makeClippableLinear(linears, tensors, lp+"self_attn.post"); err != nil {
			return err
		}
		perDimScale, err := requiredNorm(lp + "self_attn.per_dim_scale")
		if err != nil {
			return err
		}
		relProj, err := makeClippableLinear(linears, tensors, lp+"self_attn.relative_k_proj")
		if err != nil {
			return err
		}
		// The reference applies softplus and the rel-position projection in
		// the checkpoint dtype and lifts the results to f32 with q.
		attn.QScale = mlx.Softplus(perDimScale).AsType(mlx.DTypeFloat32)
		relK := relProj.Forward(table.AsType(mlx.DTypeBFloat16))
		relK = mlx.Reshape(relK, positions, a.NumAttentionHeads, headDim)
		relK = mlx.ExpandDims(mlx.Transpose(relK, 1, 2, 0), 0)
		attn.RelK = relK.AsType(mlx.DTypeFloat32)

		lc := layer.LConv
		if lc.PreNorm, err = requiredNorm(lp + "lconv1d.pre_layer_norm.weight"); err != nil {
			return err
		}
		if lc.Start, err = makeClippableLinear(linears, tensors, lp+"lconv1d.linear_start"); err != nil {
			return err
		}
		dw := tensors[lp+"lconv1d.depthwise_conv1d.weight"]
		if dw == nil {
			return fmt.Errorf("missing audio weight: %slconv1d.depthwise_conv1d.weight", lp)
		}
		lc.DWConv = mlx.Transpose(dw, 0, 2, 1)
		if lc.ConvNorm, err = requiredNorm(lp + "lconv1d.conv_norm.weight"); err != nil {
			return err
		}
		if lc.End, err = makeClippableLinear(linears, tensors, lp+"lconv1d.linear_end"); err != nil {
			return err
		}

		if layer.PreAttnNorm, err = requiredNorm(lp + "norm_pre_attn.weight"); err != nil {
			return err
		}
		if layer.PostAttnNorm, err = requiredNorm(lp + "norm_post_attn.weight"); err != nil {
			return err
		}
		if layer.OutNorm, err = requiredNorm(lp + "norm_out.weight"); err != nil {
			return err
		}
		tower.Layers[i] = layer
	}

	if tower.OutputProj, err = makeClippableLinear(linears, tensors, at+"output_proj"); err != nil {
		return err
	}
	projection, err := makeClippableLinear(linears, tensors, root+"embed_audio.embedding_projection")
	if err != nil {
		return err
	}

	tower.ClipMin = mlx.FromValue(-a.GradientClipping).AsType(mlx.DTypeBFloat16)
	tower.ClipMax = mlx.FromValue(a.GradientClipping).AsType(mlx.DTypeBFloat16)

	m.AudioTower = tower
	m.EmbedAudio = &MultimodalEmbedder{Projection: projection}
	return nil
}

// loadUnifiedAudioWeights loads the encoder-free variant, whose only audio
// weight is the embedding projection.
func (m *Model) loadUnifiedAudioWeights(tensors map[string]*mlx.Array, linears model.LinearFactory) error {
	for _, root := range []string{"model.", ""} {
		if tensors[root+"embed_audio.embedding_projection.weight"] == nil {
			continue
		}
		projection, err := makeClippableLinear(linears, tensors, root+"embed_audio.embedding_projection")
		if err != nil {
			return err
		}
		m.EmbedAudio = &MultimodalEmbedder{Projection: projection}
		return nil
	}
	return fmt.Errorf("config declares a gemma4_unified_audio embedder but embed_audio weights are missing from the manifest")
}

func (m *Model) audioLoaded() bool {
	return m.EmbedAudio != nil
}

func (t *AudioTower) clip(x *mlx.Array) *mlx.Array {
	return mlx.Clip(x, t.ClipMin, t.ClipMax)
}

// audioAttentionMask builds the boolean [blocks, 1, chunk, context] mask
// over n frames: block b's context slot j holds frame b*chunk + j -
// (ContextLeft-1), attendable when it exists and lies within the window.
func (m *Model) audioAttentionMask(n int) *mlx.Array {
	a := m.Audio
	chunk, left, right := int(a.ChunkSize), int(a.ContextLeft-1), int(a.ContextRight)
	ctx := chunk + left + right
	blocks := (n + chunk - 1) / chunk

	mask := make([]bool, blocks*chunk*ctx)
	i := 0
	for b := range blocks {
		for q := range chunk {
			gq := b*chunk + q
			for j := range ctx {
				gk := b*chunk + j - left
				dist := gq - gk
				mask[i] = gq < n && gk >= 0 && gk < n &&
					((dist >= 0 && dist < left) || (dist < 0 && -dist < right))
				i++
			}
		}
	}
	return mlx.FromValues(mask, blocks, 1, chunk, ctx)
}

// audioContextIndices maps block b's context slot j to index b*chunk+j of
// the padded key/value sequence.
func (m *Model) audioContextIndices(blocks int) *mlx.Array {
	a := m.Audio
	ctx := int(a.ChunkSize + a.ContextLeft - 1 + a.ContextRight)
	idx := make([]int32, blocks*ctx)
	for b := range blocks {
		for j := range ctx {
			idx[b*ctx+j] = int32(b*int(a.ChunkSize) + j)
		}
	}
	return mlx.FromValues(idx, blocks*ctx)
}

// The literal 1e-6 eps here and in AudioLayer.Forward mirrors the
// reference, which passes config eps only to the lconv and subsample norms.
func (f *AudioFeedForward) Forward(t *AudioTower, h *mlx.Array, a *AudioConfig) *mlx.Array {
	c := t.clip(h)
	c = mlx.RMSNormFn(c, f.PreNorm, 1e-6)
	c = f.Up.Forward(c)
	c = mlx.SiLU(c)
	c = f.Down.Forward(c)
	c = t.clip(c)
	c = mlx.RMSNormFn(c, f.PostNorm, 1e-6)
	return mlx.Add(h, mlx.MulScalar(c, a.ResidualWeight))
}

// Forward computes chunked local attention with relative positions over the
// pre-normed [n, hidden] frames: q/k/v lift to f32, queries block into
// chunks, keys and values gather each block's context window, and the
// rel-position scores shift from position- to offset-indexed before the
// softcap, mask, and softmax.
func (at *AudioAttention) Forward(t *AudioTower, h, mask, indices *mlx.Array, a *AudioConfig) *mlx.Array {
	n := h.Dim(0)
	heads := a.NumAttentionHeads
	headDim := a.HiddenSize / heads
	chunk, left, right := int(a.ChunkSize), int(a.ContextLeft-1), int(a.ContextRight)
	ctx := chunk + left + right
	blocks := (n + chunk - 1) / chunk

	shape := []int32{int32(n), heads, headDim}
	q := mlx.Reshape(at.QProj.Forward(h).AsType(mlx.DTypeFloat32), shape...)
	k := mlx.Reshape(at.KProj.Forward(h).AsType(mlx.DTypeFloat32), shape...)
	v := mlx.Reshape(at.VProj.Forward(h).AsType(mlx.DTypeFloat32), shape...)

	q = mlx.MulScalar(q, float32(1/(math.Sqrt(float64(headDim))*math.Ln2)))
	q = mlx.Mul(q, at.QScale)
	k = mlx.MulScalar(k, float32(math.Log(1+math.E)/math.Ln2))

	q = mlx.PadConstant(q, []int{0}, []int{0}, []int{blocks*chunk - n})
	qb := mlx.Transpose(mlx.Reshape(q, int32(blocks), int32(chunk), heads, headDim), 0, 2, 1, 3)

	pad := []int{blocks*chunk - n + right}
	kb := gatherContext(k, indices, left, pad, int32(blocks), int32(ctx), heads, headDim)
	vb := gatherContext(v, indices, left, pad, int32(blocks), int32(ctx), heads, headDim)

	ac := mlx.Matmul(qb, mlx.Transpose(kb, 0, 1, 3, 2))
	bd := mlx.Matmul(qb, at.RelK)

	// Rel shift: [.., chunk, positions] -> [.., chunk, context], turning
	// per-position columns into per-offset columns.
	positions := bd.Dim(3)
	bd = mlx.PadConstant(bd, []int{3}, []int{0}, []int{ctx + 1 - positions})
	bd = mlx.Reshape(bd, int32(blocks), heads, int32(chunk*(ctx+1)))
	bd = bd.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, chunk*ctx))
	bd = mlx.Reshape(bd, int32(blocks), heads, int32(chunk), int32(ctx))

	logits := mlx.Add(ac, bd)
	logits = mlx.LogitSoftcap(logits, mlx.FromValue(a.LogitCap))
	logits = mlx.Where(mask, logits, mlx.FromValue(a.InvalidLogit))

	weights := mlx.SoftmaxAxis(logits, -1, false)
	o := mlx.Matmul(weights, vb)
	o = mlx.Transpose(o, 0, 2, 1, 3)
	o = mlx.Reshape(o, int32(blocks*chunk), a.HiddenSize)
	o = o.Slice(mlx.Slice(0, n), mlx.Slice())
	return at.Post.Forward(o.AsType(mlx.DTypeBFloat16))
}

// gatherContext pads the [n, heads, headDim] sequence so index b*chunk+j
// addresses frame b*chunk+j-left, then gathers the per-block context
// windows into [blocks, heads, context, headDim].
func gatherContext(x, indices *mlx.Array, left int, highPad []int, blocks, ctx, heads, headDim int32) *mlx.Array {
	x = mlx.PadConstant(x, []int{0}, []int{left}, highPad)
	x = mlx.Take(x, indices, 0)
	x = mlx.Reshape(x, blocks, ctx, heads, headDim)
	return mlx.Transpose(x, 0, 2, 1, 3)
}

func (lc *AudioLightConv) Forward(t *AudioTower, h *mlx.Array, a *AudioConfig) *mlx.Array {
	n, hidden := int32(h.Dim(0)), a.HiddenSize
	c := mlx.RMSNormFn(h, lc.PreNorm, a.RMSNormEps)
	c = lc.Start.Forward(c)
	c = mlx.Mul(
		c.Slice(mlx.Slice(), mlx.Slice(0, int(hidden))),
		mlx.Sigmoid(c.Slice(mlx.Slice(), mlx.Slice(int(hidden), int(2*hidden)))))

	// Causal depthwise conv: left-pad time by kernel-1.
	c = mlx.Reshape(c, 1, n, hidden)
	c = mlx.PadConstant(c, []int{1}, []int{int(a.ConvKernelSize) - 1}, []int{0})
	c = mlx.Conv1d(c, lc.DWConv, nil, 1, 0, 1, hidden)
	c = mlx.Reshape(c, n, hidden)

	c = t.clip(c)
	c = mlx.RMSNormFn(c, lc.ConvNorm, a.RMSNormEps)
	c = mlx.SiLU(c)
	c = lc.End.Forward(c)
	return mlx.Add(h, c)
}

func (l *AudioLayer) Forward(t *AudioTower, h, mask, indices *mlx.Array, a *AudioConfig) *mlx.Array {
	h = l.FFW1.Forward(t, h, a)

	c := t.clip(h)
	c = mlx.RMSNormFn(c, l.PreAttnNorm, 1e-6)
	c = l.Attention.Forward(t, c, mask, indices, a)
	c = t.clip(c)
	c = mlx.RMSNormFn(c, l.PostAttnNorm, 1e-6)
	h = mlx.Add(h, c)

	h = l.LConv.Forward(t, h, a)
	h = l.FFW2.Forward(t, h, a)

	h = t.clip(h)
	return mlx.RMSNormFn(h, l.OutNorm, 1e-6)
}

// encodeAudio runs the conformer over one chunk's [frames, melBins] log-mel
// features, returning the lazy [numTokens, hidden] features.
func (m *Model) encodeAudio(data *mlx.Array) *mlx.Array {
	a := m.Audio
	t := m.AudioTower
	frames := data.Dim(0)

	h := mlx.Reshape(data, 1, int32(frames), audioMelBins, 1).AsType(mlx.DTypeBFloat16)
	for _, l := range t.SubsampleConv {
		h = mlx.Conv2d(h, l.Conv, 2, 2, 1, 1, 1, 1, 1)
		h = mlx.LayerNormFn(h, l.Norm, nil, a.RMSNormEps)
		h = mlx.ReLU(h)
	}
	n, w, c := h.Dim(1), h.Dim(2), h.Dim(3)
	h = mlx.Reshape(h, int32(n), int32(w*c))
	h = t.InputProj.Forward(h)

	mask := m.audioAttentionMask(n)
	indices := m.audioContextIndices((n + int(a.ChunkSize) - 1) / int(a.ChunkSize))
	for _, layer := range t.Layers {
		h = layer.Forward(t, h, mask, indices, a)
	}

	h = t.OutputProj.Forward(h)
	h = mlx.RMSNormFn(h, nil, a.RMSNormEps)
	return m.EmbedAudio.Projection.Forward(h)
}

// encodeUnifiedAudio projects raw waveform frames straight into the text
// embedding space; the unified variant has no audio tower.
func (m *Model) encodeUnifiedAudio(data *mlx.Array) *mlx.Array {
	h := data.AsType(mlx.DTypeBFloat16)
	h = mlx.RMSNormFn(h, nil, m.Audio.RMSNormEps)
	return m.EmbedAudio.Projection.Forward(h)
}

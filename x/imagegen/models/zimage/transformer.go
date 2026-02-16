//go:build mlx

// Package zimage implements the Z-Image diffusion transformer model.
package zimage

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// TransformerConfig holds Z-Image transformer configuration
type TransformerConfig struct {
	Dim            int32   `json:"dim"`
	NHeads         int32   `json:"n_heads"`
	NKVHeads       int32   `json:"n_kv_heads"`
	NLayers        int32   `json:"n_layers"`
	NRefinerLayers int32   `json:"n_refiner_layers"`
	InChannels     int32   `json:"in_channels"`
	PatchSize      int32   `json:"-"` // Computed from AllPatchSize
	CapFeatDim     int32   `json:"cap_feat_dim"`
	NormEps        float32 `json:"norm_eps"`
	RopeTheta      float32 `json:"rope_theta"`
	TScale         float32 `json:"t_scale"`
	QKNorm         bool    `json:"qk_norm"`
	AxesDims       []int32 `json:"axes_dims"`
	AxesLens       []int32 `json:"axes_lens"`
	AllPatchSize   []int32 `json:"all_patch_size"` // JSON array, PatchSize = first element
}

// TimestepEmbedder creates sinusoidal timestep embeddings
// Output dimension is 256 (fixed), used for AdaLN modulation
type TimestepEmbedder struct {
	Linear1       nn.LinearLayer `weight:"mlp.0"`
	Linear2       nn.LinearLayer `weight:"mlp.2"`
	FreqEmbedSize int32      // 256 (computed)
}

// Forward computes timestep embeddings -> [B, 256]
func (te *TimestepEmbedder) Forward(t *mlx.Array) *mlx.Array {
	// t: [B] timesteps

	// Create sinusoidal embedding
	half := te.FreqEmbedSize / 2

	// freqs = exp(-log(10000) * arange(half) / half)
	freqs := make([]float32, half)
	for i := int32(0); i < half; i++ {
		freqs[i] = float32(math.Exp(-math.Log(10000.0) * float64(i) / float64(half)))
	}
	freqsArr := mlx.NewArray(freqs, []int32{1, half})

	// t[:, None] * freqs[None, :] -> [B, half]
	tExpanded := mlx.ExpandDims(t, 1) // [B, 1]
	args := mlx.Mul(tExpanded, freqsArr)

	// embedding = [cos(args), sin(args)] -> [B, 256]
	cosArgs := mlx.Cos(args)
	sinArgs := mlx.Sin(args)
	embedding := mlx.Concatenate([]*mlx.Array{cosArgs, sinArgs}, 1)

	// MLP: linear1 -> silu -> linear2
	h := te.Linear1.Forward(embedding)
	h = mlx.SiLU(h)
	h = te.Linear2.Forward(h)

	return h
}

// XEmbedder embeds image patches to model dimension
type XEmbedder struct {
	Linear nn.LinearLayer `weight:"2-1"`
}

// Forward embeds patchified image latents
func (xe *XEmbedder) Forward(x *mlx.Array) *mlx.Array {
	// x: [B, L, in_channels * 4] -> [B, L, dim]
	return xe.Linear.Forward(x)
}

// CapEmbedder projects caption features to model dimension
type CapEmbedder struct {
	Norm     *nn.RMSNorm `weight:"0"`
	Linear   nn.LinearLayer  `weight:"1"`
	PadToken *mlx.Array  // loaded separately at root level
}

// Forward projects caption embeddings: [B, L, cap_feat_dim] -> [B, L, dim]
func (ce *CapEmbedder) Forward(capFeats *mlx.Array) *mlx.Array {
	// RMSNorm on last axis (uses 1e-6)
	h := ce.Norm.Forward(capFeats, 1e-6)
	// Linear projection
	return ce.Linear.Forward(h)
}

// FeedForward implements SwiGLU FFN
type FeedForward struct {
	W1     nn.LinearLayer `weight:"w1"` // gate projection
	W2     nn.LinearLayer `weight:"w2"` // down projection
	W3     nn.LinearLayer `weight:"w3"` // up projection
	OutDim int32      // computed from W2
}


// Forward applies SwiGLU: silu(W1(x)) * W3(x), then W2
func (ff *FeedForward) Forward(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	D := shape[2]

	// Reshape for matmul
	x = mlx.Reshape(x, B*L, D)

	gate := ff.W1.Forward(x)
	gate = mlx.SiLU(gate)
	up := ff.W3.Forward(x)
	h := mlx.Mul(gate, up)
	out := ff.W2.Forward(h)

	return mlx.Reshape(out, B, L, ff.OutDim)
}

// Attention implements multi-head attention with QK norm
type Attention struct {
	ToQ   nn.LinearLayer `weight:"to_q"`
	ToK   nn.LinearLayer `weight:"to_k"`
	ToV   nn.LinearLayer `weight:"to_v"`
	ToOut nn.LinearLayer `weight:"to_out.0"`
	NormQ *mlx.Array `weight:"norm_q.weight"` // [head_dim] for per-head RMSNorm
	NormK *mlx.Array `weight:"norm_k.weight"`
	// Fused QKV (computed at init time for efficiency, not loaded from weights)
	ToQKV nn.LinearLayer `weight:"-"` // Fused Q+K+V projection (created by FuseQKV)
	Fused bool       `weight:"-"` // Whether to use fused QKV path
	// Computed fields (not loaded from weights)
	NHeads  int32   `weight:"-"`
	HeadDim int32   `weight:"-"`
	Dim     int32   `weight:"-"`
	Scale   float32 `weight:"-"`
}

// FuseQKV creates a fused QKV projection by concatenating weights.
// This reduces 3 matmuls to 1 for a ~5-10% speedup.
// Note: Fusion is skipped for quantized weights as it would require complex
// dequant-concat-requant operations. The FP8 memory bandwidth savings outweigh
// the ~5% fusion benefit.
func (attn *Attention) FuseQKV() {
	if attn.ToQ == nil || attn.ToK == nil || attn.ToV == nil {
		return
	}

	// Skip fusion for quantized weights - type assert to check
	toQ, qOk := attn.ToQ.(*nn.Linear)
	toK, kOk := attn.ToK.(*nn.Linear)
	toV, vOk := attn.ToV.(*nn.Linear)
	if !qOk || !kOk || !vOk {
		// One or more are QuantizedLinear, skip fusion
		return
	}

	if toQ.Weight == nil || toK.Weight == nil || toV.Weight == nil {
		return
	}

	// Concatenate weights: [dim, dim] x 3 -> [3*dim, dim]
	// Weight shapes: ToQ.Weight [out_dim, in_dim], etc.
	qWeight := toQ.Weight
	kWeight := toK.Weight
	vWeight := toV.Weight

	// Concatenate along output dimension (axis 0)
	fusedWeight := mlx.Concatenate([]*mlx.Array{qWeight, kWeight, vWeight}, 0)

	// Evaluate fused weight to ensure it's materialized
	mlx.Eval(fusedWeight)

	// Create fused linear layer
	fusedLinear := &nn.Linear{Weight: fusedWeight}

	// Handle bias if present
	if toQ.Bias != nil && toK.Bias != nil && toV.Bias != nil {
		fusedBias := mlx.Concatenate([]*mlx.Array{toQ.Bias, toK.Bias, toV.Bias}, 0)
		mlx.Eval(fusedBias)
		fusedLinear.Bias = fusedBias
	}

	attn.ToQKV = fusedLinear
	attn.Fused = true
}

// Forward computes attention
func (attn *Attention) Forward(x *mlx.Array, cos, sin *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	D := shape[2]

	xFlat := mlx.Reshape(x, B*L, D)

	var q, k, v *mlx.Array
	if attn.Fused && attn.ToQKV != nil {
		// Fused QKV path: single matmul then split
		qkv := attn.ToQKV.Forward(xFlat) // [B*L, 3*dim]

		// Split into Q, K, V along last dimension
		// Each has shape [B*L, dim]
		q = mlx.Slice(qkv, []int32{0, 0}, []int32{B * L, attn.Dim})
		k = mlx.Slice(qkv, []int32{0, attn.Dim}, []int32{B * L, 2 * attn.Dim})
		v = mlx.Slice(qkv, []int32{0, 2 * attn.Dim}, []int32{B * L, 3 * attn.Dim})
	} else {
		// Separate Q, K, V projections
		q = attn.ToQ.Forward(xFlat)
		k = attn.ToK.Forward(xFlat)
		v = attn.ToV.Forward(xFlat)
	}

	// Reshape to [B, L, nheads, head_dim]
	q = mlx.Reshape(q, B, L, attn.NHeads, attn.HeadDim)
	k = mlx.Reshape(k, B, L, attn.NHeads, attn.HeadDim)
	v = mlx.Reshape(v, B, L, attn.NHeads, attn.HeadDim)

	// QK norm
	q = mlx.RMSNorm(q, attn.NormQ, 1e-5)
	k = mlx.RMSNorm(k, attn.NormK, 1e-5)

	// Apply RoPE if provided
	if cos != nil && sin != nil {
		q = applyRoPE3D(q, cos, sin)
		k = applyRoPE3D(k, cos, sin)
	}

	// Transpose to [B, nheads, L, head_dim]
	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	// SDPA
	out := mlx.ScaledDotProductAttention(q, k, v, attn.Scale, false)

	// Transpose back and reshape
	out = mlx.Transpose(out, 0, 2, 1, 3)
	out = mlx.Reshape(out, B*L, attn.Dim)
	out = attn.ToOut.Forward(out)

	return mlx.Reshape(out, B, L, attn.Dim)
}

// applyRoPE3D applies 3-axis rotary position embeddings
// x: [B, L, nheads, head_dim]
// cos, sin: [B, L, 1, head_dim/2]
func applyRoPE3D(x *mlx.Array, cos, sin *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	nheads := shape[2]
	headDim := shape[3]
	half := headDim / 2

	// Create even/odd index arrays
	evenIdx := make([]int32, half)
	oddIdx := make([]int32, half)
	for i := int32(0); i < half; i++ {
		evenIdx[i] = i * 2
		oddIdx[i] = i*2 + 1
	}
	evenIndices := mlx.NewArrayInt32(evenIdx, []int32{half})
	oddIndices := mlx.NewArrayInt32(oddIdx, []int32{half})

	// Extract x1 (even indices) and x2 (odd indices) along last axis
	x1 := mlx.Take(x, evenIndices, 3) // [B, L, nheads, half]
	x2 := mlx.Take(x, oddIndices, 3)  // [B, L, nheads, half]

	// Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
	r1 := mlx.Sub(mlx.Mul(x1, cos), mlx.Mul(x2, sin))
	r2 := mlx.Add(mlx.Mul(x1, sin), mlx.Mul(x2, cos))

	// Stack and reshape to interleave: [r1_0, r2_0, r1_1, r2_1, ...]
	r1 = mlx.ExpandDims(r1, 4)                          // [B, L, nheads, half, 1]
	r2 = mlx.ExpandDims(r2, 4)                          // [B, L, nheads, half, 1]
	stacked := mlx.Concatenate([]*mlx.Array{r1, r2}, 4) // [B, L, nheads, half, 2]
	return mlx.Reshape(stacked, B, L, nheads, headDim)
}

// TransformerBlock is a single transformer block with optional AdaLN modulation
type TransformerBlock struct {
	Attention      *Attention   `weight:"attention"`
	FeedForward    *FeedForward `weight:"feed_forward"`
	AttentionNorm1 *nn.RMSNorm  `weight:"attention_norm1"`
	AttentionNorm2 *nn.RMSNorm  `weight:"attention_norm2"`
	FFNNorm1       *nn.RMSNorm  `weight:"ffn_norm1"`
	FFNNorm2       *nn.RMSNorm  `weight:"ffn_norm2"`
	AdaLN          nn.LinearLayer   `weight:"adaLN_modulation.0,optional"` // only if modulation
	// Computed fields
	HasModulation bool
	Dim           int32
}

// Forward applies the transformer block
func (tb *TransformerBlock) Forward(x *mlx.Array, adaln *mlx.Array, cos, sin *mlx.Array, eps float32) *mlx.Array {
	if tb.AdaLN != nil && adaln != nil {
		// Compute modulation: [B, 256] -> [B, 4*dim]
		chunks := tb.AdaLN.Forward(adaln)

		// Split into 4 parts: scale_msa, gate_msa, scale_mlp, gate_mlp
		chunkShape := chunks.Shape()
		chunkDim := chunkShape[1] / 4

		scaleMSA := mlx.Slice(chunks, []int32{0, 0}, []int32{chunkShape[0], chunkDim})
		gateMSA := mlx.Slice(chunks, []int32{0, chunkDim}, []int32{chunkShape[0], chunkDim * 2})
		scaleMLP := mlx.Slice(chunks, []int32{0, chunkDim * 2}, []int32{chunkShape[0], chunkDim * 3})
		gateMLP := mlx.Slice(chunks, []int32{0, chunkDim * 3}, []int32{chunkShape[0], chunkDim * 4})

		// Expand for broadcasting: [B, 1, dim]
		scaleMSA = mlx.ExpandDims(scaleMSA, 1)
		gateMSA = mlx.ExpandDims(gateMSA, 1)
		scaleMLP = mlx.ExpandDims(scaleMLP, 1)
		gateMLP = mlx.ExpandDims(gateMLP, 1)

		// Attention with modulation
		normX := tb.AttentionNorm1.Forward(x, eps)
		normX = mlx.Mul(normX, mlx.AddScalar(scaleMSA, 1.0))
		attnOut := tb.Attention.Forward(normX, cos, sin)
		attnOut = tb.AttentionNorm2.Forward(attnOut, eps)
		x = mlx.Add(x, mlx.Mul(mlx.Tanh(gateMSA), attnOut))

		// FFN with modulation
		normFFN := tb.FFNNorm1.Forward(x, eps)
		normFFN = mlx.Mul(normFFN, mlx.AddScalar(scaleMLP, 1.0))
		ffnOut := tb.FeedForward.Forward(normFFN)
		ffnOut = tb.FFNNorm2.Forward(ffnOut, eps)
		x = mlx.Add(x, mlx.Mul(mlx.Tanh(gateMLP), ffnOut))
	} else {
		// No modulation (context refiner)
		attnOut := tb.Attention.Forward(tb.AttentionNorm1.Forward(x, eps), cos, sin)
		x = mlx.Add(x, tb.AttentionNorm2.Forward(attnOut, eps))

		ffnOut := tb.FeedForward.Forward(tb.FFNNorm1.Forward(x, eps))
		x = mlx.Add(x, tb.FFNNorm2.Forward(ffnOut, eps))
	}

	return x
}

// FinalLayer outputs the denoised patches
type FinalLayer struct {
	AdaLN  nn.LinearLayer `weight:"adaLN_modulation.1"` // [256] -> [dim]
	Output nn.LinearLayer `weight:"linear"`             // [dim] -> [out_channels]
	OutDim int32      // computed from Output
}

// Forward computes final output
func (fl *FinalLayer) Forward(x *mlx.Array, c *mlx.Array) *mlx.Array {
	// c: [B, 256] -> scale: [B, dim]
	scale := mlx.SiLU(c)
	scale = fl.AdaLN.Forward(scale)
	scale = mlx.ExpandDims(scale, 1) // [B, 1, dim]

	// LayerNorm (affine=False) then scale
	x = layerNormNoAffine(x, 1e-6)
	x = mlx.Mul(x, mlx.AddScalar(scale, 1.0))

	// Output projection
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	D := shape[2]
	x = mlx.Reshape(x, B*L, D)
	x = fl.Output.Forward(x)

	return mlx.Reshape(x, B, L, fl.OutDim)
}

// layerNormNoAffine applies layer norm without learnable parameters
func layerNormNoAffine(x *mlx.Array, eps float32) *mlx.Array {
	ndim := x.Ndim()
	lastAxis := ndim - 1

	mean := mlx.Mean(x, lastAxis, true)
	xCentered := mlx.Sub(x, mean)
	variance := mlx.Mean(mlx.Square(xCentered), lastAxis, true)
	return mlx.Div(xCentered, mlx.Sqrt(mlx.AddScalar(variance, eps)))
}

// Transformer is the full Z-Image DiT model
type Transformer struct {
	TEmbed          *TimestepEmbedder   `weight:"t_embedder"`
	XEmbed          *XEmbedder          `weight:"all_x_embedder"`
	CapEmbed        *CapEmbedder        `weight:"cap_embedder"`
	NoiseRefiners   []*TransformerBlock `weight:"noise_refiner"`
	ContextRefiners []*TransformerBlock `weight:"context_refiner"`
	Layers          []*TransformerBlock `weight:"layers"`
	FinalLayer      *FinalLayer         `weight:"all_final_layer.2-1"`
	XPadToken       *mlx.Array          `weight:"x_pad_token"`
	CapPadToken     *mlx.Array          `weight:"cap_pad_token"`
	*TransformerConfig
}

// Load loads the Z-Image transformer from ollama blob storage.
func (m *Transformer) Load(manifest *imagegen.ModelManifest) error {
	fmt.Print("  Loading transformer... ")

	// Load config from blob
	var cfg TransformerConfig
	if err := manifest.ReadConfigJSON("transformer/config.json", &cfg); err != nil {
		return fmt.Errorf("config: %w", err)
	}
	if len(cfg.AllPatchSize) > 0 {
		cfg.PatchSize = cfg.AllPatchSize[0]
	}
	m.TransformerConfig = &cfg
	m.NoiseRefiners = make([]*TransformerBlock, cfg.NRefinerLayers)
	m.ContextRefiners = make([]*TransformerBlock, cfg.NRefinerLayers)
	m.Layers = make([]*TransformerBlock, cfg.NLayers)

	weights, err := imagegen.LoadWeightsFromManifest(manifest, "transformer")
	if err != nil {
		return fmt.Errorf("weights: %w", err)
	}
	if err := weights.Load(0); err != nil {
		return fmt.Errorf("load weights: %w", err)
	}
	defer weights.ReleaseAll()

	return m.loadWeights(weights)
}

// loadWeights loads weights from any WeightSource into the model
func (m *Transformer) loadWeights(weights safetensors.WeightSource) error {
	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return fmt.Errorf("load module: %w", err)
	}
	m.initComputedFields()
	fmt.Println("✓")
	return nil
}

// initComputedFields initializes computed fields after loading weights
func (m *Transformer) initComputedFields() {
	cfg := m.TransformerConfig
	m.TEmbed.FreqEmbedSize = 256
	m.FinalLayer.OutDim = m.FinalLayer.Output.OutputDim()
	m.CapEmbed.Norm.Eps = 1e-6

	for _, block := range m.NoiseRefiners {
		initTransformerBlock(block, cfg)
	}
	for _, block := range m.ContextRefiners {
		initTransformerBlock(block, cfg)
	}
	for _, block := range m.Layers {
		initTransformerBlock(block, cfg)
	}
}

// FuseAllQKV fuses QKV projections in all attention layers for efficiency.
// This reduces 3 matmuls to 1 per attention layer, providing ~5-10% speedup.
func (m *Transformer) FuseAllQKV() {
	for _, block := range m.NoiseRefiners {
		block.Attention.FuseQKV()
	}
	for _, block := range m.ContextRefiners {
		block.Attention.FuseQKV()
	}
	for _, block := range m.Layers {
		block.Attention.FuseQKV()
	}
}

// initTransformerBlock sets computed fields on a transformer block
func initTransformerBlock(block *TransformerBlock, cfg *TransformerConfig) {
	block.Dim = cfg.Dim
	block.HasModulation = block.AdaLN != nil

	// Init attention computed fields
	attn := block.Attention
	attn.NHeads = cfg.NHeads
	attn.HeadDim = cfg.Dim / cfg.NHeads
	attn.Dim = cfg.Dim
	attn.Scale = float32(1.0 / math.Sqrt(float64(attn.HeadDim)))

	// Init feedforward OutDim
	block.FeedForward.OutDim = block.FeedForward.W2.OutputDim()

	// Set eps on all RMSNorm layers
	block.AttentionNorm1.Eps = cfg.NormEps
	block.AttentionNorm2.Eps = cfg.NormEps
	block.FFNNorm1.Eps = cfg.NormEps
	block.FFNNorm2.Eps = cfg.NormEps
}

// RoPECache holds precomputed RoPE values
type RoPECache struct {
	ImgCos     *mlx.Array
	ImgSin     *mlx.Array
	CapCos     *mlx.Array
	CapSin     *mlx.Array
	UnifiedCos *mlx.Array
	UnifiedSin *mlx.Array
	ImgLen     int32
	CapLen     int32
	GridH      int32 // Image token grid height
	GridW      int32 // Image token grid width
}

// PrepareRoPECache precomputes RoPE values for the given image and caption lengths.
// hTok and wTok are the number of tokens in each dimension (latentH/patchSize, latentW/patchSize).
func (m *Transformer) PrepareRoPECache(hTok, wTok, capLen int32) *RoPECache {
	imgLen := hTok * wTok

	// Image positions: grid over (1, H, W) starting at (capLen+1, 0, 0)
	imgPos := createCoordinateGrid(1, hTok, wTok, capLen+1, 0, 0)
	imgPos = mlx.ToBFloat16(imgPos)
	// Caption positions: grid over (capLen, 1, 1) starting at (1, 0, 0)
	capPos := createCoordinateGrid(capLen, 1, 1, 1, 0, 0)
	capPos = mlx.ToBFloat16(capPos)

	// Compute RoPE from UNIFIED positions
	unifiedPos := mlx.Concatenate([]*mlx.Array{imgPos, capPos}, 1)
	unifiedCos, unifiedSin := prepareRoPE3D(unifiedPos, m.TransformerConfig.AxesDims)

	// Slice RoPE for image and caption parts
	imgCos := mlx.Slice(unifiedCos, []int32{0, 0, 0, 0}, []int32{1, imgLen, 1, 64})
	imgSin := mlx.Slice(unifiedSin, []int32{0, 0, 0, 0}, []int32{1, imgLen, 1, 64})
	capCos := mlx.Slice(unifiedCos, []int32{0, imgLen, 0, 0}, []int32{1, imgLen + capLen, 1, 64})
	capSin := mlx.Slice(unifiedSin, []int32{0, imgLen, 0, 0}, []int32{1, imgLen + capLen, 1, 64})

	return &RoPECache{
		ImgCos:     imgCos,
		ImgSin:     imgSin,
		CapCos:     capCos,
		CapSin:     capSin,
		UnifiedCos: unifiedCos,
		UnifiedSin: unifiedSin,
		ImgLen:     imgLen,
		CapLen:     capLen,
		GridH:      hTok,
		GridW:      wTok,
	}
}

// Forward runs the Z-Image transformer with precomputed RoPE
func (m *Transformer) Forward(x *mlx.Array, t *mlx.Array, capFeats *mlx.Array, rope *RoPECache) *mlx.Array {
	imgLen := rope.ImgLen

	// Timestep embedding -> [B, 256]
	temb := m.TEmbed.Forward(mlx.MulScalar(t, m.TransformerConfig.TScale))

	// Embed image patches -> [B, L_img, dim]
	x = m.XEmbed.Forward(x)

	// Embed caption features -> [B, L_cap, dim]
	capEmb := m.CapEmbed.Forward(capFeats)

	eps := m.NormEps

	// Noise refiner: refine image patches with modulation
	for _, refiner := range m.NoiseRefiners {
		x = refiner.Forward(x, temb, rope.ImgCos, rope.ImgSin, eps)
	}

	// Context refiner: refine caption (no modulation)
	for _, refiner := range m.ContextRefiners {
		capEmb = refiner.Forward(capEmb, nil, rope.CapCos, rope.CapSin, eps)
	}

	// Concatenate image and caption for joint attention
	unified := mlx.Concatenate([]*mlx.Array{x, capEmb}, 1)

	// Main transformer layers use full unified RoPE
	for _, layer := range m.Layers {
		unified = layer.Forward(unified, temb, rope.UnifiedCos, rope.UnifiedSin, eps)
	}

	// Extract image tokens only
	unifiedShape := unified.Shape()
	B := unifiedShape[0]
	imgOut := mlx.Slice(unified, []int32{0, 0, 0}, []int32{B, imgLen, unifiedShape[2]})

	// Final layer
	return m.FinalLayer.Forward(imgOut, temb)
}

// ForwardWithCache runs the transformer with layer caching for faster inference.
// On refresh steps (step % cacheInterval == 0), all layers are computed and cached.
// On other steps, shallow layers (0 to cacheLayers-1) reuse cached outputs.
func (m *Transformer) ForwardWithCache(
	x *mlx.Array,
	t *mlx.Array,
	capFeats *mlx.Array,
	rope *RoPECache,
	stepCache *cache.StepCache,
	step int,
	cacheInterval int,
) *mlx.Array {
	imgLen := rope.ImgLen
	cacheLayers := stepCache.NumLayers()
	eps := m.NormEps

	// Timestep embedding -> [B, 256]
	temb := m.TEmbed.Forward(mlx.MulScalar(t, m.TransformerConfig.TScale))

	// Embed image patches -> [B, L_img, dim]
	x = m.XEmbed.Forward(x)

	// Context refiners: compute once on step 0, reuse forever
	// (caption embedding doesn't depend on timestep or latents)
	var capEmb *mlx.Array
	if stepCache.GetConstant() != nil {
		capEmb = stepCache.GetConstant()
	} else {
		capEmb = m.CapEmbed.Forward(capFeats)
		for _, refiner := range m.ContextRefiners {
			capEmb = refiner.Forward(capEmb, nil, rope.CapCos, rope.CapSin, eps)
		}
		stepCache.SetConstant(capEmb)
	}

	// Noise refiners: always compute (depend on x which changes each step)
	for _, refiner := range m.NoiseRefiners {
		x = refiner.Forward(x, temb, rope.ImgCos, rope.ImgSin, eps)
	}

	// Concatenate image and caption for joint attention
	unified := mlx.Concatenate([]*mlx.Array{x, capEmb}, 1)

	// Determine if this is a cache refresh step
	refreshCache := stepCache.ShouldRefresh(step, cacheInterval)

	// Main transformer layers with caching
	for i, layer := range m.Layers {
		if i < cacheLayers && !refreshCache && stepCache.Get(i) != nil {
			// Use cached output for shallow layers
			unified = stepCache.Get(i)
		} else {
			// Compute layer
			unified = layer.Forward(unified, temb, rope.UnifiedCos, rope.UnifiedSin, eps)
			// Cache shallow layer outputs on refresh steps
			if i < cacheLayers && refreshCache {
				stepCache.Set(i, unified)
			}
		}
	}

	// Extract image tokens only
	unifiedShape := unified.Shape()
	B := unifiedShape[0]
	imgOut := mlx.Slice(unified, []int32{0, 0, 0}, []int32{B, imgLen, unifiedShape[2]})

	// Final layer
	return m.FinalLayer.Forward(imgOut, temb)
}

// createCoordinateGrid creates 3D position grid [1, d0*d1*d2, 3]
func createCoordinateGrid(d0, d1, d2, s0, s1, s2 int32) *mlx.Array {
	// Create meshgrid and stack
	total := d0 * d1 * d2
	coords := make([]float32, total*3)

	idx := 0
	for i := int32(0); i < d0; i++ {
		for j := int32(0); j < d1; j++ {
			for k := int32(0); k < d2; k++ {
				coords[idx*3+0] = float32(s0 + i)
				coords[idx*3+1] = float32(s1 + j)
				coords[idx*3+2] = float32(s2 + k)
				idx++
			}
		}
	}

	return mlx.NewArray(coords, []int32{1, total, 3})
}

// prepareRoPE3D computes cos/sin for 3-axis RoPE
// positions: [B, L, 3] with (h, w, t) coordinates
// axesDims: [32, 48, 48] - dimensions for each axis
// Returns: cos, sin each [B, L, 1, head_dim/2]
func prepareRoPE3D(positions *mlx.Array, axesDims []int32) (*mlx.Array, *mlx.Array) {
	// Compute frequencies for each axis
	// dims = [32, 48, 48], so halves = [16, 24, 24]
	ropeTheta := float32(256.0)

	freqs := make([]*mlx.Array, 3)
	for axis := 0; axis < 3; axis++ {
		half := axesDims[axis] / 2
		f := make([]float32, half)
		for i := int32(0); i < half; i++ {
			f[i] = float32(math.Exp(-math.Log(float64(ropeTheta)) * float64(i) / float64(half)))
		}
		freqs[axis] = mlx.NewArray(f, []int32{1, 1, 1, half})
	}

	// Extract position coordinates
	shape := positions.Shape()
	B := shape[0]
	L := shape[1]

	// positions[:, :, 0] -> h positions
	posH := mlx.Slice(positions, []int32{0, 0, 0}, []int32{B, L, 1})
	posW := mlx.Slice(positions, []int32{0, 0, 1}, []int32{B, L, 2})
	posT := mlx.Slice(positions, []int32{0, 0, 2}, []int32{B, L, 3})

	// Compute args: pos * freqs for each axis
	posH = mlx.ExpandDims(posH, 3) // [B, L, 1, 1]
	posW = mlx.ExpandDims(posW, 3)
	posT = mlx.ExpandDims(posT, 3)

	argsH := mlx.Mul(posH, freqs[0]) // [B, L, 1, 16]
	argsW := mlx.Mul(posW, freqs[1]) // [B, L, 1, 24]
	argsT := mlx.Mul(posT, freqs[2]) // [B, L, 1, 24]

	// Concatenate: [B, L, 1, 16+24+24=64]
	args := mlx.Concatenate([]*mlx.Array{argsH, argsW, argsT}, 3)

	// Compute cos and sin
	return mlx.Cos(args), mlx.Sin(args)
}

// PatchifyLatents converts latents [B, C, H, W] to patches [B, L, C*patch^2]
// Matches Python: x.reshape(C, 1, 1, H_tok, 2, W_tok, 2).transpose(1,2,3,5,4,6,0).reshape(1,-1,C*4)
func PatchifyLatents(latents *mlx.Array, patchSize int32) *mlx.Array {
	shape := latents.Shape()
	C := shape[1]
	H := shape[2]
	W := shape[3]

	pH := H / patchSize // H_tok
	pW := W / patchSize // W_tok

	// Match Python exactly: reshape treating B=1 as part of contiguous data
	// [1, C, H, W] -> [C, 1, 1, pH, 2, pW, 2]
	x := mlx.Reshape(latents, C, 1, 1, pH, patchSize, pW, patchSize)

	// Python: transpose(1, 2, 3, 5, 4, 6, 0)
	// [C, 1, 1, pH, 2, pW, 2] -> [1, 1, pH, pW, 2, 2, C]
	x = mlx.Transpose(x, 1, 2, 3, 5, 4, 6, 0)

	// [1, 1, pH, pW, 2, 2, C] -> [1, pH*pW, C*4]
	return mlx.Reshape(x, 1, pH*pW, C*patchSize*patchSize)
}

// UnpatchifyLatents converts patches [B, L, C*patch^2] back to [B, C, H, W]
// Matches Python: out.reshape(1,1,H_tok,W_tok,2,2,C).transpose(6,0,1,2,4,3,5).reshape(1,C,H,W)
func UnpatchifyLatents(patches *mlx.Array, patchSize, H, W, C int32) *mlx.Array {
	pH := H / patchSize
	pW := W / patchSize

	// [1, L, C*4] -> [1, 1, pH, pW, 2, 2, C]
	x := mlx.Reshape(patches, 1, 1, pH, pW, patchSize, patchSize, C)

	// Python: transpose(6, 0, 1, 2, 4, 3, 5)
	// [1, 1, pH, pW, 2, 2, C] -> [C, 1, 1, pH, 2, pW, 2]
	x = mlx.Transpose(x, 6, 0, 1, 2, 4, 3, 5)

	// [C, 1, 1, pH, 2, pW, 2] -> [1, C, H, W]
	return mlx.Reshape(x, 1, C, H, W)
}

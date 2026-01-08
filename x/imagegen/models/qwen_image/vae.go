//go:build mlx

package qwen_image

import (
	"fmt"
	"math"
	"path/filepath"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// VAEConfig holds Qwen-Image VAE configuration
type VAEConfig struct {
	ZDim               int32     `json:"z_dim"`               // 16
	BaseDim            int32     `json:"base_dim"`            // 96
	DimMult            []int32   `json:"dim_mult"`            // [1, 2, 4, 4]
	NumResBlocks       int32     `json:"num_res_blocks"`      // 2
	LatentsMean        []float32 `json:"latents_mean"`        // 16 values
	LatentsStd         []float32 `json:"latents_std"`         // 16 values
	TemperalDownsample []bool    `json:"temperal_downsample"` // [false, true, true]
}

// defaultVAEConfig returns config for Qwen-Image VAE
func defaultVAEConfig() *VAEConfig {
	return &VAEConfig{
		ZDim:         16,
		BaseDim:      96,
		DimMult:      []int32{1, 2, 4, 4},
		NumResBlocks: 2,
		LatentsMean: []float32{
			-0.7571, -0.7089, -0.9113, 0.1075,
			-0.1745, 0.9653, -0.1517, 1.5508,
			0.4134, -0.0715, 0.5517, -0.3632,
			-0.1922, -0.9497, 0.2503, -0.2921,
		},
		LatentsStd: []float32{
			2.8184, 1.4541, 2.3275, 2.6558,
			1.2196, 1.7708, 2.6052, 2.0743,
			3.2687, 2.1526, 2.8652, 1.5579,
			1.6382, 1.1253, 2.8251, 1.916,
		},
		TemperalDownsample: []bool{false, true, true},
	}
}

// CausalConv3d is a causal 3D convolution (for temporal causality)
type CausalConv3d struct {
	Weight       *mlx.Array
	Bias         *mlx.Array
	BiasReshaped *mlx.Array // [1, C, 1, 1, 1]
	KernelT      int32
}

// newCausalConv3d creates a 3D causal conv
func newCausalConv3d(weights *safetensors.ModelWeights, prefix string) (*CausalConv3d, error) {
	weight, err := weights.Get(prefix + ".weight")
	if err != nil {
		return nil, fmt.Errorf("weight not found: %s", prefix)
	}
	bias, _ := weights.Get(prefix + ".bias")

	kernelT := weight.Shape()[2]
	outC := weight.Shape()[0]

	var biasReshaped *mlx.Array
	if bias != nil {
		biasReshaped = mlx.Reshape(bias, 1, outC, 1, 1, 1)
	}

	return &CausalConv3d{
		Weight:       weight,
		Bias:         bias,
		BiasReshaped: biasReshaped,
		KernelT:      kernelT,
	}, nil
}

// Forward applies causal 3D convolution
// x: [B, T, H, W, C] (channels-last, MLX format)
func (c *CausalConv3d) Forward(x *mlx.Array) *mlx.Array {
	shape := c.Weight.Shape() // PyTorch format: [O, I, kT, kH, kW]
	kernelT := shape[2]
	kernelH := shape[3]
	kernelW := shape[4]

	// Causal temporal padding, same spatial padding
	// Input is channels-last: [B, T, H, W, C]
	padT := kernelT - 1
	padH := kernelH / 2
	padW := kernelW / 2

	// Stage 1: Pad
	{
			x = pad3DChannelsLast(x, padT, 0, padH, padH, padW, padW)
		mlx.Eval(x)
	}

	// Stage 2: Conv + bias
	var out *mlx.Array
	{
			prev := x
		weight := mlx.Transpose(c.Weight, 0, 2, 3, 4, 1)
		out = mlx.Conv3d(x, weight, 1, 1, 1, 0, 0, 0)
		if c.Bias != nil {
			bias := mlx.Reshape(c.Bias, 1, 1, 1, 1, c.Bias.Dim(0))
			out = mlx.Add(out, bias)
		}
		prev.Free()
		mlx.Eval(out)
	}

	return out
}

// RMSNorm3D applies RMS normalization over channels
// Works with channels-last [B, T, H, W, C] format
type RMSNorm3D struct {
	Gamma *mlx.Array // [1, 1, 1, 1, C] for broadcasting
}

// newRMSNorm3D creates an RMS norm
func newRMSNorm3D(weights *safetensors.ModelWeights, prefix string, dim int32) (*RMSNorm3D, error) {
	gamma, err := weights.Get(prefix + ".gamma")
	if err != nil {
		return nil, err
	}
	// Reshape for channels-last broadcasting: [1, 1, 1, 1, C]
	gamma = mlx.Reshape(gamma, 1, 1, 1, 1, gamma.Dim(0))
	return &RMSNorm3D{Gamma: gamma}, nil
}

// Forward applies RMS norm to channels-last input [B, T, H, W, C]
func (n *RMSNorm3D) Forward(x *mlx.Array) *mlx.Array {
	// RMSNorm: x * rsqrt(mean(x^2) + eps) * gamma
	normalized := mlx.RMSNormNoWeight(x, 1e-6)
	return mlx.Mul(normalized, n.Gamma)
}

// ResBlock is a residual block with RMS norm and causal convs
type ResBlock struct {
	Norm1    *RMSNorm3D
	Conv1    *CausalConv3d
	Norm2    *RMSNorm3D
	Conv2    *CausalConv3d
	Shortcut *CausalConv3d
}

// newResBlock creates a residual block
func newResBlock(weights *safetensors.ModelWeights, prefix string, inDim, outDim int32) (*ResBlock, error) {
	norm1, err := newRMSNorm3D(weights, prefix+".norm1", inDim)
	if err != nil {
		return nil, err
	}
	conv1, err := newCausalConv3d(weights, prefix+".conv1")
	if err != nil {
		return nil, err
	}
	norm2, err := newRMSNorm3D(weights, prefix+".norm2", outDim)
	if err != nil {
		return nil, err
	}
	conv2, err := newCausalConv3d(weights, prefix+".conv2")
	if err != nil {
		return nil, err
	}

	var shortcut *CausalConv3d
	if inDim != outDim {
		shortcut, err = newCausalConv3d(weights, prefix+".conv_shortcut")
		if err != nil {
			return nil, err
		}
	}

	return &ResBlock{
		Norm1:    norm1,
		Conv1:    conv1,
		Norm2:    norm2,
		Conv2:    conv2,
		Shortcut: shortcut,
	}, nil
}

// Forward applies the residual block
func (r *ResBlock) Forward(x *mlx.Array) *mlx.Array {
	// Use h as working variable, keep x intact for residual (caller will free x)
	// Conv handles its own pools, so we just need pools for non-conv operations
	var h *mlx.Array

	// Keep x so it survives Eval() cleanup - needed for residual connection
	mlx.Keep(x)

	// Stage 1: norm1 + silu
	{
			h = r.Norm1.Forward(x)
		h = silu3D(h)
		mlx.Eval(h)
	}

	// Stage 2: conv1 (handles its own pools)
	{
		prev := h
		h = r.Conv1.Forward(h)
		prev.Free()
	}

	// Stage 3: norm2 + silu
	{
			prev := h
		h = r.Norm2.Forward(h)
		h = silu3D(h)
		prev.Free()
		mlx.Eval(h)
	}

	// Stage 4: conv2 (handles its own pools)
	{
		prev := h
		h = r.Conv2.Forward(h)
		prev.Free()
	}

	// Residual connection (shortcut handles its own pools if present)
	if r.Shortcut != nil {
		shortcut := r.Shortcut.Forward(x)
			h = mlx.Add(h, shortcut)
		mlx.Eval(h)
	} else {
			h = mlx.Add(h, x)
		mlx.Eval(h)
	}

	return h
}

// AttentionBlock is a 2D attention block
type AttentionBlock struct {
	Norm      *RMSNorm3D
	ToQKV     *mlx.Array
	ToQKVBias *mlx.Array
	Proj      *mlx.Array
	ProjBias  *mlx.Array
	Dim       int32
}

// newAttentionBlock creates an attention block
func newAttentionBlock(weights *safetensors.ModelWeights, prefix string, dim int32) (*AttentionBlock, error) {
	norm, err := newRMSNorm3D(weights, prefix+".norm", dim)
	if err != nil {
		return nil, err
	}
	toQKV, _ := weights.Get(prefix + ".to_qkv.weight")
	toQKVBias, _ := weights.Get(prefix + ".to_qkv.bias")
	proj, _ := weights.Get(prefix + ".proj.weight")
	projBias, _ := weights.Get(prefix + ".proj.bias")

	return &AttentionBlock{
		Norm:      norm,
		ToQKV:     toQKV,
		ToQKVBias: toQKVBias,
		Proj:      proj,
		ProjBias:  projBias,
		Dim:       dim,
	}, nil
}

// Forward applies 2D attention
// Input: [B, T, H, W, C] (channels-last)
func (a *AttentionBlock) Forward(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	T := shape[1]
	H := shape[2]
	W := shape[3]
	C := shape[4]

	identity := x

	// Flatten to [B*T, 1, H, W, C] for norm
	x = mlx.Reshape(x, B*T, 1, H, W, C)
	x = a.Norm.Forward(x)
	x = mlx.Reshape(x, B*T, H, W, C)

	// Flatten spatial to [B*T, H*W, C]
	x = mlx.Reshape(x, B*T, H*W, C)

	// Linear to get Q, K, V: [B*T, H*W, 3*C]
	// Weight is [outC, inC] or [outC, inC, 1, 1]
	wShape := a.ToQKV.Shape()
	var w *mlx.Array
	if len(wShape) == 4 {
		w = mlx.Reshape(a.ToQKV, wShape[0], wShape[1])
	} else {
		w = a.ToQKV
	}
	w = mlx.Transpose(w, 1, 0) // [inC, outC]

	qkv := mlx.Linear(x, w) // [B*T, H*W, 3*C]
	if a.ToQKVBias != nil {
		qkv = mlx.Add(qkv, a.ToQKVBias)
	}
	qkv = mlx.Reshape(qkv, B*T, 1, H*W, 3*C)

	q := mlx.Slice(qkv, []int32{0, 0, 0, 0}, []int32{B * T, 1, H * W, C})
	k := mlx.Slice(qkv, []int32{0, 0, 0, C}, []int32{B * T, 1, H * W, 2 * C})
	v := mlx.Slice(qkv, []int32{0, 0, 0, 2 * C}, []int32{B * T, 1, H * W, 3 * C})

	scale := float32(1.0 / math.Sqrt(float64(C)))
	out := mlx.ScaledDotProductAttention(q, k, v, scale, false)

	// out: [B*T, 1, H*W, C]
	out = mlx.Reshape(out, B*T, H*W, C)

	// Project back
	pShape := a.Proj.Shape()
	var p *mlx.Array
	if len(pShape) == 4 {
		p = mlx.Reshape(a.Proj, pShape[0], pShape[1])
	} else {
		p = a.Proj
	}
	p = mlx.Transpose(p, 1, 0) // [inC, outC]
	out = mlx.Linear(out, p) // [B*T, H*W, C]
	if a.ProjBias != nil {
		out = mlx.Add(out, a.ProjBias)
	}

	out = mlx.Reshape(out, B, T, H, W, C)
	return mlx.Add(out, identity)
}

// UpBlock handles upsampling in decoder
type UpBlock struct {
	ResBlocks []*ResBlock
	Upsampler *Upsample
}

// newUpBlock creates an up block
func newUpBlock(weights *safetensors.ModelWeights, prefix string, inDim, outDim int32, numBlocks int32, upsampleMode string) (*UpBlock, error) {
	resBlocks := make([]*ResBlock, numBlocks+1)

	currentDim := inDim
	for i := int32(0); i <= numBlocks; i++ {
		resPrefix := fmt.Sprintf("%s.resnets.%d", prefix, i)
		block, err := newResBlock(weights, resPrefix, currentDim, outDim)
		if err != nil {
			return nil, err
		}
		resBlocks[i] = block
		currentDim = outDim
	}

	var upsampler *Upsample
	if upsampleMode != "" {
		upsampler = newUpsample(weights, prefix+".upsamplers.0", outDim, upsampleMode)
	}

	return &UpBlock{
		ResBlocks: resBlocks,
		Upsampler: upsampler,
	}, nil
}

// Forward applies up block with staged memory management
func (u *UpBlock) Forward(x *mlx.Array) *mlx.Array {
	// ResBlocks handle their own pools
	for _, block := range u.ResBlocks {
		prev := x
		x = block.Forward(x)
		prev.Free()
	}

	// Upsampler handles its own pools
	if u.Upsampler != nil {
		prev := x
		x = u.Upsampler.Forward(x)
		prev.Free()
	}
	return x
}

// Upsample handles spatial upsampling
type Upsample struct {
	Conv *mlx.Array
	Bias *mlx.Array
	Mode string
}

// newUpsample creates an upsampler
func newUpsample(weights *safetensors.ModelWeights, prefix string, dim int32, mode string) *Upsample {
	conv, _ := weights.Get(prefix + ".resample.1.weight")
	bias, _ := weights.Get(prefix + ".resample.1.bias")
	return &Upsample{
		Conv: conv,
		Bias: bias,
		Mode: mode,
	}
}

// Forward applies upsampling to channels-last input [B, T, H, W, C]
// Uses staged pools to reduce peak memory during 2x upsampling
func (u *Upsample) Forward(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	T := shape[1]
	H := shape[2]
	W := shape[3]
	C := shape[4]
	outC := u.Conv.Shape()[0]

	// Stage 1: 2x nearest neighbor upsample
	{
			x = mlx.Reshape(x, B*T, H, W, C)
		x = upsample2xChannelsLast(x)
		mlx.Eval(x)
	}

	// Stage 2: Conv + bias
	{
			prev := x
		weight := mlx.Transpose(u.Conv, 0, 2, 3, 1)
		x = conv2D3x3PaddedChannelsLast(x, weight)
		if u.Bias != nil {
			bias := mlx.Reshape(u.Bias, 1, 1, 1, outC)
			x = mlx.Add(x, bias)
		}
		x = mlx.Reshape(x, B, T, H*2, W*2, outC)
		prev.Free()
		mlx.Eval(x)
	}

	return x
}

// MidBlock is the middle block of decoder
type MidBlock struct {
	ResBlock1 *ResBlock
	Attention *AttentionBlock
	ResBlock2 *ResBlock
}

// newMidBlock creates a mid block
func newMidBlock(weights *safetensors.ModelWeights, prefix string, dim int32) (*MidBlock, error) {
	res1, err := newResBlock(weights, prefix+".resnets.0", dim, dim)
	if err != nil {
		return nil, err
	}
	attn, err := newAttentionBlock(weights, prefix+".attentions.0", dim)
	if err != nil {
		return nil, err
	}
	res2, err := newResBlock(weights, prefix+".resnets.1", dim, dim)
	if err != nil {
		return nil, err
	}

	return &MidBlock{
		ResBlock1: res1,
		Attention: attn,
		ResBlock2: res2,
	}, nil
}

// Forward applies mid block
func (m *MidBlock) Forward(x *mlx.Array) *mlx.Array {
	// Each component handles its own pools; we just free inputs
	prev := x
	x = m.ResBlock1.Forward(x)
	prev.Free()

	prev = x
	x = m.Attention.Forward(x)
	prev.Free()

	prev = x
	x = m.ResBlock2.Forward(x)
	prev.Free()

	return x
}

// VAEDecoder is the full VAE decoder
type VAEDecoder struct {
	Config *VAEConfig

	PostQuantConv *CausalConv3d
	ConvIn        *CausalConv3d
	MidBlock      *MidBlock
	UpBlocks      []*UpBlock
	NormOut       *RMSNorm3D
	ConvOut       *CausalConv3d
}

// Load loads the VAE decoder from a directory
func (m *VAEDecoder) Load(path string) error {
	fmt.Println("Loading Qwen-Image VAE decoder...")

	cfg := defaultVAEConfig()
	m.Config = cfg

	weights, err := safetensors.LoadModelWeights(path)
	if err != nil {
		return fmt.Errorf("weights: %w", err)
	}

	// Bulk load all weights as bf16
	fmt.Print("  Loading weights as bf16... ")
	if err := weights.Load(mlx.DtypeBFloat16); err != nil {
		return fmt.Errorf("failed to load weights: %w", err)
	}
	fmt.Printf("✓ (%.1f GB)\n", float64(mlx.MetalGetActiveMemory())/(1024*1024*1024))

	fmt.Print("  Loading post_quant_conv... ")
	postQuantConv, err := newCausalConv3d(weights, "post_quant_conv")
	if err != nil {
		return err
	}
	m.PostQuantConv = postQuantConv
	fmt.Println("✓")

	fmt.Print("  Loading conv_in... ")
	convIn, err := newCausalConv3d(weights, "decoder.conv_in")
	if err != nil {
		return err
	}
	m.ConvIn = convIn
	fmt.Println("✓")

	// Mid block (dim = base_dim * dim_mult[-1] = 96 * 4 = 384)
	fmt.Print("  Loading mid_block... ")
	midDim := cfg.BaseDim * cfg.DimMult[len(cfg.DimMult)-1]
	midBlock, err := newMidBlock(weights, "decoder.mid_block", midDim)
	if err != nil {
		return err
	}
	m.MidBlock = midBlock
	fmt.Println("✓")

	// Up blocks (reversed dim_mult)
	fmt.Print("  Loading up_blocks... ")
	numUpBlocks := len(cfg.DimMult)
	m.UpBlocks = make([]*UpBlock, numUpBlocks)

	dimsMult := make([]int32, numUpBlocks+1)
	dimsMult[0] = cfg.DimMult[numUpBlocks-1]
	for i := 0; i < numUpBlocks; i++ {
		dimsMult[i+1] = cfg.DimMult[numUpBlocks-1-i]
	}

	temporalUpsample := make([]bool, len(cfg.TemperalDownsample))
	for i := range cfg.TemperalDownsample {
		temporalUpsample[i] = cfg.TemperalDownsample[len(cfg.TemperalDownsample)-1-i]
	}

	for i := 0; i < numUpBlocks; i++ {
		inDim := cfg.BaseDim * dimsMult[i]
		outDim := cfg.BaseDim * dimsMult[i+1]

		if i > 0 {
			inDim = inDim / 2
		}

		upsampleMode := ""
		if i < numUpBlocks-1 {
			if temporalUpsample[i] {
				upsampleMode = "upsample3d"
			} else {
				upsampleMode = "upsample2d"
			}
		}

		prefix := fmt.Sprintf("decoder.up_blocks.%d", i)
		upBlock, err := newUpBlock(weights, prefix, inDim, outDim, cfg.NumResBlocks, upsampleMode)
		if err != nil {
			return err
		}
		m.UpBlocks[i] = upBlock
	}
	fmt.Printf("✓ [%d blocks]\n", numUpBlocks)

	fmt.Print("  Loading output layers... ")
	normOut, err := newRMSNorm3D(weights, "decoder.norm_out", cfg.BaseDim)
	if err != nil {
		return err
	}
	m.NormOut = normOut
	convOut, err := newCausalConv3d(weights, "decoder.conv_out")
	if err != nil {
		return err
	}
	m.ConvOut = convOut
	fmt.Println("✓")

	weights.ReleaseAll()
	return nil
}

// LoadVAEDecoderFromPath is a convenience function to load VAE from path
func LoadVAEDecoderFromPath(path string) (*VAEDecoder, error) {
	m := &VAEDecoder{}
	if err := m.Load(filepath.Join(path, "vae")); err != nil {
		return nil, err
	}
	return m, nil
}

// Decode converts latents to image
// z: [B, C, T, H, W] normalized latents
// Uses staged pools to free intermediate arrays and reduce peak memory.
func (vae *VAEDecoder) Decode(z *mlx.Array) *mlx.Array {
	var x *mlx.Array

	// Stage 1a: Denormalize and transpose
	{
			z = vae.Denormalize(z)
		// Convert from channels-first [N, C, T, H, W] to channels-last [N, T, H, W, C]
		z = mlx.Contiguous(mlx.Transpose(z, 0, 2, 3, 4, 1))
		mlx.Eval(z)
	}

	// Stage 1b: PostQuantConv (handles its own pools)
	x = vae.PostQuantConv.Forward(z)
	z.Free()

	// Stage 1c: ConvIn (handles its own pools)
	{
		prev := x
		x = vae.ConvIn.Forward(x)
		prev.Free()
	}

	// Stage 2: Mid block (handles its own pools)
	x = vae.MidBlock.Forward(x)

	// Stage 3: Up blocks (each handles its own pools)
	for _, upBlock := range vae.UpBlocks {
		x = upBlock.Forward(x)
	}

	// Stage 4a: NormOut + silu
	{
			prev := x
		x = vae.NormOut.Forward(x)
		x = silu3D(x)
		prev.Free()
		mlx.Eval(x)
	}

	// Stage 4b: ConvOut (handles its own pools)
	{
		prev := x
		x = vae.ConvOut.Forward(x)
		prev.Free()
	}

	// Stage 4c: Post-processing
	{
			prev := x
		// Clamp to [-1, 1]
		x = mlx.ClipScalar(x, -1.0, 1.0, true, true)
		// Convert back from channels-last to channels-first
		x = mlx.Contiguous(mlx.Transpose(x, 0, 4, 1, 2, 3))
		prev.Free()
		mlx.Eval(x)
	}

	return x
}

// Denormalize reverses the normalization applied during encoding
func (vae *VAEDecoder) Denormalize(z *mlx.Array) *mlx.Array {
	shape := z.Shape()
	C := shape[1]

	mean := mlx.NewArray(vae.Config.LatentsMean[:C], []int32{1, C, 1, 1, 1})
	std := mlx.NewArray(vae.Config.LatentsStd[:C], []int32{1, C, 1, 1, 1})

	mean = mlx.ToBFloat16(mean)
	std = mlx.ToBFloat16(std)

	return mlx.Add(mlx.Mul(z, std), mean)
}

// Helper functions

func silu3D(x *mlx.Array) *mlx.Array {
	return mlx.Mul(x, mlx.Sigmoid(x))
}

// pad3DChannelsLast pads a channels-last [B, T, H, W, C] tensor
func pad3DChannelsLast(x *mlx.Array, tBefore, tAfter, hBefore, hAfter, wBefore, wAfter int32) *mlx.Array {
	if tBefore == 0 && tAfter == 0 && hBefore == 0 && hAfter == 0 && wBefore == 0 && wAfter == 0 {
		return x
	}
	// Pad dims: [B before, B after, T before, T after, H before, H after, W before, W after, C before, C after]
	return mlx.Pad(x, []int32{0, 0, tBefore, tAfter, hBefore, hAfter, wBefore, wAfter, 0, 0})
}

func pad2D(x *mlx.Array, hBefore, hAfter, wBefore, wAfter int32) *mlx.Array {
	if hBefore == 0 && hAfter == 0 && wBefore == 0 && wAfter == 0 {
		return x
	}
	return mlx.Pad(x, []int32{0, 0, 0, 0, hBefore, hAfter, wBefore, wAfter})
}

func conv2D1x1(x, weight *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	H := shape[2]
	W := shape[3]

	x = mlx.Transpose(x, 0, 2, 3, 1)
	x = mlx.Reshape(x, B*H*W, shape[1])

	wShape := weight.Shape()
	var w *mlx.Array
	if len(wShape) == 4 {
		w = mlx.Reshape(weight, wShape[0], wShape[1])
	} else {
		w = weight
	}
	w = mlx.Transpose(w, 1, 0)

	out := mlx.Linear(x, w)
	outC := w.Dim(1)
	out = mlx.Reshape(out, B, H, W, outC)
	return mlx.Transpose(out, 0, 3, 1, 2)
}

func conv2D3x3Padded(x, weight *mlx.Array) *mlx.Array {
	x = pad2D(x, 1, 1, 1, 1)
	return conv2D(x, weight, 1, 1)
}

func conv2D(x, w *mlx.Array, strideH, strideW int32) *mlx.Array {
	x = mlx.Transpose(x, 0, 2, 3, 1)
	w = mlx.Transpose(w, 0, 2, 3, 1)

	shape := x.Shape()
	B := shape[0]
	H := shape[1]
	W := shape[2]

	wShape := w.Shape()
	Cout := wShape[0]
	kH := wShape[1]
	kW := wShape[2]

	outH := (H-kH)/strideH + 1
	outW := (W-kW)/strideW + 1

	patches := extractPatches2D(x, kH, kW, strideH, strideW)
	wFlat := mlx.Reshape(w, Cout, -1)
	patches = mlx.Reshape(patches, B*outH*outW, -1)
	out := mlx.Linear(patches, mlx.Transpose(wFlat, 1, 0))
	out = mlx.Reshape(out, B, outH, outW, Cout)
	return mlx.Transpose(out, 0, 3, 1, 2)
}

func extractPatches2D(x *mlx.Array, kH, kW, strideH, strideW int32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	H := shape[1]
	W := shape[2]
	C := shape[3]

	outH := (H-kH)/strideH + 1
	outW := (W-kW)/strideW + 1

	patches := make([]*mlx.Array, outH*outW)
	idx := 0
	for i := int32(0); i < outH; i++ {
		for j := int32(0); j < outW; j++ {
			startH := i * strideH
			startW := j * strideW
			patch := mlx.Slice(x, []int32{0, startH, startW, 0}, []int32{B, startH + kH, startW + kW, C})
			patch = mlx.Reshape(patch, B, kH*kW*C)
			patches[idx] = patch
			idx++
		}
	}

	for i := range patches {
		patches[i] = mlx.ExpandDims(patches[i], 1)
	}
	stacked := mlx.Concatenate(patches, 1)
	return mlx.Reshape(stacked, B, outH, outW, kH*kW*C)
}

func upsample2x(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	H := shape[2]
	W := shape[3]

	rowIdxData := make([]int32, H*2)
	for i := int32(0); i < H; i++ {
		rowIdxData[i*2] = i
		rowIdxData[i*2+1] = i
	}
	rowIdx := mlx.NewArrayInt32(rowIdxData, []int32{H * 2})

	colIdxData := make([]int32, W*2)
	for i := int32(0); i < W; i++ {
		colIdxData[i*2] = i
		colIdxData[i*2+1] = i
	}
	colIdx := mlx.NewArrayInt32(colIdxData, []int32{W * 2})

	x = mlx.Take(x, rowIdx, 2)
	x = mlx.Take(x, colIdx, 3)

	return x
}

// upsample2xChannelsLast upsamples channels-last input [B, H, W, C] by 2x
func upsample2xChannelsLast(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	H := shape[1]
	W := shape[2]

	// Create repeat indices for rows
	rowIdxData := make([]int32, H*2)
	for i := int32(0); i < H; i++ {
		rowIdxData[i*2] = i
		rowIdxData[i*2+1] = i
	}
	rowIdx := mlx.NewArrayInt32(rowIdxData, []int32{H * 2})

	// Create repeat indices for columns
	colIdxData := make([]int32, W*2)
	for i := int32(0); i < W; i++ {
		colIdxData[i*2] = i
		colIdxData[i*2+1] = i
	}
	colIdx := mlx.NewArrayInt32(colIdxData, []int32{W * 2})

	// Take along H (axis 1) then W (axis 2)
	x = mlx.Take(x, rowIdx, 1)
	x = mlx.Take(x, colIdx, 2)

	return x
}

// conv2D3x3PaddedChannelsLast applies 3x3 conv with padding to channels-last input [B, H, W, C]
// weight: [outC, kH, kW, inC] (MLX channels-last format)
func conv2D3x3PaddedChannelsLast(x, weight *mlx.Array) *mlx.Array {
	// Pad spatial dims: [B, H, W, C] -> pad H and W by 1 each side
	x = mlx.Pad(x, []int32{0, 0, 1, 1, 1, 1, 0, 0})
	// Conv2d expects: input [B, H, W, inC], weight [outC, kH, kW, inC]
	// stride=1, padding=0 (we already padded manually)
	return mlx.Conv2d(x, weight, 1, 0)
}

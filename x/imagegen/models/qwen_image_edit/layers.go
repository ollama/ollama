//go:build mlx

package qwen_image_edit

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

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

// Forward applies causal 3D convolution (or 2D if weight is 4D)
// x: [B, T, H, W, C] (channels-last, MLX format)
func (c *CausalConv3d) Forward(x *mlx.Array) *mlx.Array {
	shape := c.Weight.Shape()

	// Handle both 5D (3D conv) and 4D (2D conv) weights
	if len(shape) == 4 {
		// 2D conv: [O, I, kH, kW] - need to apply per-frame
		return c.forward2D(x)
	}

	// 3D conv: [O, I, kT, kH, kW]
	kernelT := shape[2]
	kernelH := shape[3]
	kernelW := shape[4]

	// Causal temporal padding, same spatial padding
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

// forward2D applies 2D conv per-frame for [B, T, H, W, C] input
func (c *CausalConv3d) forward2D(x *mlx.Array) *mlx.Array {
	xShape := x.Shape()
	B := xShape[0]
	T := xShape[1]
	H := xShape[2]
	W := xShape[3]
	C := xShape[4]

	wShape := c.Weight.Shape() // [O, I, kH, kW]
	kernelH := wShape[2]
	kernelW := wShape[3]
	outC := wShape[0]

	padH := kernelH / 2
	padW := kernelW / 2

	// Reshape to [B*T, H, W, C] for 2D conv
	x = mlx.Reshape(x, B*T, H, W, C)

	// Pad spatially
	x = mlx.Pad(x, []int32{0, 0, padH, padH, padW, padW, 0, 0})

	// Apply 2D conv
	weight := mlx.Transpose(c.Weight, 0, 2, 3, 1) // [O, I, kH, kW] -> [O, kH, kW, I]
	x = mlx.Conv2d(x, weight, 1, 0)

	if c.Bias != nil {
		bias := mlx.Reshape(c.Bias, 1, 1, 1, outC)
		x = mlx.Add(x, bias)
	}

	// Get output spatial dims
	outH := H
	outW := W

	// Reshape back to [B, T, H, W, C]
	x = mlx.Reshape(x, B, T, outH, outW, outC)
	mlx.Eval(x)

	return x
}

// RMSNorm3D applies RMS normalization over channels
type RMSNorm3D struct {
	Gamma *mlx.Array // [1, 1, 1, 1, C] for broadcasting
}

// newRMSNorm3D creates an RMS norm
func newRMSNorm3D(weights *safetensors.ModelWeights, prefix string, dim int32) (*RMSNorm3D, error) {
	gamma, err := weights.Get(prefix + ".gamma")
	if err != nil {
		return nil, err
	}
	gamma = mlx.Reshape(gamma, 1, 1, 1, 1, gamma.Dim(0))
	return &RMSNorm3D{Gamma: gamma}, nil
}

// Forward applies RMS norm to channels-last input [B, T, H, W, C]
func (n *RMSNorm3D) Forward(x *mlx.Array) *mlx.Array {
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
	var h *mlx.Array

	mlx.Keep(x)

	// Stage 1: norm1 + silu
	{
		h = r.Norm1.Forward(x)
		h = silu3D(h)
		mlx.Eval(h)
	}

	// Stage 2: conv1
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

	// Stage 4: conv2
	{
		prev := h
		h = r.Conv2.Forward(h)
		prev.Free()
	}

	// Residual connection
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

	// Linear to get Q, K, V
	wShape := a.ToQKV.Shape()
	var w *mlx.Array
	if len(wShape) == 4 {
		w = mlx.Reshape(a.ToQKV, wShape[0], wShape[1])
	} else {
		w = a.ToQKV
	}
	w = mlx.Transpose(w, 1, 0)

	qkv := mlx.Linear(x, w)
	if a.ToQKVBias != nil {
		qkv = mlx.Add(qkv, a.ToQKVBias)
	}
	qkv = mlx.Reshape(qkv, B*T, 1, H*W, 3*C)

	q := mlx.Slice(qkv, []int32{0, 0, 0, 0}, []int32{B * T, 1, H * W, C})
	k := mlx.Slice(qkv, []int32{0, 0, 0, C}, []int32{B * T, 1, H * W, 2 * C})
	v := mlx.Slice(qkv, []int32{0, 0, 0, 2 * C}, []int32{B * T, 1, H * W, 3 * C})

	scale := float32(1.0 / math.Sqrt(float64(C)))
	out := mlx.ScaledDotProductAttention(q, k, v, scale, false)

	out = mlx.Reshape(out, B*T, H*W, C)

	// Project back
	pShape := a.Proj.Shape()
	var p *mlx.Array
	if len(pShape) == 4 {
		p = mlx.Reshape(a.Proj, pShape[0], pShape[1])
	} else {
		p = a.Proj
	}
	p = mlx.Transpose(p, 1, 0)
	out = mlx.Linear(out, p)
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

// Forward applies up block
func (u *UpBlock) Forward(x *mlx.Array) *mlx.Array {
	for _, block := range u.ResBlocks {
		prev := x
		x = block.Forward(x)
		prev.Free()
	}

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

// MidBlock is the middle block
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

// Helper functions

func silu3D(x *mlx.Array) *mlx.Array {
	return mlx.Mul(x, mlx.Sigmoid(x))
}

// pad3DChannelsLast pads a channels-last [B, T, H, W, C] tensor
func pad3DChannelsLast(x *mlx.Array, tBefore, tAfter, hBefore, hAfter, wBefore, wAfter int32) *mlx.Array {
	if tBefore == 0 && tAfter == 0 && hBefore == 0 && hAfter == 0 && wBefore == 0 && wAfter == 0 {
		return x
	}
	return mlx.Pad(x, []int32{0, 0, tBefore, tAfter, hBefore, hAfter, wBefore, wAfter, 0, 0})
}

// upsample2xChannelsLast upsamples channels-last input [B, H, W, C] by 2x
func upsample2xChannelsLast(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	H := shape[1]
	W := shape[2]

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

	x = mlx.Take(x, rowIdx, 1)
	x = mlx.Take(x, colIdx, 2)

	return x
}

// conv2D3x3PaddedChannelsLast applies 3x3 conv with padding to channels-last input [B, H, W, C]
func conv2D3x3PaddedChannelsLast(x, weight *mlx.Array) *mlx.Array {
	x = mlx.Pad(x, []int32{0, 0, 1, 1, 1, 1, 0, 0})
	return mlx.Conv2d(x, weight, 1, 0)
}

// conv2DStrided applies conv with stride > 1 using manual patch extraction
// x: [B, H, W, C] (channels-last), weight: [O, kH, kW, I]
func conv2DStrided(x, weight *mlx.Array, stride int32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	H := shape[1]
	W := shape[2]

	wShape := weight.Shape()
	Cout := wShape[0]
	kH := wShape[1]
	kW := wShape[2]

	outH := (H - kH) / stride + 1
	outW := (W - kW) / stride + 1

	patches := extractPatches2DStrided(x, kH, kW, stride)
	wFlat := mlx.Reshape(weight, Cout, -1)
	patches = mlx.Reshape(patches, B*outH*outW, -1)
	out := mlx.Linear(patches, mlx.Transpose(wFlat, 1, 0))
	return mlx.Reshape(out, B, outH, outW, Cout)
}

// conv3DStrided applies 3D conv with strides using manual patch extraction
// x: [B, T, H, W, C] (channels-last), weight: [O, I, kT, kH, kW] (PyTorch format)
// strideT, strideH, strideW are the strides for each dimension
// Patches are extracted in [C, T, H, W] order to match Python's preprocessing
func conv3DStrided(x, weight *mlx.Array, strideT, strideH, strideW int32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	T := shape[1]
	H := shape[2]
	W := shape[3]
	C := shape[4]

	wShape := weight.Shape()
	Cout := wShape[0]
	// I := wShape[1]
	kT := wShape[2]
	kH := wShape[3]
	kW := wShape[4]

	// For temporal: if T < kT, we need to repeat frames temporally
	// For single image with T=1 and kT=2, we duplicate the frame to T=kT
	// Python Qwen2.5-VL duplicates the frame, not zero-pads
	if T < kT {
		// Tile along T dimension: [B, T, H, W, C] -> [B, kT, H, W, C]
		x = mlx.Tile(x, []int32{1, kT, 1, 1, 1})
		T = kT
	}

	outT := (T - kT) / strideT + 1
	outH := (H - kH) / strideH + 1
	outW := (W - kW) / strideW + 1

	// Extract 3D patches in [C, T, H, W] order to match Python
	patches := extractPatches3DStrided(x, kT, kH, kW, strideT, strideH, strideW)
	// patches shape: [B, outT, outH, outW, C*kT*kH*kW]

	// Weight is [O, I, kT, kH, kW] - flatten to [O, I*kT*kH*kW] to match patch order [C, T, H, W]
	wFlat := mlx.Reshape(weight, Cout, -1) // [Cout, I*kT*kH*kW]
	patches = mlx.Reshape(patches, B*outT*outH*outW, C*kT*kH*kW)
	out := mlx.Linear(patches, mlx.Transpose(wFlat, 1, 0))
	return mlx.Reshape(out, B, outT, outH, outW, Cout)
}

// extractPatches3DStrided extracts 3D patches with given strides
// Returns patches with values in [C, T, H, W] order to match Python's preprocessing
func extractPatches3DStrided(x *mlx.Array, kT, kH, kW, strideT, strideH, strideW int32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	T := shape[1]
	H := shape[2]
	W := shape[3]
	C := shape[4]

	outT := (T - kT) / strideT + 1
	outH := (H - kH) / strideH + 1
	outW := (W - kW) / strideW + 1

	numPatches := outT * outH * outW
	patches := make([]*mlx.Array, numPatches)
	idx := 0
	for t := int32(0); t < outT; t++ {
		for i := int32(0); i < outH; i++ {
			for j := int32(0); j < outW; j++ {
				startT := t * strideT
				startH := i * strideH
				startW := j * strideW
				// Extract patch: [B, kT, kH, kW, C]
				patch := mlx.Slice(x,
					[]int32{0, startT, startH, startW, 0},
					[]int32{B, startT + kT, startH + kH, startW + kW, C})
				// Transpose from [B, T, H, W, C] to [B, C, T, H, W] to match Python's order
				patch = mlx.Transpose(patch, 0, 4, 1, 2, 3)
				// Flatten to [B, C*T*H*W]
				patch = mlx.Reshape(patch, B, C*kT*kH*kW)
				patches[idx] = patch
				idx++
			}
		}
	}

	for i := range patches {
		patches[i] = mlx.ExpandDims(patches[i], 1)
	}
	stacked := mlx.Concatenate(patches, 1)
	return mlx.Reshape(stacked, B, outT, outH, outW, C*kT*kH*kW)
}

// extractPatches2DStrided extracts patches with given stride
func extractPatches2DStrided(x *mlx.Array, kH, kW, stride int32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	H := shape[1]
	W := shape[2]
	C := shape[3]

	outH := (H - kH) / stride + 1
	outW := (W - kW) / stride + 1

	patches := make([]*mlx.Array, outH*outW)
	idx := 0
	for i := int32(0); i < outH; i++ {
		for j := int32(0); j < outW; j++ {
			startH := i * stride
			startW := j * stride
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

// layerNormNoAffine applies layer norm without learnable parameters
func layerNormNoAffine(x *mlx.Array, eps float32) *mlx.Array {
	ndim := x.Ndim()
	lastAxis := ndim - 1
	mean := mlx.Mean(x, lastAxis, true)
	xCentered := mlx.Sub(x, mean)
	variance := mlx.Mean(mlx.Square(xCentered), lastAxis, true)
	return mlx.Div(xCentered, mlx.Sqrt(mlx.AddScalar(variance, eps)))
}

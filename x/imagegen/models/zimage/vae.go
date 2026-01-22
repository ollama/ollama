//go:build mlx

package zimage

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/safetensors"
	"github.com/ollama/ollama/x/imagegen/vae"
)

// VAEConfig holds VAE decoder configuration
type VAEConfig struct {
	InChannels       int32   `json:"in_channels"`
	OutChannels      int32   `json:"out_channels"`
	LatentChannels   int32   `json:"latent_channels"`
	BlockOutChannels []int32 `json:"block_out_channels"`
	LayersPerBlock   int32   `json:"layers_per_block"`
	NormNumGroups    int32   `json:"norm_num_groups"`
	ScalingFactor    float32 `json:"scaling_factor"`
	ShiftFactor      float32 `json:"shift_factor"`
}

// GroupNormLayer implements group normalization
type GroupNormLayer struct {
	Weight    *mlx.Array
	Bias      *mlx.Array
	NumGroups int32
	Eps       float32
}

// NewGroupNorm creates a group norm layer
func NewGroupNorm(weight, bias *mlx.Array, numGroups int32) *GroupNormLayer {
	return &GroupNormLayer{
		Weight:    weight,
		Bias:      bias,
		NumGroups: numGroups,
		Eps:       1e-5,
	}
}

// Forward applies group normalization
// Input and output are in NHWC format [B, H, W, C]
func (gn *GroupNormLayer) Forward(x *mlx.Array) *mlx.Array {
	// x: [B, H, W, C] (NHWC format)
	shape := x.Shape()
	B := shape[0]
	H := shape[1]
	W := shape[2]
	C := shape[3]

	// For large spatial sizes, use tiled computation to avoid CUDA grid limits
	// CUDA grid.y max is 65535, so H*W/16 must be <= 65535, meaning H*W <= ~1M
	// To be safe, tile when H*W > 512*512 = 262144
	if H*W > 512*512 {
		return gn.forwardTiled(x, B, H, W, C)
	}

	return gn.forwardSmall(x, B, H, W, C)
}

// forwardSmall is the standard GroupNorm for tensors that fit within CUDA grid limits
func (gn *GroupNormLayer) forwardSmall(x *mlx.Array, B, H, W, C int32) *mlx.Array {
	// Reshape to [B, H, W, groups, C/groups]
	groupSize := C / gn.NumGroups
	x = mlx.Reshape(x, B, H, W, gn.NumGroups, groupSize)

	// Compute mean and variance per group (over H, W, and C/groups dimensions)
	mean := mlx.Mean(x, 1, true)
	mean = mlx.Mean(mean, 2, true)
	mean = mlx.Mean(mean, 4, true)

	xCentered := mlx.Sub(x, mean)

	// Variance over same axes
	sq := mlx.Square(xCentered)
	variance := mlx.Mean(sq, 1, true)
	variance = mlx.Mean(variance, 2, true)
	variance = mlx.Mean(variance, 4, true)

	// Normalize
	xNorm := mlx.Div(xCentered, mlx.Sqrt(mlx.AddScalar(variance, gn.Eps)))

	// Reshape back to [B, H, W, C]
	xNorm = mlx.Reshape(xNorm, B, H, W, C)

	// Scale and shift (weight and bias are [C])
	if gn.Weight != nil {
		weight := mlx.Reshape(gn.Weight, 1, 1, 1, C)
		xNorm = mlx.Mul(xNorm, weight)
	}
	if gn.Bias != nil {
		bias := mlx.Reshape(gn.Bias, 1, 1, 1, C)
		xNorm = mlx.Add(xNorm, bias)
	}

	return xNorm
}

// forwardTiled handles large tensors by processing in H-tiles to avoid CUDA grid limits
func (gn *GroupNormLayer) forwardTiled(x *mlx.Array, B, H, W, C int32) *mlx.Array {
	groupSize := C / gn.NumGroups

	// Keep the input - we need it for slicing tiles later
	// Track if we were the ones who kept it, so we can restore state after
	wasKept := x.Kept()
	mlx.Keep(x)

	// Compute per-group mean and variance using flattened spatial dimensions
	// Build the entire compute graph first, then eval once
	// Reshape to [B, H*W, groups, groupSize]
	xFlat := mlx.Reshape(x, B, H*W, gn.NumGroups, groupSize)

	// Mean over spatial (axis 1) and groupSize (axis 3) dimensions
	// Result shape: [B, 1, groups, 1]
	mean1 := mlx.Mean(xFlat, 1, true)
	mean := mlx.Mean(mean1, 3, true)

	// Variance using E[X^2] - E[X]^2
	xSq := mlx.Square(xFlat)
	meanSq1 := mlx.Mean(xSq, 1, true)
	meanSq := mlx.Mean(meanSq1, 3, true)
	meanSquared := mlx.Square(mean)
	variance := mlx.Sub(meanSq, meanSquared)

	// invStd = 1/sqrt(var + eps)
	varPlusEps := mlx.AddScalar(variance, gn.Eps)
	stdDev := mlx.Sqrt(varPlusEps)
	one := mlx.Full(1.0, 1)
	invStd := mlx.Div(one, stdDev)

	// Eval mean and invStd together - these are what we need for the tile loop
	mlx.Keep(mean, invStd)
	mlx.Eval(mean, invStd)

	// Tile along H dimension
	tileH := int32(512 * 512 / W)
	if tileH < 1 {
		tileH = 1
	}
	if tileH > H {
		tileH = H
	}

	// Prepare weight and bias reshaped for 4D broadcast [1, 1, groups, groupSize]
	var weightGN, biasGN *mlx.Array
	if gn.Weight != nil {
		weightGN = mlx.Reshape(gn.Weight, 1, 1, gn.NumGroups, groupSize)
		mlx.Keep(weightGN)
		mlx.Eval(weightGN)
	}
	if gn.Bias != nil {
		biasGN = mlx.Reshape(gn.Bias, 1, 1, gn.NumGroups, groupSize)
		mlx.Keep(biasGN)
		mlx.Eval(biasGN)
	}

	var tiles []*mlx.Array
	for hStart := int32(0); hStart < H; hStart += tileH {
		hEnd := hStart + tileH
		if hEnd > H {
			hEnd = H
		}
		tileHeight := hEnd - hStart
		spatialSize := tileHeight * W

		// Build the compute graph for this tile (no intermediate Evals)
		// Extract tile and flatten spatial dims: [B, tileH*W, groups, groupSize]
		tile := mlx.Slice(x, []int32{0, hStart, 0, 0}, []int32{B, hEnd, W, C})
		tileFlat := mlx.Reshape(tile, B, spatialSize, gn.NumGroups, groupSize)

		// Normalize: (x - mean) * invStd
		tileCentered := mlx.Sub(tileFlat, mean)
		tileNorm := mlx.Mul(tileCentered, invStd)

		// Apply scale and shift in 4D space
		if weightGN != nil {
			tileNorm = mlx.Mul(tileNorm, weightGN)
		}
		if biasGN != nil {
			tileNorm = mlx.Add(tileNorm, biasGN)
		}

		// Reshape back to [B, tileH, W, C]
		tileOut := mlx.Reshape(tileNorm, B, tileHeight, W, C)

		// Now eval and keep this tile
		mlx.Keep(tileOut)
		mlx.Eval(tileOut)

		tiles = append(tiles, tileOut)
	}

	// Concatenate tiles along H axis
	var result *mlx.Array
	if len(tiles) == 1 {
		result = tiles[0]
	} else {
		result = mlx.Concatenate(tiles, 1)
		mlx.Eval(result)
		// Free the individual tiles now that they're concatenated
		for _, t := range tiles {
			t.Free()
		}
	}

	// Clean up kept arrays
	// Restore x's kept state - only free if we were the ones who kept it
	if !wasKept {
		x.Free()
	}
	mean.Free()
	invStd.Free()
	if weightGN != nil {
		weightGN.Free()
	}
	if biasGN != nil {
		biasGN.Free()
	}

	return result
}

// Conv2D represents a 2D convolution layer
// Works natively in NHWC format (MLX's native format)
type Conv2D struct {
	Weight  *mlx.Array // [out_channels, kH, kW, in_channels] (OHWI for MLX)
	Bias    *mlx.Array // [out_channels]
	Stride  int32
	Padding int32
}

// NewConv2D creates a Conv2D layer
// weight comes in as [out_channels, in_channels, kH, kW] (OIHW from PyTorch)
// we transpose to [out_channels, kH, kW, in_channels] (OHWI for MLX)
func NewConv2D(weight, bias *mlx.Array, stride, padding int32) *Conv2D {
	// Transpose weight from OIHW to OHWI
	// [O, I, H, W] -> [O, H, W, I]
	weightOHWI := mlx.Transpose(weight, 0, 2, 3, 1)
	return &Conv2D{
		Weight:  weightOHWI,
		Bias:    bias,
		Stride:  stride,
		Padding: padding,
	}
}

// Forward applies convolution
// Input and output are in NHWC format [N, H, W, C]
func (conv *Conv2D) Forward(x *mlx.Array) *mlx.Array {
	// Conv in NHWC format (MLX native)
	out := mlx.Conv2d(x, conv.Weight, conv.Stride, conv.Padding)

	if conv.Bias != nil {
		// Bias is [C], reshape to [1, 1, 1, C] for NHWC broadcast
		bias := mlx.Reshape(conv.Bias, 1, 1, 1, conv.Bias.Dim(0))
		out = mlx.Add(out, bias)
	}

	return out
}

// ResnetBlock2D implements a ResNet block for VAE
type ResnetBlock2D struct {
	Norm1        *GroupNormLayer
	Conv1        *Conv2D
	Norm2        *GroupNormLayer
	Conv2        *Conv2D
	ConvShortcut *Conv2D // nil if in_channels == out_channels
}

// NewResnetBlock2D creates a ResNet block
func NewResnetBlock2D(weights safetensors.WeightSource, prefix string, numGroups int32) (*ResnetBlock2D, error) {
	norm1Weight, err := weights.GetTensor(prefix + ".norm1.weight")
	if err != nil {
		return nil, err
	}
	norm1Bias, err := weights.GetTensor(prefix + ".norm1.bias")
	if err != nil {
		return nil, err
	}

	conv1Weight, err := weights.GetTensor(prefix + ".conv1.weight")
	if err != nil {
		return nil, err
	}
	conv1Bias, err := weights.GetTensor(prefix + ".conv1.bias")
	if err != nil {
		return nil, err
	}

	norm2Weight, err := weights.GetTensor(prefix + ".norm2.weight")
	if err != nil {
		return nil, err
	}
	norm2Bias, err := weights.GetTensor(prefix + ".norm2.bias")
	if err != nil {
		return nil, err
	}

	conv2Weight, err := weights.GetTensor(prefix + ".conv2.weight")
	if err != nil {
		return nil, err
	}
	conv2Bias, err := weights.GetTensor(prefix + ".conv2.bias")
	if err != nil {
		return nil, err
	}

	block := &ResnetBlock2D{
		Norm1: NewGroupNorm(norm1Weight, norm1Bias, numGroups),
		Conv1: NewConv2D(conv1Weight, conv1Bias, 1, 1),
		Norm2: NewGroupNorm(norm2Weight, norm2Bias, numGroups),
		Conv2: NewConv2D(conv2Weight, conv2Bias, 1, 1),
	}

	if weights.HasTensor(prefix + ".conv_shortcut.weight") {
		shortcutWeight, err := weights.GetTensor(prefix + ".conv_shortcut.weight")
		if err != nil {
			return nil, err
		}
		shortcutBias, err := weights.GetTensor(prefix + ".conv_shortcut.bias")
		if err != nil {
			return nil, err
		}
		block.ConvShortcut = NewConv2D(shortcutWeight, shortcutBias, 1, 0)
	}

	return block, nil
}

// Forward applies the ResNet block with staged evaluation
func (rb *ResnetBlock2D) Forward(x *mlx.Array) *mlx.Array {
	var h *mlx.Array

	// Stage 1: norm1
	{
		h = rb.Norm1.Forward(x)
		mlx.Eval(h)
	}

	// Stage 2: silu + conv1
	{
		prev := h
		h = mlx.SiLU(h)
		h = rb.Conv1.Forward(h)
		prev.Free()
		mlx.Eval(h)
	}

	// Stage 3: norm2
	{
		prev := h
		h = rb.Norm2.Forward(h)
		prev.Free()
		mlx.Eval(h)
	}

	// Stage 4: silu + conv2
	{
		prev := h
		h = mlx.SiLU(h)
		h = rb.Conv2.Forward(h)
		prev.Free()
		mlx.Eval(h)
	}

	// Residual connection
	{
		prev := h
		if rb.ConvShortcut != nil {
			shortcut := rb.ConvShortcut.Forward(x)
			h = mlx.Add(h, shortcut)
		} else {
			h = mlx.Add(h, x)
		}
		prev.Free()
		mlx.Eval(h)
	}

	return h
}

// VAEAttentionBlock implements self-attention for VAE
type VAEAttentionBlock struct {
	GroupNorm   *GroupNormLayer
	ToQWeight   *mlx.Array
	ToQBias     *mlx.Array
	ToKWeight   *mlx.Array
	ToKBias     *mlx.Array
	ToVWeight   *mlx.Array
	ToVBias     *mlx.Array
	ToOutWeight *mlx.Array
	ToOutBias   *mlx.Array
	NumHeads    int32
}

// NewVAEAttentionBlock creates an attention block
func NewVAEAttentionBlock(weights safetensors.WeightSource, prefix string, numGroups int32) (*VAEAttentionBlock, error) {
	normWeight, err := weights.GetTensor(prefix + ".group_norm.weight")
	if err != nil {
		return nil, err
	}
	normBias, err := weights.GetTensor(prefix + ".group_norm.bias")
	if err != nil {
		return nil, err
	}

	toQWeight, err := weights.GetTensor(prefix + ".to_q.weight")
	if err != nil {
		return nil, err
	}
	toQBias, err := weights.GetTensor(prefix + ".to_q.bias")
	if err != nil {
		return nil, err
	}

	toKWeight, err := weights.GetTensor(prefix + ".to_k.weight")
	if err != nil {
		return nil, err
	}
	toKBias, err := weights.GetTensor(prefix + ".to_k.bias")
	if err != nil {
		return nil, err
	}

	toVWeight, err := weights.GetTensor(prefix + ".to_v.weight")
	if err != nil {
		return nil, err
	}
	toVBias, err := weights.GetTensor(prefix + ".to_v.bias")
	if err != nil {
		return nil, err
	}

	toOutWeight, err := weights.GetTensor(prefix + ".to_out.0.weight")
	if err != nil {
		return nil, err
	}
	toOutBias, err := weights.GetTensor(prefix + ".to_out.0.bias")
	if err != nil {
		return nil, err
	}

	return &VAEAttentionBlock{
		GroupNorm:   NewGroupNorm(normWeight, normBias, numGroups),
		ToQWeight:   mlx.Transpose(toQWeight, 1, 0),
		ToQBias:     toQBias,
		ToKWeight:   mlx.Transpose(toKWeight, 1, 0),
		ToKBias:     toKBias,
		ToVWeight:   mlx.Transpose(toVWeight, 1, 0),
		ToVBias:     toVBias,
		ToOutWeight: mlx.Transpose(toOutWeight, 1, 0),
		ToOutBias:   toOutBias,
		NumHeads:    1,
	}, nil
}

// Forward applies attention with staged evaluation
// Input and output are in NHWC format [B, H, W, C]
func (ab *VAEAttentionBlock) Forward(x *mlx.Array) *mlx.Array {
	residual := x
	shape := x.Shape()
	B := shape[0]
	H := shape[1]
	W := shape[2]
	C := shape[3]

	var h *mlx.Array

	// Stage 1: GroupNorm + reshape to [B, H*W, C]
	{
		h = ab.GroupNorm.Forward(x)
		h = mlx.Reshape(h, B, H*W, C)
		mlx.Eval(h)
	}

	var out *mlx.Array

	// Stage 2: Q, K, V projections + attention
	{
		q := mlx.Linear(h, ab.ToQWeight)
		q = mlx.Add(q, ab.ToQBias)
		k := mlx.Linear(h, ab.ToKWeight)
		k = mlx.Add(k, ab.ToKBias)
		v := mlx.Linear(h, ab.ToVWeight)
		v = mlx.Add(v, ab.ToVBias)
		h.Free()

		q = mlx.ExpandDims(q, 1)
		k = mlx.ExpandDims(k, 1)
		v = mlx.ExpandDims(v, 1)

		scale := float32(1.0 / math.Sqrt(float64(C)))
		out = mlx.ScaledDotProductAttention(q, k, v, scale, false)
		out = mlx.Squeeze(out, 1)
		mlx.Eval(out)
	}

	// Stage 3: Output projection + reshape + residual
	{
		prev := out
		out = mlx.Linear(out, ab.ToOutWeight)
		out = mlx.Add(out, ab.ToOutBias)
		out = mlx.Reshape(out, B, H, W, C)
		out = mlx.Add(out, residual)
		prev.Free()
		mlx.Eval(out)
	}

	return out
}

// UpDecoderBlock2D implements an upsampling decoder block
type UpDecoderBlock2D struct {
	ResnetBlocks []*ResnetBlock2D
	Upsample     *Conv2D
}

// NewUpDecoderBlock2D creates an up decoder block
func NewUpDecoderBlock2D(weights safetensors.WeightSource, prefix string, numLayers, numGroups int32, hasUpsample bool) (*UpDecoderBlock2D, error) {
	resnets := make([]*ResnetBlock2D, numLayers)
	for i := int32(0); i < numLayers; i++ {
		resPrefix := fmt.Sprintf("%s.resnets.%d", prefix, i)
		resnet, err := NewResnetBlock2D(weights, resPrefix, numGroups)
		if err != nil {
			return nil, err
		}
		resnets[i] = resnet
	}

	var upsample *Conv2D
	if hasUpsample {
		upWeight, err := weights.GetTensor(prefix + ".upsamplers.0.conv.weight")
		if err != nil {
			return nil, err
		}
		upBias, err := weights.GetTensor(prefix + ".upsamplers.0.conv.bias")
		if err != nil {
			return nil, err
		}
		upsample = NewConv2D(upWeight, upBias, 1, 1)
	}

	return &UpDecoderBlock2D{
		ResnetBlocks: resnets,
		Upsample:     upsample,
	}, nil
}

// Forward applies the up decoder block with staged evaluation to reduce peak memory
func (ub *UpDecoderBlock2D) Forward(x *mlx.Array) *mlx.Array {
	for _, resnet := range ub.ResnetBlocks {
		prev := x
		x = resnet.Forward(x) // ResNet handles its own pools
		prev.Free()
	}

	if ub.Upsample != nil {
		// Stage 1: Upsample2x (nearest neighbor)
		{
					prev := x
			x = Upsample2x(x)
			prev.Free()
			mlx.Eval(x)
		}

		// Stage 2: Upsample conv
		{
					prev := x
			x = ub.Upsample.Forward(x)
			prev.Free()
			mlx.Eval(x)
		}
	}

	return x
}

// VAEMidBlock is the middle block with attention
type VAEMidBlock struct {
	Resnet1   *ResnetBlock2D
	Attention *VAEAttentionBlock
	Resnet2   *ResnetBlock2D
}

// NewVAEMidBlock creates the mid block
func NewVAEMidBlock(weights safetensors.WeightSource, prefix string, numGroups int32) (*VAEMidBlock, error) {
	resnet1, err := NewResnetBlock2D(weights, prefix+".resnets.0", numGroups)
	if err != nil {
		return nil, err
	}

	attention, err := NewVAEAttentionBlock(weights, prefix+".attentions.0", numGroups)
	if err != nil {
		return nil, err
	}

	resnet2, err := NewResnetBlock2D(weights, prefix+".resnets.1", numGroups)
	if err != nil {
		return nil, err
	}

	return &VAEMidBlock{
		Resnet1:   resnet1,
		Attention: attention,
		Resnet2:   resnet2,
	}, nil
}

// Forward applies the mid block with staged evaluation
func (mb *VAEMidBlock) Forward(x *mlx.Array) *mlx.Array {
	prev := x
	x = mb.Resnet1.Forward(x) // ResNet handles its own pools
	prev.Free()

	// Attention handles its own pools
	prev = x
	x = mb.Attention.Forward(x)
	prev.Free()

	prev = x
	x = mb.Resnet2.Forward(x) // ResNet handles its own pools
	prev.Free()

	return x
}

// VAEDecoder is the full VAE decoder
type VAEDecoder struct {
	Config      *VAEConfig
	ConvIn      *Conv2D
	MidBlock    *VAEMidBlock
	UpBlocks    []*UpDecoderBlock2D
	ConvNormOut *GroupNormLayer
	ConvOut     *Conv2D

	// Tiling configuration (nil = no tiling)
	Tiling *vae.TilingConfig
}

// Load loads the VAE decoder from ollama blob storage.
func (m *VAEDecoder) Load(manifest *imagegen.ModelManifest) error {
	// Load config from blob
	var cfg VAEConfig
	if err := manifest.ReadConfigJSON("vae/config.json", &cfg); err != nil {
		return fmt.Errorf("config: %w", err)
	}
	m.Config = &cfg

	// Load weights from tensor blobs
	weights, err := imagegen.LoadWeightsFromManifest(manifest, "vae")
	if err != nil {
		return fmt.Errorf("weights: %w", err)
	}
	if err := weights.Load(0); err != nil {
		return fmt.Errorf("load weights: %w", err)
	}
	defer weights.ReleaseAll()

	return m.loadWeights(weights, &cfg)
}

// loadWeights loads VAE weights from any WeightSource
func (m *VAEDecoder) loadWeights(weights safetensors.WeightSource, cfg *VAEConfig) error {
	var err error

	// Load conv_in
	fmt.Print("  Loading conv_in... ")
	convInWeight, err := weights.GetTensor("decoder.conv_in.weight")
	if err != nil {
		return err
	}
	convInBias, err := weights.GetTensor("decoder.conv_in.bias")
	if err != nil {
		return err
	}
	m.ConvIn = NewConv2D(convInWeight, convInBias, 1, 1)
	fmt.Println("✓")

	// Load mid block
	fmt.Print("  Loading mid block... ")
	m.MidBlock, err = NewVAEMidBlock(weights, "decoder.mid_block", cfg.NormNumGroups)
	if err != nil {
		return err
	}
	fmt.Println("✓")

	// Load up blocks
	fmt.Print("  Loading up blocks... ")
	numBlocks := len(cfg.BlockOutChannels)
	m.UpBlocks = make([]*UpDecoderBlock2D, numBlocks)
	for i := 0; i < numBlocks; i++ {
		prefix := fmt.Sprintf("decoder.up_blocks.%d", i)
		hasUpsample := i < numBlocks-1
		m.UpBlocks[i], err = NewUpDecoderBlock2D(weights, prefix, cfg.LayersPerBlock+1, cfg.NormNumGroups, hasUpsample)
		if err != nil {
			return err
		}
	}
	fmt.Printf("✓ [%d blocks]\n", numBlocks)

	// Load conv_norm_out
	fmt.Print("  Loading conv_norm_out... ")
	normWeight, err := weights.GetTensor("decoder.conv_norm_out.weight")
	if err != nil {
		return err
	}
	normBias, err := weights.GetTensor("decoder.conv_norm_out.bias")
	if err != nil {
		return err
	}
	m.ConvNormOut = NewGroupNorm(normWeight, normBias, cfg.NormNumGroups)
	fmt.Println("✓")

	// Load conv_out
	fmt.Print("  Loading conv_out... ")
	convOutWeight, err := weights.GetTensor("decoder.conv_out.weight")
	if err != nil {
		return err
	}
	convOutBias, err := weights.GetTensor("decoder.conv_out.bias")
	if err != nil {
		return err
	}
	m.ConvOut = NewConv2D(convOutWeight, convOutBias, 1, 1)
	fmt.Println("✓")

	return nil
}

// Decode decodes latents to images.
// Input latents are in NCHW format, output is in NCHW format.
// If Tiling is set, uses tiled decoding to reduce memory for large images.
func (v *VAEDecoder) Decode(latents *mlx.Array) *mlx.Array {
	// Scale latents
	z := mlx.DivScalar(latents, v.Config.ScalingFactor)
	z = mlx.AddScalar(z, v.Config.ShiftFactor)
	// Convert NCHW -> NHWC for internal processing
	z = mlx.Transpose(z, 0, 2, 3, 1)

	// Use tiled decoding if enabled
	if v.Tiling != nil {
		mlx.Eval(z)
		return vae.DecodeTiled(z, v.Tiling, v.decodeTile)
	}

	// Direct decode
	h := v.decodeTile(z)
	h = mlx.ClipScalar(h, 0.0, 1.0, true, true)
	// Convert NHWC -> NCHW for output
	h = mlx.Transpose(h, 0, 3, 1, 2)
	mlx.Eval(h)
	return h
}

// decodeTile decodes a single latent tile to pixels.
// Input: [B, H, W, C] latent tile in NHWC format (already scaled)
// Output: [B, H*8, W*8, 3] pixel tile in NHWC format
func (v *VAEDecoder) decodeTile(z *mlx.Array) *mlx.Array {
	h := v.ConvIn.Forward(z)
	mlx.Eval(h)

	prev := h
	h = v.MidBlock.Forward(h)
	prev.Free()

	for _, upBlock := range v.UpBlocks {
		prev = h
		h = upBlock.Forward(h)
		prev.Free()
	}

	prev = h
	h = v.ConvNormOut.Forward(h)
	mlx.Eval(h) // Eval after GroupNorm to avoid grid dimension issues
	prev.Free()

	prev = h
	h = mlx.SiLU(h)
	h = v.ConvOut.Forward(h)
	mlx.Eval(h)
	prev.Free()

	// VAE outputs [-1, 1], convert to [0, 1]
	h = mlx.MulScalar(h, 0.5)
	h = mlx.AddScalar(h, 0.5)

	return h
}

// Upsample2x performs 2x nearest neighbor upsampling using Take.
// Input and output are in NHWC format: [B, H, W, C] -> [B, H*2, W*2, C]
// Uses Take with repeated indices to produce contiguous output.
func Upsample2x(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	H := shape[1]
	W := shape[2]

	// Create indices [0, 0, 1, 1, 2, 2, ...] for nearest neighbor
	// For H dimension
	hIdx := mlx.ArangeInt(0, H, 1, mlx.DtypeInt32)
	hIdx = mlx.Reshape(hIdx, H, 1)
	hIdx = mlx.BroadcastTo(hIdx, []int32{H, 2})
	hIdx = mlx.Reshape(hIdx, H*2)

	// For W dimension
	wIdx := mlx.ArangeInt(0, W, 1, mlx.DtypeInt32)
	wIdx = mlx.Reshape(wIdx, W, 1)
	wIdx = mlx.BroadcastTo(wIdx, []int32{W, 2})
	wIdx = mlx.Reshape(wIdx, W*2)

	// Take along H axis (axis 1 in NHWC)
	x = mlx.Take(x, hIdx, 1)
	// Take along W axis (axis 2 in NHWC)
	x = mlx.Take(x, wIdx, 2)

	return x
}

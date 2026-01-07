package zimage

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/safetensors"
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

// loadVAEConfig loads VAE config from a JSON file
func loadVAEConfig(path string) (*VAEConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var cfg VAEConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	return &cfg, nil
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
func (gn *GroupNormLayer) Forward(x *mlx.Array) *mlx.Array {
	// x: [B, C, H, W]
	shape := x.Shape()
	B := shape[0]
	C := shape[1]
	H := shape[2]
	W := shape[3]

	// Reshape to [B, groups, C/groups, H, W]
	groupSize := C / gn.NumGroups
	x = mlx.Reshape(x, B, gn.NumGroups, groupSize, H, W)

	// Compute mean and variance per group
	mean := mlx.Mean(x, 2, true)
	mean = mlx.Mean(mean, 3, true)
	mean = mlx.Mean(mean, 4, true)

	xCentered := mlx.Sub(x, mean)
	variance := mlx.Mean(mlx.Square(xCentered), 2, true)
	variance = mlx.Mean(variance, 3, true)
	variance = mlx.Mean(variance, 4, true)

	// Normalize
	xNorm := mlx.Div(xCentered, mlx.Sqrt(mlx.AddScalar(variance, gn.Eps)))

	// Reshape back to [B, C, H, W]
	xNorm = mlx.Reshape(xNorm, B, C, H, W)

	// Scale and shift (weight and bias are [C])
	if gn.Weight != nil {
		weight := mlx.Reshape(gn.Weight, 1, C, 1, 1)
		xNorm = mlx.Mul(xNorm, weight)
	}
	if gn.Bias != nil {
		bias := mlx.Reshape(gn.Bias, 1, C, 1, 1)
		xNorm = mlx.Add(xNorm, bias)
	}

	return xNorm
}

// Conv2D represents a 2D convolution layer
// MLX uses NHWC format, but we store weights in OHWI format for MLX conv
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
// Input x is in NCHW format, we convert to NHWC for MLX, then back to NCHW
func (conv *Conv2D) Forward(x *mlx.Array) *mlx.Array {
	// x: [N, C, H, W] -> [N, H, W, C]
	xNHWC := mlx.Transpose(x, 0, 2, 3, 1)

	// Conv in NHWC format
	outNHWC := mlx.Conv2d(xNHWC, conv.Weight, conv.Stride, conv.Padding)

	// Convert back to NCHW: [N, H, W, C] -> [N, C, H, W]
	out := mlx.Transpose(outNHWC, 0, 3, 1, 2)

	if conv.Bias != nil {
		bias := mlx.Reshape(conv.Bias, 1, conv.Bias.Dim(0), 1, 1)
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
func NewResnetBlock2D(weights *safetensors.ModelWeights, prefix string, numGroups int32) (*ResnetBlock2D, error) {
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
func NewVAEAttentionBlock(weights *safetensors.ModelWeights, prefix string, numGroups int32) (*VAEAttentionBlock, error) {
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
func (ab *VAEAttentionBlock) Forward(x *mlx.Array) *mlx.Array {
	residual := x
	shape := x.Shape()
	B := shape[0]
	C := shape[1]
	H := shape[2]
	W := shape[3]

	var h *mlx.Array

	// Stage 1: GroupNorm + reshape
	{
			h = ab.GroupNorm.Forward(x)
		h = mlx.Transpose(h, 0, 2, 3, 1)
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
		out = mlx.Transpose(out, 0, 3, 1, 2)
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
func NewUpDecoderBlock2D(weights *safetensors.ModelWeights, prefix string, numLayers, numGroups int32, hasUpsample bool) (*UpDecoderBlock2D, error) {
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
func NewVAEMidBlock(weights *safetensors.ModelWeights, prefix string, numGroups int32) (*VAEMidBlock, error) {
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
}

// Load loads the VAE decoder from a directory
func (m *VAEDecoder) Load(path string) error {
	fmt.Println("Loading VAE decoder...")

	// Load config
	cfg, err := loadVAEConfig(filepath.Join(path, "config.json"))
	if err != nil {
		return fmt.Errorf("config: %w", err)
	}
	m.Config = cfg

	// Load weights
	weights, err := safetensors.LoadModelWeights(path)
	if err != nil {
		return fmt.Errorf("weights: %w", err)
	}

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

	weights.ReleaseAll()
	return nil
}

// Decode decodes latents to images.
// Uses staged pools to free intermediate arrays and reduce peak memory.
func (vae *VAEDecoder) Decode(latents *mlx.Array) *mlx.Array {
	var h *mlx.Array
	{
		z := mlx.DivScalar(latents, vae.Config.ScalingFactor)
		z = mlx.AddScalar(z, vae.Config.ShiftFactor)
		h = vae.ConvIn.Forward(z)
		mlx.Eval(h)
	}

	h = vae.MidBlock.Forward(h)

	for _, upBlock := range vae.UpBlocks {
		h = upBlock.Forward(h)
	}

	{
			prev := h
		h = vae.ConvNormOut.Forward(h)
		h = mlx.SiLU(h)
		h = vae.ConvOut.Forward(h)
		// VAE outputs [-1, 1], convert to [0, 1]
		h = mlx.AddScalar(mlx.MulScalar(h, 0.5), 0.5)
		h = mlx.ClipScalar(h, 0.0, 1.0, true, true)
		prev.Free()
		mlx.Eval(h)
	}

	return h
}

// Upsample2x performs 2x nearest neighbor upsampling using broadcast.
// x: [B, C, H, W] -> [B, C, H*2, W*2]
func Upsample2x(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	C := shape[1]
	H := shape[2]
	W := shape[3]

	// [B, C, H, W] -> [B, C, H, 1, W, 1]
	x = mlx.Reshape(x, B, C, H, 1, W, 1)
	// Broadcast to [B, C, H, 2, W, 2]
	x = mlx.BroadcastTo(x, []int32{B, C, H, 2, W, 2})
	// Reshape to [B, C, H*2, W*2]
	x = mlx.Reshape(x, B, C, H*2, W*2)

	return x
}

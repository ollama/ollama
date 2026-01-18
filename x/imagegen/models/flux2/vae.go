//go:build mlx

package flux2

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// VAEConfig holds AutoencoderKLFlux2 configuration
type VAEConfig struct {
	ActFn             string  `json:"act_fn"`              // "silu"
	BatchNormEps      float32 `json:"batch_norm_eps"`      // 0.0001
	BatchNormMomentum float32 `json:"batch_norm_momentum"` // 0.1
	BlockOutChannels  []int32 `json:"block_out_channels"`  // [128, 256, 512, 512]
	ForceUpcast       bool    `json:"force_upcast"`        // true
	InChannels        int32   `json:"in_channels"`         // 3
	LatentChannels    int32   `json:"latent_channels"`     // 32
	LayersPerBlock    int32   `json:"layers_per_block"`    // 2
	MidBlockAddAttn   bool    `json:"mid_block_add_attention"` // true
	NormNumGroups     int32   `json:"norm_num_groups"`     // 32
	OutChannels       int32   `json:"out_channels"`        // 3
	PatchSize         []int32 `json:"patch_size"`          // [2, 2]
	SampleSize        int32   `json:"sample_size"`         // 1024
	UsePostQuantConv  bool    `json:"use_post_quant_conv"` // true
	UseQuantConv      bool    `json:"use_quant_conv"`      // true
}

// BatchNorm2D implements 2D batch normalization with running statistics
type BatchNorm2D struct {
	RunningMean *mlx.Array // [C]
	RunningVar  *mlx.Array // [C]
	Weight      *mlx.Array // [C] gamma
	Bias        *mlx.Array // [C] beta
	Eps         float32
	Momentum    float32
}

// Forward applies batch normalization (inference mode - uses running stats)
// Input and output are in NHWC format [B, H, W, C]
func (bn *BatchNorm2D) Forward(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	C := shape[3]

	// Reshape stats for broadcasting [1, 1, 1, C]
	mean := mlx.Reshape(bn.RunningMean, 1, 1, 1, C)
	variance := mlx.Reshape(bn.RunningVar, 1, 1, 1, C)

	// Normalize: (x - mean) / sqrt(var + eps)
	xNorm := mlx.Sub(x, mean)
	xNorm = mlx.Div(xNorm, mlx.Sqrt(mlx.AddScalar(variance, bn.Eps)))

	// Scale and shift (only if affine=True)
	if bn.Weight != nil {
		weight := mlx.Reshape(bn.Weight, 1, 1, 1, C)
		xNorm = mlx.Mul(xNorm, weight)
	}
	if bn.Bias != nil {
		bias := mlx.Reshape(bn.Bias, 1, 1, 1, C)
		xNorm = mlx.Add(xNorm, bias)
	}

	return xNorm
}

// Denormalize inverts the batch normalization
// Used when decoding latents
func (bn *BatchNorm2D) Denormalize(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	C := shape[3]

	// Reshape stats for broadcasting [1, 1, 1, C]
	mean := mlx.Reshape(bn.RunningMean, 1, 1, 1, C)
	variance := mlx.Reshape(bn.RunningVar, 1, 1, 1, C)

	// Inverse: first undo affine, then undo normalization
	// For affine=False: x_denorm = x * sqrt(var + eps) + mean
	if bn.Bias != nil {
		bias := mlx.Reshape(bn.Bias, 1, 1, 1, C)
		x = mlx.Sub(x, bias)
	}
	if bn.Weight != nil {
		weight := mlx.Reshape(bn.Weight, 1, 1, 1, C)
		x = mlx.Div(x, weight)
	}
	x = mlx.Mul(x, mlx.Sqrt(mlx.AddScalar(variance, bn.Eps)))
	x = mlx.Add(x, mean)

	return x
}

// GroupNormLayer implements group normalization
// Reused from zimage package pattern
type GroupNormLayer struct {
	Weight    *mlx.Array
	Bias      *mlx.Array
	NumGroups int32
	Eps       float32
}

// Forward applies group normalization
// Input and output are in NHWC format [B, H, W, C]
func (gn *GroupNormLayer) Forward(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	H := shape[1]
	W := shape[2]
	C := shape[3]

	// Reshape to [B, H, W, groups, C/groups]
	groupSize := C / gn.NumGroups
	x = mlx.Reshape(x, B, H, W, gn.NumGroups, groupSize)

	// Compute mean and variance per group
	mean := mlx.Mean(x, 1, true)
	mean = mlx.Mean(mean, 2, true)
	mean = mlx.Mean(mean, 4, true)

	xCentered := mlx.Sub(x, mean)

	sq := mlx.Square(xCentered)
	variance := mlx.Mean(sq, 1, true)
	variance = mlx.Mean(variance, 2, true)
	variance = mlx.Mean(variance, 4, true)

	// Normalize
	xNorm := mlx.Div(xCentered, mlx.Sqrt(mlx.AddScalar(variance, gn.Eps)))

	// Reshape back to [B, H, W, C]
	xNorm = mlx.Reshape(xNorm, B, H, W, C)

	// Scale and shift
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

// Conv2D represents a 2D convolution layer (reused pattern)
type Conv2D struct {
	Weight  *mlx.Array
	Bias    *mlx.Array
	Stride  int32
	Padding int32
}

// NewConv2D creates a Conv2D layer
// weight comes in as [out_channels, in_channels, kH, kW] (OIHW from PyTorch)
func NewConv2D(weight, bias *mlx.Array, stride, padding int32) *Conv2D {
	// Transpose weight from OIHW to OHWI for MLX
	weightOHWI := mlx.Transpose(weight, 0, 2, 3, 1)
	return &Conv2D{
		Weight:  weightOHWI,
		Bias:    bias,
		Stride:  stride,
		Padding: padding,
	}
}

// Forward applies convolution (NHWC format)
func (conv *Conv2D) Forward(x *mlx.Array) *mlx.Array {
	out := mlx.Conv2d(x, conv.Weight, conv.Stride, conv.Padding)

	if conv.Bias != nil {
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
	ConvShortcut *Conv2D
}

// Forward applies the ResNet block
func (rb *ResnetBlock2D) Forward(x *mlx.Array) *mlx.Array {
	h := rb.Norm1.Forward(x)
	h = mlx.SiLU(h)
	h = rb.Conv1.Forward(h)

	h = rb.Norm2.Forward(h)
	h = mlx.SiLU(h)
	h = rb.Conv2.Forward(h)

	if rb.ConvShortcut != nil {
		x = rb.ConvShortcut.Forward(x)
	}

	return mlx.Add(h, x)
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

// Forward applies attention (NHWC format)
func (ab *VAEAttentionBlock) Forward(x *mlx.Array) *mlx.Array {
	residual := x
	shape := x.Shape()
	B := shape[0]
	H := shape[1]
	W := shape[2]
	C := shape[3]

	h := ab.GroupNorm.Forward(x)
	h = mlx.Reshape(h, B, H*W, C)

	q := mlx.Linear(h, ab.ToQWeight)
	q = mlx.Add(q, ab.ToQBias)
	k := mlx.Linear(h, ab.ToKWeight)
	k = mlx.Add(k, ab.ToKBias)
	v := mlx.Linear(h, ab.ToVWeight)
	v = mlx.Add(v, ab.ToVBias)

	q = mlx.ExpandDims(q, 1)
	k = mlx.ExpandDims(k, 1)
	v = mlx.ExpandDims(v, 1)

	scale := float32(1.0 / math.Sqrt(float64(C)))
	out := mlx.ScaledDotProductAttention(q, k, v, scale, false)
	out = mlx.Squeeze(out, 1)

	out = mlx.Linear(out, ab.ToOutWeight)
	out = mlx.Add(out, ab.ToOutBias)
	out = mlx.Reshape(out, B, H, W, C)
	out = mlx.Add(out, residual)

	return out
}

// UpDecoderBlock2D implements an upsampling decoder block
type UpDecoderBlock2D struct {
	ResnetBlocks []*ResnetBlock2D
	Upsample     *Conv2D
}

// Forward applies the up decoder block
func (ub *UpDecoderBlock2D) Forward(x *mlx.Array) *mlx.Array {
	for _, resnet := range ub.ResnetBlocks {
		x = resnet.Forward(x)
	}

	if ub.Upsample != nil {
		x = upsample2x(x)
		x = ub.Upsample.Forward(x)
	}

	return x
}

// upsample2x performs 2x nearest neighbor upsampling
func upsample2x(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	H := shape[1]
	W := shape[2]

	hIdx := mlx.ArangeInt(0, H, 1, mlx.DtypeInt32)
	hIdx = mlx.Reshape(hIdx, H, 1)
	hIdx = mlx.BroadcastTo(hIdx, []int32{H, 2})
	hIdx = mlx.Reshape(hIdx, H*2)

	wIdx := mlx.ArangeInt(0, W, 1, mlx.DtypeInt32)
	wIdx = mlx.Reshape(wIdx, W, 1)
	wIdx = mlx.BroadcastTo(wIdx, []int32{W, 2})
	wIdx = mlx.Reshape(wIdx, W*2)

	x = mlx.Take(x, hIdx, 1)
	x = mlx.Take(x, wIdx, 2)

	return x
}

// VAEMidBlock is the middle block with attention
type VAEMidBlock struct {
	Resnet1   *ResnetBlock2D
	Attention *VAEAttentionBlock
	Resnet2   *ResnetBlock2D
}

// Forward applies the mid block
func (mb *VAEMidBlock) Forward(x *mlx.Array) *mlx.Array {
	x = mb.Resnet1.Forward(x)
	x = mb.Attention.Forward(x)
	x = mb.Resnet2.Forward(x)
	return x
}

// AutoencoderKLFlux2 is the Flux2 VAE with BatchNorm
type AutoencoderKLFlux2 struct {
	Config *VAEConfig

	// Encoder components (for image editing)
	EncoderConvIn  *Conv2D
	EncoderMid     *VAEMidBlock
	EncoderDown    []*DownEncoderBlock2D
	EncoderNormOut *GroupNormLayer
	EncoderConvOut *Conv2D

	// Decoder components
	DecoderConvIn  *Conv2D
	DecoderMid     *VAEMidBlock
	DecoderUp      []*UpDecoderBlock2D
	DecoderNormOut *GroupNormLayer
	DecoderConvOut *Conv2D

	// Quant conv layers
	QuantConv     *Conv2D
	PostQuantConv *Conv2D

	// BatchNorm for latent normalization
	LatentBN *BatchNorm2D
}

// DownEncoderBlock2D implements a downsampling encoder block
type DownEncoderBlock2D struct {
	ResnetBlocks []*ResnetBlock2D
	Downsample   *Conv2D
}

// Forward applies the down encoder block
func (db *DownEncoderBlock2D) Forward(x *mlx.Array) *mlx.Array {
	for _, resnet := range db.ResnetBlocks {
		x = resnet.Forward(x)
	}

	if db.Downsample != nil {
		// Pad then conv with stride 2
		x = mlx.Pad(x, []int32{0, 0, 0, 1, 0, 1, 0, 0})
		x = db.Downsample.Forward(x)
	}

	return x
}

// Load loads the Flux2 VAE from ollama blob storage.
func (m *AutoencoderKLFlux2) Load(manifest *imagegen.ModelManifest) error {
	fmt.Print("  Loading VAE... ")

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
func (m *AutoencoderKLFlux2) loadWeights(weights safetensors.WeightSource, cfg *VAEConfig) error {
	var err error

	// Load encoder components (for image conditioning)
	if err := m.loadEncoderWeights(weights, cfg); err != nil {
		return fmt.Errorf("encoder: %w", err)
	}

	// Load decoder conv_in
	convInWeight, err := weights.GetTensor("decoder.conv_in.weight")
	if err != nil {
		return fmt.Errorf("decoder.conv_in.weight: %w", err)
	}
	convInBias, err := weights.GetTensor("decoder.conv_in.bias")
	if err != nil {
		return fmt.Errorf("decoder.conv_in.bias: %w", err)
	}
	m.DecoderConvIn = NewConv2D(convInWeight, convInBias, 1, 1)

	// Load mid block
	m.DecoderMid, err = loadVAEMidBlock(weights, "decoder.mid_block", cfg.NormNumGroups)
	if err != nil {
		return fmt.Errorf("decoder.mid_block: %w", err)
	}

	// Load up blocks
	numBlocks := len(cfg.BlockOutChannels)
	m.DecoderUp = make([]*UpDecoderBlock2D, numBlocks)
	for i := 0; i < numBlocks; i++ {
		prefix := fmt.Sprintf("decoder.up_blocks.%d", i)
		hasUpsample := i < numBlocks-1
		m.DecoderUp[i], err = loadUpDecoderBlock2D(weights, prefix, cfg.LayersPerBlock+1, cfg.NormNumGroups, hasUpsample)
		if err != nil {
			return fmt.Errorf("%s: %w", prefix, err)
		}
	}

	// Load decoder conv_norm_out and conv_out
	normWeight, err := weights.GetTensor("decoder.conv_norm_out.weight")
	if err != nil {
		return fmt.Errorf("decoder.conv_norm_out.weight: %w", err)
	}
	normBias, err := weights.GetTensor("decoder.conv_norm_out.bias")
	if err != nil {
		return fmt.Errorf("decoder.conv_norm_out.bias: %w", err)
	}
	m.DecoderNormOut = &GroupNormLayer{
		Weight:    normWeight,
		Bias:      normBias,
		NumGroups: cfg.NormNumGroups,
		Eps:       1e-5,
	}

	convOutWeight, err := weights.GetTensor("decoder.conv_out.weight")
	if err != nil {
		return fmt.Errorf("decoder.conv_out.weight: %w", err)
	}
	convOutBias, err := weights.GetTensor("decoder.conv_out.bias")
	if err != nil {
		return fmt.Errorf("decoder.conv_out.bias: %w", err)
	}
	m.DecoderConvOut = NewConv2D(convOutWeight, convOutBias, 1, 1)

	// Load post_quant_conv
	if cfg.UsePostQuantConv {
		postQuantWeight, err := weights.GetTensor("post_quant_conv.weight")
		if err != nil {
			return fmt.Errorf("post_quant_conv.weight: %w", err)
		}
		postQuantBias, err := weights.GetTensor("post_quant_conv.bias")
		if err != nil {
			return fmt.Errorf("post_quant_conv.bias: %w", err)
		}
		m.PostQuantConv = NewConv2D(postQuantWeight, postQuantBias, 1, 0)
	}

	// Load latent BatchNorm (affine=False, so no weight/bias)
	bnMean, err := weights.GetTensor("bn.running_mean")
	if err != nil {
		return fmt.Errorf("bn.running_mean: %w", err)
	}
	bnVar, err := weights.GetTensor("bn.running_var")
	if err != nil {
		return fmt.Errorf("bn.running_var: %w", err)
	}
	m.LatentBN = &BatchNorm2D{
		RunningMean: bnMean,
		RunningVar:  bnVar,
		Weight:      nil, // affine=False
		Bias:        nil, // affine=False
		Eps:         cfg.BatchNormEps,
		Momentum:    cfg.BatchNormMomentum,
	}

	fmt.Println("✓")
	return nil
}

// loadVAEMidBlock loads the mid block
func loadVAEMidBlock(weights safetensors.WeightSource, prefix string, numGroups int32) (*VAEMidBlock, error) {
	resnet1, err := loadResnetBlock2D(weights, prefix+".resnets.0", numGroups)
	if err != nil {
		return nil, err
	}

	attention, err := loadVAEAttentionBlock(weights, prefix+".attentions.0", numGroups)
	if err != nil {
		return nil, err
	}

	resnet2, err := loadResnetBlock2D(weights, prefix+".resnets.1", numGroups)
	if err != nil {
		return nil, err
	}

	return &VAEMidBlock{
		Resnet1:   resnet1,
		Attention: attention,
		Resnet2:   resnet2,
	}, nil
}

// loadResnetBlock2D loads a ResNet block
func loadResnetBlock2D(weights safetensors.WeightSource, prefix string, numGroups int32) (*ResnetBlock2D, error) {
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
		Norm1: &GroupNormLayer{Weight: norm1Weight, Bias: norm1Bias, NumGroups: numGroups, Eps: 1e-5},
		Conv1: NewConv2D(conv1Weight, conv1Bias, 1, 1),
		Norm2: &GroupNormLayer{Weight: norm2Weight, Bias: norm2Bias, NumGroups: numGroups, Eps: 1e-5},
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

// loadVAEAttentionBlock loads an attention block
func loadVAEAttentionBlock(weights safetensors.WeightSource, prefix string, numGroups int32) (*VAEAttentionBlock, error) {
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
		GroupNorm:   &GroupNormLayer{Weight: normWeight, Bias: normBias, NumGroups: numGroups, Eps: 1e-5},
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

// loadUpDecoderBlock2D loads an up decoder block
func loadUpDecoderBlock2D(weights safetensors.WeightSource, prefix string, numLayers, numGroups int32, hasUpsample bool) (*UpDecoderBlock2D, error) {
	resnets := make([]*ResnetBlock2D, numLayers)
	for i := int32(0); i < numLayers; i++ {
		resPrefix := fmt.Sprintf("%s.resnets.%d", prefix, i)
		resnet, err := loadResnetBlock2D(weights, resPrefix, numGroups)
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

// Patchify converts latents [B, C, H, W] to patches [B, H*W/4, C*4] using 2x2 patches
// This is the inverse of the VAE's patchify for feeding to transformer
func (vae *AutoencoderKLFlux2) Patchify(latents *mlx.Array) *mlx.Array {
	shape := latents.Shape()
	B := shape[0]
	C := shape[1]
	H := shape[2]
	W := shape[3]

	patchH := vae.Config.PatchSize[0]
	patchW := vae.Config.PatchSize[1]

	pH := H / patchH
	pW := W / patchW

	// [B, C, H, W] -> [B, C, pH, patchH, pW, patchW]
	x := mlx.Reshape(latents, B, C, pH, patchH, pW, patchW)
	// [B, C, pH, patchH, pW, patchW] -> [B, pH, pW, C, patchH, patchW]
	x = mlx.Transpose(x, 0, 2, 4, 1, 3, 5)
	// [B, pH, pW, C, patchH, patchW] -> [B, pH*pW, C*patchH*patchW]
	return mlx.Reshape(x, B, pH*pW, C*patchH*patchW)
}

// Unpatchify converts patches [B, L, C*4] back to [B, C, H, W]
func (vae *AutoencoderKLFlux2) Unpatchify(patches *mlx.Array, pH, pW, C int32) *mlx.Array {
	shape := patches.Shape()
	B := shape[0]

	patchH := vae.Config.PatchSize[0]
	patchW := vae.Config.PatchSize[1]

	// [B, pH*pW, C*patchH*patchW] -> [B, pH, pW, C, patchH, patchW]
	x := mlx.Reshape(patches, B, pH, pW, C, patchH, patchW)
	// [B, pH, pW, C, patchH, patchW] -> [B, C, pH, patchH, pW, patchW]
	x = mlx.Transpose(x, 0, 3, 1, 4, 2, 5)
	// [B, C, pH, patchH, pW, patchW] -> [B, C, H, W]
	H := pH * patchH
	W := pW * patchW
	return mlx.Reshape(x, B, C, H, W)
}

// denormalizePatchified applies inverse batch normalization to patchified latents.
// Input: [B, L, 128] where 128 = 32 latent channels * 4 (2x2 patch)
// Output: [B, L, 128] denormalized
func (vae *AutoencoderKLFlux2) denormalizePatchified(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	C := shape[2] // 128

	// Reshape stats for broadcasting [1, 1, C]
	mean := mlx.Reshape(vae.LatentBN.RunningMean, 1, 1, C)
	variance := mlx.Reshape(vae.LatentBN.RunningVar, 1, 1, C)

	// Inverse BN (affine=False): x_denorm = x * sqrt(var + eps) + mean
	if vae.LatentBN.Bias != nil {
		bias := mlx.Reshape(vae.LatentBN.Bias, 1, 1, C)
		x = mlx.Sub(x, bias)
	}
	if vae.LatentBN.Weight != nil {
		weight := mlx.Reshape(vae.LatentBN.Weight, 1, 1, C)
		x = mlx.Div(x, weight)
	}
	x = mlx.Mul(x, mlx.Sqrt(mlx.AddScalar(variance, vae.LatentBN.Eps)))
	x = mlx.Add(x, mean)

	return x
}

// Decode decodes latent patches to images.
// latents: [B, L, C*4] patchified latents from transformer
// pH, pW: patch grid dimensions
// Returns: [B, 3, H, W] image tensor
func (vae *AutoencoderKLFlux2) Decode(latents *mlx.Array, pH, pW int32) *mlx.Array {
	// Denormalize patchified latents using BatchNorm
	// latents: [B, L, 128] where 128 = 32 latent channels * 4 (2x2 patch)
	// BatchNorm has 128 channels matching this dimension
	z := vae.denormalizePatchified(latents)

	// Unpatchify: [B, L, C*4] -> [B, C, H, W]
	z = vae.Unpatchify(z, pH, pW, vae.Config.LatentChannels)

	// Convert NCHW -> NHWC for processing
	z = mlx.Transpose(z, 0, 2, 3, 1)

	// Post-quant conv
	if vae.PostQuantConv != nil {
		z = vae.PostQuantConv.Forward(z)
	}

	// Decoder
	h := vae.DecoderConvIn.Forward(z)
	h = vae.DecoderMid.Forward(h)

	for _, upBlock := range vae.DecoderUp {
		h = upBlock.Forward(h)
	}

	h = vae.DecoderNormOut.Forward(h)
	h = mlx.SiLU(h)
	h = vae.DecoderConvOut.Forward(h)

	// VAE outputs [-1, 1], convert to [0, 1]
	h = mlx.MulScalar(h, 0.5)
	h = mlx.AddScalar(h, 0.5)
	h = mlx.ClipScalar(h, 0.0, 1.0, true, true)

	// Convert NHWC -> NCHW for output
	h = mlx.Transpose(h, 0, 3, 1, 2)

	return h
}

// loadEncoderWeights loads the encoder components for image conditioning
func (m *AutoencoderKLFlux2) loadEncoderWeights(weights safetensors.WeightSource, cfg *VAEConfig) error {
	// Load encoder conv_in
	convInWeight, err := weights.GetTensor("encoder.conv_in.weight")
	if err != nil {
		return fmt.Errorf("encoder.conv_in.weight: %w", err)
	}
	convInBias, err := weights.GetTensor("encoder.conv_in.bias")
	if err != nil {
		return fmt.Errorf("encoder.conv_in.bias: %w", err)
	}
	m.EncoderConvIn = NewConv2D(convInWeight, convInBias, 1, 1)

	// Load encoder down blocks
	numBlocks := len(cfg.BlockOutChannels)
	m.EncoderDown = make([]*DownEncoderBlock2D, numBlocks)
	for i := 0; i < numBlocks; i++ {
		prefix := fmt.Sprintf("encoder.down_blocks.%d", i)
		hasDownsample := i < numBlocks-1
		m.EncoderDown[i], err = loadDownEncoderBlock2D(weights, prefix, cfg.LayersPerBlock, cfg.NormNumGroups, hasDownsample)
		if err != nil {
			return fmt.Errorf("%s: %w", prefix, err)
		}
	}

	// Load encoder mid block
	m.EncoderMid, err = loadVAEMidBlock(weights, "encoder.mid_block", cfg.NormNumGroups)
	if err != nil {
		return fmt.Errorf("encoder.mid_block: %w", err)
	}

	// Load encoder conv_norm_out and conv_out
	normWeight, err := weights.GetTensor("encoder.conv_norm_out.weight")
	if err != nil {
		return fmt.Errorf("encoder.conv_norm_out.weight: %w", err)
	}
	normBias, err := weights.GetTensor("encoder.conv_norm_out.bias")
	if err != nil {
		return fmt.Errorf("encoder.conv_norm_out.bias: %w", err)
	}
	m.EncoderNormOut = &GroupNormLayer{
		Weight:    normWeight,
		Bias:      normBias,
		NumGroups: cfg.NormNumGroups,
		Eps:       1e-5,
	}

	convOutWeight, err := weights.GetTensor("encoder.conv_out.weight")
	if err != nil {
		return fmt.Errorf("encoder.conv_out.weight: %w", err)
	}
	convOutBias, err := weights.GetTensor("encoder.conv_out.bias")
	if err != nil {
		return fmt.Errorf("encoder.conv_out.bias: %w", err)
	}
	m.EncoderConvOut = NewConv2D(convOutWeight, convOutBias, 1, 1)

	// Load quant_conv (for encoding)
	if cfg.UseQuantConv {
		quantWeight, err := weights.GetTensor("quant_conv.weight")
		if err != nil {
			return fmt.Errorf("quant_conv.weight: %w", err)
		}
		quantBias, err := weights.GetTensor("quant_conv.bias")
		if err != nil {
			return fmt.Errorf("quant_conv.bias: %w", err)
		}
		m.QuantConv = NewConv2D(quantWeight, quantBias, 1, 0)
	}

	return nil
}

// loadDownEncoderBlock2D loads a down encoder block
func loadDownEncoderBlock2D(weights safetensors.WeightSource, prefix string, numLayers, numGroups int32, hasDownsample bool) (*DownEncoderBlock2D, error) {
	resnets := make([]*ResnetBlock2D, numLayers)
	for i := int32(0); i < numLayers; i++ {
		resPrefix := fmt.Sprintf("%s.resnets.%d", prefix, i)
		resnet, err := loadResnetBlock2D(weights, resPrefix, numGroups)
		if err != nil {
			return nil, err
		}
		resnets[i] = resnet
	}

	var downsample *Conv2D
	if hasDownsample {
		downWeight, err := weights.GetTensor(prefix + ".downsamplers.0.conv.weight")
		if err != nil {
			return nil, err
		}
		downBias, err := weights.GetTensor(prefix + ".downsamplers.0.conv.bias")
		if err != nil {
			return nil, err
		}
		downsample = NewConv2D(downWeight, downBias, 2, 0)
	}

	return &DownEncoderBlock2D{
		ResnetBlocks: resnets,
		Downsample:   downsample,
	}, nil
}

// EncodeImage encodes an image to normalized latents.
// image: [B, 3, H, W] image tensor in [-1, 1]
// Returns: [B, L, C*4] patchified normalized latents
func (vae *AutoencoderKLFlux2) EncodeImage(image *mlx.Array) *mlx.Array {
	// Convert NCHW -> NHWC
	x := mlx.Transpose(image, 0, 2, 3, 1)

	// Encoder
	h := vae.EncoderConvIn.Forward(x)

	for _, downBlock := range vae.EncoderDown {
		h = downBlock.Forward(h)
	}

	h = vae.EncoderMid.Forward(h)
	h = vae.EncoderNormOut.Forward(h)
	h = mlx.SiLU(h)
	h = vae.EncoderConvOut.Forward(h)

	// Quant conv outputs [B, H, W, 2*latent_channels] (mean + logvar)
	if vae.QuantConv != nil {
		h = vae.QuantConv.Forward(h)
	}

	// Take only the mean (first latent_channels) - deterministic encoding
	// h is [B, H, W, 64] -> take first 32 channels for mean
	shape := h.Shape()
	latentChannels := vae.Config.LatentChannels // 32
	h = mlx.Slice(h, []int32{0, 0, 0, 0}, []int32{shape[0], shape[1], shape[2], latentChannels})

	// Convert NHWC -> NCHW for patchifying
	h = mlx.Transpose(h, 0, 3, 1, 2)

	// Patchify: [B, C, H, W] -> [B, L, C*4]
	h = vae.Patchify(h)

	// Apply BatchNorm on patchified latents [B, L, 128]
	// The BatchNorm has 128 channels matching the patchified dimension
	h = vae.normalizePatchified(h)

	return h
}

// normalizePatchified applies batch normalization to patchified latents.
// Input: [B, L, 128] where 128 = 32 latent channels * 4 (2x2 patch)
// Output: [B, L, 128] normalized
func (vae *AutoencoderKLFlux2) normalizePatchified(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	C := shape[2] // 128

	// Reshape stats for broadcasting [1, 1, C]
	mean := mlx.Reshape(vae.LatentBN.RunningMean, 1, 1, C)
	variance := mlx.Reshape(vae.LatentBN.RunningVar, 1, 1, C)

	// Normalize: (x - mean) / sqrt(var + eps)
	xNorm := mlx.Sub(x, mean)
	xNorm = mlx.Div(xNorm, mlx.Sqrt(mlx.AddScalar(variance, vae.LatentBN.Eps)))

	// Scale and shift (only if affine=True)
	if vae.LatentBN.Weight != nil {
		weight := mlx.Reshape(vae.LatentBN.Weight, 1, 1, C)
		xNorm = mlx.Mul(xNorm, weight)
	}
	if vae.LatentBN.Bias != nil {
		bias := mlx.Reshape(vae.LatentBN.Bias, 1, 1, C)
		xNorm = mlx.Add(xNorm, bias)
	}

	return xNorm
}

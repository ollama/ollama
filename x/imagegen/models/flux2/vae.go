//go:build mlx

package flux2

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
	"github.com/ollama/ollama/x/imagegen/safetensors"
	"github.com/ollama/ollama/x/imagegen/vae"
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
	Weight    *mlx.Array `weight:"weight"`
	Bias      *mlx.Array `weight:"bias"`
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
	Weight  *mlx.Array `weight:"weight"`
	Bias    *mlx.Array `weight:"bias,optional"`
	Stride  int32
	Padding int32
}

// Transform implements safetensors.Transformer to transpose weights from PyTorch's OIHW to MLX's OHWI.
func (conv *Conv2D) Transform(field string, arr *mlx.Array) *mlx.Array {
	if field == "Weight" {
		return mlx.Transpose(arr, 0, 2, 3, 1)
	}
	return arr
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
	Norm1        *GroupNormLayer `weight:"norm1"`
	Conv1        *Conv2D         `weight:"conv1"`
	Norm2        *GroupNormLayer `weight:"norm2"`
	Conv2        *Conv2D         `weight:"conv2"`
	ConvShortcut *Conv2D         `weight:"conv_shortcut,optional"`
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
	GroupNorm *GroupNormLayer `weight:"group_norm"`
	ToQ       nn.LinearLayer  `weight:"to_q"`
	ToK       nn.LinearLayer  `weight:"to_k"`
	ToV       nn.LinearLayer  `weight:"to_v"`
	ToOut     nn.LinearLayer  `weight:"to_out.0"`
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

	q := ab.ToQ.Forward(h)
	k := ab.ToK.Forward(h)
	v := ab.ToV.Forward(h)

	q = mlx.ExpandDims(q, 1)
	k = mlx.ExpandDims(k, 1)
	v = mlx.ExpandDims(v, 1)

	scale := float32(1.0 / math.Sqrt(float64(C)))
	out := mlx.ScaledDotProductAttention(q, k, v, scale, false)
	out = mlx.Squeeze(out, 1)

	out = ab.ToOut.Forward(out)
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

// DefaultTilingConfig returns reasonable defaults for tiled decoding
// Matches diffusers: tile_latent_min_size=64, tile_overlap_factor=0.25
func DefaultTilingConfig() *vae.TilingConfig {
	return vae.DefaultTilingConfig()
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

	// Tiling configuration (nil = no tiling)
	Tiling *vae.TilingConfig
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
	m.DecoderConvIn = &Conv2D{Stride: 1, Padding: 1}
	if err := safetensors.LoadModule(m.DecoderConvIn, weights, "decoder.conv_in"); err != nil {
		return fmt.Errorf("decoder.conv_in: %w", err)
	}

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
	m.DecoderNormOut = &GroupNormLayer{NumGroups: cfg.NormNumGroups, Eps: 1e-5}
	if err := safetensors.LoadModule(m.DecoderNormOut, weights, "decoder.conv_norm_out"); err != nil {
		return fmt.Errorf("decoder.conv_norm_out: %w", err)
	}

	m.DecoderConvOut = &Conv2D{Stride: 1, Padding: 1}
	if err := safetensors.LoadModule(m.DecoderConvOut, weights, "decoder.conv_out"); err != nil {
		return fmt.Errorf("decoder.conv_out: %w", err)
	}

	// Load post_quant_conv
	if cfg.UsePostQuantConv {
		m.PostQuantConv = &Conv2D{Stride: 1, Padding: 0}
		if err := safetensors.LoadModule(m.PostQuantConv, weights, "post_quant_conv"); err != nil {
			return fmt.Errorf("post_quant_conv: %w", err)
		}
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

// loadVAEMidBlock loads the mid block.
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

// loadResnetBlock2D loads a ResNet block.
func loadResnetBlock2D(weights safetensors.WeightSource, prefix string, numGroups int32) (*ResnetBlock2D, error) {
	block := &ResnetBlock2D{
		Norm1:        &GroupNormLayer{NumGroups: numGroups, Eps: 1e-5},
		Conv1:        &Conv2D{Stride: 1, Padding: 1},
		Norm2:        &GroupNormLayer{NumGroups: numGroups, Eps: 1e-5},
		Conv2:        &Conv2D{Stride: 1, Padding: 1},
		ConvShortcut: &Conv2D{Stride: 1, Padding: 0}, // Pre-allocate for optional loading
	}
	if err := safetensors.LoadModule(block, weights, prefix); err != nil {
		return nil, err
	}
	// If ConvShortcut wasn't loaded (no weights found), nil it out
	if block.ConvShortcut.Weight == nil {
		block.ConvShortcut = nil
	}
	return block, nil
}

// loadVAEAttentionBlock loads an attention block using LoadModule.
func loadVAEAttentionBlock(weights safetensors.WeightSource, prefix string, numGroups int32) (*VAEAttentionBlock, error) {
	ab := &VAEAttentionBlock{
		GroupNorm: &GroupNormLayer{NumGroups: numGroups, Eps: 1e-5},
	}
	if err := safetensors.LoadModule(ab, weights, prefix); err != nil {
		return nil, err
	}
	return ab, nil
}

// loadUpDecoderBlock2D loads an up decoder block.
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
		upsample = &Conv2D{Stride: 1, Padding: 1}
		if err := safetensors.LoadModule(upsample, weights, prefix+".upsamplers.0.conv"); err != nil {
			return nil, err
		}
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
// If Tiling is set, uses tiled decoding to reduce memory for large images.
// latents: [B, L, C*4] patchified latents from transformer
// pH, pW: patch grid dimensions
// Returns: [B, 3, H, W] image tensor
func (v *AutoencoderKLFlux2) Decode(latents *mlx.Array, pH, pW int32) *mlx.Array {
	// Denormalize patchified latents
	z := v.denormalizePatchified(latents)

	// Unpatchify: [B, L, C*4] -> [B, C, H, W]
	z = v.Unpatchify(z, pH, pW, v.Config.LatentChannels)

	// Convert NCHW -> NHWC for processing
	z = mlx.Transpose(z, 0, 2, 3, 1)

	// Use tiled decoding if enabled
	if v.Tiling != nil {
		mlx.Eval(z)
		return vae.DecodeTiled(z, v.Tiling, v.decodeTile)
	}

	// Direct decode (no tiling)
	h := v.decodeTile(z)
	h = mlx.ClipScalar(h, 0.0, 1.0, true, true)
	h = mlx.Transpose(h, 0, 3, 1, 2)
	return h
}

// decodeTile decodes a single latent tile to pixels (internal helper)
// z: [B, H, W, C] latent tile in NHWC format
// Returns: [B, H*8, W*8, 3] pixel tile in NHWC format (before clipping)
func (vae *AutoencoderKLFlux2) decodeTile(z *mlx.Array) *mlx.Array {
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

	return h
}

// loadEncoderWeights loads the encoder components for image conditioning
func (m *AutoencoderKLFlux2) loadEncoderWeights(weights safetensors.WeightSource, cfg *VAEConfig) error {
	var err error

	// Load encoder conv_in
	m.EncoderConvIn = &Conv2D{Stride: 1, Padding: 1}
	if err := safetensors.LoadModule(m.EncoderConvIn, weights, "encoder.conv_in"); err != nil {
		return fmt.Errorf("encoder.conv_in: %w", err)
	}

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
	m.EncoderNormOut = &GroupNormLayer{NumGroups: cfg.NormNumGroups, Eps: 1e-5}
	if err := safetensors.LoadModule(m.EncoderNormOut, weights, "encoder.conv_norm_out"); err != nil {
		return fmt.Errorf("encoder.conv_norm_out: %w", err)
	}

	m.EncoderConvOut = &Conv2D{Stride: 1, Padding: 1}
	if err := safetensors.LoadModule(m.EncoderConvOut, weights, "encoder.conv_out"); err != nil {
		return fmt.Errorf("encoder.conv_out: %w", err)
	}

	// Load quant_conv (for encoding)
	if cfg.UseQuantConv {
		m.QuantConv = &Conv2D{Stride: 1, Padding: 0}
		if err := safetensors.LoadModule(m.QuantConv, weights, "quant_conv"); err != nil {
			return fmt.Errorf("quant_conv: %w", err)
		}
	}

	return nil
}

// loadDownEncoderBlock2D loads a down encoder block.
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
		downsample = &Conv2D{Stride: 2, Padding: 0}
		if err := safetensors.LoadModule(downsample, weights, prefix+".downsamplers.0.conv"); err != nil {
			return nil, err
		}
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

//go:build mlx

package glm_image

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// VAEConfig holds VAE decoder configuration
type VAEConfig struct {
	InChannels       int32     `json:"in_channels"`        // 3
	OutChannels      int32     `json:"out_channels"`       // 3
	LatentChannels   int32     `json:"latent_channels"`    // 16
	BlockOutChannels []int32   `json:"block_out_channels"` // [128, 512, 1024, 1024]
	LayersPerBlock   int32     `json:"layers_per_block"`   // 3
	NormNumGroups    int32     `json:"norm_num_groups"`    // 32
	ScalingFactor    float32   `json:"scaling_factor"`     // 0.18215
	ShiftFactor      *float32  `json:"shift_factor"`       // null
	LatentsMean      []float32 `json:"latents_mean"`       // [16 values]
	LatentsStd       []float32 `json:"latents_std"`        // [16 values]
}

// VAEDecoder is the VAE latent decoder
type VAEDecoder struct {
	Config *VAEConfig

	// Decoder components
	ConvIn  *VAEConv2d    `weight:"decoder.conv_in"`
	MidBlock *VAEMidBlock `weight:"decoder.mid_block"`
	UpBlocks []*VAEUpBlock `weight:"decoder.up_blocks"`
	ConvNormOut *GroupNorm `weight:"decoder.conv_norm_out"`
	ConvOut *VAEConv2d    `weight:"decoder.conv_out"`
}

// VAEConv2d is a 2D convolution layer
type VAEConv2d struct {
	Weight *mlx.Array `weight:"weight"`
	Bias   *mlx.Array `weight:"bias"`
	Stride int32
	Padding int32
}

// GroupNorm is group normalization
type GroupNorm struct {
	Weight    *mlx.Array `weight:"weight"`
	Bias      *mlx.Array `weight:"bias"`
	NumGroups int32
	Eps       float32
}

// VAEMidBlock is the middle block of the VAE
type VAEMidBlock struct {
	Resnets []*VAEResnetBlock `weight:"resnets"`
}

// VAEUpBlock is an upsampling block
type VAEUpBlock struct {
	Resnets []*VAEResnetBlock `weight:"resnets"`
	Upsamplers []*VAEUpsampler `weight:"upsamplers"`
}

// VAEResnetBlock is a residual block
type VAEResnetBlock struct {
	Norm1   *GroupNorm `weight:"norm1"`
	Conv1   *VAEConv2d `weight:"conv1"`
	Norm2   *GroupNorm `weight:"norm2"`
	Conv2   *VAEConv2d `weight:"conv2"`
	ConvShortcut *VAEConv2d `weight:"conv_shortcut,optional"` // Optional, for channel mismatch
}

// VAEUpsampler is an upsampling layer
type VAEUpsampler struct {
	Conv *VAEConv2d `weight:"conv"`
}

// Load loads the VAE decoder from manifest
func (m *VAEDecoder) Load(manifest *imagegen.ModelManifest) error {
	fmt.Print("  Loading VAE decoder... ")

	// Load config
	var cfg VAEConfig
	if err := manifest.ReadConfigJSON("vae/config.json", &cfg); err != nil {
		return fmt.Errorf("config: %w", err)
	}
	m.Config = &cfg

	// Initialize structure based on config
	numBlocks := len(cfg.BlockOutChannels)
	m.UpBlocks = make([]*VAEUpBlock, numBlocks)

	// Pre-allocate MidBlock resnets (VAE mid_block typically has 2 resnets)
	m.MidBlock = &VAEMidBlock{
		Resnets: make([]*VAEResnetBlock, 2),
	}

	// Pre-allocate UpBlocks with their resnets and upsamplers
	// VAE decoder has layers_per_block+1 resnets per up_block (to match encoder)
	// And all but the last up_block has an upsampler
	for i := 0; i < numBlocks; i++ {
		numResnets := cfg.LayersPerBlock + 1 // typically 4 resnets
		m.UpBlocks[i] = &VAEUpBlock{
			Resnets: make([]*VAEResnetBlock, numResnets),
		}
		// All but the last block has upsamplers
		if i < numBlocks-1 {
			m.UpBlocks[i].Upsamplers = make([]*VAEUpsampler, 1)
		}
	}

	// Load weights
	weights, err := imagegen.LoadWeightsFromManifest(manifest, "vae")
	if err != nil {
		return fmt.Errorf("weights: %w", err)
	}
	if err := weights.Load(mlx.DtypeBFloat16); err != nil {
		return fmt.Errorf("load weights: %w", err)
	}
	defer weights.ReleaseAll()

	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return fmt.Errorf("load module: %w", err)
	}

	// Initialize GroupNorm parameters
	m.initGroupNorms()

	fmt.Println("✓")
	return nil
}

// LoadFromPath loads the VAE decoder from a directory path
func (m *VAEDecoder) LoadFromPath(path string) error {
	fmt.Print("  Loading VAE decoder... ")

	// Load config
	var cfg VAEConfig
	configPath := filepath.Join(path, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("read config: %w", err)
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return fmt.Errorf("parse config: %w", err)
	}
	m.Config = &cfg

	// Initialize structure based on config
	numBlocks := len(cfg.BlockOutChannels)
	m.UpBlocks = make([]*VAEUpBlock, numBlocks)

	// Pre-allocate MidBlock resnets (VAE mid_block typically has 2 resnets)
	m.MidBlock = &VAEMidBlock{
		Resnets: make([]*VAEResnetBlock, 2),
	}

	// Pre-allocate UpBlocks with their resnets and upsamplers
	for i := 0; i < numBlocks; i++ {
		numResnets := cfg.LayersPerBlock + 1
		m.UpBlocks[i] = &VAEUpBlock{
			Resnets: make([]*VAEResnetBlock, numResnets),
		}
		if i < numBlocks-1 {
			m.UpBlocks[i].Upsamplers = make([]*VAEUpsampler, 1)
		}
	}

	// Load weights from safetensors files
	weights, err := safetensors.LoadModelWeights(path)
	if err != nil {
		return fmt.Errorf("weights: %w", err)
	}
	if err := weights.Load(mlx.DtypeBFloat16); err != nil {
		return fmt.Errorf("load weights: %w", err)
	}
	defer weights.ReleaseAll()

	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return fmt.Errorf("load module: %w", err)
	}

	// Initialize GroupNorm parameters
	m.initGroupNorms()

	fmt.Println("✓")
	return nil
}

func (m *VAEDecoder) initGroupNorms() {
	cfg := m.Config
	numGroups := cfg.NormNumGroups
	eps := float32(1e-6) // Must match diffusers VAE (1e-6, not 1e-5)

	if m.ConvNormOut != nil {
		m.ConvNormOut.NumGroups = numGroups
		m.ConvNormOut.Eps = eps
	}

	if m.MidBlock != nil {
		for _, resnet := range m.MidBlock.Resnets {
			if resnet.Norm1 != nil {
				resnet.Norm1.NumGroups = numGroups
				resnet.Norm1.Eps = eps
			}
			if resnet.Norm2 != nil {
				resnet.Norm2.NumGroups = numGroups
				resnet.Norm2.Eps = eps
			}
		}
	}

	for _, upBlock := range m.UpBlocks {
		if upBlock == nil {
			continue
		}
		for _, resnet := range upBlock.Resnets {
			if resnet == nil {
				continue
			}
			if resnet.Norm1 != nil {
				resnet.Norm1.NumGroups = numGroups
				resnet.Norm1.Eps = eps
			}
			if resnet.Norm2 != nil {
				resnet.Norm2.NumGroups = numGroups
				resnet.Norm2.Eps = eps
			}
		}
	}
}

// Decode decodes latents to an image
func (m *VAEDecoder) Decode(latents *mlx.Array) *mlx.Array {
	cfg := m.Config

	// Apply latent denormalization if mean/std are provided
	// This matches diffusers GLM-Image: latents = latents * std + mean
	// Note: GLM-Image does NOT divide by scaling_factor (unlike standard SD VAEs)
	if len(cfg.LatentsMean) > 0 && len(cfg.LatentsStd) > 0 {
		latents = m.denormalizeLatents(latents)
	}

	// Convert from NCHW to NHWC for processing
	// [B, C, H, W] -> [B, H, W, C]
	x := mlx.Transpose(latents, 0, 2, 3, 1)

	// Initial convolution
	x = m.ConvIn.Forward(x)

	// Mid block
	x = m.MidBlock.Forward(x)

	// Up blocks (forward order - index 0 is at lowest resolution/highest channels)
	for i := 0; i < len(m.UpBlocks); i++ {
		if m.UpBlocks[i] != nil {
			x = m.UpBlocks[i].Forward(x)
		}
	}

	// Final normalization and convolution
	x = m.ConvNormOut.Forward(x)
	x = mlx.SiLU(x)
	x = m.ConvOut.Forward(x)

	// Convert back to NCHW
	// [B, H, W, C] -> [B, C, H, W]
	x = mlx.Transpose(x, 0, 3, 1, 2)

	// Clamp to valid range and convert to [0, 1]
	x = mlx.ClipScalar(x, -1.0, 1.0, true, true)
	x = mlx.AddScalar(x, 1.0)
	x = mlx.DivScalar(x, 2.0)

	return x
}

// denormalizeLatents applies the latent mean/std denormalization
func (m *VAEDecoder) denormalizeLatents(latents *mlx.Array) *mlx.Array {
	cfg := m.Config

	// Create mean and std arrays [1, C, 1, 1] for broadcasting
	mean := mlx.NewArray(cfg.LatentsMean, []int32{1, int32(len(cfg.LatentsMean)), 1, 1})
	std := mlx.NewArray(cfg.LatentsStd, []int32{1, int32(len(cfg.LatentsStd)), 1, 1})

	// Denormalize: latents * std + mean
	latents = mlx.Mul(latents, std)
	latents = mlx.Add(latents, mean)

	return latents
}

// Forward for VAEConv2d
func (c *VAEConv2d) Forward(x *mlx.Array) *mlx.Array {
	// x: [B, H, W, C_in] (NHWC)
	// PyTorch weight: [C_out, C_in, kH, kW] (OIHW)
	// MLX conv2d expects weight: [C_out, kH, kW, C_in] (OHWI)
	// So we need to transpose from OIHW to OHWI

	stride := c.Stride
	if stride == 0 {
		stride = 1
	}
	padding := c.Padding
	if padding == 0 {
		// Default to same padding for 3x3 kernels
		wShape := c.Weight.Shape()
		if len(wShape) >= 3 && wShape[2] == 3 {
			padding = 1
		}
	}

	// Transpose weight from OIHW [out, in, h, w] to OHWI [out, h, w, in]
	weight := mlx.Transpose(c.Weight, 0, 2, 3, 1)

	out := mlx.Conv2d(x, weight, stride, padding)
	if c.Bias != nil {
		// Bias: [C_out] -> [1, 1, 1, C_out]
		bias := mlx.Reshape(c.Bias, 1, 1, 1, -1)
		out = mlx.Add(out, bias)
	}
	return out
}

// Forward for GroupNorm
func (gn *GroupNorm) Forward(x *mlx.Array) *mlx.Array {
	// x: [B, H, W, C] (NHWC)
	shape := x.Shape()
	B := shape[0]
	H := shape[1]
	W := shape[2]
	C := shape[3]

	numGroups := gn.NumGroups
	if numGroups == 0 {
		numGroups = 32
	}
	groupSize := C / numGroups

	// Reshape to [B, H, W, groups, groupSize]
	x = mlx.Reshape(x, B, H, W, numGroups, groupSize)

	// Compute mean and variance per group
	mean := mlx.Mean(x, 1, true)
	mean = mlx.Mean(mean, 2, true)
	mean = mlx.Mean(mean, 4, true)

	xCentered := mlx.Sub(x, mean)
	variance := mlx.Mean(mlx.Square(xCentered), 1, true)
	variance = mlx.Mean(variance, 2, true)
	variance = mlx.Mean(variance, 4, true)

	// Normalize
	xNorm := mlx.Div(xCentered, mlx.Sqrt(mlx.AddScalar(variance, gn.Eps)))

	// Reshape back
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

// Forward for VAEMidBlock
func (mb *VAEMidBlock) Forward(x *mlx.Array) *mlx.Array {
	for _, resnet := range mb.Resnets {
		x = resnet.Forward(x)
	}
	return x
}

// Forward for VAEUpBlock
func (ub *VAEUpBlock) Forward(x *mlx.Array) *mlx.Array {
	// Apply resnets
	for _, resnet := range ub.Resnets {
		if resnet != nil {
			x = resnet.Forward(x)
		}
	}

	// Apply upsamplers
	for _, upsampler := range ub.Upsamplers {
		if upsampler != nil {
			x = upsampler.Forward(x)
		}
	}

	return x
}

// Forward for VAEResnetBlock
func (rb *VAEResnetBlock) Forward(x *mlx.Array) *mlx.Array {
	residual := x

	// First norm + activation + conv
	h := rb.Norm1.Forward(x)
	h = mlx.SiLU(h)
	h = rb.Conv1.Forward(h)

	// Second norm + activation + conv
	h = rb.Norm2.Forward(h)
	h = mlx.SiLU(h)
	h = rb.Conv2.Forward(h)

	// Shortcut for channel mismatch
	if rb.ConvShortcut != nil {
		residual = rb.ConvShortcut.Forward(residual)
	}

	return mlx.Add(h, residual)
}

// Forward for VAEUpsampler (2x nearest neighbor upsample + conv)
func (us *VAEUpsampler) Forward(x *mlx.Array) *mlx.Array {
	// x: [B, H, W, C]
	// 2x nearest neighbor upsample
	x = upsample2x(x)

	// Conv
	if us.Conv != nil {
		x = us.Conv.Forward(x)
	}

	return x
}

// upsample2x performs 2x nearest neighbor upsampling.
// Input and output are in NHWC format: [B, H, W, C] -> [B, H*2, W*2, C]
func upsample2x(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	H := shape[1]
	W := shape[2]
	C := shape[3]

	// Create indices [0, 0, 1, 1, 2, 2, ...] for nearest neighbor
	hIndices := make([]int32, H*2)
	for i := int32(0); i < H; i++ {
		hIndices[i*2] = i
		hIndices[i*2+1] = i
	}
	wIndices := make([]int32, W*2)
	for i := int32(0); i < W; i++ {
		wIndices[i*2] = i
		wIndices[i*2+1] = i
	}

	hIdx := mlx.NewArrayInt32(hIndices, []int32{H * 2})
	wIdx := mlx.NewArrayInt32(wIndices, []int32{W * 2})

	// Take along height axis
	x = mlx.Reshape(x, B*H, W, C)
	x = mlx.Take(x, wIdx, 1) // [B*H, W*2, C]
	x = mlx.Reshape(x, B, H, W*2, C)

	// Take along width axis - transpose to [B, W*2, H, C], take, transpose back
	x = mlx.Transpose(x, 0, 2, 1, 3) // [B, W*2, H, C]
	x = mlx.Reshape(x, B*(W*2), H, C)
	x = mlx.Take(x, hIdx, 1) // [B*(W*2), H*2, C]
	x = mlx.Reshape(x, B, W*2, H*2, C)
	x = mlx.Transpose(x, 0, 2, 1, 3) // [B, H*2, W*2, C]

	return x
}

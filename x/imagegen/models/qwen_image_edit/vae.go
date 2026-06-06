//go:build mlx

package qwen_image_edit

import (
	"fmt"

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

// VAE is the full VAE with encoder and decoder
type VAE struct {
	Config  *VAEConfig
	Encoder *VAEEncoder
	Decoder *VAEDecoder
}

// Load loads the VAE from a directory
func (m *VAE) Load(path string) error {
	fmt.Println("Loading Qwen-Image-Edit VAE (encoder + decoder)...")

	cfg := defaultVAEConfig()
	m.Config = cfg

	weights, err := safetensors.LoadModelWeights(path)
	if err != nil {
		return fmt.Errorf("weights: %w", err)
	}

	// Load weights as f32 for quality (matches Python default behavior)
	// VAE decoder precision is critical for final image quality
	fmt.Print("  Loading weights as f32... ")
	if err := weights.Load(mlx.DtypeFloat32); err != nil {
		return fmt.Errorf("failed to load weights: %w", err)
	}
	fmt.Printf("✓ (%.1f GB)\n", float64(mlx.MetalGetActiveMemory())/(1024*1024*1024))

	// Load encoder
	fmt.Print("  Loading encoder... ")
	m.Encoder = &VAEEncoder{}
	if err := m.Encoder.loadFromWeights(weights, cfg); err != nil {
		return fmt.Errorf("encoder: %w", err)
	}
	fmt.Println("✓")

	// Load decoder
	fmt.Print("  Loading decoder... ")
	m.Decoder = &VAEDecoder{}
	if err := m.Decoder.loadFromWeights(weights, cfg); err != nil {
		return fmt.Errorf("decoder: %w", err)
	}
	fmt.Println("✓")

	weights.ReleaseAll()
	return nil
}

// Encode encodes an image to latents
// x: [B, C, T, H, W] image tensor in [-1, 1] range
// Returns: [B, C, T, H/8, W/8] latents (unnormalized)
func (m *VAE) Encode(x *mlx.Array) *mlx.Array {
	return m.Encoder.Encode(x)
}

// Decode decodes latents to image
// z: [B, C, T, H, W] latents (denormalized)
// Returns: [B, C, T, H*8, W*8] image in [-1, 1]
func (m *VAE) Decode(z *mlx.Array) *mlx.Array {
	return m.Decoder.Decode(z)
}

// Normalize applies latent normalization
// Input z should be f32 (from VAE encoder), output is f32 for transformer
func (m *VAE) Normalize(z *mlx.Array) *mlx.Array {
	shape := z.Shape()
	C := shape[1]

	mean := mlx.NewArray(m.Config.LatentsMean[:C], []int32{1, C, 1, 1, 1})
	std := mlx.NewArray(m.Config.LatentsStd[:C], []int32{1, C, 1, 1, 1})

	// Mean/std are f32, will match z dtype through broadcasting
	return mlx.Div(mlx.Sub(z, mean), std)
}

// Denormalize reverses latent normalization
// Input z is bf16 (from transformer), output converted to f32 for VAE decoder
func (m *VAE) Denormalize(z *mlx.Array) *mlx.Array {
	shape := z.Shape()
	C := shape[1]

	// Convert latents to f32 for VAE decoder quality
	z = mlx.AsType(z, mlx.DtypeFloat32)

	mean := mlx.NewArray(m.Config.LatentsMean[:C], []int32{1, C, 1, 1, 1})
	std := mlx.NewArray(m.Config.LatentsStd[:C], []int32{1, C, 1, 1, 1})

	return mlx.Add(mlx.Mul(z, std), mean)
}

// VAEEncoder is the encoder part of the VAE
// The encoder uses a flat structure where down_blocks contains a mix of ResBlocks and Downsamplers:
// - Blocks 0,1: ResBlocks (base_dim)
// - Block 2: Downsample
// - Blocks 3,4: ResBlocks (base_dim*2)
// - Block 5: Downsample + temporal
// - Blocks 6,7: ResBlocks (base_dim*4)
// - Block 8: Downsample + temporal
// - Blocks 9,10: ResBlocks (base_dim*4)
type VAEEncoder struct {
	Config *VAEConfig

	ConvIn     *CausalConv3d
	Blocks     []EncoderBlock // Flat list of ResBlocks and Downsamplers
	MidBlock   *MidBlock
	NormOut    *RMSNorm3D
	ConvOut    *CausalConv3d
	QuantConv  *CausalConv3d
}

// EncoderBlock is either a ResBlock or a Downsample
type EncoderBlock interface {
	Forward(x *mlx.Array) *mlx.Array
	IsDownsample() bool
}

// EncoderResBlock wraps ResBlock
type EncoderResBlock struct {
	*ResBlock
}

func (b *EncoderResBlock) IsDownsample() bool { return false }

// EncoderDownsample is a downsample layer
type EncoderDownsample struct {
	Resample *CausalConv3d
	TimeConv *CausalConv3d // Optional temporal downsample
}

func (d *EncoderDownsample) IsDownsample() bool { return true }

func (d *EncoderDownsample) Forward(x *mlx.Array) *mlx.Array {
	// Spatial downsample with stride 2
	// WAN VAE uses: ZeroPad2d(0,1,0,1) + Conv2d(3x3, stride=2)
	x = d.forwardSpatialDownsample(x)

	// NOTE: In WAN VAE, time_conv is ONLY used in streaming/chunked mode
	// with feat_cache. For single-frame encoding (T=1), time_conv is skipped.
	// The Python forward checks: if feat_cache is not None ... then use time_conv
	// Since we don't support streaming, we skip time_conv entirely.
	return x
}

// forwardSpatialDownsample applies 2D conv with stride 2 for spatial downsampling
func (d *EncoderDownsample) forwardSpatialDownsample(x *mlx.Array) *mlx.Array {
	xShape := x.Shape()
	B := xShape[0]
	T := xShape[1]
	H := xShape[2]
	W := xShape[3]
	C := xShape[4]

	wShape := d.Resample.Weight.Shape()
	outC := wShape[0]

	// Reshape to [B*T, H, W, C] for 2D conv
	x = mlx.Reshape(x, B*T, H, W, C)

	// Asymmetric padding: pad right and bottom by 1 (WAN VAE style)
	// ZeroPad2d(0, 1, 0, 1) means (left=0, right=1, top=0, bottom=1)
	x = mlx.Pad(x, []int32{0, 0, 0, 1, 0, 1, 0, 0}) // [B, H, W, C] -> pad H and W

	// Apply 2D conv with stride 2
	weight := mlx.Transpose(d.Resample.Weight, 0, 2, 3, 1) // [O, I, kH, kW] -> [O, kH, kW, I]
	x = conv2DStrided(x, weight, 2)

	if d.Resample.Bias != nil {
		bias := mlx.Reshape(d.Resample.Bias, 1, 1, 1, outC)
		x = mlx.Add(x, bias)
	}

	// Output dims after stride 2: (H+1)/2, (W+1)/2
	outH := (H + 1) / 2
	outW := (W + 1) / 2

	// Reshape back to [B, T, H', W', C]
	x = mlx.Reshape(x, B, T, outH, outW, outC)
	mlx.Eval(x)

	return x
}

// loadFromWeights loads the encoder from pre-loaded weights
func (e *VAEEncoder) loadFromWeights(weights *safetensors.ModelWeights, cfg *VAEConfig) error {
	e.Config = cfg

	// Conv in
	convIn, err := newCausalConv3d(weights, "encoder.conv_in")
	if err != nil {
		return err
	}
	e.ConvIn = convIn

	// Encoder uses flat block structure:
	// dim_mult = [1, 2, 4, 4], num_res_blocks = 2, temporal_downsample = [false, true, true]
	// Block layout: res,res,down, res,res,down+t, res,res,down+t, res,res
	// That's 11 blocks: 0,1=res, 2=down, 3,4=res, 5=down+t, 6,7=res, 8=down+t, 9,10=res
	e.Blocks = make([]EncoderBlock, 0, 11)

	// Track dimensions
	dims := []int32{cfg.BaseDim, cfg.BaseDim * 2, cfg.BaseDim * 4, cfg.BaseDim * 4}
	blockIdx := 0

	for stage := 0; stage < len(cfg.DimMult); stage++ {
		inDim := cfg.BaseDim
		if stage > 0 {
			inDim = dims[stage-1]
		}
		outDim := dims[stage]

		// ResBlocks for this stage (num_res_blocks per stage)
		for r := int32(0); r < cfg.NumResBlocks; r++ {
			prefix := fmt.Sprintf("encoder.down_blocks.%d", blockIdx)
			currentInDim := inDim
			if r > 0 {
				currentInDim = outDim
			}
			block, err := newEncoderResBlock(weights, prefix, currentInDim, outDim)
			if err != nil {
				return fmt.Errorf("encoder res block %d: %w", blockIdx, err)
			}
			e.Blocks = append(e.Blocks, block)
			blockIdx++
		}

		// Downsample after each stage except the last
		if stage < len(cfg.DimMult)-1 {
			prefix := fmt.Sprintf("encoder.down_blocks.%d", blockIdx)
			down, err := newEncoderDownsample(weights, prefix, cfg.TemperalDownsample[stage])
			if err != nil {
				return fmt.Errorf("encoder downsample %d: %w", blockIdx, err)
			}
			e.Blocks = append(e.Blocks, down)
			blockIdx++
		}
	}

	// Mid block
	midDim := cfg.BaseDim * cfg.DimMult[len(cfg.DimMult)-1]
	midBlock, err := newMidBlock(weights, "encoder.mid_block", midDim)
	if err != nil {
		return err
	}
	e.MidBlock = midBlock

	// Norm out
	normOut, err := newRMSNorm3D(weights, "encoder.norm_out", midDim)
	if err != nil {
		return err
	}
	e.NormOut = normOut

	// Conv out
	convOut, err := newCausalConv3d(weights, "encoder.conv_out")
	if err != nil {
		return err
	}
	e.ConvOut = convOut

	// Quant conv
	quantConv, err := newCausalConv3d(weights, "quant_conv")
	if err != nil {
		return err
	}
	e.QuantConv = quantConv

	return nil
}

// newEncoderResBlock creates a ResBlock for the encoder (flat structure)
func newEncoderResBlock(weights *safetensors.ModelWeights, prefix string, inDim, outDim int32) (*EncoderResBlock, error) {
	block, err := newResBlock(weights, prefix, inDim, outDim)
	if err != nil {
		return nil, err
	}
	return &EncoderResBlock{block}, nil
}

// newEncoderDownsample creates a downsample layer for the encoder
func newEncoderDownsample(weights *safetensors.ModelWeights, prefix string, temporal bool) (*EncoderDownsample, error) {
	resample, err := newCausalConv3d(weights, prefix+".resample.1")
	if err != nil {
		return nil, err
	}

	var timeConv *CausalConv3d
	if temporal {
		timeConv, _ = newCausalConv3d(weights, prefix+".time_conv")
	}

	return &EncoderDownsample{
		Resample: resample,
		TimeConv: timeConv,
	}, nil
}

// Encode encodes an image to latents
// x: [B, C, T, H, W] image tensor (channels-first)
// Returns: [B, latent_C, T, H/8, W/8] latent distribution mode
func (e *VAEEncoder) Encode(x *mlx.Array) *mlx.Array {
	// Convert from channels-first [N, C, T, H, W] to channels-last [N, T, H, W, C]
	x = mlx.Contiguous(mlx.Transpose(x, 0, 2, 3, 4, 1))
	mlx.Eval(x)

	// Conv in
	x = e.ConvIn.Forward(x)

	// Encoder blocks (mix of ResBlocks and Downsamplers)
	for _, block := range e.Blocks {
		prev := x
		x = block.Forward(x)
		prev.Free()
	}

	// Mid block
	x = e.MidBlock.Forward(x)

	// Norm + silu
	{
		prev := x
		x = e.NormOut.Forward(x)
		x = silu3D(x)
		prev.Free()
		mlx.Eval(x)
	}

	// Conv out
	{
		prev := x
		x = e.ConvOut.Forward(x)
		prev.Free()
	}

	// Quant conv
	{
		prev := x
		x = e.QuantConv.Forward(x)
		prev.Free()
	}

	// Get mode from distribution (first half of channels = mean)
	// Output is [B, T, H, W, 2*latent_C], we take first latent_C channels
	shape := x.Shape()
	latentC := shape[4] / 2
	x = mlx.Slice(x, []int32{0, 0, 0, 0, 0}, []int32{shape[0], shape[1], shape[2], shape[3], latentC})

	// Convert back to channels-first [N, C, T, H, W]
	x = mlx.Contiguous(mlx.Transpose(x, 0, 4, 1, 2, 3))
	mlx.Eval(x)

	return x
}

// VAEDecoder is the decoder part of the VAE
type VAEDecoder struct {
	Config *VAEConfig

	PostQuantConv *CausalConv3d
	ConvIn        *CausalConv3d
	MidBlock      *MidBlock
	UpBlocks      []*UpBlock
	NormOut       *RMSNorm3D
	ConvOut       *CausalConv3d
}

// loadFromWeights loads the decoder from pre-loaded weights
func (d *VAEDecoder) loadFromWeights(weights *safetensors.ModelWeights, cfg *VAEConfig) error {
	d.Config = cfg

	postQuantConv, err := newCausalConv3d(weights, "post_quant_conv")
	if err != nil {
		return err
	}
	d.PostQuantConv = postQuantConv

	convIn, err := newCausalConv3d(weights, "decoder.conv_in")
	if err != nil {
		return err
	}
	d.ConvIn = convIn

	// Mid block
	midDim := cfg.BaseDim * cfg.DimMult[len(cfg.DimMult)-1]
	midBlock, err := newMidBlock(weights, "decoder.mid_block", midDim)
	if err != nil {
		return err
	}
	d.MidBlock = midBlock

	// Up blocks (reversed dim_mult)
	numUpBlocks := len(cfg.DimMult)
	d.UpBlocks = make([]*UpBlock, numUpBlocks)

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
		d.UpBlocks[i] = upBlock
	}

	normOut, err := newRMSNorm3D(weights, "decoder.norm_out", cfg.BaseDim)
	if err != nil {
		return err
	}
	d.NormOut = normOut

	convOut, err := newCausalConv3d(weights, "decoder.conv_out")
	if err != nil {
		return err
	}
	d.ConvOut = convOut

	return nil
}

// Decode converts latents to image
// z: [B, C, T, H, W] denormalized latents
func (d *VAEDecoder) Decode(z *mlx.Array) *mlx.Array {
	var x *mlx.Array

	// Convert from channels-first to channels-last
	{
		z = mlx.Contiguous(mlx.Transpose(z, 0, 2, 3, 4, 1))
		mlx.Eval(z)
	}

	// PostQuantConv
	x = d.PostQuantConv.Forward(z)
	z.Free()

	// ConvIn
	{
		prev := x
		x = d.ConvIn.Forward(x)
		prev.Free()
	}

	// Mid block
	x = d.MidBlock.Forward(x)

	// Up blocks
	for _, upBlock := range d.UpBlocks {
		x = upBlock.Forward(x)
	}

	// NormOut + silu
	{
		prev := x
		x = d.NormOut.Forward(x)
		x = silu3D(x)
		prev.Free()
		mlx.Eval(x)
	}

	// ConvOut
	{
		prev := x
		x = d.ConvOut.Forward(x)
		prev.Free()
	}

	// Post-processing: clamp and convert back to channels-first
	{
		prev := x
		x = mlx.ClipScalar(x, -1.0, 1.0, true, true)
		x = mlx.Contiguous(mlx.Transpose(x, 0, 4, 1, 2, 3))
		prev.Free()
		mlx.Eval(x)
	}

	return x
}

// DownBlock handles downsampling in encoder
type DownBlock struct {
	ResBlocks   []*ResBlock
	Downsampler *Downsample
}

// newDownBlock creates a down block
func newDownBlock(weights *safetensors.ModelWeights, prefix string, inDim, outDim int32, numBlocks int32, downsampleMode string) (*DownBlock, error) {
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

	var downsampler *Downsample
	if downsampleMode != "" {
		downsampler = newDownsample(weights, prefix+".downsamplers.0", outDim, downsampleMode)
	}

	return &DownBlock{
		ResBlocks:   resBlocks,
		Downsampler: downsampler,
	}, nil
}

// Forward applies down block
func (d *DownBlock) Forward(x *mlx.Array) *mlx.Array {
	for _, block := range d.ResBlocks {
		prev := x
		x = block.Forward(x)
		prev.Free()
	}

	if d.Downsampler != nil {
		prev := x
		x = d.Downsampler.Forward(x)
		prev.Free()
	}
	return x
}

// Downsample handles spatial downsampling
type Downsample struct {
	Conv *mlx.Array
	Bias *mlx.Array
	Mode string
}

// newDownsample creates a downsampler
func newDownsample(weights *safetensors.ModelWeights, prefix string, dim int32, mode string) *Downsample {
	conv, _ := weights.Get(prefix + ".resample.1.weight")
	bias, _ := weights.Get(prefix + ".resample.1.bias")
	return &Downsample{
		Conv: conv,
		Bias: bias,
		Mode: mode,
	}
}

// Forward applies downsampling to channels-last input [B, T, H, W, C]
func (d *Downsample) Forward(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	T := shape[1]
	H := shape[2]
	W := shape[3]
	C := shape[4]
	outC := d.Conv.Shape()[0]

	// Reshape to [B*T, H, W, C] for 2D conv
	x = mlx.Reshape(x, B*T, H, W, C)

	// Pad for stride-2 conv: need (3-1)/2 = 1 on each side, but for stride 2 we need specific padding
	// For 3x3 stride 2: pad 1 on all sides
	x = mlx.Pad(x, []int32{0, 0, 1, 1, 1, 1, 0, 0})

	// Conv with stride 2 using manual strided patching
	weight := mlx.Transpose(d.Conv, 0, 2, 3, 1)
	x = conv2DStrided(x, weight, 2)
	if d.Bias != nil {
		bias := mlx.Reshape(d.Bias, 1, 1, 1, outC)
		x = mlx.Add(x, bias)
	}

	x = mlx.Reshape(x, B, T, H/2, W/2, outC)
	mlx.Eval(x)

	return x
}

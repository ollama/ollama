//go:build mlx

// Package flux2 implements the FLUX.2 Klein diffusion transformer model.
// Klein is a 4B parameter distilled model that supports sub-second inference.
package flux2

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"math"
	"time"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/models/qwen3"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
	"golang.org/x/image/draw"
)

// GenerateConfig holds all options for image generation.
type GenerateConfig struct {
	Prompt        string
	Width         int32                 // Image width (default: 1024)
	Height        int32                 // Image height (default: 1024)
	Steps         int                   // Denoising steps (default: 4 for Klein)
	GuidanceScale float32               // Guidance scale (default: 1.0, Klein doesn't need CFG)
	Seed          int64                 // Random seed
	Progress      func(step, totalSteps int) // Optional progress callback
	CapturePath   string                // GPU capture path (debug)
	InputImages   []image.Image         // Reference images for image conditioning (already loaded)
}

// Model represents a FLUX.2 Klein model.
type Model struct {
	ModelName       string
	Tokenizer       *tokenizer.Tokenizer
	TextEncoder     *qwen3.TextEncoder
	Transformer     *Flux2Transformer2DModel
	VAE             *AutoencoderKLFlux2
	SchedulerConfig *SchedulerConfig
}

// TextEncoderLayerIndices are the layers from which to extract text embeddings.
// Diffusers uses hidden_states[9, 18, 27]. In Python, hidden_states[0] is the embedding
// output before any layers, so hidden_states[9] = after layer 8 (0-indexed).
// Go's ForwardWithLayerOutputs captures after layer i runs, so we use [8, 17, 26].
var TextEncoderLayerIndices = []int{8, 17, 26}

// Load loads the FLUX.2 Klein model from ollama blob storage.
func (m *Model) Load(modelName string) error {
	fmt.Printf("Loading FLUX.2 Klein model from manifest: %s...\n", modelName)
	start := time.Now()

	if mlx.GPUIsAvailable() {
		mlx.SetDefaultDeviceGPU()
		mlx.EnableCompile()
	}

	m.ModelName = modelName

	// Load manifest
	manifest, err := imagegen.LoadManifest(modelName)
	if err != nil {
		return fmt.Errorf("load manifest: %w", err)
	}

	// Load tokenizer
	fmt.Print("  Loading tokenizer... ")
	tokData, err := manifest.ReadConfig("tokenizer/tokenizer.json")
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}

	tokConfig := &tokenizer.TokenizerConfig{}
	if data, err := manifest.ReadConfig("tokenizer/tokenizer_config.json"); err == nil {
		tokConfig.TokenizerConfigJSON = data
	}
	if data, err := manifest.ReadConfig("tokenizer/generation_config.json"); err == nil {
		tokConfig.GenerationConfigJSON = data
	}
	if data, err := manifest.ReadConfig("tokenizer/special_tokens_map.json"); err == nil {
		tokConfig.SpecialTokensMapJSON = data
	}

	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokConfig)
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}
	m.Tokenizer = tok
	fmt.Println("✓")

	// Load text encoder
	m.TextEncoder = &qwen3.TextEncoder{}
	if err := m.TextEncoder.Load(manifest, "text_encoder/config.json"); err != nil {
		return fmt.Errorf("text encoder: %w", err)
	}

	// Load transformer
	m.Transformer = &Flux2Transformer2DModel{}
	if err := m.Transformer.Load(manifest); err != nil {
		return fmt.Errorf("transformer: %w", err)
	}

	// Load VAE
	m.VAE = &AutoencoderKLFlux2{}
	if err := m.VAE.Load(manifest); err != nil {
		return fmt.Errorf("VAE: %w", err)
	}

	// Evaluate all weights in a single batch (reduces GPU sync overhead)
	fmt.Print("  Evaluating weights... ")
	allWeights := mlx.Collect(m.TextEncoder)
	allWeights = append(allWeights, mlx.Collect(m.Transformer)...)
	allWeights = append(allWeights, mlx.Collect(m.VAE)...)
	mlx.Eval(allWeights...)
	fmt.Println("✓")

	// Load scheduler config
	m.SchedulerConfig = DefaultSchedulerConfig()
	if schedData, err := manifest.ReadConfig("scheduler/scheduler_config.json"); err == nil {
		if err := json.Unmarshal(schedData, m.SchedulerConfig); err != nil {
			fmt.Printf("  Warning: failed to parse scheduler config: %v\n", err)
		}
	}

	mem := mlx.MetalGetActiveMemory()
	fmt.Printf("  Loaded in %.2fs (%.1f GB VRAM)\n", time.Since(start).Seconds(), float64(mem)/(1024*1024*1024))

	return nil
}

// Generate creates an image from a prompt.
func (m *Model) Generate(prompt string, width, height int32, steps int, seed int64) (*mlx.Array, error) {
	return m.GenerateFromConfig(context.Background(), &GenerateConfig{
		Prompt: prompt,
		Width:  width,
		Height: height,
		Steps:  steps,
		Seed:   seed,
	})
}

// GenerateWithProgress creates an image with progress callback.
func (m *Model) GenerateWithProgress(prompt string, width, height int32, steps int, seed int64, progress func(step, totalSteps int)) (*mlx.Array, error) {
	return m.GenerateFromConfig(context.Background(), &GenerateConfig{
		Prompt:   prompt,
		Width:    width,
		Height:   height,
		Steps:    steps,
		Seed:     seed,
		Progress: progress,
	})
}

// GenerateFromConfig generates an image using the unified config struct.
func (m *Model) GenerateFromConfig(ctx context.Context, cfg *GenerateConfig) (*mlx.Array, error) {
	start := time.Now()
	result, err := m.generate(ctx, cfg)
	if err != nil {
		return nil, err
	}
	fmt.Printf("Generated in %.2fs (%d steps)\n", time.Since(start).Seconds(), cfg.Steps)
	return result, nil
}

// GenerateImage implements runner.ImageModel interface.
func (m *Model) GenerateImage(ctx context.Context, prompt string, width, height int32, steps int, seed int64, progress func(step, total int)) (*mlx.Array, error) {
	return m.GenerateFromConfig(ctx, &GenerateConfig{
		Prompt:   prompt,
		Width:    width,
		Height:   height,
		Steps:    steps,
		Seed:     seed,
		Progress: progress,
	})
}

// MaxOutputPixels is the maximum output resolution (4 megapixels, ~2048x2048)
const MaxOutputPixels = 2048 * 2048

// MaxRefPixels is the maximum resolution for reference images (smaller to reduce attention memory)
const MaxRefPixels = 728 * 728

// generate is the internal denoising pipeline.
func (m *Model) generate(ctx context.Context, cfg *GenerateConfig) (*mlx.Array, error) {
	// Enable MLX compilation for fused kernels
	mlx.EnableCompile()

	// Apply defaults
	if cfg.Steps <= 0 {
		cfg.Steps = 4 // Klein default: 4 steps for distilled model
	}
	if cfg.GuidanceScale <= 0 {
		cfg.GuidanceScale = 1.0 // Klein doesn't need guidance
	}

	// Determine output dimensions
	if len(cfg.InputImages) > 0 {
		// With input images, compute missing dimension from aspect ratio
		// Images are already EXIF-rotated by the caller
		bounds := cfg.InputImages[0].Bounds()
		imgW, imgH := bounds.Dx(), bounds.Dy()
		aspectRatio := float64(imgH) / float64(imgW)
		if cfg.Width > 0 && cfg.Height <= 0 {
			// Width specified, compute height
			cfg.Height = int32(math.Round(float64(cfg.Width)*aspectRatio/16) * 16)
		} else if cfg.Height > 0 && cfg.Width <= 0 {
			// Height specified, compute width
			cfg.Width = int32(math.Round(float64(cfg.Height)/aspectRatio/16) * 16)
		} else if cfg.Width <= 0 && cfg.Height <= 0 {
			// Neither specified, use input dimensions
			cfg.Width = int32(imgW)
			cfg.Height = int32(imgH)
		}
	}
	if cfg.Width <= 0 {
		cfg.Width = 1024
	}
	if cfg.Height <= 0 {
		cfg.Height = 1024
	}

	// Cap to max pixels, preserve aspect ratio, round to multiple of 16
	pixels := int(cfg.Width) * int(cfg.Height)
	if pixels > MaxOutputPixels {
		scale := math.Sqrt(float64(MaxOutputPixels) / float64(pixels))
		cfg.Width = int32(math.Round(float64(cfg.Width) * scale / 16) * 16)
		cfg.Height = int32(math.Round(float64(cfg.Height) * scale / 16) * 16)
	}
	cfg.Height = int32((cfg.Height + 8) / 16 * 16) // round to nearest 16
	cfg.Width = int32((cfg.Width + 8) / 16 * 16)
	fmt.Printf("  Output: %dx%d\n", cfg.Width, cfg.Height)

	tcfg := m.Transformer.TransformerConfig
	patchSize := m.VAE.Config.PatchSize

	// Latent dimensions: image / 8 (VAE downscale) / patch_size
	latentH := cfg.Height / 8
	latentW := cfg.Width / 8
	patchH := latentH / patchSize[0]
	patchW := latentW / patchSize[1]
	imgSeqLen := patchH * patchW

	// Text encoding with multi-layer extraction (no padding, use true sequence length)
	fmt.Print("  Encoding prompt... ")
	promptEmbeds, textLen := m.TextEncoder.EncodePromptWithLayers(m.Tokenizer, cfg.Prompt, 512, TextEncoderLayerIndices, false)
	fmt.Println("✓")

	// Encode reference images if provided
	var refTokens *ImageCondTokens
	var refHeights, refWidths []int32
	if len(cfg.InputImages) > 0 {
		fmt.Printf("  Encoding %d reference image(s):\n", len(cfg.InputImages))

		var err error
		refTokens, err = m.EncodeImageRefs(cfg.InputImages)
		if err != nil {
			return nil, fmt.Errorf("encode reference images: %w", err)
		}

		// Extract heights/widths for RoPE computation (same limits as EncodeImageRefs)
		limitPixels := MaxRefPixels
		if len(cfg.InputImages) > 1 {
			limitPixels = MaxRefPixels / 2
		}
		for _, img := range cfg.InputImages {
			_, w, h := PrepareImage(img, limitPixels)
			refHeights = append(refHeights, int32(h/16))
			refWidths = append(refWidths, int32(w/16))
		}
	}

	// Scheduler
	scheduler := NewFlowMatchScheduler(m.SchedulerConfig)
	scheduler.SetTimestepsWithMu(cfg.Steps, CalculateShift(imgSeqLen, cfg.Steps))

	// Init latents in packed form [B, C*4, H/2, W/2] like diffusers
	// diffusers creates noise in [B, 128, 64, 64] and packs to [B, 4096, 128]
	latentChannels := m.VAE.Config.LatentChannels
	packedChannels := latentChannels * 4 // 32 * 4 = 128
	latents := scheduler.InitNoise([]int32{1, packedChannels, patchH, patchW}, cfg.Seed)

	// Pack latents (transpose): [B, C, H, W] -> [B, H*W, C]
	// This matches diffusers' _pack_latents
	patches := packLatents(latents)
	noiseSeqLen := patches.Shape()[1]

	// RoPE cache - includes reference images if present
	rope := PrepareRoPECache(textLen, patchH, patchW, tcfg.AxesDimsRoPE, tcfg.RopeTheta, refHeights, refWidths, ImageRefScale)

	// Cleanup setup arrays when done
	defer func() {
		rope.Cos.Free()
		rope.Sin.Free()
		promptEmbeds.Free()
		if refTokens != nil {
			refTokens.Tokens.Free()
		}
	}()

	// Pre-compute all timesteps before the loop to avoid per-step tensor creation
	timesteps := make([]*mlx.Array, cfg.Steps)
	for i := 0; i < cfg.Steps; i++ {
		tCurr := scheduler.Timesteps[i] / float32(m.SchedulerConfig.NumTrainTimesteps)
		timesteps[i] = mlx.ToBFloat16(mlx.NewArray([]float32{tCurr}, []int32{1}))
	}

	// Evaluate setup arrays
	fmt.Print("  Evaluating setup... ")
	setupStart := time.Now()
	toEval := []*mlx.Array{promptEmbeds, patches, rope.Cos, rope.Sin}
	toEval = append(toEval, timesteps...)
	if refTokens != nil {
		toEval = append(toEval, refTokens.Tokens)
	}
	mlx.Eval(toEval...)
	mlx.MetalResetPeakMemory() // Reset peak to measure generation separately
	fmt.Printf("✓ (%.2fs, %.1f GB)\n", time.Since(setupStart).Seconds(),
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024))

	if cfg.Progress != nil {
		cfg.Progress(0, cfg.Steps)
	}

	loopStart := time.Now()
	stepStart := time.Now()

	// Denoising loop
	for i := 0; i < cfg.Steps; i++ {
		// Check for cancellation
		if ctx != nil {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			default:
			}
		}

		// GPU capture on step 2 if requested
		if cfg.CapturePath != "" && i == 1 {
			mlx.MetalStartCapture(cfg.CapturePath)
		}

		timestep := timesteps[i]

		// Prepare input - concatenate noise patches with reference tokens if present
		imgInput := patches
		if refTokens != nil {
			imgInput = mlx.Concatenate([]*mlx.Array{patches, refTokens.Tokens}, 1)
		}

		// Transformer forward pass
		output := m.Transformer.Forward(imgInput, promptEmbeds, timestep, rope)

		// If we concatenated reference tokens, slice to only get noise portion
		if refTokens != nil {
			output = mlx.Slice(output, []int32{0, 0, 0}, []int32{1, noiseSeqLen, output.Shape()[2]})
		}

		// Scheduler step (keep reference to old patches for the computation graph)
		newPatches := scheduler.Step(output, patches, i)

		if cfg.CapturePath != "" && i == 1 {
			mlx.MetalStopCapture()
		}

		mlx.Eval(newPatches)
		patches = newPatches

		elapsed := time.Since(stepStart).Seconds()
		peakGB := float64(mlx.MetalGetPeakMemory()) / (1024 * 1024 * 1024)
		if i == 0 {
			fmt.Printf("    step %d: %.2fs (JIT warmup), peak %.1f GB\n", i+1, elapsed, peakGB)
		} else {
			fmt.Printf("    step %d: %.2fs, peak %.1f GB\n", i+1, elapsed, peakGB)
		}
		stepStart = time.Now()
		if cfg.Progress != nil {
			cfg.Progress(i+1, cfg.Steps)
		}
	}

	loopTime := time.Since(loopStart).Seconds()
	peakMem := float64(mlx.MetalGetPeakMemory()) / (1024 * 1024 * 1024)
	fmt.Printf("  Denoised %d steps in %.2fs (%.2fs/step), peak %.1f GB\n",
		cfg.Steps, loopTime, loopTime/float64(cfg.Steps), peakMem)

	// Free timesteps now that denoising is done
	for _, ts := range timesteps {
		ts.Free()
	}

	// VAE decode with tiling for larger images
	fmt.Print("  Decoding VAE... ")
	vaeStart := time.Now()
	// Enable tiling for images > 512x512 (latent > 64x64)
	// VAE attention is O(n²) on latent pixels, tiling reduces memory significantly
	if patchH*2 > 64 || patchW*2 > 64 {
		m.VAE.Tiling = DefaultTilingConfig()
	}
	decoded := m.VAE.Decode(patches, patchH, patchW)
	mlx.Eval(decoded)

	// Free patches now that decode is done
	patches.Free()

	fmt.Printf("✓ (%.2fs, peak %.1f GB)\n", time.Since(vaeStart).Seconds(),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	return decoded, nil
}

// packLatents converts [B, C, H, W] to [B, H*W, C] (matches diffusers _pack_latents)
func packLatents(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	C := shape[1]
	H := shape[2]
	W := shape[3]
	// [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
	x = mlx.Reshape(x, B, C, H*W)
	return mlx.Transpose(x, 0, 2, 1)
}

// LoadPersistent loads the model and keeps it in memory for repeated use.
func LoadPersistent(modelName string) (*Model, error) {
	m := &Model{}
	if err := m.Load(modelName); err != nil {
		return nil, err
	}
	return m, nil
}

// ImageRefScale is the time coordinate offset between reference images (matches diffusers scale=10)
const ImageRefScale = 10

// PrepareImage resizes and crops an image to be a multiple of 16, with optional pixel limit.
// Returns the processed image and its dimensions.
func PrepareImage(img image.Image, limitPixels int) (image.Image, int, int) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	// Cap pixels if needed (like diffusers cap_pixels)
	if limitPixels > 0 && w*h > limitPixels {
		scale := math.Sqrt(float64(limitPixels) / float64(w*h))
		w = int(float64(w) * scale)
		h = int(float64(h) * scale)
	}

	// Round down to multiple of 16
	w = (w / 16) * 16
	h = (h / 16) * 16

	if w < 16 {
		w = 16
	}
	if h < 16 {
		h = 16
	}

	// Resize using high-quality bicubic interpolation (matches diffusers' default lanczos)
	resized := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.CatmullRom.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Over, nil)

	return resized, w, h
}

// ImageToTensor converts an image to a tensor in [-1, 1] range with shape [1, C, H, W].
func ImageToTensor(img image.Image) *mlx.Array {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	// Convert to float32 array in NCHW format [1, 3, H, W] with values in [-1, 1]
	data := make([]float32, 3*h*w)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			// RGBA returns 16-bit values, convert to [-1, 1]
			data[0*h*w+y*w+x] = float32(r>>8)/127.5 - 1.0
			data[1*h*w+y*w+x] = float32(g>>8)/127.5 - 1.0
			data[2*h*w+y*w+x] = float32(b>>8)/127.5 - 1.0
		}
	}

	arr := mlx.NewArrayFloat32(data, []int32{1, 3, int32(h), int32(w)})
	return arr
}

// ImageCondTokens holds encoded reference image tokens.
type ImageCondTokens struct {
	Tokens *mlx.Array // [1, total_tokens, C] - concatenated reference tokens
}

// EncodeImageRefs encodes reference images using the VAE.
func (m *Model) EncodeImageRefs(images []image.Image) (*ImageCondTokens, error) {
	if len(images) == 0 {
		return nil, nil
	}

	// Limit reference images to reduce attention memory
	limitPixels := MaxRefPixels
	if len(images) > 1 {
		limitPixels = MaxRefPixels / 2
	}

	var allTokens []*mlx.Array

	for _, img := range images {
		// Prepare image (resize, crop to multiple of 16)
		prepared, prepW, prepH := PrepareImage(img, limitPixels)
		fmt.Printf("    Encoding %dx%d image... ", prepW, prepH)

		// Convert to tensor [-1, 1]
		tensor := ImageToTensor(prepared)

		// Encode with VAE - returns [1, L, 128]
		encoded := m.VAE.EncodeImage(tensor)
		squeezed := mlx.Squeeze(encoded, 0) // [L, C]

		// Defer eval - will be done with other setup arrays
		allTokens = append(allTokens, squeezed)
		fmt.Println("✓")
	}

	// For single image, just add batch dimension directly
	// For multiple images, concatenate first
	var tokens *mlx.Array
	if len(allTokens) == 1 {
		tokens = mlx.ExpandDims(allTokens[0], 0) // [1, L, C]
	} else {
		tokens = mlx.Concatenate(allTokens, 0) // [total_L, C]
		tokens = mlx.ExpandDims(tokens, 0)     // [1, total_L, C]
	}

	return &ImageCondTokens{Tokens: tokens}, nil
}

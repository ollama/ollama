//go:build mlx

// Package flux2 implements the FLUX.2 Klein diffusion transformer model.
// Klein is a 4B parameter distilled model that supports sub-second inference.
package flux2

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
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
	Width         int32        // Image width (default: 1024)
	Height        int32        // Image height (default: 1024)
	Steps         int          // Denoising steps (default: 4 for Klein)
	GuidanceScale float32      // Guidance scale (default: 1.0, Klein doesn't need CFG)
	Seed          int64        // Random seed
	Progress      ProgressFunc // Optional progress callback
	CapturePath   string       // GPU capture path (debug)
	InputImages   []string     // Paths to reference images for image conditioning
}

// ProgressFunc is called during generation with step progress.
type ProgressFunc func(step, totalSteps int)

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
	mlx.Eval(mlx.Collect(m.TextEncoder)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	// Load transformer
	m.Transformer = &Flux2Transformer2DModel{}
	if err := m.Transformer.Load(manifest); err != nil {
		return fmt.Errorf("transformer: %w", err)
	}
	mlx.Eval(mlx.Collect(m.Transformer)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	// Load VAE
	m.VAE = &AutoencoderKLFlux2{}
	if err := m.VAE.Load(manifest); err != nil {
		return fmt.Errorf("VAE: %w", err)
	}
	mlx.Eval(mlx.Collect(m.VAE)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

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
func (m *Model) GenerateWithProgress(prompt string, width, height int32, steps int, seed int64, progress ProgressFunc) (*mlx.Array, error) {
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

// GenerateImage implements model.ImageModel interface.
func (m *Model) GenerateImage(ctx context.Context, prompt string, width, height int32, steps int, seed int64) (*mlx.Array, error) {
	return m.Generate(prompt, width, height, steps, seed)
}

// generate is the internal denoising pipeline.
func (m *Model) generate(ctx context.Context, cfg *GenerateConfig) (*mlx.Array, error) {
	// Apply defaults
	if cfg.Steps <= 0 {
		cfg.Steps = 4 // Klein default: 4 steps for distilled model
	}
	if cfg.GuidanceScale <= 0 {
		cfg.GuidanceScale = 1.0 // Klein doesn't need guidance
	}

	// If input images are provided and dimensions not explicitly set, match first input image
	if len(cfg.InputImages) > 0 && cfg.Width <= 0 && cfg.Height <= 0 {
		img, err := LoadImage(cfg.InputImages[0])
		if err == nil {
			bounds := img.Bounds()
			cfg.Width = int32(bounds.Dx())
			cfg.Height = int32(bounds.Dy())
			fmt.Printf("  Matching output size to input image: %dx%d\n", cfg.Width, cfg.Height)
		}
	}

	// Apply dimension defaults if still not set
	if cfg.Width <= 0 {
		cfg.Width = 1024
	}
	if cfg.Height <= 0 {
		cfg.Height = 1024
	}

	// Clamp dimensions to multiples of 16 (vae_scale_factor * 2)
	vaeScaleFactor := int32(8)
	newHeight := (cfg.Height / (vaeScaleFactor * 2)) * (vaeScaleFactor * 2)
	newWidth := (cfg.Width / (vaeScaleFactor * 2)) * (vaeScaleFactor * 2)
	if newHeight != cfg.Height || newWidth != cfg.Width {
		fmt.Printf("  Note: dimensions adjusted from %dx%d to %dx%d (must be multiple of 16)\n",
			cfg.Width, cfg.Height, newWidth, newHeight)
	}
	cfg.Height = newHeight
	cfg.Width = newWidth

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
	mlx.Keep(promptEmbeds)
	mlx.Eval(promptEmbeds)
	fmt.Println("✓")

	// Load and encode reference images if provided
	var refTokens *ImageCondTokens
	var refHeights, refWidths []int32
	if len(cfg.InputImages) > 0 {
		fmt.Printf("  Encoding %d reference image(s)... ", len(cfg.InputImages))
		var images []image.Image
		for _, path := range cfg.InputImages {
			img, err := LoadImage(path)
			if err != nil {
				return nil, fmt.Errorf("load image %s: %w", path, err)
			}
			images = append(images, img)
		}

		var err error
		refTokens, err = m.EncodeImageRefs(images)
		if err != nil {
			return nil, fmt.Errorf("encode reference images: %w", err)
		}

		// Extract heights/widths for RoPE computation
		// Pixel limit for multiple images
		limitPixels := 2024 * 2024
		if len(images) > 1 {
			limitPixels = 1024 * 1024
		}
		for _, img := range images {
			_, w, h := PrepareImage(img, limitPixels)
			refHeights = append(refHeights, int32(h/16))
			refWidths = append(refWidths, int32(w/16))
		}
		fmt.Println("✓")
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
	var rope *RoPECache
	if refTokens != nil {
		rope = PrepareRoPECacheWithImages(textLen, patchH, patchW, refHeights, refWidths, ImageRefScale, tcfg.AxesDimsRoPE, tcfg.RopeTheta)
	} else {
		rope = PrepareRoPECache(textLen, patchH, patchW, tcfg.AxesDimsRoPE, tcfg.RopeTheta)
	}

	// Evaluate all arrays together
	toEval := []*mlx.Array{patches, rope.Cos, rope.Sin}
	if refTokens != nil {
		toEval = append(toEval, refTokens.Tokens)
	}
	mlx.Eval(toEval...)

	// Cleanup function
	cleanup := func() {
		promptEmbeds.Free()
		rope.Cos.Free()
		rope.Sin.Free()
		latents.Free()
	}

	// Denoising loop - work entirely in sequence form [B, L, C]
	// patches is [B, 4096, 128], transformer output is [B, 4096, 128]
	// Scheduler step operates directly on sequence form
	if cfg.Progress != nil {
		cfg.Progress(0, cfg.Steps)
	}

	for i := 0; i < cfg.Steps; i++ {
		// Check for cancellation
		if ctx != nil {
			select {
			case <-ctx.Done():
				cleanup()
				return nil, ctx.Err()
			default:
			}
		}
		stepStart := time.Now()

		// GPU capture on step 2 if requested
		if cfg.CapturePath != "" && i == 1 {
			mlx.MetalStartCapture(cfg.CapturePath)
		}

		tCurr := scheduler.Timesteps[i] / float32(m.SchedulerConfig.NumTrainTimesteps)
		// Flow matching: transformer expects t in [0,1]; it multiplies by 1000 internally
		timestep := mlx.ToBFloat16(mlx.NewArray([]float32{tCurr}, []int32{1}))

		// Prepare input - concatenate noise patches with reference tokens if present
		// This matches diffusers: img_input = torch.cat((img, img_cond_seq), dim=1)
		var imgInput *mlx.Array
		if refTokens != nil {
			imgInput = mlx.Concatenate([]*mlx.Array{patches, refTokens.Tokens}, 1)
		} else {
			imgInput = patches
		}

		// Transformer forward: [B, L, C] -> [B, L, C]
		output := m.Transformer.Forward(imgInput, promptEmbeds, timestep, rope)

		// If we concatenated reference tokens, slice to only get noise portion
		// This matches diffusers: pred = pred[:, : img.shape[1]]
		if refTokens != nil {
			output = mlx.Slice(output, []int32{0, 0, 0}, []int32{1, noiseSeqLen, output.Shape()[2]})
			imgInput.Free() // Free the concatenated input
		}

		// Scheduler step directly on sequence form [B, L, C]
		// No unpatchify/patchify - everything stays in sequence form
		oldPatches := patches
		patches = scheduler.Step(output, patches, i)
		mlx.Eval(patches)
		oldPatches.Free()

		if cfg.CapturePath != "" && i == 1 {
			mlx.MetalStopCapture()
		}

		activeMem := float64(mlx.MetalGetActiveMemory()) / (1024 * 1024 * 1024)
		peakMem := float64(mlx.MetalGetPeakMemory()) / (1024 * 1024 * 1024)
		fmt.Printf("  Step %d/%d: t=%.4f (%.2fs) [%.1f GB active, %.1f GB peak]\n",
			i+1, cfg.Steps, tCurr, time.Since(stepStart).Seconds(), activeMem, peakMem)

		if cfg.Progress != nil {
			cfg.Progress(i+1, cfg.Steps)
		}
	}

	// Free denoising temporaries (but not patches - need it for decode)
	promptEmbeds.Free()
	rope.Cos.Free()
	rope.Sin.Free()
	latents.Free() // Free the spatial-form latents we created initially
	if refTokens != nil {
		refTokens.Tokens.Free()
	}

	// VAE decode - patches is already in [B, L, 128] sequence form
	// VAE.Decode handles denormalization and unpatchify internally
	fmt.Print("  Decoding... ")
	decoded := m.VAE.Decode(patches, patchH, patchW)
	patches.Free()
	mlx.Eval(decoded)
	fmt.Println("✓")

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

// LoadImage loads an image from disk.
func LoadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open image: %w", err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	return img, nil
}

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

	// Resize using bilinear interpolation
	resized := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.BiLinear.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Over, nil)

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

	// Pixel limit depends on number of images (like diffusers)
	limitPixels := 2024 * 2024
	if len(images) > 1 {
		limitPixels = 1024 * 1024
	}

	var allTokens []*mlx.Array

	for _, img := range images {
		// Prepare image (resize, crop to multiple of 16)
		prepared, _, _ := PrepareImage(img, limitPixels)

		// Convert to tensor [-1, 1]
		tensor := ImageToTensor(prepared)

		// Encode with VAE - returns [1, L, 128]
		encoded := m.VAE.EncodeImage(tensor)

		allTokens = append(allTokens, mlx.Squeeze(encoded, 0)) // [L, C]
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

//go:build mlx

// Package glm_image implements the GLM-Image hybrid AR + diffusion model.
package glm_image

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"time"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/mlx"
)

// ByT5Tokenizer is a simple byte-level tokenizer for ByT5
// ByT5 uses bytes as tokens: each byte (0-255) maps to token ID (3-258)
// Special tokens: 0=pad, 1=eos, 2=unk
type ByT5Tokenizer struct {
	PadTokenID int32
	EOSTokenID int32
	UNKTokenID int32
}

// NewByT5Tokenizer creates a new ByT5 tokenizer
func NewByT5Tokenizer() *ByT5Tokenizer {
	return &ByT5Tokenizer{
		PadTokenID: 0,
		EOSTokenID: 1,
		UNKTokenID: 2,
	}
}

// Encode converts a string to token IDs
func (t *ByT5Tokenizer) Encode(text string) []int32 {
	bytes := []byte(text)
	tokens := make([]int32, len(bytes))
	for i, b := range bytes {
		// Standard ByT5 tokenization: bytes 0-255 map to tokens 3-258
		// (tokens 0, 1, 2 are PAD, EOS, UNK)
		tokens[i] = int32(b) + 3
	}
	return tokens
}

// Decode converts token IDs back to a string
func (t *ByT5Tokenizer) Decode(tokens []int32) string {
	bytes := make([]byte, 0, len(tokens))
	for _, tok := range tokens {
		if tok >= 3 && tok < 259 {
			bytes = append(bytes, byte(tok-3))
		}
	}
	return string(bytes)
}

// GenerateConfig holds all options for image generation.
type GenerateConfig struct {
	Prompt         string
	NegativePrompt string       // For CFG (optional, not typically used with GLM-Image)
	GuidanceScale  float32      // Guidance scale (default: 1.5)
	Width          int32        // Image width (default: 1024, must be divisible by 32)
	Height         int32        // Image height (default: 1024, must be divisible by 32)
	Steps          int          // Diffusion denoising steps (default: 50)
	Seed           int64        // Random seed
	Progress       ProgressFunc // Optional progress callback

	// AR generation options
	MaxVisualTokens int32   // Max visual tokens to generate (default: 256)
	Temperature     float32 // AR sampling temperature (default: 0.9)
	TopP            float32 // Nucleus sampling (default: 0.75)
}

// ProgressFunc is called during generation with stage and step progress.
type ProgressFunc func(stage string, step, totalSteps int)

// Model represents a GLM-Image hybrid model.
type Model struct {
	ModelName             string
	Tokenizer             *ByT5Tokenizer   // For T5 text encoder (glyph embeddings)
	GLMTokenizer          *GLMTokenizer    // For AR model (visual token generation)
	TextEncoder           *T5TextEncoder
	VisionLanguageEncoder *VisionLanguageEncoder
	Transformer           *DiffusionTransformer
	VAEDecoder            *VAEDecoder
}

// Load loads the GLM-Image model from ollama blob storage.
func (m *Model) Load(modelName string) error {
	fmt.Printf("Loading GLM-Image model from manifest: %s...\n", modelName)
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

	// Create ByT5 tokenizer (byte-level, no vocabulary file needed)
	// Used for T5 text encoder (glyph embeddings)
	fmt.Print("  Creating ByT5 tokenizer... ")
	m.Tokenizer = NewByT5Tokenizer()
	fmt.Println("✓")

	// Load GLM tokenizer for AR model (visual token generation)
	fmt.Print("  Loading GLM tokenizer... ")
	glmTok, err := NewGLMTokenizer(manifest)
	if err != nil {
		return fmt.Errorf("glm tokenizer: %w", err)
	}
	m.GLMTokenizer = glmTok
	fmt.Println("✓")

	// Load T5 text encoder (~830MB)
	m.TextEncoder = &T5TextEncoder{}
	if err := m.TextEncoder.Load(manifest); err != nil {
		return fmt.Errorf("text encoder: %w", err)
	}
	mlx.Eval(mlx.Collect(m.TextEncoder)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	// Load vision-language encoder (~19GB, 9B params)
	m.VisionLanguageEncoder = &VisionLanguageEncoder{}
	if err := m.VisionLanguageEncoder.Load(manifest); err != nil {
		return fmt.Errorf("vision language encoder: %w", err)
	}
	mlx.Eval(mlx.Collect(m.VisionLanguageEncoder)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	// Load diffusion transformer (~13GB, 7B params)
	m.Transformer = &DiffusionTransformer{}
	if err := m.Transformer.Load(manifest); err != nil {
		return fmt.Errorf("transformer: %w", err)
	}
	mlx.Eval(mlx.Collect(m.Transformer)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	// Load VAE decoder (~775MB)
	m.VAEDecoder = &VAEDecoder{}
	if err := m.VAEDecoder.Load(manifest); err != nil {
		return fmt.Errorf("VAE decoder: %w", err)
	}
	mlx.Eval(mlx.Collect(m.VAEDecoder)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	mem := mlx.MetalGetActiveMemory()
	fmt.Printf("  Loaded in %.2fs (%.1f GB VRAM)\n", time.Since(start).Seconds(), float64(mem)/(1024*1024*1024))

	return nil
}

// LoadFromPath loads the model from a directory path (not ollama manifest)
func (m *Model) LoadFromPath(modelPath string) error {
	fmt.Printf("Loading GLM-Image model from path: %s...\n", modelPath)
	start := time.Now()

	if mlx.GPUIsAvailable() {
		mlx.SetDefaultDeviceGPU()
		mlx.EnableCompile()
	}

	m.ModelName = modelPath

	// Create ByT5 tokenizer (byte-level, no vocabulary file needed)
	fmt.Print("  Creating ByT5 tokenizer... ")
	m.Tokenizer = NewByT5Tokenizer()
	fmt.Println("✓")

	// Load GLM tokenizer for AR model (visual token generation)
	fmt.Print("  Loading GLM tokenizer... ")
	glmTok, err := NewGLMTokenizerFromPath(modelPath)
	if err != nil {
		return fmt.Errorf("glm tokenizer: %w", err)
	}
	m.GLMTokenizer = glmTok
	fmt.Println("✓")

	// Load T5 text encoder
	m.TextEncoder = &T5TextEncoder{}
	if err := m.TextEncoder.LoadFromPath(filepath.Join(modelPath, "text_encoder")); err != nil {
		return fmt.Errorf("text encoder: %w", err)
	}
	mlx.Eval(mlx.Collect(m.TextEncoder)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	// Load vision-language encoder
	m.VisionLanguageEncoder = &VisionLanguageEncoder{}
	if err := m.VisionLanguageEncoder.LoadFromPath(filepath.Join(modelPath, "vision_language_encoder")); err != nil {
		return fmt.Errorf("vision language encoder: %w", err)
	}
	mlx.Eval(mlx.Collect(m.VisionLanguageEncoder)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	// Load diffusion transformer
	m.Transformer = &DiffusionTransformer{}
	if err := m.Transformer.LoadFromPath(filepath.Join(modelPath, "transformer")); err != nil {
		return fmt.Errorf("transformer: %w", err)
	}
	mlx.Eval(mlx.Collect(m.Transformer)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	// Load VAE decoder
	m.VAEDecoder = &VAEDecoder{}
	if err := m.VAEDecoder.LoadFromPath(filepath.Join(modelPath, "vae")); err != nil {
		return fmt.Errorf("VAE decoder: %w", err)
	}
	mlx.Eval(mlx.Collect(m.VAEDecoder)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

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
	fmt.Printf("Generated in %.2fs (%d diffusion steps)\n", time.Since(start).Seconds(), cfg.Steps)
	return result, nil
}

// GenerateImage implements model.ImageModel interface.
func (m *Model) GenerateImage(ctx context.Context, prompt string, width, height int32, steps int, seed int64) (*mlx.Array, error) {
	return m.Generate(prompt, width, height, steps, seed)
}

// generate is the internal generation pipeline.
func (m *Model) generate(ctx context.Context, cfg *GenerateConfig) (*mlx.Array, error) {
	// Apply defaults
	if cfg.Width <= 0 {
		cfg.Width = 1024
	}
	if cfg.Height <= 0 {
		cfg.Height = 1024
	}
	if cfg.Steps <= 0 {
		cfg.Steps = 50
	}
	if cfg.GuidanceScale <= 0 {
		cfg.GuidanceScale = 1.5
	}
	// Calculate MaxVisualTokens based on image dimensions
	// GLM-Image generates TWO grids of visual tokens:
	//   1. First: prev (small) grid - prevTokenH × prevTokenW tokens
	//   2. Then: target (large) grid - tokenH × tokenW tokens
	// After generation, we extract only the TARGET grid tokens for diffusion.
	factor := int32(32)
	tokenH := cfg.Height / factor
	tokenW := cfg.Width / factor
	targetGridTokens := tokenH * tokenW

	// Compute prev grid dimensions using diffusers formula:
	// ratio = token_h / token_w
	// prev_token_h = int(sqrt(ratio) * 16)
	// prev_token_w = int(sqrt(1/ratio) * 16)
	ratio := float64(tokenH) / float64(tokenW)
	prevTokenH := int32(math.Sqrt(ratio) * 16)
	prevTokenW := int32(math.Sqrt(1/ratio) * 16)
	prevGridTokens := prevTokenH * prevTokenW

	// Total tokens to generate = prev grid + target grid
	// (diffusers does max_new_tokens = total + 1 for EOS, but we stop on EOS anyway)
	cfg.MaxVisualTokens = prevGridTokens + targetGridTokens
	if cfg.Temperature <= 0 {
		cfg.Temperature = 0.9
	}
	if cfg.TopP <= 0 {
		cfg.TopP = 0.75
	}

	// Ensure dimensions are divisible by 32
	cfg.Width = (cfg.Width / 32) * 32
	cfg.Height = (cfg.Height / 32) * 32

	tcfg := m.Transformer.Config
	latentH := cfg.Height / 8
	latentW := cfg.Width / 8

	// Progress callback helper
	progress := func(stage string, step, total int) {
		if cfg.Progress != nil {
			cfg.Progress(stage, step, total)
		}
	}

	// === PHASE 1: T5 Text Encoding ===
	fmt.Println("[T5] Encoding glyph text...")
	progress("text_encoding", 0, 1)
	textEmbed := m.TextEncoder.EncodePrompt(m.Tokenizer, cfg.Prompt)
	mlx.Keep(textEmbed)
	mlx.Eval(textEmbed)
	fmt.Printf("[T5] Done, shape: %v\n", textEmbed.Shape())
	progress("text_encoding", 1, 1)

	// === PHASE 2: AR Visual Token Generation ===
	fmt.Printf("[AR] Generating %d visual tokens...\n", cfg.MaxVisualTokens)
	progress("ar_generation", 0, int(cfg.MaxVisualTokens))
	visualTokens := m.VisionLanguageEncoder.Generate(
		cfg.Prompt,
		m.GLMTokenizer,
		cfg.MaxVisualTokens,
		cfg.Temperature,
		cfg.TopP,
		cfg.Seed,
		cfg.Height,
		cfg.Width,
		func(step int) {
			if step%100 == 0 || step < 10 {
				fmt.Printf("[AR] Step %d/%d\n", step, cfg.MaxVisualTokens)
			}
			progress("ar_generation", step, int(cfg.MaxVisualTokens))
		},
	)
	mlx.Keep(visualTokens)
	mlx.Eval(visualTokens)
	fmt.Printf("[AR] Done generating visual tokens\n")
	progress("ar_generation", int(cfg.MaxVisualTokens), int(cfg.MaxVisualTokens))

	vtShape := visualTokens.Shape()
	totalGenerated := vtShape[1]
	fmt.Printf("[AR] Generated %d tokens total\n", totalGenerated)

	// Extract only the TARGET grid tokens (skip the prev grid tokens)
	// diffusers: large_image_tokens = outputs[input_length + large_image_start_offset : ...]
	// large_image_start_offset = prev_grid_size
	var targetGridVisualTokens *mlx.Array
	if totalGenerated >= prevGridTokens+targetGridTokens {
		// Full generation completed - extract target grid
		targetGridVisualTokens = mlx.Slice(visualTokens,
			[]int32{0, prevGridTokens},
			[]int32{1, prevGridTokens + targetGridTokens})
		mlx.Keep(targetGridVisualTokens)
		mlx.Eval(targetGridVisualTokens)
	} else if totalGenerated > prevGridTokens {
		// Partial target grid - take what we have
		actualTargetTokens := totalGenerated - prevGridTokens
		targetGridVisualTokens = mlx.Slice(visualTokens,
			[]int32{0, prevGridTokens},
			[]int32{1, totalGenerated})
		mlx.Keep(targetGridVisualTokens)
		mlx.Eval(targetGridVisualTokens)
		fmt.Printf("WARNING: Partial target grid: got %d/%d target tokens\n",
			actualTargetTokens, targetGridTokens)
	} else {
		// Not enough tokens - EOS came too early
		return nil, fmt.Errorf("AR generation stopped too early: got %d tokens, need at least %d (prev grid) + 1",
			totalGenerated, prevGridTokens)
	}

	// === PHASE 3: Diffusion Decoding ===
	// Setup scheduler with dynamic shift based on image size
	scheduler := NewFlowMatchScheduler(DefaultSchedulerConfig())
	imgSeqLen := (latentH / tcfg.PatchSize) * (latentW / tcfg.PatchSize)
	scheduler.SetTimestepsWithDynamicShift(cfg.Steps, imgSeqLen)

	// Initialize noise latents [B, C, H, W]
	latents := scheduler.InitNoise([]int32{1, tcfg.InChannels, latentH, latentW}, cfg.Seed)
	mlx.Eval(latents)

	// Upsample TARGET grid visual tokens 2x to match patch count (matching diffusers)
	// target_grid tokens -> 2x upsample -> patch_count
	// e.g., 32x32=1024 tokens -> 64x64=4096 patches for 1024x1024
	visualTokensUpsampled := upsampleTokens(targetGridVisualTokens, tokenH, tokenW, 2)

	// Prepare prior embeddings from upsampled visual tokens (VQ codebook lookup + projection)
	priorEmbed := m.Transformer.EmbedPriorTokens(visualTokensUpsampled)
	mlx.Keep(priorEmbed)
	mlx.Eval(priorEmbed)

	// Prepare text conditioning (project T5 embeddings)
	textCond := m.Transformer.ProjectTextEmbeddings(textEmbed)
	mlx.Keep(textCond)
	mlx.Eval(textCond)

	// === CFG Setup ===
	// For classifier-free guidance, we need unconditional (negative) text embeddings
	// GLM-Image uses empty string "" for negative prompt
	doCFG := cfg.GuidanceScale > 1.0
	var negativeTextCond *mlx.Array
	if doCFG {
		// Encode empty string for negative prompt
		negativeTextEmbed := m.TextEncoder.EncodePrompt(m.Tokenizer, "")
		mlx.Keep(negativeTextEmbed)
		mlx.Eval(negativeTextEmbed)
		negativeTextCond = m.Transformer.ProjectTextEmbeddings(negativeTextEmbed)
		mlx.Keep(negativeTextCond)
		mlx.Eval(negativeTextCond)
		negativeTextEmbed.Free()
	}

	// Prepare conditioning inputs
	targetSize := mlx.NewArray([]float32{float32(cfg.Height), float32(cfg.Width)}, []int32{1, 2})
	cropCoords := mlx.NewArray([]float32{0, 0}, []int32{1, 2}) // Default: no crop offset
	targetSize = mlx.ToBFloat16(targetSize)
	cropCoords = mlx.ToBFloat16(cropCoords)
	mlx.Keep(targetSize)
	mlx.Keep(cropCoords)
	mlx.Eval(targetSize, cropCoords)

	pH := latentH / tcfg.PatchSize
	pW := latentW / tcfg.PatchSize

	// Denoising loop
	fmt.Printf("[Diffusion] Starting %d denoising steps...\n", cfg.Steps)
	progress("diffusion", 0, cfg.Steps)
	for i := 0; i < cfg.Steps; i++ {
		fmt.Printf("[Diffusion] Step %d/%d (timestep=%.1f)\n", i+1, cfg.Steps, scheduler.Timesteps[i]-1)
		// Check for cancellation
		if ctx != nil {
			select {
			case <-ctx.Done():
				textEmbed.Free()
				visualTokens.Free()
				// visualTokensUpsampled points to visualTokens, don't double-free
				priorEmbed.Free()
				textCond.Free()
				latents.Free()
				return nil, ctx.Err()
			default:
			}
		}

		// Get timestep value for the transformer
		// scheduler.Timesteps contains raw timestep values (1000 down to ~20)
		// Pass timestep - 1 to match diffusers: timestep = t.expand(latents.shape[0]) - 1
		timestepVal := scheduler.Timesteps[i] - 1
		timestep := mlx.ToBFloat16(mlx.NewArray([]float32{timestepVal}, []int32{1}))

		// Patchify latents [B, C, H, W] -> [B, L, C*p*p]
		patches := PatchifyLatents(latents, tcfg.PatchSize)

		// Transformer forward with MMDiT architecture
		// Conditional pass (with text + prior embeddings)
		outputCond := m.Transformer.ForwardWithPriorDrop(
			patches,
			priorEmbed,
			textCond,
			timestep,
			targetSize,
			cropCoords,
			pH,
			pW,
			false, // priorTokenDrop = false for conditional
		)

		// Unpatchify [B, L, C*p*p] -> [B, C, H, W]
		noisePredCond := UnpatchifyLatents(outputCond, latentH, latentW, tcfg.PatchSize, tcfg.OutChannels)

		var noisePred *mlx.Array
		if doCFG {
			// Unconditional pass (empty text, dropped prior embeddings)
			outputUncond := m.Transformer.ForwardWithPriorDrop(
				patches,
				priorEmbed, // Still passed but will be ignored due to priorTokenDrop=true
				negativeTextCond,
				timestep,
				targetSize,
				cropCoords,
				pH,
				pW,
				true, // priorTokenDrop = true for unconditional
			)
			noisePredUncond := UnpatchifyLatents(outputUncond, latentH, latentW, tcfg.PatchSize, tcfg.OutChannels)

			// CFG formula: noise_pred = uncond + guidance_scale * (cond - uncond)
			diff := mlx.Sub(noisePredCond, noisePredUncond)
			scaled := mlx.MulScalar(diff, cfg.GuidanceScale)
			noisePred = mlx.Add(noisePredUncond, scaled)
		} else {
			noisePred = noisePredCond
		}

		// Scheduler step
		oldLatents := latents
		latents = scheduler.Step(noisePred, latents, i)
		mlx.Eval(latents)
		oldLatents.Free()

		progress("diffusion", i+1, cfg.Steps)
	}

	// Cleanup intermediate arrays
	textEmbed.Free()
	visualTokens.Free()
	// visualTokensUpsampled points to visualTokens, don't double-free
	priorEmbed.Free()
	textCond.Free()
	if negativeTextCond != nil {
		negativeTextCond.Free()
	}
	targetSize.Free()
	cropCoords.Free()

	// === PHASE 4: VAE Decode ===
	progress("vae_decode", 0, 1)
	decoded := m.VAEDecoder.Decode(latents)
	mlx.Eval(decoded)
	latents.Free()
	progress("vae_decode", 1, 1)

	return decoded, nil
}

// upsampleTokens performs nearest-neighbor upsampling of visual tokens
// Converts from prev_grid (e.g., 16x16) to target_grid (e.g., 32x32 for 2x, 64x64 for 4x)
// scale must be 2 or 4
//
// Handles early EOS gracefully: if tokens has fewer than prevH*prevW elements,
// missing tokens are padded with 0 (visual token padding value).
func upsampleTokens(tokens *mlx.Array, prevH, prevW int32, scale int32) *mlx.Array {
	// tokens: [1, N] where N <= prevH*prevW (may be shorter if early EOS)
	// Each token at (i, j) becomes scale*scale tokens in the output

	mlx.Eval(tokens)
	tokenData := tokens.DataInt32()
	numTokens := int32(len(tokenData))
	expectedTokens := prevH * prevW

	// Warn if we got fewer tokens than expected (early EOS)
	if numTokens < expectedTokens {
		fmt.Printf("WARNING: upsampleTokens got %d tokens, expected %d (padding with 0)\n",
			numTokens, expectedTokens)
	}

	targetH := prevH * scale
	targetW := prevW * scale
	upsampled := make([]int32, targetH*targetW)

	for i := int32(0); i < prevH; i++ {
		for j := int32(0); j < prevW; j++ {
			srcIdx := i*prevW + j

			// Handle early EOS: use 0 (padding) for missing tokens
			var val int32
			if srcIdx < numTokens {
				val = tokenData[srcIdx]
			} else {
				val = 0 // Padding token
			}

			// Place in scale*scale positions
			dstI := i * scale
			dstJ := j * scale
			for di := int32(0); di < scale; di++ {
				for dj := int32(0); dj < scale; dj++ {
					upsampled[(dstI+di)*targetW+(dstJ+dj)] = val
				}
			}
		}
	}

	return mlx.NewArrayInt32(upsampled, []int32{1, targetH * targetW})
}

// PatchifyLatents converts [B, C, H, W] to [B, L, C*p*p]
func PatchifyLatents(latents *mlx.Array, patchSize int32) *mlx.Array {
	shape := latents.Shape()
	B := shape[0]
	C := shape[1]
	H := shape[2]
	W := shape[3]

	pH := H / patchSize
	pW := W / patchSize

	// Reshape: [B, C, H, W] -> [B, C, pH, p, pW, p]
	x := mlx.Reshape(latents, B, C, pH, patchSize, pW, patchSize)
	// Transpose: -> [B, pH, pW, C, p, p]
	x = mlx.Transpose(x, 0, 2, 4, 1, 3, 5)
	// Flatten: -> [B, pH*pW, C*p*p]
	return mlx.Reshape(x, B, pH*pW, C*patchSize*patchSize)
}

// UnpatchifyLatents converts [B, L, C*p*p] back to [B, C, H, W]
func UnpatchifyLatents(patches *mlx.Array, H, W, patchSize, channels int32) *mlx.Array {
	shape := patches.Shape()
	B := shape[0]

	pH := H / patchSize
	pW := W / patchSize

	// Reshape: [B, L, C*p*p] -> [B, pH, pW, C, p, p]
	x := mlx.Reshape(patches, B, pH, pW, channels, patchSize, patchSize)
	// Transpose: -> [B, C, pH, p, pW, p]
	x = mlx.Transpose(x, 0, 3, 1, 4, 2, 5)
	// Reshape: -> [B, C, H, W]
	return mlx.Reshape(x, B, channels, pH*patchSize, pW*patchSize)
}

// CalculateShift computes the dynamic shift for flow matching based on image sequence length.
func CalculateShift(imgSeqLen int32) float32 {
	cfg := DefaultSchedulerConfig()
	if !cfg.UseDynamicShifting {
		return 0
	}

	// Sqrt-based shift calculation (matches diffusers)
	m := float32(math.Sqrt(float64(imgSeqLen) / float64(cfg.BaseImageSeqLen)))
	return m*cfg.MaxShift + cfg.BaseShift
}

// UpsampleTokens2x upsamples token IDs by 2x using nearest neighbor interpolation
// tokens: [B, H*W] -> [B, (H*2)*(W*2)]
// This matches diffusers' _upsample_token_ids function
func UpsampleTokens2x(tokens *mlx.Array, gridH, gridW int32) *mlx.Array {
	shape := tokens.Shape()
	B := shape[0]

	// Reshape to [B, 1, H, W] for interpolation
	tokens = mlx.Reshape(tokens, B, 1, gridH, gridW)

	// Convert to float for interpolation
	tokensFloat := mlx.AsType(tokens, mlx.DtypeFloat32)

	// 2x nearest neighbor upsample
	// [B, 1, H, W] -> [B, 1, H*2, W*2]
	upsampled := nearestUpsample2x(tokensFloat)

	// Convert back to int and reshape to [B, H*2*W*2]
	upsampled = mlx.AsType(upsampled, mlx.DtypeInt32)
	return mlx.Reshape(upsampled, B, gridH*2*gridW*2)
}

// nearestUpsample2x performs 2x nearest neighbor upsampling on NCHW tensor
func nearestUpsample2x(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	C := shape[1]
	H := shape[2]
	W := shape[3]

	// Repeat each element 2x2
	// [B, C, H, W] -> [B, C, H, 1, W, 1] -> [B, C, H, 2, W, 2] -> [B, C, H*2, W*2]
	x = mlx.Reshape(x, B, C, H, 1, W, 1)

	// Tile to repeat each pixel 2x2
	x = mlx.Tile(x, []int32{1, 1, 1, 2, 1, 2})

	// Reshape to final size
	return mlx.Reshape(x, B, C, H*2, W*2)
}

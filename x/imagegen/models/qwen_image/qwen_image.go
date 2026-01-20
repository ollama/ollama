//go:build mlx

// Package qwen_image implements the Qwen-Image diffusion transformer model.
package qwen_image

import (
	"context"
	"fmt"
	"path/filepath"
	"time"

	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
)

// GenerateConfig holds all options for image generation.
type GenerateConfig struct {
	Prompt         string
	NegativePrompt string                // Empty = no CFG
	CFGScale       float32               // Only used if NegativePrompt is set (default: 4.0)
	Width          int32                 // Image width (default: 1024)
	Height         int32                 // Image height (default: 1024)
	Steps          int                   // Denoising steps (default: 30)
	Seed           int64                 // Random seed
	Progress       func(step, totalSteps int) // Optional progress callback

	// Layer caching (DeepCache/Learning-to-Cache speedup)
	LayerCache    bool // Enable layer caching (default: false)
	CacheInterval int  // Refresh cache every N steps (default: 3)
	CacheLayers   int  // Number of shallow layers to cache (default: 25)
}

// Model represents a Qwen-Image diffusion model.
type Model struct {
	ModelPath   string
	Tokenizer   *tokenizer.Tokenizer
	TextEncoder *Qwen25VL
	Transformer *Transformer
	VAEDecoder  *VAEDecoder
}

// Load loads the Qwen-Image model from a directory.
func (m *Model) Load(modelPath string) error {
	fmt.Println("Loading Qwen-Image model...")
	start := time.Now()

	if mlx.GPUIsAvailable() {
		mlx.SetDefaultDeviceGPU()
		mlx.EnableCompile()
	}

	m.ModelPath = modelPath

	// Load tokenizer
	fmt.Print("  Loading tokenizer... ")
	tokenizerPath := filepath.Join(modelPath, "tokenizer")
	tok, err := tokenizer.Load(tokenizerPath)
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}
	m.Tokenizer = tok
	fmt.Println("✓")

	// Load text encoder (Qwen2.5-VL in text-only mode - skip vision tower for efficiency)
	m.TextEncoder = &Qwen25VL{}
	if err := m.TextEncoder.LoadTextOnly(filepath.Join(modelPath, "text_encoder")); err != nil {
		return fmt.Errorf("text encoder: %w", err)
	}
	mlx.Eval(mlx.Collect(m.TextEncoder)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	// Load transformer
	m.Transformer = &Transformer{}
	if err := m.Transformer.Load(filepath.Join(modelPath, "transformer")); err != nil {
		return fmt.Errorf("transformer: %w", err)
	}
	mlx.Eval(mlx.Collect(m.Transformer)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	// Load VAE decoder
	m.VAEDecoder = &VAEDecoder{}
	if err := m.VAEDecoder.Load(filepath.Join(modelPath, "vae")); err != nil {
		return fmt.Errorf("VAE decoder: %w", err)
	}
	mlx.Eval(mlx.Collect(m.VAEDecoder)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	mem := mlx.MetalGetActiveMemory()
	peak := mlx.MetalGetPeakMemory()
	fmt.Printf("  Loaded in %.2fs (%.1f GB active, %.1f GB peak)\n",
		time.Since(start).Seconds(),
		float64(mem)/(1024*1024*1024),
		float64(peak)/(1024*1024*1024))

	return nil
}

// Generate creates an image from a prompt.
func (m *Model) Generate(prompt string, width, height int32, steps int, seed int64) (*mlx.Array, error) {
	return m.GenerateFromConfig(&GenerateConfig{
		Prompt: prompt,
		Width:  width,
		Height: height,
		Steps:  steps,
		Seed:   seed,
	})
}

// GenerateWithProgress creates an image with progress callback.
func (m *Model) GenerateWithProgress(prompt string, width, height int32, steps int, seed int64, progress func(step, totalSteps int)) (*mlx.Array, error) {
	return m.GenerateFromConfig(&GenerateConfig{
		Prompt:   prompt,
		Width:    width,
		Height:   height,
		Steps:    steps,
		Seed:     seed,
		Progress: progress,
	})
}

// GenerateWithCFG creates an image with classifier-free guidance.
func (m *Model) GenerateWithCFG(prompt, negativePrompt string, width, height int32, steps int, seed int64, cfgScale float32, progress func(step, totalSteps int)) (*mlx.Array, error) {
	return m.GenerateFromConfig(&GenerateConfig{
		Prompt:         prompt,
		NegativePrompt: negativePrompt,
		CFGScale:       cfgScale,
		Width:          width,
		Height:         height,
		Steps:          steps,
		Seed:           seed,
		Progress:       progress,
	})
}

// GenerateFromConfig generates an image using the unified config struct.
func (m *Model) GenerateFromConfig(cfg *GenerateConfig) (*mlx.Array, error) {
	start := time.Now()
	result, err := m.generate(cfg)
	if err != nil {
		return nil, err
	}
	if cfg.NegativePrompt != "" {
		fmt.Printf("Generated with CFG (scale=%.1f) in %.2fs (%d steps)\n", cfg.CFGScale, time.Since(start).Seconds(), cfg.Steps)
	} else {
		fmt.Printf("Generated in %.2fs (%d steps)\n", time.Since(start).Seconds(), cfg.Steps)
	}
	return result, nil
}

// GenerateImage implements model.ImageModel interface.
func (m *Model) GenerateImage(ctx context.Context, prompt string, width, height int32, steps int, seed int64) (*mlx.Array, error) {
	return m.Generate(prompt, width, height, steps, seed)
}

// generate is the internal denoising pipeline.
func (m *Model) generate(cfg *GenerateConfig) (*mlx.Array, error) {
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
	if cfg.CFGScale <= 0 {
		cfg.CFGScale = 4.0
	}
	if cfg.CacheInterval <= 0 {
		cfg.CacheInterval = 3
	}
	if cfg.CacheLayers <= 0 {
		cfg.CacheLayers = 25 // ~42% of 60 layers (similar ratio to Z-Image's 15/38)
	}

	useCFG := cfg.NegativePrompt != ""
	tcfg := m.Transformer.Config
	latentH := cfg.Height / 8
	latentW := cfg.Width / 8
	pH := latentH / tcfg.PatchSize
	pW := latentW / tcfg.PatchSize
	imgSeqLen := pH * pW

	// Text encoding
	var posEmb, negEmb *mlx.Array
	{
		posEmb = m.TextEncoder.EncodePrompt(m.Tokenizer, cfg.Prompt)
		if useCFG {
			negEmb = m.TextEncoder.EncodePrompt(m.Tokenizer, cfg.NegativePrompt)
			mlx.Keep(posEmb, negEmb)
			mlx.Eval(posEmb, negEmb)
		} else {
			mlx.Keep(posEmb)
			mlx.Eval(posEmb)
		}
	}

	// Pad sequences to same length for CFG
	txtLen := posEmb.Shape()[1]
	if useCFG {
		negLen := negEmb.Shape()[1]
		if negLen > txtLen {
			txtLen = negLen
		}
		if posEmb.Shape()[1] < txtLen {
			posEmb = padSequence(posEmb, txtLen)
		}
		if negEmb.Shape()[1] < txtLen {
			negEmb = padSequence(negEmb, txtLen)
		}
		mlx.Keep(posEmb, negEmb)
	}

	// Pre-compute batched embeddings for CFG (single forward pass optimization)
	var batchedEmb *mlx.Array
	if useCFG {
		batchedEmb = mlx.Concatenate([]*mlx.Array{posEmb, negEmb}, 0)
		mlx.Keep(batchedEmb)
		mlx.Eval(batchedEmb)
	}

	// Scheduler
	scheduler := NewFlowMatchScheduler(DefaultSchedulerConfig())
	scheduler.SetTimesteps(cfg.Steps, imgSeqLen)

	// Init latents [B, C, T, H, W]
	var latents *mlx.Array
	{
		latents = scheduler.InitNoise([]int32{1, tcfg.OutChannels, 1, latentH, latentW}, cfg.Seed)
		mlx.Eval(latents)
	}

	// RoPE cache
	var ropeCache *RoPECache
	{
		ropeCache = PrepareRoPE(pH, pW, txtLen, tcfg.AxesDimsRope)
		mlx.Keep(ropeCache.ImgFreqs, ropeCache.TxtFreqs)
		mlx.Eval(ropeCache.ImgFreqs)
	}

	// Layer cache for DeepCache/Learning-to-Cache speedup
	var stepCache *cache.StepCache
	if cfg.LayerCache {
		stepCache = cache.NewStepCache(cfg.CacheLayers)
		fmt.Printf("  Layer caching: %d layers, refresh every %d steps\n", cfg.CacheLayers, cfg.CacheInterval)
	}

	// Denoising loop
	for i := 0; i < cfg.Steps; i++ {
		stepStart := time.Now()
		if cfg.Progress != nil {
			cfg.Progress(i+1, cfg.Steps)
		}

		t := scheduler.Timesteps[i]
		timestep := mlx.ToBFloat16(mlx.NewArray([]float32{t}, []int32{1}))

		// Squeeze temporal dim: [B, C, T, H, W] -> [B, C, H, W]
		latents2D := mlx.Squeeze(latents, 2)
		patches := PackLatents(latents2D, tcfg.PatchSize)

		var output *mlx.Array
		if useCFG {
			// CFG Batching: single forward pass with batch=2
			// Note: layer caching with CFG is not supported yet (would need 2 caches)
			batchedPatches := mlx.Tile(patches, []int32{2, 1, 1})
			batchedTimestep := mlx.Tile(timestep, []int32{2})

			// Single batched forward pass
			batchedOutput := m.Transformer.Forward(batchedPatches, batchedEmb, batchedTimestep, ropeCache.ImgFreqs, ropeCache.TxtFreqs)

			// Split output: [2, L, D] -> pos [1, L, D], neg [1, L, D]
			L := batchedOutput.Shape()[1]
			D := batchedOutput.Shape()[2]
			posOutput := mlx.Slice(batchedOutput, []int32{0, 0, 0}, []int32{1, L, D})
			negOutput := mlx.Slice(batchedOutput, []int32{1, 0, 0}, []int32{2, L, D})

			diff := mlx.Sub(posOutput, negOutput)
			scaledDiff := mlx.MulScalar(diff, cfg.CFGScale)
			combPred := mlx.Add(negOutput, scaledDiff)

			// Norm rescaling: rescale combined prediction to match conditional prediction's norm
			condNorm := mlx.Sqrt(mlx.Sum(mlx.Square(posOutput), -1, true))
			combNorm := mlx.Sqrt(mlx.Sum(mlx.Square(combPred), -1, true))
			output = mlx.Mul(combPred, mlx.Div(condNorm, combNorm))
		} else if stepCache != nil {
			output = m.Transformer.ForwardWithCache(patches, posEmb, timestep, ropeCache.ImgFreqs, ropeCache.TxtFreqs,
				stepCache, i, cfg.CacheInterval, cfg.CacheLayers)
		} else {
			output = m.Transformer.Forward(patches, posEmb, timestep, ropeCache.ImgFreqs, ropeCache.TxtFreqs)
		}

		noisePred := UnpackLatents(output, latentH, latentW, tcfg.PatchSize)
		oldLatents := latents
		latents = scheduler.Step(noisePred, latents, i)

		// Keep cached arrays alive across cleanup
		if stepCache != nil {
			mlx.Keep(stepCache.Arrays()...)
		}
		mlx.Eval(latents)
		oldLatents.Free()

		activeMem := float64(mlx.MetalGetActiveMemory()) / (1024 * 1024 * 1024)
		peakMem := float64(mlx.MetalGetPeakMemory()) / (1024 * 1024 * 1024)
		fmt.Printf("  Step %d/%d: t=%.4f (%.2fs) [%.1f GB active, %.1f GB peak]\n", i+1, cfg.Steps, t, time.Since(stepStart).Seconds(), activeMem, peakMem)
	}

	// Free denoising temporaries before VAE decode
	posEmb.Free()
	if negEmb != nil {
		negEmb.Free()
	}
	if batchedEmb != nil {
		batchedEmb.Free()
	}
	ropeCache.ImgFreqs.Free()
	ropeCache.TxtFreqs.Free()
	if stepCache != nil {
		stepCache.Free()
	}

	// VAE decode (Decode manages its own pools for staged memory)
	decoded := m.VAEDecoder.Decode(latents)
	latents.Free()
	// Post-process: squeeze temporal dim and rescale to [0, 1]
	{
		decoded = mlx.Squeeze(decoded, 2)
		decoded = mlx.AddScalar(decoded, 1.0)
		decoded = mlx.DivScalar(decoded, 2.0)
		mlx.Eval(decoded)
	}

	fmt.Printf("  Peak memory: %.2f GB\n", float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	return decoded, nil
}

// padSequence pads a sequence tensor to the target length with zeros
func padSequence(x *mlx.Array, targetLen int32) *mlx.Array {
	shape := x.Shape()
	currentLen := shape[1]
	if currentLen >= targetLen {
		return x
	}
	padLen := targetLen - currentLen
	// Pad on sequence dimension (axis 1)
	return mlx.Pad(x, []int32{0, 0, 0, padLen, 0, 0})
}

// LoadPersistent is an alias for backward compatibility.
// Use m := &Model{}; m.Load(path) instead.
func LoadPersistent(modelPath string) (*Model, error) {
	m := &Model{}
	if err := m.Load(modelPath); err != nil {
		return nil, err
	}
	return m, nil
}

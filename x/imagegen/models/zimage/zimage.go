// Package zimage implements the Z-Image diffusion transformer model.
package zimage

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
	NegativePrompt string       // Empty = no CFG
	CFGScale       float32      // Only used if NegativePrompt is set (default: 4.0)
	Width          int32        // Image width (default: 1024)
	Height         int32        // Image height (default: 1024)
	Steps          int          // Denoising steps (default: 9 for turbo)
	Seed           int64        // Random seed
	Progress       ProgressFunc // Optional progress callback
	CapturePath    string       // GPU capture path (debug)

	// Layer caching options (speedup via shallow layer reuse)
	LayerCache    bool // Enable layer caching (default: false)
	CacheInterval int  // Refresh cache every N steps (default: 3)
	CacheLayers   int  // Number of shallow layers to cache (default: 15)
}

// ProgressFunc is called during generation with step progress.
type ProgressFunc func(step, totalSteps int)

// Model represents a Z-Image diffusion model.
type Model struct {
	ModelPath   string
	Tokenizer   *tokenizer.Tokenizer
	TextEncoder *Qwen3TextEncoder
	Transformer *Transformer
	VAEDecoder  *VAEDecoder
}

// Load loads the Z-Image model from a directory.
func (m *Model) Load(modelPath string) error {
	fmt.Println("Loading Z-Image model...")
	start := time.Now()

	if mlx.GPUIsAvailable() {
		mlx.SetDefaultDeviceGPU()
		mlx.EnableCompile()
	}

	m.ModelPath = modelPath

	// Load tokenizer
	fmt.Print("  Loading tokenizer... ")
	tokenizerPath := filepath.Join(modelPath, "tokenizer", "tokenizer.json")
	tok, err := tokenizer.Load(tokenizerPath)
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}
	m.Tokenizer = tok
	fmt.Println("✓")

	// Load text encoder
	m.TextEncoder = &Qwen3TextEncoder{}
	if err := m.TextEncoder.Load(filepath.Join(modelPath, "text_encoder")); err != nil {
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
	fmt.Printf("  Loaded in %.2fs (%.1f GB VRAM)\n", time.Since(start).Seconds(), float64(mem)/(1024*1024*1024))

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
func (m *Model) GenerateWithProgress(prompt string, width, height int32, steps int, seed int64, progress ProgressFunc) (*mlx.Array, error) {
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
func (m *Model) GenerateWithCFG(prompt, negativePrompt string, width, height int32, steps int, seed int64, cfgScale float32, progress ProgressFunc) (*mlx.Array, error) {
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
		cfg.Steps = 9 // Turbo default
	}
	if cfg.CFGScale <= 0 {
		cfg.CFGScale = 4.0
	}
	if cfg.LayerCache {
		if cfg.CacheInterval <= 0 {
			cfg.CacheInterval = 3
		}
		if cfg.CacheLayers <= 0 {
			cfg.CacheLayers = 15 // Half of 30 layers
		}
	}

	useCFG := cfg.NegativePrompt != ""
	tcfg := m.Transformer.TransformerConfig
	latentH := cfg.Height / 8
	latentW := cfg.Width / 8
	hTok := latentH / tcfg.PatchSize
	wTok := latentW / tcfg.PatchSize

	// Text encoding with padding to multiple of 32
	var posEmb, negEmb *mlx.Array
	{
		posEmb, _ = m.TextEncoder.EncodePrompt(m.Tokenizer, cfg.Prompt, 512)
		if useCFG {
			negEmb, _ = m.TextEncoder.EncodePrompt(m.Tokenizer, cfg.NegativePrompt, 512)
		}

		// Pad both to same length (multiple of 32)
		maxLen := posEmb.Shape()[1]
		if useCFG && negEmb.Shape()[1] > maxLen {
			maxLen = negEmb.Shape()[1]
		}
		if pad := (32 - (maxLen % 32)) % 32; pad > 0 {
			maxLen += pad
		}

		posEmb = padToLength(posEmb, maxLen)
		if useCFG {
			negEmb = padToLength(negEmb, maxLen)
			mlx.Keep(posEmb, negEmb)
			mlx.Eval(posEmb, negEmb)
		} else {
			mlx.Keep(posEmb)
			mlx.Eval(posEmb)
		}
	}

	// Scheduler
	scheduler := NewFlowMatchEulerScheduler(DefaultFlowMatchSchedulerConfig())
	scheduler.SetTimestepsWithMu(cfg.Steps, CalculateShift(hTok*wTok))

	// Init latents [B, C, H, W]
	var latents *mlx.Array
	{
		latents = scheduler.InitNoise([]int32{1, tcfg.InChannels, latentH, latentW}, cfg.Seed)
		mlx.Eval(latents)
	}

	// RoPE cache
	var ropeCache *RoPECache
	{
		ropeCache = m.Transformer.PrepareRoPECache(hTok, wTok, posEmb.Shape()[1])
		mlx.Keep(ropeCache.ImgCos, ropeCache.ImgSin, ropeCache.CapCos, ropeCache.CapSin,
			ropeCache.UnifiedCos, ropeCache.UnifiedSin)
		mlx.Eval(ropeCache.UnifiedCos)
	}

	// Step cache for shallow layer reuse (DeepCache/Learning-to-Cache style)
	var stepCache *cache.StepCache
	if cfg.LayerCache {
		stepCache = cache.NewStepCache(cfg.CacheLayers)
		fmt.Printf("  Layer caching enabled: %d layers, refresh every %d steps\n",
			cfg.CacheLayers, cfg.CacheInterval)
	}

	// Denoising loop
	for i := 0; i < cfg.Steps; i++ {
		stepStart := time.Now()
		if cfg.Progress != nil {
			cfg.Progress(i+1, cfg.Steps)
		}

		// GPU capture on step 2 if requested
		if cfg.CapturePath != "" && i == 1 {
			mlx.MetalStartCapture(cfg.CapturePath)
		}

		tCurr := scheduler.Timesteps[i]
		timestep := mlx.ToBFloat16(mlx.NewArray([]float32{1.0 - tCurr}, []int32{1}))

		patches := PatchifyLatents(latents, tcfg.PatchSize)

		var output *mlx.Array
		if stepCache != nil {
			// Use layer caching for faster inference
			if useCFG {
				posOutput := m.Transformer.ForwardWithCache(patches, timestep, posEmb, ropeCache,
					stepCache, i, cfg.CacheInterval)
				// Note: CFG with layer cache shares the cache between pos/neg
				// This is approximate but fast - neg prompt uses same cached shallow layers
				negOutput := m.Transformer.ForwardWithCache(patches, timestep, negEmb, ropeCache,
					stepCache, i, cfg.CacheInterval)
				diff := mlx.Sub(posOutput, negOutput)
				scaledDiff := mlx.MulScalar(diff, cfg.CFGScale)
				output = mlx.Add(negOutput, scaledDiff)
			} else {
				output = m.Transformer.ForwardWithCache(patches, timestep, posEmb, ropeCache,
					stepCache, i, cfg.CacheInterval)
			}
		} else {
			// Standard forward without caching
			if useCFG {
				posOutput := m.Transformer.Forward(patches, timestep, posEmb, ropeCache)
				negOutput := m.Transformer.Forward(patches, timestep, negEmb, ropeCache)
				diff := mlx.Sub(posOutput, negOutput)
				scaledDiff := mlx.MulScalar(diff, cfg.CFGScale)
				output = mlx.Add(negOutput, scaledDiff)
			} else {
				output = m.Transformer.Forward(patches, timestep, posEmb, ropeCache)
			}
		}

		noisePred := UnpatchifyLatents(output, tcfg.PatchSize, latentH, latentW, tcfg.InChannels)
		noisePred = mlx.Neg(noisePred)
		oldLatents := latents
		latents = scheduler.Step(noisePred, latents, i)

		// Keep latents and any cached arrays
		if stepCache != nil {
			mlx.Keep(stepCache.Arrays()...)
		}
		mlx.Eval(latents)
		oldLatents.Free()

		if cfg.CapturePath != "" && i == 1 {
			mlx.MetalStopCapture()
		}

		activeMem := float64(mlx.MetalGetActiveMemory()) / (1024 * 1024 * 1024)
		peakMem := float64(mlx.MetalGetPeakMemory()) / (1024 * 1024 * 1024)
		fmt.Printf("  Step %d/%d: t=%.4f (%.2fs) [%.1f GB active, %.1f GB peak]\n",
			i+1, cfg.Steps, tCurr, time.Since(stepStart).Seconds(), activeMem, peakMem)
	}

	// Free denoising temporaries before VAE decode
	posEmb.Free()
	if negEmb != nil {
		negEmb.Free()
	}
	ropeCache.ImgCos.Free()
	ropeCache.ImgSin.Free()
	ropeCache.CapCos.Free()
	ropeCache.CapSin.Free()
	ropeCache.UnifiedCos.Free()
	ropeCache.UnifiedSin.Free()
	if stepCache != nil {
		stepCache.Free()
	}

	// VAE decode
	decoded := m.VAEDecoder.Decode(latents)
	latents.Free()

	return decoded, nil
}

// padToLength pads a sequence tensor to the target length by repeating the last token.
func padToLength(x *mlx.Array, targetLen int32) *mlx.Array {
	shape := x.Shape()
	currentLen := shape[1]
	if currentLen >= targetLen {
		return x
	}
	padLen := targetLen - currentLen
	lastToken := mlx.Slice(x, []int32{0, currentLen - 1, 0}, []int32{shape[0], currentLen, shape[2]})
	padding := mlx.Tile(lastToken, []int32{1, padLen, 1})
	return mlx.Concatenate([]*mlx.Array{x, padding}, 1)
}

// CalculateShift computes the mu shift value for dynamic scheduling
func CalculateShift(imgSeqLen int32) float32 {
	baseSeqLen := float32(256)
	maxSeqLen := float32(4096)
	baseShift := float32(0.5)
	maxShift := float32(1.15)

	m := (maxShift - baseShift) / (maxSeqLen - baseSeqLen)
	b := baseShift - m*baseSeqLen
	return float32(imgSeqLen)*m + b
}

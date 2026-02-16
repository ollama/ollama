//go:build mlx

// Package zimage implements the Z-Image diffusion transformer model.
package zimage

import (
	"context"
	"fmt"
	"time"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
	"github.com/ollama/ollama/x/imagegen/vae"
)

// GenerateConfig holds all options for image generation.
type GenerateConfig struct {
	Prompt         string
	NegativePrompt string                // Empty = no CFG
	CFGScale       float32               // Only used if NegativePrompt is set (default: 4.0)
	Width          int32                 // Image width (default: 1024)
	Height         int32                 // Image height (default: 1024)
	Steps          int                   // Denoising steps (default: 9 for turbo)
	Seed           int64                 // Random seed
	Progress       func(step, totalSteps int) // Optional progress callback
	CapturePath    string                // GPU capture path (debug)

	// TeaCache options (timestep embedding aware caching)
	TeaCache          bool    // TeaCache is always enabled for faster inference
	TeaCacheThreshold float32 // Threshold for cache reuse (default: 0.1, lower = more aggressive)

	// Fused QKV (fuse Q/K/V projections into single matmul)
	FusedQKV bool // Enable fused QKV projection (default: false)
}

// Model represents a Z-Image diffusion model.
type Model struct {
	ModelName   string
	Tokenizer   *tokenizer.Tokenizer
	TextEncoder *Qwen3TextEncoder
	Transformer *Transformer
	VAEDecoder  *VAEDecoder
	qkvFused    bool // Track if QKV has been fused (do only once)
}

// Load loads the Z-Image model from ollama blob storage.
func (m *Model) Load(modelName string) error {
	fmt.Printf("Loading Z-Image model from manifest: %s...\n", modelName)
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

	// Load tokenizer from manifest with config
	fmt.Print("  Loading tokenizer... ")
	tokData, err := manifest.ReadConfig("tokenizer/tokenizer.json")
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}

	// Try to read tokenizer config files from manifest
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
	m.TextEncoder = &Qwen3TextEncoder{}
	if err := m.TextEncoder.Load(manifest, "text_encoder/config.json"); err != nil {
		return fmt.Errorf("text encoder: %w", err)
	}
	mlx.Eval(mlx.Collect(m.TextEncoder)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	// Load transformer
	m.Transformer = &Transformer{}
	if err := m.Transformer.Load(manifest); err != nil {
		return fmt.Errorf("transformer: %w", err)
	}
	mlx.Eval(mlx.Collect(m.Transformer)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	// Load VAE decoder
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

// GenerateWithCFG creates an image with classifier-free guidance.
func (m *Model) GenerateWithCFG(prompt, negativePrompt string, width, height int32, steps int, seed int64, cfgScale float32, progress func(step, totalSteps int)) (*mlx.Array, error) {
	return m.GenerateFromConfig(context.Background(), &GenerateConfig{
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
func (m *Model) GenerateFromConfig(ctx context.Context, cfg *GenerateConfig) (*mlx.Array, error) {
	start := time.Now()
	result, err := m.generate(ctx, cfg)
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

// generate is the internal denoising pipeline.
func (m *Model) generate(ctx context.Context, cfg *GenerateConfig) (*mlx.Array, error) {
	// Apply defaults
	if cfg.Width <= 0 {
		cfg.Width = 1024
	}
	if cfg.Height <= 0 {
		cfg.Height = 1024
	}
	if cfg.Steps <= 0 {
		cfg.Steps = 9 // Z-Image turbo default
	}
	if cfg.CFGScale <= 0 {
		cfg.CFGScale = 4.0
	}
	// TeaCache enabled by default
	cfg.TeaCache = true
	if cfg.TeaCacheThreshold <= 0 {
		cfg.TeaCacheThreshold = 0.15
	}

	// Enable fused QKV if requested (only fuse once)
	if cfg.FusedQKV && !m.qkvFused {
		m.Transformer.FuseAllQKV()
		m.qkvFused = true
		fmt.Println("  Fused QKV enabled")
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
		posEmb, _ = m.TextEncoder.EncodePrompt(m.Tokenizer, cfg.Prompt, 512, false)
		if useCFG {
			negEmb, _ = m.TextEncoder.EncodePrompt(m.Tokenizer, cfg.NegativePrompt, 512, false)
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

	// Pre-compute batched embeddings for CFG (outside the loop for efficiency)
	var batchedEmb *mlx.Array
	if useCFG {
		// Concatenate embeddings once: [1, L, D] + [1, L, D] -> [2, L, D]
		batchedEmb = mlx.Concatenate([]*mlx.Array{posEmb, negEmb}, 0)
		mlx.Keep(batchedEmb)
		mlx.Eval(batchedEmb)
	}

	// TeaCache for timestep-aware caching
	// For CFG mode, we cache pos/neg separately, skip early steps, and always compute CFG fresh
	var teaCache *cache.TeaCache
	if cfg.TeaCache {
		skipEarly := 0
		if useCFG {
			skipEarly = 3 // Skip first 3 steps for CFG to preserve structure
		}
		teaCache = cache.NewTeaCache(&cache.TeaCacheConfig{
			Threshold:      cfg.TeaCacheThreshold,
			RescaleFactor:  1.0,
			SkipEarlySteps: skipEarly,
		})
		if useCFG {
			fmt.Printf("  TeaCache enabled (CFG mode): threshold=%.2f, skip first %d steps\n", cfg.TeaCacheThreshold, skipEarly)
		} else {
			fmt.Printf("  TeaCache enabled: threshold=%.2f\n", cfg.TeaCacheThreshold)
		}
	}

	// cleanup frees all kept arrays when we need to abort early
	cleanup := func() {
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
		if batchedEmb != nil {
			batchedEmb.Free()
		}
		if teaCache != nil {
			teaCache.Free()
		}
		latents.Free()
	}

	// Denoising loop
	if cfg.Progress != nil {
		cfg.Progress(0, cfg.Steps) // Start at 0%
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

		tCurr := scheduler.Timesteps[i]
		var noisePred *mlx.Array

		// TeaCache: check if we should compute or reuse cached output
		shouldCompute := teaCache == nil || teaCache.ShouldCompute(i, tCurr)

		if shouldCompute {
			timestep := mlx.ToBFloat16(mlx.NewArray([]float32{1.0 - tCurr}, []int32{1}))
			patches := PatchifyLatents(latents, tcfg.PatchSize)

			var output *mlx.Array
			if useCFG {
				// CFG Batching: single forward pass with batch=2
				// Tile patches: [1, L, D] -> [2, L, D]
				batchedPatches := mlx.Tile(patches, []int32{2, 1, 1})
				// Tile timestep: [1] -> [2]
				batchedTimestep := mlx.Tile(timestep, []int32{2})

				// Single batched forward pass (RoPE broadcasts from [1,L,H,D] to [2,L,H,D])
				batchedOutput := m.Transformer.Forward(batchedPatches, batchedTimestep, batchedEmb, ropeCache)

				// Split output: [2, L, D] -> pos [1, L, D], neg [1, L, D]
				outputShape := batchedOutput.Shape()
				L := outputShape[1]
				D := outputShape[2]
				posOutput := mlx.Slice(batchedOutput, []int32{0, 0, 0}, []int32{1, L, D})
				negOutput := mlx.Slice(batchedOutput, []int32{1, 0, 0}, []int32{2, L, D})

				// Convert to noise predictions (unpatchify and negate)
				posPred := UnpatchifyLatents(posOutput, tcfg.PatchSize, latentH, latentW, tcfg.InChannels)
				posPred = mlx.Neg(posPred)
				negPred := UnpatchifyLatents(negOutput, tcfg.PatchSize, latentH, latentW, tcfg.InChannels)
				negPred = mlx.Neg(negPred)

				// Cache pos/neg separately for TeaCache
				if teaCache != nil {
					teaCache.UpdateCFGCache(posPred, negPred, tCurr)
					mlx.Keep(teaCache.Arrays()...)
				}

				// Apply CFG: noisePred = neg + scale * (pos - neg)
				diff := mlx.Sub(posPred, negPred)
				scaledDiff := mlx.MulScalar(diff, cfg.CFGScale)
				noisePred = mlx.Add(negPred, scaledDiff)
			} else {
				// Non-CFG forward pass
				output = m.Transformer.Forward(patches, timestep, posEmb, ropeCache)
				noisePred = UnpatchifyLatents(output, tcfg.PatchSize, latentH, latentW, tcfg.InChannels)
				noisePred = mlx.Neg(noisePred)

				// Update TeaCache
				if teaCache != nil {
					teaCache.UpdateCache(noisePred, tCurr)
					mlx.Keep(teaCache.Arrays()...)
				}
			}
		} else if useCFG && teaCache != nil && teaCache.HasCFGCache() {
			// CFG mode: get cached pos/neg and compute CFG fresh
			posPred, negPred := teaCache.GetCFGCached()
			diff := mlx.Sub(posPred, negPred)
			scaledDiff := mlx.MulScalar(diff, cfg.CFGScale)
			noisePred = mlx.Add(negPred, scaledDiff)
			fmt.Printf("    [TeaCache: reusing cached pos/neg outputs]\n")
		} else {
			// Non-CFG mode: reuse cached noise prediction
			noisePred = teaCache.GetCached()
			fmt.Printf("    [TeaCache: reusing cached output]\n")
		}

		oldLatents := latents
		latents = scheduler.Step(noisePred, latents, i)

		mlx.Eval(latents)
		oldLatents.Free()

		if cfg.CapturePath != "" && i == 1 {
			mlx.MetalStopCapture()
		}

		activeMem := float64(mlx.MetalGetActiveMemory()) / (1024 * 1024 * 1024)
		peakMem := float64(mlx.MetalGetPeakMemory()) / (1024 * 1024 * 1024)
		fmt.Printf("  Step %d/%d: t=%.4f (%.2fs) [%.1f GB active, %.1f GB peak]\n",
			i+1, cfg.Steps, tCurr, time.Since(stepStart).Seconds(), activeMem, peakMem)

		if cfg.Progress != nil {
			cfg.Progress(i+1, cfg.Steps) // Report completed step
		}
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
	if batchedEmb != nil {
		batchedEmb.Free()
	}
	if teaCache != nil {
		hits, misses := teaCache.Stats()
		fmt.Printf("  TeaCache stats: %d hits, %d misses (%.1f%% cache rate)\n",
			hits, misses, float64(hits)/float64(hits+misses)*100)
		teaCache.Free()
	}

	// VAE decode - enable tiling for larger images to reduce memory
	// VAE attention is O(n²) on latent pixels, tiling helps significantly
	if latentH > 64 || latentW > 64 {
		m.VAEDecoder.Tiling = vae.DefaultTilingConfig()
	}
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

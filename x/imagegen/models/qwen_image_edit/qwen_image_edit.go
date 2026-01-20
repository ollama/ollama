//go:build mlx

// Package qwen_image_edit implements the Qwen-Image-Edit diffusion model for image editing.
// It reuses components from qwen_image where possible.
package qwen_image_edit

import (
	"context"
	"fmt"
	"path/filepath"
	"time"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/models/qwen_image"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
)

// GenerateConfig holds all options for image editing.
type GenerateConfig struct {
	Prompt         string
	NegativePrompt string                // Unconditional prompt for CFG (empty string "" is valid)
	CFGScale       float32               // CFG enabled when > 1.0 (default: 4.0)
	Width          int32                 // Output width (default: from input image)
	Height         int32                 // Output height (default: from input image)
	Steps          int                   // Denoising steps (default: 50)
	Seed           int64                 // Random seed
	Progress       func(step, totalSteps int) // Optional progress callback
}

// Model represents a Qwen-Image-Edit diffusion model.
type Model struct {
	ModelPath     string
	Tokenizer     *tokenizer.Tokenizer
	Processor     *Processor                // Image processor for vision encoder
	TextEncoder   *qwen_image.Qwen25VL      // Qwen2.5-VL vision-language encoder (from qwen_image)
	Transformer   *qwen_image.Transformer   // Reuse qwen_image transformer
	VAE           *VAE                      // Combined encoder + decoder
}

// Load loads the Qwen-Image-Edit model from a directory.
func (m *Model) Load(modelPath string) error {
	fmt.Println("Loading Qwen-Image-Edit model...")
	start := time.Now()

	if mlx.GPUIsAvailable() {
		mlx.SetDefaultDeviceGPU()
		mlx.EnableCompile()
	}

	m.ModelPath = modelPath

	// Load tokenizer from processor directory
	fmt.Print("  Loading tokenizer... ")
	processorPath := filepath.Join(modelPath, "processor")
	tok, err := tokenizer.Load(processorPath)
	if err != nil {
		// Fallback to tokenizer directory
		tokenizerPath := filepath.Join(modelPath, "tokenizer")
		tok, err = tokenizer.Load(tokenizerPath)
		if err != nil {
			return fmt.Errorf("tokenizer: %w", err)
		}
	}
	m.Tokenizer = tok
	fmt.Println("✓")

	// Load processor (image preprocessing config)
	fmt.Print("  Loading processor... ")
	m.Processor = &Processor{}
	if err := m.Processor.Load(processorPath); err != nil {
		return fmt.Errorf("processor: %w", err)
	}
	fmt.Println("✓")

	// Load vision-language text encoder (Qwen2.5-VL from qwen_image package)
	m.TextEncoder = &qwen_image.Qwen25VL{}
	if err := m.TextEncoder.Load(filepath.Join(modelPath, "text_encoder")); err != nil {
		return fmt.Errorf("text encoder: %w", err)
	}
	mlx.Eval(mlx.Collect(m.TextEncoder)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	// Load transformer (reuse qwen_image)
	m.Transformer = &qwen_image.Transformer{}
	if err := m.Transformer.Load(filepath.Join(modelPath, "transformer")); err != nil {
		return fmt.Errorf("transformer: %w", err)
	}
	mlx.Eval(mlx.Collect(m.Transformer)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	// Load VAE (encoder + decoder)
	m.VAE = &VAE{}
	if err := m.VAE.Load(filepath.Join(modelPath, "vae")); err != nil {
		return fmt.Errorf("VAE: %w", err)
	}
	mlx.Eval(mlx.Collect(m.VAE)...)
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

// Edit edits an image based on a text prompt.
// inputImagePath: path to input image
// prompt: text description of desired edit
func (m *Model) Edit(inputImagePath string, prompt string, width, height int32, steps int, seed int64) (*mlx.Array, error) {
	return m.EditFromConfig([]string{inputImagePath}, &GenerateConfig{
		Prompt: prompt,
		Width:  width,
		Height: height,
		Steps:  steps,
		Seed:   seed,
	})
}

// EditFromConfig edits images using the unified config struct.
// Accepts one or more input images.
func (m *Model) EditFromConfig(inputImagePaths []string, cfg *GenerateConfig) (*mlx.Array, error) {
	if len(inputImagePaths) == 0 {
		return nil, fmt.Errorf("no input images provided")
	}

	start := time.Now()
	result, err := m.edit(inputImagePaths, cfg)
	if err != nil {
		return nil, err
	}

	if cfg.NegativePrompt != "" {
		fmt.Printf("Edited %d image(s) with CFG (scale=%.1f) in %.2fs (%d steps)\n",
			len(inputImagePaths), cfg.CFGScale, time.Since(start).Seconds(), cfg.Steps)
	} else {
		fmt.Printf("Edited %d image(s) in %.2fs (%d steps)\n",
			len(inputImagePaths), time.Since(start).Seconds(), cfg.Steps)
	}
	return result, nil
}

// EditImage implements model.ImageEditModel interface.
func (m *Model) EditImage(ctx context.Context, inputImagePath, prompt string, width, height int32, steps int, seed int64) (*mlx.Array, error) {
	return m.Edit(inputImagePath, prompt, width, height, steps, seed)
}

// EditMultiImage edits using multiple source images.
// This matches diffusers' QwenImageEditPlusPipeline behavior.
func (m *Model) EditMultiImage(inputImagePaths []string, cfg *GenerateConfig) (*mlx.Array, error) {
	return m.EditFromConfig(inputImagePaths, cfg)
}

// edit is the internal editing pipeline that handles one or more images.
func (m *Model) edit(inputImagePaths []string, cfg *GenerateConfig) (*mlx.Array, error) {
	// Apply defaults
	if cfg.Steps <= 0 {
		cfg.Steps = 50
	}
	if cfg.CFGScale <= 0 {
		cfg.CFGScale = 4.0
	}

	// Load and preprocess all input images
	fmt.Printf("Loading %d image(s)...\n", len(inputImagePaths))
	condImages, vaeImages, inputDims, err := m.Processor.LoadAndPreprocessMultiple(inputImagePaths)
	if err != nil {
		return nil, fmt.Errorf("preprocess images: %w", err)
	}
	for _, img := range condImages {
		mlx.Keep(img)
	}
	for _, img := range vaeImages {
		mlx.Keep(img)
	}
	mlx.Eval(append(condImages, vaeImages...)...)

	useCFG := cfg.NegativePrompt != ""
	tcfg := m.Transformer.Config
	vaeScaleFactor := int32(8)

	// Output dimensions - if not specified, use first input image dimensions
	if cfg.Width <= 0 {
		cfg.Width = inputDims[0].VaeW
	}
	if cfg.Height <= 0 {
		cfg.Height = inputDims[0].VaeH
	}

	// Output (noise) latent dimensions
	outLatentH := cfg.Height / vaeScaleFactor
	outLatentW := cfg.Width / vaeScaleFactor
	outPH := outLatentH / tcfg.PatchSize
	outPW := outLatentW / tcfg.PatchSize
	noiseSeqLen := outPH * outPW
	imgSeqLen := noiseSeqLen

	// Encode prompt with all images for conditioning
	posEmb, _, _, err := m.TextEncoder.EncodePromptWithImages(m.Tokenizer, cfg.Prompt, condImages)
	if err != nil {
		return nil, fmt.Errorf("encoding prompt: %w", err)
	}
	mlx.Keep(posEmb)
	mlx.Eval(posEmb)

	var negEmb *mlx.Array
	if useCFG {
		negEmb, _, _, err = m.TextEncoder.EncodePromptWithImages(m.Tokenizer, cfg.NegativePrompt, condImages)
		if err != nil {
			return nil, fmt.Errorf("encoding negative prompt: %w", err)
		}
		mlx.Keep(negEmb)
		mlx.Eval(negEmb)
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
		mlx.Eval(posEmb, negEmb)
	}

	// Pre-compute batched embeddings for CFG (single forward pass optimization)
	var batchedEmb *mlx.Array
	if useCFG {
		batchedEmb = mlx.Concatenate([]*mlx.Array{posEmb, negEmb}, 0)
		mlx.Keep(batchedEmb)
		mlx.Eval(batchedEmb)
	}

	// Encode all input images to latents and concatenate
	fmt.Println("Encoding images to latents...")
	allImageLatentsPacked := make([]*mlx.Array, len(vaeImages))
	for i, vaeImage := range vaeImages {
		imageLatents := m.VAE.Encode(vaeImage)
		imageLatents = m.VAE.Normalize(imageLatents)
		imageLatents2D := mlx.Squeeze(imageLatents, 2)
		packed := qwen_image.PackLatents(imageLatents2D, tcfg.PatchSize)
		mlx.Keep(packed)
		mlx.Eval(packed)
		allImageLatentsPacked[i] = packed
	}

	imageLatentsPacked := mlx.Concatenate(allImageLatentsPacked, 1)
	mlx.Keep(imageLatentsPacked)
	mlx.Eval(imageLatentsPacked)

	// Scheduler
	scheduler := qwen_image.NewFlowMatchScheduler(qwen_image.DefaultSchedulerConfig())
	scheduler.SetTimesteps(cfg.Steps, noiseSeqLen)

	// Init noise latents in packed format
	packedChannels := tcfg.OutChannels * tcfg.PatchSize * tcfg.PatchSize
	packedNoise := scheduler.InitNoisePacked(1, noiseSeqLen, packedChannels, cfg.Seed)
	latents := qwen_image.UnpackLatents(packedNoise, outLatentH, outLatentW, tcfg.PatchSize)
	mlx.Eval(latents)

	// RoPE cache
	ropeCache := PrepareRoPEMultiImage(outPH, outPW, inputDims, txtLen, tcfg.AxesDimsRope)
	mlx.Keep(ropeCache.ImgFreqs, ropeCache.TxtFreqs)
	mlx.Eval(ropeCache.ImgFreqs, ropeCache.TxtFreqs)

	// Denoising loop
	fmt.Printf("Running denoising (%d steps)...\n", cfg.Steps)
	for i := 0; i < cfg.Steps; i++ {
		stepStart := time.Now()
		if cfg.Progress != nil {
			cfg.Progress(i+1, cfg.Steps)
		}

		t := scheduler.Timesteps[i]
		timestep := mlx.ToBFloat16(mlx.NewArray([]float32{t}, []int32{1}))
		mlx.Eval(timestep)

		latents2D := mlx.Squeeze(latents, 2)
		patches := qwen_image.PackLatents(latents2D, tcfg.PatchSize)
		latentInput := mlx.Concatenate([]*mlx.Array{patches, imageLatentsPacked}, 1)

		var output *mlx.Array
		if useCFG {
			// CFG Batching: single forward pass with batch=2
			// Tile inputs: [1, L, D] -> [2, L, D]
			batchedLatentInput := mlx.Tile(latentInput, []int32{2, 1, 1})
			batchedTimestep := mlx.Tile(timestep, []int32{2})

			// Single batched forward pass
			batchedOutput := m.Transformer.Forward(batchedLatentInput, batchedEmb, batchedTimestep, ropeCache.ImgFreqs, ropeCache.TxtFreqs)

			// Split output: [2, L, D] -> pos [1, L, D], neg [1, L, D]
			D := batchedOutput.Shape()[2]
			posOutput := mlx.Slice(batchedOutput, []int32{0, 0, 0}, []int32{1, imgSeqLen, D})
			negOutput := mlx.Slice(batchedOutput, []int32{1, 0, 0}, []int32{2, imgSeqLen, D})

			output = applyCFGWithNormRescale(posOutput, negOutput, cfg.CFGScale)
		} else {
			output = m.Transformer.Forward(latentInput, posEmb, timestep, ropeCache.ImgFreqs, ropeCache.TxtFreqs)
			output = mlx.Slice(output, []int32{0, 0, 0}, []int32{1, imgSeqLen, output.Shape()[2]})
		}

		noisePred := qwen_image.UnpackLatents(output, outLatentH, outLatentW, tcfg.PatchSize)
		oldLatents := latents
		latents = scheduler.Step(noisePred, latents, i)
		mlx.Eval(latents)
		oldLatents.Free()

		fmt.Printf("  Step %d/%d: t=%.4f (%.2fs)\n", i+1, cfg.Steps, t, time.Since(stepStart).Seconds())
	}

	// Free denoising temporaries
	posEmb.Free()
	if negEmb != nil {
		negEmb.Free()
	}
	if batchedEmb != nil {
		batchedEmb.Free()
	}
	ropeCache.ImgFreqs.Free()
	ropeCache.TxtFreqs.Free()
	imageLatentsPacked.Free()

	// Decode latents
	decoded := m.decodeAndPostprocess(latents)
	latents.Free()

	fmt.Printf("  Peak memory: %.2f GB\n", float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))
	return decoded, nil
}

// applyCFGWithNormRescale applies classifier-free guidance with norm rescaling.
// This prevents CFG from inflating magnitude too much.
func applyCFGWithNormRescale(posOutput, negOutput *mlx.Array, scale float32) *mlx.Array {
	// Upcast to float32 for precision
	posF32 := mlx.AsType(posOutput, mlx.DtypeFloat32)
	negF32 := mlx.AsType(negOutput, mlx.DtypeFloat32)

	// CFG: pred = neg + scale * (pos - neg)
	diff := mlx.Sub(posF32, negF32)
	scaledDiff := mlx.MulScalar(diff, scale)
	combPred := mlx.Add(negF32, scaledDiff)

	// Norm rescaling: rescale combined prediction to match conditional norm
	condNorm := mlx.Sqrt(mlx.Sum(mlx.Square(posF32), -1, true))
	combNorm := mlx.Sqrt(mlx.Sum(mlx.Square(combPred), -1, true))
	output := mlx.Mul(combPred, mlx.Div(condNorm, combNorm))

	mlx.Eval(output)
	return mlx.ToBFloat16(output)
}

// decodeAndPostprocess denormalizes latents, decodes through VAE, and scales to [0,1].
func (m *Model) decodeAndPostprocess(latents *mlx.Array) *mlx.Array {
	latents = m.VAE.Denormalize(latents)
	decoded := m.VAE.Decode(latents)

	// Post-process: squeeze temporal dim and rescale to [0, 1]
	decoded = mlx.Squeeze(decoded, 2)
	decoded = mlx.AddScalar(decoded, 1.0)
	decoded = mlx.DivScalar(decoded, 2.0)
	decoded = mlx.ClipScalar(decoded, 0.0, 1.0, true, true)
	mlx.Eval(decoded)
	return decoded
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
func LoadPersistent(modelPath string) (*Model, error) {
	m := &Model{}
	if err := m.Load(modelPath); err != nil {
		return nil, err
	}
	return m, nil
}

// PrepareRoPEMultiImage computes RoPE with interpolation for image editing.
// Handles single or multiple input images with different resolutions.
//
// Parameters:
//   - outPH, outPW: output patch dimensions (noise latent resolution)
//   - inputDims: patch dimensions for each input image [(pH1, pW1), (pH2, pW2), ...]
//   - txtLen: text sequence length
//   - axesDims: RoPE axis dimensions [16, 56, 56]
//
// Returns RoPE cache where:
//   - ImgFreqs has (outPH*outPW + sum(inPH*inPW for each image)) positions
//   - First outPH*outPW positions are for noise latents (standard RoPE at output res)
//   - Following positions are for each input image (interpolated from output res)
func PrepareRoPEMultiImage(outPH, outPW int32, inputDims []ImageDims, txtLen int32, axesDims []int32) *qwen_image.RoPECache {
	theta := float64(10000)
	maxIdx := int32(4096)

	// Compute base frequencies for each axis dimension
	freqsT := qwen_image.ComputeAxisFreqs(axesDims[0], theta)
	freqsH := qwen_image.ComputeAxisFreqs(axesDims[1], theta)
	freqsW := qwen_image.ComputeAxisFreqs(axesDims[2], theta)

	// Build frequency lookup tables
	posFreqsT := qwen_image.MakeFreqTable(maxIdx, freqsT, false)
	posFreqsH := qwen_image.MakeFreqTable(maxIdx, freqsH, false)
	posFreqsW := qwen_image.MakeFreqTable(maxIdx, freqsW, false)
	negFreqsT := qwen_image.MakeFreqTable(maxIdx, freqsT, true) // For frame -1 on last condition image
	negFreqsH := qwen_image.MakeFreqTable(maxIdx, freqsH, true)
	negFreqsW := qwen_image.MakeFreqTable(maxIdx, freqsW, true)

	headDim := int32(len(freqsT)+len(freqsH)+len(freqsW)) * 2

	// Helper to compute RoPE for a single position at output resolution with scale_rope
	computePosFreqs := func(framePos, y, x int32) []float32 {
		row := make([]float32, headDim)
		idx := 0

		// Frame position
		for i := 0; i < len(freqsT)*2; i++ {
			row[idx+i] = posFreqsT[framePos][i]
		}
		idx += len(freqsT) * 2

		// Height with scale_rope centering (using OUTPUT dimensions)
		outHHalf := outPH / 2
		hNegCount := outPH - outHHalf
		if y < hNegCount {
			negTableIdx := maxIdx - hNegCount + y
			for i := 0; i < len(freqsH)*2; i++ {
				row[idx+i] = negFreqsH[negTableIdx][i]
			}
		} else {
			posIdx := y - hNegCount
			for i := 0; i < len(freqsH)*2; i++ {
				row[idx+i] = posFreqsH[posIdx][i]
			}
		}
		idx += len(freqsH) * 2

		// Width with scale_rope centering (using OUTPUT dimensions)
		outWHalf := outPW / 2
		wNegCount := outPW - outWHalf
		if x < wNegCount {
			negTableIdx := maxIdx - wNegCount + x
			for i := 0; i < len(freqsW)*2; i++ {
				row[idx+i] = negFreqsW[negTableIdx][i]
			}
		} else {
			posIdx := x - wNegCount
			for i := 0; i < len(freqsW)*2; i++ {
				row[idx+i] = posFreqsW[posIdx][i]
			}
		}

		return row
	}

	// Helper to compute RoPE for frame -1 (used for last condition image)
	// This matches Python's _compute_condition_freqs which uses freqs_neg[0][-1:]
	computeNegFrameFreqs := func(y, x int32) []float32 {
		row := make([]float32, headDim)
		idx := 0

		// Frame -1: use last row of negative frame frequencies
		negFrameIdx := maxIdx - 1
		for i := 0; i < len(freqsT)*2; i++ {
			row[idx+i] = negFreqsT[negFrameIdx][i]
		}
		idx += len(freqsT) * 2

		// Height with scale_rope centering (using OUTPUT dimensions)
		outHHalf := outPH / 2
		hNegCount := outPH - outHHalf
		if y < hNegCount {
			negTableIdx := maxIdx - hNegCount + y
			for i := 0; i < len(freqsH)*2; i++ {
				row[idx+i] = negFreqsH[negTableIdx][i]
			}
		} else {
			posIdx := y - hNegCount
			for i := 0; i < len(freqsH)*2; i++ {
				row[idx+i] = posFreqsH[posIdx][i]
			}
		}
		idx += len(freqsH) * 2

		// Width with scale_rope centering (using OUTPUT dimensions)
		outWHalf := outPW / 2
		wNegCount := outPW - outWHalf
		if x < wNegCount {
			negTableIdx := maxIdx - wNegCount + x
			for i := 0; i < len(freqsW)*2; i++ {
				row[idx+i] = negFreqsW[negTableIdx][i]
			}
		} else {
			posIdx := x - wNegCount
			for i := 0; i < len(freqsW)*2; i++ {
				row[idx+i] = posFreqsW[posIdx][i]
			}
		}

		return row
	}

	// Total image sequence length: noise + all input images
	noiseSeqLen := outPH * outPW
	totalImgLen := noiseSeqLen
	for _, dims := range inputDims {
		totalImgLen += dims.PatchH * dims.PatchW
	}

	imgFreqsData := make([]float32, totalImgLen*headDim)
	idx := int32(0)

	// Segment 0: Noise latents - standard RoPE at output resolution (frame 0)
	for y := int32(0); y < outPH; y++ {
		for x := int32(0); x < outPW; x++ {
			row := computePosFreqs(0, y, x)
			copy(imgFreqsData[idx:], row)
			idx += headDim
		}
	}

	// Segments 1..N: Edit image latents - INTERPOLATED RoPE
	// For single image: use frame 1 (matches original PrepareRoPEInterpolated)
	// For multiple images: Python uses frame -1 for the LAST condition image
	// (_compute_condition_freqs), positive indices for others.
	numImages := len(inputDims)
	lastImgIdx := numImages - 1
	for imgIdx, dims := range inputDims {
		inPH := dims.PatchH
		inPW := dims.PatchW

		// Determine frame index for this image
		// Single image case: use frame 1 (like original PrepareRoPEInterpolated)
		// Multi-image case: last image uses frame -1, others use frame 1, 2, etc.
		useNegFrame := numImages > 1 && imgIdx == lastImgIdx

		// Map each input position to an output position using linear interpolation
		for y := int32(0); y < inPH; y++ {
			for x := int32(0); x < inPW; x++ {
				// Interpolate: map input (y, x) to output grid position
				// This is the key fix from DiffSynth's forward_sampling
				var yOut, xOut int32
				if inPH == 1 {
					yOut = 0
				} else {
					// Linear interpolation: y_out = y * (outPH - 1) / (inPH - 1)
					yOut = y * (outPH - 1) / (inPH - 1)
				}
				if inPW == 1 {
					xOut = 0
				} else {
					xOut = x * (outPW - 1) / (inPW - 1)
				}

				var row []float32
				if useNegFrame {
					// Last image in multi-image uses frame -1
					row = computeNegFrameFreqs(yOut, xOut)
				} else {
					// Single image uses frame 1, multi-image uses frame 1, 2, etc.
					frameIdx := int32(imgIdx + 1)
					row = computePosFreqs(frameIdx, yOut, xOut)
				}
				copy(imgFreqsData[idx:], row)
				idx += headDim
			}
		}
	}

	imgFreqs := mlx.NewArray(imgFreqsData, []int32{totalImgLen, headDim})
	imgFreqs = mlx.ToBFloat16(imgFreqs)

	// Text frequencies - start after max video index
	maxVidIdx := max(outPH/2, outPW/2)

	txtFreqsData := make([]float32, txtLen*headDim)
	idx = 0
	for t := int32(0); t < txtLen; t++ {
		pos := maxVidIdx + t
		for i := 0; i < len(freqsT)*2; i++ {
			txtFreqsData[idx+int32(i)] = posFreqsT[pos][i]
		}
		idx += int32(len(freqsT) * 2)
		for i := 0; i < len(freqsH)*2; i++ {
			txtFreqsData[idx+int32(i)] = posFreqsH[pos][i]
		}
		idx += int32(len(freqsH) * 2)
		for i := 0; i < len(freqsW)*2; i++ {
			txtFreqsData[idx+int32(i)] = posFreqsW[pos][i]
		}
		idx += int32(len(freqsW) * 2)
	}

	txtFreqs := mlx.NewArray(txtFreqsData, []int32{txtLen, headDim})
	txtFreqs = mlx.ToBFloat16(txtFreqs)

	return &qwen_image.RoPECache{
		ImgFreqs: imgFreqs,
		TxtFreqs: txtFreqs,
	}
}

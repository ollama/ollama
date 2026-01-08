//go:build mlx

package qwen_image

import (
	"errors"
	"fmt"
	"math"
	"path/filepath"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/safetensors"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
)

// Qwen25VLConfig holds Qwen2.5-VL configuration
type Qwen25VLConfig struct {
	// Text model config
	HiddenSize        int32   `json:"hidden_size"`         // 3584
	NumHiddenLayers   int32   `json:"num_hidden_layers"`   // 28
	IntermediateSize  int32   `json:"intermediate_size"`   // 18944
	NumAttentionHeads int32   `json:"num_attention_heads"` // 28
	NumKeyValueHeads  int32   `json:"num_key_value_heads"` // 4
	VocabSize         int32   `json:"vocab_size"`          // 152064
	RMSNormEps        float32 `json:"rms_norm_eps"`        // 1e-6
	RopeTheta         float32 `json:"rope_theta"`          // 1000000
	HeadDim           int32   // Calculated: HiddenSize / NumAttentionHeads
	MRoPESection      []int32 // [16, 24, 24] for temporal, height, width

	// Vision config
	VisionHiddenSize    int32   `json:"vision_hidden_size"`    // 1280
	VisionNumLayers     int32   `json:"vision_num_layers"`     // 32
	VisionNumHeads      int32   `json:"vision_num_heads"`      // 16
	VisionIntermSize    int32   `json:"vision_intermediate"`   // 3420
	VisionPatchSize     int32   `json:"vision_patch_size"`     // 14
	VisionOutHiddenSize int32   `json:"vision_out_hidden"`     // 3584
	VisionSpatialMerge  int32   `json:"vision_spatial_merge"`  // 2
	VisionWindowSize    int32   `json:"vision_window_size"`    // 112
	VisionFullAttIdx    []int32 // [7, 15, 23, 31]

	// Special tokens
	ImageTokenID       int32 // 151655
	VisionStartTokenID int32 // 151652
	VisionEndTokenID   int32 // 151653
}

// defaultQwen25VLConfig returns default config
func defaultQwen25VLConfig() *Qwen25VLConfig {
	cfg := &Qwen25VLConfig{
		// Text
		HiddenSize:        3584,
		NumHiddenLayers:   28,
		IntermediateSize:  18944,
		NumAttentionHeads: 28,
		NumKeyValueHeads:  4,
		VocabSize:         152064,
		RMSNormEps:        1e-6,
		RopeTheta:         1000000,
		MRoPESection:      []int32{16, 24, 24},

		// Vision
		VisionHiddenSize:    1280,
		VisionNumLayers:     32,
		VisionNumHeads:      16,
		VisionIntermSize:    3420,
		VisionPatchSize:     14,
		VisionOutHiddenSize: 3584,
		VisionSpatialMerge:  2,
		VisionWindowSize:    112,
		VisionFullAttIdx:    []int32{7, 15, 23, 31},

		// Special tokens
		ImageTokenID:       151655,
		VisionStartTokenID: 151652,
		VisionEndTokenID:   153653,
	}
	cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	return cfg
}

// Qwen25VL is the Qwen2.5-VL vision-language encoder
type Qwen25VL struct {
	Config *Qwen25VLConfig

	// Text model
	Embedding *mlx.Array
	Blocks    []*VLTextBlock
	FinalNorm *mlx.Array

	// Vision tower (optional - nil for text-only models)
	VisionPatchEmbed *VisionPatchEmbed
	VisionBlocks     []*VisionBlock
	VisionMerger     *VisionMerger
	HasVision        bool // True if vision tower is loaded
}

// LoadTextOnly loads only the text encoder components (skips vision tower)
// Use this for text-to-image generation where vision components are not needed
func (m *Qwen25VL) LoadTextOnly(path string) error {
	return m.load(path, false)
}

// Load loads the vision-language encoder from a directory
// Vision components are loaded if weights exist
func (m *Qwen25VL) Load(path string) error {
	return m.load(path, true)
}

// load is the internal loading function
func (m *Qwen25VL) load(path string, loadVision bool) error {
	fmt.Println("Loading Qwen2.5-VL encoder...")

	cfg := defaultQwen25VLConfig()
	m.Config = cfg

	weights, err := safetensors.LoadModelWeights(path)
	if err != nil {
		return fmt.Errorf("weights: %w", err)
	}

	// Bulk load all weights as bf16
	fmt.Print("  Loading weights as bf16... ")
	if err := weights.Load(mlx.DtypeBFloat16); err != nil {
		return fmt.Errorf("failed to load weights: %w", err)
	}
	fmt.Printf("✓ (%.1f GB)\n", float64(mlx.MetalGetActiveMemory())/(1024*1024*1024))

	// Load text embedding
	fmt.Print("  Loading text embeddings... ")
	embedding, err := weights.Get("model.embed_tokens.weight")
	if err != nil {
		return err
	}
	m.Embedding = embedding
	fmt.Printf("✓ [%v]\n", embedding.Shape())

	// Load text blocks
	m.Blocks = make([]*VLTextBlock, cfg.NumHiddenLayers)
	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		fmt.Printf("\r  Loading text blocks... %d/%d", i+1, cfg.NumHiddenLayers)
		block, err := newVLTextBlock(weights, int(i), cfg)
		if err != nil {
			return fmt.Errorf("failed to load text block %d: %w", i, err)
		}
		m.Blocks[i] = block
	}
	fmt.Printf("\r  Loading text blocks... ✓ [%d blocks]          \n", cfg.NumHiddenLayers)

	// Load final norm
	fmt.Print("  Loading final norm... ")
	finalNorm, err := weights.Get("model.norm.weight")
	if err != nil {
		return err
	}
	m.FinalNorm = finalNorm
	fmt.Println("✓")

	// Try to load vision tower (optional)
	m.HasVision = false
	if loadVision {
		if _, err := weights.Get("visual.patch_embed.proj.weight"); err == nil {
			fmt.Print("  Loading vision patch embed... ")
			m.VisionPatchEmbed, err = newVisionPatchEmbed(weights, cfg)
			if err != nil {
				return fmt.Errorf("vision patch embed: %w", err)
			}
			fmt.Println("✓")

			m.VisionBlocks = make([]*VisionBlock, cfg.VisionNumLayers)
			for i := int32(0); i < cfg.VisionNumLayers; i++ {
				fmt.Printf("\r  Loading vision blocks... %d/%d", i+1, cfg.VisionNumLayers)
				block, err := newVisionBlock(weights, int(i), cfg)
				if err != nil {
					return fmt.Errorf("failed to load vision block %d: %w", i, err)
				}
				m.VisionBlocks[i] = block
			}
			fmt.Printf("\r  Loading vision blocks... ✓ [%d blocks]          \n", cfg.VisionNumLayers)

			fmt.Print("  Loading vision merger... ")
			m.VisionMerger, err = newVisionMerger(weights, cfg)
			if err != nil {
				return fmt.Errorf("vision merger: %w", err)
			}
			fmt.Println("✓")

			m.HasVision = true
		} else {
			fmt.Println("  (No vision tower - text-only mode)")
		}
	} else {
		fmt.Println("  (Skipping vision tower)")
	}

	weights.ReleaseAll()
	return nil
}

// EncodePrompt encodes a text prompt for image generation (text-only mode)
// Uses the Qwen-Image template and drops the first 34 tokens (system prefix)
func (m *Qwen25VL) EncodePrompt(tok *tokenizer.Tokenizer, prompt string) *mlx.Array {
	cfg := m.Config

	// Template from Python: prompt_template_encode (for image generation)
	template := "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
	formattedPrompt := fmt.Sprintf(template, prompt)

	// Tokenize
	tokens := tok.Encode(formattedPrompt, false)

	// Create token array
	seqLen := int32(len(tokens))
	tokenArr := mlx.NewArrayInt32(tokens, []int32{1, seqLen})

	// Get text embeddings
	textEmbed := mlx.EmbeddingLookup(m.Embedding, tokenArr)

	// Compute RoPE
	cossin := m.computeTextRoPE(seqLen, 1)

	// Forward through ALL text blocks
	x := textEmbed
	for _, block := range m.Blocks {
		x = block.Forward(x, cossin)
	}

	// Apply final norm
	x = mlx.RMSNorm(x, m.FinalNorm, cfg.RMSNormEps)

	// Drop first 34 tokens (system prefix)
	// prompt_template_encode_start_idx = 34
	dropIdx := int32(34)
	if x.Shape()[1] > dropIdx {
		x = mlx.Slice(x, []int32{0, dropIdx, 0}, []int32{1, x.Shape()[1], cfg.HiddenSize})
	}

	return x
}

// EncodePromptWithImage encodes a text prompt with an image
// Returns: embeddings [B, L, hidden_size], mask [B, L], error
func (m *Qwen25VL) EncodePromptWithImage(tok *tokenizer.Tokenizer, prompt string, image *mlx.Array) (*mlx.Array, *mlx.Array, error) {
	if !m.HasVision {
		return nil, nil, errors.New("EncodePromptWithImage called on text-only model")
	}

	cfg := m.Config

	// Template from Python diffusers pipeline: prompt_template_encode
	// Python's _get_qwen_prompt_embeds adds "Picture 1: " before vision tokens
	template := "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|vision_end|>%s<|im_end|>\n<|im_start|>assistant\n"
	formattedPrompt := fmt.Sprintf(template, prompt)

	// Tokenize
	tokens := tok.Encode(formattedPrompt, false)

	// Process vision if image provided
	var visionEmbeddings *mlx.Array
	var numImageTokens int32
	var visionH, visionW int32 // Grid dims in patches (before spatial merge)
	if image != nil {
		visionEmbeddings = m.encodeVision(image)
		numImageTokens = visionEmbeddings.Shape()[1]
		// Get original grid dimensions from image shape
		imgShape := image.Shape()
		visionH = imgShape[2] / cfg.VisionPatchSize // Height in patches
		visionW = imgShape[3] / cfg.VisionPatchSize // Width in patches
	}

	// Find image token position and expand
	expandedTokens := make([]int32, 0, len(tokens)+int(numImageTokens))
	imageTokenPos := int32(-1)
	textAfterCount := int32(0)
	for i, t := range tokens {
		if t == cfg.ImageTokenID {
			imageTokenPos = int32(len(expandedTokens))
			// Insert placeholder tokens for image
			for j := int32(0); j < numImageTokens; j++ {
				expandedTokens = append(expandedTokens, cfg.ImageTokenID)
			}
			// Count remaining tokens after image
			textAfterCount = int32(len(tokens) - i - 1)
		} else {
			expandedTokens = append(expandedTokens, t)
		}
	}

	// Create token array
	seqLen := int32(len(expandedTokens))
	tokenArr := mlx.NewArrayInt32(expandedTokens, []int32{1, seqLen})

	// Get text embeddings
	textEmbed := mlx.EmbeddingLookup(m.Embedding, tokenArr) // [1, L, hidden]

	// Replace image token embeddings with vision embeddings
	if visionEmbeddings != nil && imageTokenPos >= 0 {
		// Split, replace, concat
		before := mlx.Slice(textEmbed, []int32{0, 0, 0}, []int32{1, imageTokenPos, cfg.HiddenSize})
		after := mlx.Slice(textEmbed, []int32{0, imageTokenPos + numImageTokens, 0}, []int32{1, seqLen, cfg.HiddenSize})
		textEmbed = mlx.Concatenate([]*mlx.Array{before, visionEmbeddings, after}, 1)
	}

	// Compute RoPE - use multimodal RoPE when image is present
	var cossin [2]*mlx.Array
	if image != nil && imageTokenPos >= 0 {
		cossin = m.ComputeMultimodalRoPE(imageTokenPos, visionH, visionW, textAfterCount, cfg.VisionSpatialMerge)
	} else {
		cossin = m.computeTextRoPE(seqLen, 1)
	}

	// Forward through ALL text blocks
	// Python uses hidden_states[-1] (LAST layer output, not second-to-last!)
	x := textEmbed
	for _, block := range m.Blocks {
		x = block.Forward(x, cossin)
	}

	// Apply final norm (Python DOES apply this for the output)
	x = mlx.RMSNorm(x, m.FinalNorm, cfg.RMSNormEps)

	// Drop first N tokens (system prefix)
	// prompt_template_encode_start_idx = 64
	dropIdx := int32(64)
	if x.Shape()[1] > dropIdx {
		x = mlx.Slice(x, []int32{0, dropIdx, 0}, []int32{1, x.Shape()[1], cfg.HiddenSize})
	}

	// Create attention mask (all ones for now)
	mask := mlx.Ones(1, x.Shape()[1])

	return x, mask, nil
}

// EncodeVision encodes an image through the vision tower (exported for testing)
// image: [B, C, H, W] normalized image tensor
// Returns: [B, num_tokens, hidden_size] vision embeddings
func (m *Qwen25VL) EncodeVision(image *mlx.Array) *mlx.Array {
	return m.encodeVision(image)
}

// VisionRegion describes where vision embeddings are inserted in the sequence
type VisionRegion struct {
	StartPos   int32 // Position in sequence where vision tokens start
	NumTokens  int32 // Number of vision tokens
	GridH      int32 // Vision grid height (in patches, after spatial merge)
	GridW      int32 // Vision grid width (in patches, after spatial merge)
}

// EncodePromptWithImages encodes a text prompt with multiple images
// Returns: embeddings [B, L, hidden_size], mask [B, L], regions []VisionRegion, error
func (m *Qwen25VL) EncodePromptWithImages(tok *tokenizer.Tokenizer, prompt string, images []*mlx.Array) (*mlx.Array, *mlx.Array, []VisionRegion, error) {
	if !m.HasVision {
		return nil, nil, nil, errors.New("EncodePromptWithImages called on text-only model")
	}
	if len(images) == 0 {
		return nil, nil, nil, errors.New("EncodePromptWithImages called with no images")
	}

	cfg := m.Config

	// Build image prompt prefix: "Picture 1: <vision>...Picture N: <vision>..."
	imgPromptTemplate := "Picture %d: <|vision_start|><|image_pad|><|vision_end|>"
	imgPrompt := ""
	for i := range images {
		imgPrompt += fmt.Sprintf(imgPromptTemplate, i+1)
	}

	// Template from Python diffusers pipeline: prompt_template_encode
	template := "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n%s%s<|im_end|>\n<|im_start|>assistant\n"
	formattedPrompt := fmt.Sprintf(template, imgPrompt, prompt)

	// Tokenize
	tokens := tok.Encode(formattedPrompt, false)

	// Process each image through vision tower
	visionEmbeddings := make([]*mlx.Array, len(images))
	numImageTokens := make([]int32, len(images))
	visionGridH := make([]int32, len(images))
	visionGridW := make([]int32, len(images))

	for i, image := range images {
		visionEmbeddings[i] = m.encodeVision(image)
		numImageTokens[i] = visionEmbeddings[i].Shape()[1]
		// Get original grid dimensions from image shape
		imgShape := image.Shape()
		visionH := imgShape[2] / cfg.VisionPatchSize // Height in patches
		visionW := imgShape[3] / cfg.VisionPatchSize // Width in patches
		// After spatial merge, grid is halved
		visionGridH[i] = visionH / cfg.VisionSpatialMerge
		visionGridW[i] = visionW / cfg.VisionSpatialMerge
	}

	// Find all image token positions and expand tokens
	expandedTokens := make([]int32, 0, len(tokens)+int(sum(numImageTokens)))
	imagePositions := make([]int32, 0, len(images)) // Start position for each image's tokens
	imageIdx := 0

	for _, t := range tokens {
		if t == cfg.ImageTokenID {
			if imageIdx < len(images) {
				imagePositions = append(imagePositions, int32(len(expandedTokens)))
				// Insert placeholder tokens for this image
				for j := int32(0); j < numImageTokens[imageIdx]; j++ {
					expandedTokens = append(expandedTokens, cfg.ImageTokenID)
				}
				imageIdx++
			}
		} else {
			expandedTokens = append(expandedTokens, t)
		}
	}

	// Create token array
	seqLen := int32(len(expandedTokens))
	tokenArr := mlx.NewArrayInt32(expandedTokens, []int32{1, seqLen})

	// Get text embeddings
	textEmbed := mlx.EmbeddingLookup(m.Embedding, tokenArr) // [1, L, hidden]

	// Replace image token embeddings with vision embeddings
	// Build list of segments to concatenate
	segments := make([]*mlx.Array, 0, len(images)*2+1)
	regions := make([]VisionRegion, len(images))
	lastEnd := int32(0)

	for i, imgPos := range imagePositions {
		// Text segment before this image
		if imgPos > lastEnd {
			segments = append(segments, mlx.Slice(textEmbed, []int32{0, lastEnd, 0}, []int32{1, imgPos, cfg.HiddenSize}))
		}
		// Vision embeddings for this image
		segments = append(segments, visionEmbeddings[i])
		regions[i] = VisionRegion{
			StartPos:  imgPos,
			NumTokens: numImageTokens[i],
			GridH:     visionGridH[i],
			GridW:     visionGridW[i],
		}
		lastEnd = imgPos + numImageTokens[i]
	}
	// Remaining text after last image
	if lastEnd < seqLen {
		segments = append(segments, mlx.Slice(textEmbed, []int32{0, lastEnd, 0}, []int32{1, seqLen, cfg.HiddenSize}))
	}

	// Concatenate all segments
	textEmbed = mlx.Concatenate(segments, 1)

	// Compute RoPE - use multimodal RoPE for multiple images
	cossin, err := m.ComputeMultiImageRoPE(imagePositions, visionGridH, visionGridW, numImageTokens, seqLen)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("computing RoPE: %w", err)
	}

	// Forward through ALL text blocks
	x := textEmbed
	for _, block := range m.Blocks {
		x = block.Forward(x, cossin)
	}

	// Apply final norm
	x = mlx.RMSNorm(x, m.FinalNorm, cfg.RMSNormEps)

	// Drop first N tokens (system prefix)
	// prompt_template_encode_start_idx = 64
	dropIdx := int32(64)
	if x.Shape()[1] > dropIdx {
		x = mlx.Slice(x, []int32{0, dropIdx, 0}, []int32{1, x.Shape()[1], cfg.HiddenSize})
		// Adjust region positions
		for i := range regions {
			regions[i].StartPos -= dropIdx
		}
	}

	// Create attention mask (all ones)
	mask := mlx.Ones(1, x.Shape()[1])

	return x, mask, regions, nil
}

// sum returns the sum of int32 slice
func sum(arr []int32) int32 {
	var s int32
	for _, v := range arr {
		s += v
	}
	return s
}

// EncodeTextOnly encodes text tokens through all text blocks (exported for testing)
// tokens: array of token IDs
// Returns: [B, L, hidden_size] text embeddings after all blocks
func (m *Qwen25VL) EncodeTextOnly(tokens []int32) *mlx.Array {
	seqLen := int32(len(tokens))
	tokenArr := mlx.NewArrayInt32(tokens, []int32{1, seqLen})

	// Get text embeddings
	textEmbed := mlx.EmbeddingLookup(m.Embedding, tokenArr) // [1, L, hidden]

	// Compute RoPE
	cossin := m.computeTextRoPE(seqLen, 1)

	// Forward through ALL text blocks (unlike Encode which stops at second-to-last)
	x := textEmbed
	for _, block := range m.Blocks {
		x = block.Forward(x, cossin)
	}

	// Apply final norm
	x = mlx.RMSNorm(x, m.FinalNorm, m.Config.RMSNormEps)

	return x
}

// encodeVision encodes an image through the vision tower
// image: [B, C, H, W] normalized image tensor
// Returns: [B, num_tokens, hidden_size] vision embeddings
func (m *Qwen25VL) encodeVision(image *mlx.Array) *mlx.Array {
	cfg := m.Config

	// Calculate grid dimensions from image
	imgShape := image.Shape()
	imgH := imgShape[2]
	imgW := imgShape[3]
	pH := imgH / cfg.VisionPatchSize // grid height in patches
	pW := imgW / cfg.VisionPatchSize // grid width in patches

	// Patch embed
	x := m.VisionPatchEmbed.Forward(image)
	mlx.Eval(x)

	// Get window reordering info
	winInfo := m.getWindowInfo(pH, pW)

	// Compute vision RoPE embeddings (already in 2x2-block order)
	posEmb := m.computeVisionRoPE(pH, pW)

	shape := x.Shape()
	B := shape[0]
	L := shape[1] // num patches = pH * pW
	D := shape[2]
	spatialMergeUnit := winInfo.SpatialMergeUnit
	spatialMerge := cfg.VisionSpatialMerge

	// Convert patch embed from row-major to 2x2-block order
	// Row-major: (0,0), (0,1), (0,2), ..., (1,0), (1,1), ...
	// 2x2-block: (0,0), (0,1), (1,0), (1,1), (0,2), (0,3), (1,2), (1,3), ...
	llmGridH := pH / spatialMerge
	llmGridW := pW / spatialMerge
	blockReorderIdx := make([]int32, L)
	idx := int32(0)
	for hBlock := int32(0); hBlock < llmGridH; hBlock++ {
		for wBlock := int32(0); wBlock < llmGridW; wBlock++ {
			for dh := int32(0); dh < spatialMerge; dh++ {
				for dw := int32(0); dw < spatialMerge; dw++ {
					h := hBlock*spatialMerge + dh
					w := wBlock*spatialMerge + dw
					rowMajorIdx := h*pW + w
					blockReorderIdx[idx] = rowMajorIdx
					idx++
				}
			}
		}
	}
	blockIdxArr := mlx.NewArrayInt32(blockReorderIdx, []int32{L})
	x = mlx.Take(x, blockIdxArr, 1) // Reorder patches to 2x2-block order

	// Window reorder hidden states and RoPE before blocks
	// Python: reshape to [L/4, 4, D], reorder dim 0, reshape back
	// Reshape x: [B, L, D] -> [B, L/4, 4, D]
	x = mlx.Reshape(x, B, L/spatialMergeUnit, spatialMergeUnit, D)
	// Reorder using window index
	winIdxArr := mlx.NewArrayInt32(winInfo.WindowIndex, []int32{int32(len(winInfo.WindowIndex))})
	x = mlx.Take(x, winIdxArr, 1) // Take along axis 1
	// Reshape back: [B, L/4, 4, D] -> [B, L, D]
	x = mlx.Reshape(x, B, L, D)

	// Similarly reorder RoPE: [L, headDim] -> [L/4, 4, headDim] -> reorder -> [L, headDim]
	cosShape := posEmb[0].Shape()
	ropeL := cosShape[0]
	ropeD := cosShape[1]
	cos := mlx.Reshape(posEmb[0], ropeL/spatialMergeUnit, spatialMergeUnit, ropeD)
	sin := mlx.Reshape(posEmb[1], ropeL/spatialMergeUnit, spatialMergeUnit, ropeD)
	cos = mlx.Take(cos, winIdxArr, 0)
	sin = mlx.Take(sin, winIdxArr, 0)
	cos = mlx.Reshape(cos, ropeL, ropeD)
	sin = mlx.Reshape(sin, ropeL, ropeD)
	posEmb = [2]*mlx.Array{cos, sin}

	// Materialize to prevent freeing during block evaluations
	mlx.Eval(x, posEmb[0], posEmb[1])

	// Full sequence cu_seqlens for full attention blocks
	cuSeqlensFull := []int32{0, L}

	// Vision blocks - use window attention except at full attention indices
	for i, block := range m.VisionBlocks {
		useFullAttention := false
		for _, idx := range cfg.VisionFullAttIdx {
			if int32(i) == idx {
				useFullAttention = true
				break
			}
		}

		var cuSeqlens []int32
		if useFullAttention {
			cuSeqlens = cuSeqlensFull
		} else {
			cuSeqlens = winInfo.CuWindowSeqlens
		}

		x = block.Forward(x, posEmb, cuSeqlens)
	}

	// Spatial merge (2x2 -> 1)
	x = m.VisionMerger.ForwardWithDims(x, pH, pW)

	// Reverse window reorder after merger
	revIdxArr := mlx.NewArrayInt32(winInfo.ReverseIndex, []int32{int32(len(winInfo.ReverseIndex))})
	x = mlx.Take(x, revIdxArr, 1)

	return x
}

// WindowInfo holds window reordering and attention boundary info
type WindowInfo struct {
	WindowIndex      []int32 // Reordering indices for merged tokens
	ReverseIndex     []int32 // Reverse reordering indices
	CuWindowSeqlens  []int32 // Cumulative window boundaries in UNMERGED sequence
	SpatialMergeUnit int32   // Number of patches per merged token (4 = 2x2)
}

// getWindowInfo computes window reordering indices and attention boundaries
// pH, pW: patch grid dimensions before 2x2 merge
func (m *Qwen25VL) getWindowInfo(pH, pW int32) *WindowInfo {
	cfg := m.Config
	spatialMergeUnit := cfg.VisionSpatialMerge * cfg.VisionSpatialMerge // 4

	// After 2x2 merge
	llmGridH := pH / cfg.VisionSpatialMerge
	llmGridW := pW / cfg.VisionSpatialMerge
	numTokens := llmGridH * llmGridW

	// Window size in merged tokens
	// window_size=112, spatial_merge_size=2, patch_size=14
	// vit_merger_window_size = 112 / 2 / 14 = 4
	vitMergerWindowSize := cfg.VisionWindowSize / cfg.VisionSpatialMerge / cfg.VisionPatchSize

	// Calculate padding and number of windows
	padH := vitMergerWindowSize - llmGridH%vitMergerWindowSize
	if padH == vitMergerWindowSize {
		padH = 0
	}
	padW := vitMergerWindowSize - llmGridW%vitMergerWindowSize
	if padW == vitMergerWindowSize {
		padW = 0
	}

	numWindowsH := (llmGridH + padH) / vitMergerWindowSize
	numWindowsW := (llmGridW + padW) / vitMergerWindowSize

	// Create padded grid with -1 for padding
	paddedH := llmGridH + padH
	paddedW := llmGridW + padW
	grid := make([]int32, paddedH*paddedW)
	for i := range grid {
		grid[i] = -1
	}
	for h := int32(0); h < llmGridH; h++ {
		for w := int32(0); w < llmGridW; w++ {
			grid[h*paddedW+w] = h*llmGridW + w
		}
	}

	// Reorder into windows and track window sizes
	windowIndex := make([]int32, 0, numTokens)
	windowSizes := make([]int32, 0, numWindowsH*numWindowsW)
	ws := vitMergerWindowSize

	for wh := int32(0); wh < numWindowsH; wh++ {
		for ww := int32(0); ww < numWindowsW; ww++ {
			windowStart := len(windowIndex)
			// Extract window
			for h := int32(0); h < ws; h++ {
				for w := int32(0); w < ws; w++ {
					idx := (wh*ws+h)*paddedW + (ww*ws + w)
					if grid[idx] >= 0 {
						windowIndex = append(windowIndex, grid[idx])
					}
				}
			}
			windowSize := int32(len(windowIndex) - windowStart)
			windowSizes = append(windowSizes, windowSize)
		}
	}

	// Create reverse index (argsort of windowIndex)
	reverseIndex := make([]int32, numTokens)
	for i, idx := range windowIndex {
		reverseIndex[idx] = int32(i)
	}

	// Compute cumulative sequence lengths in UNMERGED sequence
	// Each merged token corresponds to spatialMergeUnit patches
	cuWindowSeqlens := make([]int32, len(windowSizes)+1)
	cuWindowSeqlens[0] = 0
	for i, size := range windowSizes {
		cuWindowSeqlens[i+1] = cuWindowSeqlens[i] + size*spatialMergeUnit
	}

	return &WindowInfo{
		WindowIndex:      windowIndex,
		ReverseIndex:     reverseIndex,
		CuWindowSeqlens:  cuWindowSeqlens,
		SpatialMergeUnit: spatialMergeUnit,
	}
}

// ComputeMultiImageRoPE computes M-RoPE for combined text + multiple vision regions + text sequences
// This extends ComputeMultimodalRoPE to handle N images instead of just one.
//
// Parameters:
//   - imagePositions: starting position of each image's tokens in the sequence
//   - visionGridH, visionGridW: grid dimensions for each image (after spatial merge)
//   - numImageTokens: number of tokens for each image
//   - totalLen: total sequence length
func (m *Qwen25VL) ComputeMultiImageRoPE(imagePositions []int32, visionGridH, visionGridW, numImageTokens []int32, totalLen int32) ([2]*mlx.Array, error) {
	numImages := len(imagePositions)

	// Build 3D position IDs: [3, 1, totalLen]
	// Dimension 0: temporal, Dimension 1: height, Dimension 2: width
	posIDs := make([]float32, 3*totalLen)

	// Process sequence in order
	stIdx := int32(0) // Running text position counter
	seqIdx := int32(0)

	for i := 0; i < numImages; i++ {
		imgPos := imagePositions[i]
		gridH := visionGridH[i]
		gridW := visionGridW[i]
		numTokens := numImageTokens[i]

		// Text segment before this image
		for seqIdx < imgPos {
			posIDs[0*totalLen+seqIdx] = float32(stIdx)
			posIDs[1*totalLen+seqIdx] = float32(stIdx)
			posIDs[2*totalLen+seqIdx] = float32(stIdx)
			stIdx++
			seqIdx++
		}

		// Vision tokens for this image
		// Python uses stIdx as base offset for all position dimensions
		for h := int32(0); h < gridH; h++ {
			for w := int32(0); w < gridW; w++ {
				posIDs[0*totalLen+seqIdx] = float32(stIdx)     // temporal: constant = stIdx
				posIDs[1*totalLen+seqIdx] = float32(stIdx + h) // height: stIdx + row_index
				posIDs[2*totalLen+seqIdx] = float32(stIdx + w) // width: stIdx + col_index
				seqIdx++
			}
		}

		// Verify we processed the expected number of tokens
		if seqIdx != imgPos+numTokens {
			return [2]*mlx.Array{}, fmt.Errorf("mismatch: processed %d but expected %d tokens for image %d", seqIdx-imgPos, numTokens, i)
		}

		// Update stIdx for next text segment: max(temporal, height, width) + 1
		maxVisionPos := stIdx // temporal max
		if stIdx+gridH-1 > maxVisionPos {
			maxVisionPos = stIdx + gridH - 1
		}
		if stIdx+gridW-1 > maxVisionPos {
			maxVisionPos = stIdx + gridW - 1
		}
		stIdx = maxVisionPos + 1
	}

	// Text after last image
	for seqIdx < totalLen {
		posIDs[0*totalLen+seqIdx] = float32(stIdx)
		posIDs[1*totalLen+seqIdx] = float32(stIdx)
		posIDs[2*totalLen+seqIdx] = float32(stIdx)
		stIdx++
		seqIdx++
	}

	posIDsArr := mlx.NewArray(posIDs, []int32{3, 1, totalLen})
	return m.computeRoPEFromPositions(posIDsArr, totalLen, 1), nil
}

// computeTextRoPE computes M-RoPE for text-only sequences
func (m *Qwen25VL) computeTextRoPE(L, B int32) [2]*mlx.Array {
	// For text-only, all 3 dims use same positions [0, 1, 2, ..., L-1]
	posArr := make([]float32, L*3)
	for d := 0; d < 3; d++ {
		for i := int32(0); i < L; i++ {
			posArr[int32(d)*L+i] = float32(i)
		}
	}
	posIDs := mlx.NewArray(posArr, []int32{3, 1, L})
	posIDs = mlx.Tile(posIDs, []int32{1, B, 1})
	return m.computeRoPEFromPositions(posIDs, L, B)
}

// ComputeMultimodalRoPE computes M-RoPE for combined text + vision + text sequences
// This matches Python's get_rope_index behavior exactly.
// Exported for testing.
//
// Python pattern discovered from testing:
//
//	Vision row 1: temporal=stIdx, height=stIdx, width=[stIdx, stIdx+1, ..., stIdx+gridW-1]
//	Vision row 2: temporal=stIdx, height=stIdx+1, width=[stIdx, stIdx+1, ..., stIdx+gridW-1]
//	Text after: temporal=stIdx+1+i, height=stIdx+gridH+i, width=stIdx+gridW+i
func (m *Qwen25VL) ComputeMultimodalRoPE(textBefore, visionH, visionW, textAfter int32, spatialMerge int32) [2]*mlx.Array {
	// Vision grid after spatial merge
	llmGridH := visionH / spatialMerge
	llmGridW := visionW / spatialMerge
	visionLen := llmGridH * llmGridW
	totalLen := textBefore + visionLen + textAfter

	// Build 3D position IDs: [3, 1, totalLen]
	// Dimension 0: temporal, Dimension 1: height, Dimension 2: width
	posIDs := make([]float32, 3*totalLen)

	// Text before vision: all dims same [0, 1, 2, ..., textBefore-1]
	for d := 0; d < 3; d++ {
		for i := int32(0); i < textBefore; i++ {
			posIDs[int32(d)*totalLen+i] = float32(i)
		}
	}

	// Vision tokens: 3D grid positions
	// Python uses stIdx (textBefore) as base offset for all position dimensions
	stIdx := textBefore
	for h := int32(0); h < llmGridH; h++ {
		for w := int32(0); w < llmGridW; w++ {
			idx := stIdx + h*llmGridW + w
			posIDs[0*totalLen+idx] = float32(stIdx)     // temporal: constant = stIdx
			posIDs[1*totalLen+idx] = float32(stIdx + h) // height: stIdx + row_index
			posIDs[2*totalLen+idx] = float32(stIdx + w) // width: stIdx + col_index
		}
	}

	// Text after vision: ALL dimensions continue from max(temporal, height, width) + 1
	// max is max(stIdx, stIdx+llmGridH-1, stIdx+llmGridW-1) = stIdx + max(0, llmGridH-1, llmGridW-1)
	// Then st_idx = max + 1
	maxVisionPos := stIdx // temporal max
	if stIdx+llmGridH-1 > maxVisionPos {
		maxVisionPos = stIdx + llmGridH - 1
	}
	if stIdx+llmGridW-1 > maxVisionPos {
		maxVisionPos = stIdx + llmGridW - 1
	}
	textAfterStart := maxVisionPos + 1
	for i := int32(0); i < textAfter; i++ {
		seqIdx := textBefore + visionLen + i
		posIDs[0*totalLen+seqIdx] = float32(textAfterStart + i) // temporal
		posIDs[1*totalLen+seqIdx] = float32(textAfterStart + i) // height
		posIDs[2*totalLen+seqIdx] = float32(textAfterStart + i) // width
	}

	posIDsArr := mlx.NewArray(posIDs, []int32{3, 1, totalLen})
	return m.computeRoPEFromPositions(posIDsArr, totalLen, 1)
}

// computeRoPEFromPositions computes cos/sin from 3D position IDs
// posIDs: [3, B, L] where dim 0 is temporal, 1 is height, 2 is width
func (m *Qwen25VL) computeRoPEFromPositions(posIDs *mlx.Array, L, B int32) [2]*mlx.Array {
	cfg := m.Config
	half := cfg.HeadDim / 2

	// Compute inv_freq
	invFreqArr := make([]float32, half)
	for i := int32(0); i < half; i++ {
		invFreqArr[i] = float32(1.0 / math.Pow(float64(cfg.RopeTheta), 2.0*float64(i)/float64(cfg.HeadDim)))
	}
	invFreq := mlx.NewArray(invFreqArr, []int32{half})

	// Process each position dimension
	var cosAll, sinAll []*mlx.Array
	for d := int32(0); d < 3; d++ {
		// Get positions for this dimension: [B, L]
		pos := mlx.Slice(posIDs, []int32{d, 0, 0}, []int32{d + 1, B, L})
		pos = mlx.Squeeze(pos, 0) // [B, L]

		posExp := mlx.ExpandDims(pos, 2)               // [B, L, 1]
		invFreqExp := mlx.Reshape(invFreq, 1, 1, half) // [1, 1, half]
		freqs := mlx.Mul(posExp, invFreqExp)           // [B, L, half]
		emb := mlx.Tile(freqs, []int32{1, 1, 2})       // [B, L, D]

		cosAll = append(cosAll, mlx.ExpandDims(mlx.Cos(emb), 0))
		sinAll = append(sinAll, mlx.ExpandDims(mlx.Sin(emb), 0))
	}

	cos := mlx.Concatenate(cosAll, 0) // [3, B, L, D]
	sin := mlx.Concatenate(sinAll, 0)

	return [2]*mlx.Array{cos, sin}
}

// computeVisionRoPE computes RoPE embeddings for vision patches
// pH, pW: grid dimensions in patches
// Returns: [2]*mlx.Array containing (cos, sin) each of shape [numPatches, headDim]
func (m *Qwen25VL) computeVisionRoPE(pH, pW int32) [2]*mlx.Array {
	cfg := m.Config
	headDim := cfg.VisionHiddenSize / cfg.VisionNumHeads // 80 for 1280/16
	halfDim := headDim / 2                               // 40
	quarterDim := halfDim / 2                            // 20
	spatialMerge := cfg.VisionSpatialMerge               // 2

	// Python Qwen2_5_VisionRotaryEmbedding uses dim=head_dim/2=40
	// inv_freq = 1.0 / (theta ** (arange(0, dim, 2) / dim)) -> 20 elements
	theta := float64(10000.0)
	invFreqArr := make([]float32, quarterDim)
	for i := int32(0); i < quarterDim; i++ {
		invFreqArr[i] = float32(1.0 / math.Pow(theta, float64(2*i)/float64(halfDim)))
	}
	invFreq := mlx.NewArray(invFreqArr, []int32{quarterDim})

	// Create position IDs matching Python's 2x2 block ordering:
	// Python does: reshape(h//2, 2, w//2, 2), permute(0, 2, 1, 3), flatten
	// This groups patches by 2x2 merged token blocks
	numPatches := pH * pW
	hPosArr := make([]float32, numPatches)
	wPosArr := make([]float32, numPatches)

	// Number of merged token blocks
	llmGridH := pH / spatialMerge
	llmGridW := pW / spatialMerge

	idx := int32(0)
	for hBlock := int32(0); hBlock < llmGridH; hBlock++ {
		for wBlock := int32(0); wBlock < llmGridW; wBlock++ {
			// Within each 2x2 block: (0,0), (0,1), (1,0), (1,1)
			for dh := int32(0); dh < spatialMerge; dh++ {
				for dw := int32(0); dw < spatialMerge; dw++ {
					h := hBlock*spatialMerge + dh
					w := wBlock*spatialMerge + dw
					hPosArr[idx] = float32(h)
					wPosArr[idx] = float32(w)
					idx++
				}
			}
		}
	}

	hPos := mlx.NewArray(hPosArr, []int32{numPatches, 1})
	wPos := mlx.NewArray(wPosArr, []int32{numPatches, 1})
	invFreqExp := mlx.Reshape(invFreq, 1, quarterDim)

	// Compute freqs: [numPatches, quarterDim] for each of h and w
	hFreqs := mlx.Mul(hPos, invFreqExp) // [L, 20]
	wFreqs := mlx.Mul(wPos, invFreqExp) // [L, 20]

	// Concatenate h and w freqs: [numPatches, halfDim] = [L, 40]
	freqs := mlx.Concatenate([]*mlx.Array{hFreqs, wFreqs}, 1)

	// Double for cos/sin application: [L, 40] -> [L, 80] = [L, headDim]
	emb := mlx.Concatenate([]*mlx.Array{freqs, freqs}, 1)

	cos := mlx.Cos(emb)
	sin := mlx.Sin(emb)

	return [2]*mlx.Array{cos, sin}
}

// VLTextBlock is a single Qwen2.5 transformer block (for VL model)
type VLTextBlock struct {
	Attention         *VLTextAttention
	MLP               *VLTextMLP
	InputLayerNorm    *mlx.Array
	PostAttnLayerNorm *mlx.Array
	NormEps           float32
}

// newVLTextBlock creates a text block
func newVLTextBlock(weights *safetensors.ModelWeights, layerIdx int, cfg *Qwen25VLConfig) (*VLTextBlock, error) {
	prefix := fmt.Sprintf("model.layers.%d", layerIdx)

	inputNorm, err := weights.Get(prefix + ".input_layernorm.weight")
	if err != nil {
		return nil, err
	}
	postAttnNorm, err := weights.Get(prefix + ".post_attention_layernorm.weight")
	if err != nil {
		return nil, err
	}

	attention, err := newVLTextAttention(weights, prefix, cfg)
	if err != nil {
		return nil, err
	}

	mlpLayer, err := newVLTextMLP(weights, prefix)
	if err != nil {
		return nil, err
	}

	return &VLTextBlock{
		Attention:         attention,
		MLP:               mlpLayer,
		InputLayerNorm:    inputNorm,
		PostAttnLayerNorm: postAttnNorm,
		NormEps:           cfg.RMSNormEps,
	}, nil
}

// Forward applies the block
func (tb *VLTextBlock) Forward(x *mlx.Array, cossin [2]*mlx.Array) *mlx.Array {
	h := mlx.RMSNorm(x, tb.InputLayerNorm, tb.NormEps)
	attnOut := tb.Attention.Forward(h, cossin)
	x = mlx.Add(x, attnOut)

	h = mlx.RMSNorm(x, tb.PostAttnLayerNorm, tb.NormEps)
	mlpOut := tb.MLP.Forward(h)
	x = mlx.Add(x, mlpOut)

	return x
}

// VLTextAttention implements Qwen2.5 attention with M-RoPE
type VLTextAttention struct {
	QProj        *mlx.Array
	KProj        *mlx.Array
	VProj        *mlx.Array
	OProj        *mlx.Array
	QBias        *mlx.Array
	KBias        *mlx.Array
	VBias        *mlx.Array
	NHeads       int32
	NKVHeads     int32
	HeadDim      int32
	Scale        float32
	MRoPESection []int32
}

// newVLTextAttention creates a text attention layer
func newVLTextAttention(weights *safetensors.ModelWeights, prefix string, cfg *Qwen25VLConfig) (*VLTextAttention, error) {
	qProj, err := weights.Get(prefix + ".self_attn.q_proj.weight")
	if err != nil {
		return nil, err
	}
	kProj, err := weights.Get(prefix + ".self_attn.k_proj.weight")
	if err != nil {
		return nil, err
	}
	vProj, err := weights.Get(prefix + ".self_attn.v_proj.weight")
	if err != nil {
		return nil, err
	}
	oProj, err := weights.Get(prefix + ".self_attn.o_proj.weight")
	if err != nil {
		return nil, err
	}

	qBias, _ := weights.Get(prefix + ".self_attn.q_proj.bias")
	kBias, _ := weights.Get(prefix + ".self_attn.k_proj.bias")
	vBias, _ := weights.Get(prefix + ".self_attn.v_proj.bias")

	return &VLTextAttention{
		QProj:        mlx.Transpose(qProj, 1, 0),
		KProj:        mlx.Transpose(kProj, 1, 0),
		VProj:        mlx.Transpose(vProj, 1, 0),
		OProj:        mlx.Transpose(oProj, 1, 0),
		QBias:        qBias,
		KBias:        kBias,
		VBias:        vBias,
		NHeads:       cfg.NumAttentionHeads,
		NKVHeads:     cfg.NumKeyValueHeads,
		HeadDim:      cfg.HeadDim,
		Scale:        float32(1.0 / math.Sqrt(float64(cfg.HeadDim))),
		MRoPESection: cfg.MRoPESection,
	}, nil
}

// Forward computes attention
func (attn *VLTextAttention) Forward(x *mlx.Array, cossin [2]*mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]

	q := mlx.Linear(x, attn.QProj)
	if attn.QBias != nil {
		q = mlx.Add(q, attn.QBias)
	}
	k := mlx.Linear(x, attn.KProj)
	if attn.KBias != nil {
		k = mlx.Add(k, attn.KBias)
	}
	v := mlx.Linear(x, attn.VProj)
	if attn.VBias != nil {
		v = mlx.Add(v, attn.VBias)
	}

	q = mlx.Reshape(q, B, L, attn.NHeads, attn.HeadDim)
	k = mlx.Reshape(k, B, L, attn.NKVHeads, attn.HeadDim)
	v = mlx.Reshape(v, B, L, attn.NKVHeads, attn.HeadDim)

	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	// Apply M-RoPE
	if cossin[0] != nil && cossin[1] != nil {
		q = applyMRoPE(q, cossin[0], cossin[1], attn.MRoPESection)
		k = applyMRoPE(k, cossin[0], cossin[1], attn.MRoPESection)
	}

	// Repeat KV for GQA
	if attn.NKVHeads < attn.NHeads {
		repeats := attn.NHeads / attn.NKVHeads
		k = repeatKV(k, repeats)
		v = repeatKV(v, repeats)
	}

	out := mlx.ScaledDotProductAttention(q, k, v, attn.Scale, true)

	out = mlx.Transpose(out, 0, 2, 1, 3)
	out = mlx.Reshape(out, B, L, attn.NHeads*attn.HeadDim)

	return mlx.Linear(out, attn.OProj)
}

// applyMRoPE applies Multi-Resolution RoPE
func applyMRoPE(x *mlx.Array, cos, sin *mlx.Array, section []int32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	H := shape[1]
	L := shape[2]
	D := shape[3]
	half := D / 2

	fullSection := make([]int32, len(section))
	for i, s := range section {
		fullSection[i] = s * 2
	}

	var cosParts, sinParts []*mlx.Array
	offset := int32(0)
	for i, size := range fullSection {
		posDim := int32(i % 3)
		cosSection := mlx.Slice(cos, []int32{posDim, 0, 0, offset}, []int32{posDim + 1, B, L, offset + size})
		sinSection := mlx.Slice(sin, []int32{posDim, 0, 0, offset}, []int32{posDim + 1, B, L, offset + size})
		cosSection = mlx.Squeeze(cosSection, 0)
		sinSection = mlx.Squeeze(sinSection, 0)
		cosParts = append(cosParts, cosSection)
		sinParts = append(sinParts, sinSection)
		offset += size
	}

	cosFlat := mlx.Concatenate(cosParts, 2)
	sinFlat := mlx.Concatenate(sinParts, 2)

	cosFlat = mlx.Reshape(cosFlat, B, 1, L, D)
	sinFlat = mlx.Reshape(sinFlat, B, 1, L, D)

	x1 := mlx.Slice(x, []int32{0, 0, 0, 0}, []int32{B, H, L, half})
	x2 := mlx.Slice(x, []int32{0, 0, 0, half}, []int32{B, H, L, D})
	negX2 := mlx.MulScalar(x2, -1)
	rotatedX := mlx.Concatenate([]*mlx.Array{negX2, x1}, 3)

	return mlx.Add(mlx.Mul(x, cosFlat), mlx.Mul(rotatedX, sinFlat))
}

// repeatKV repeats key/value heads for GQA
func repeatKV(x *mlx.Array, repeats int32) *mlx.Array {
	if repeats == 1 {
		return x
	}
	shape := x.Shape()
	x = mlx.ExpandDims(x, 2)
	x = mlx.Tile(x, []int32{1, 1, repeats, 1, 1})
	return mlx.Reshape(x, shape[0], shape[1]*repeats, shape[2], shape[3])
}

// VLTextMLP implements Qwen2.5 SwiGLU MLP
type VLTextMLP struct {
	GateProj *mlx.Array
	UpProj   *mlx.Array
	DownProj *mlx.Array
}

// newVLTextMLP creates a text MLP layer
func newVLTextMLP(weights *safetensors.ModelWeights, prefix string) (*VLTextMLP, error) {
	gateProj, err := weights.Get(prefix + ".mlp.gate_proj.weight")
	if err != nil {
		return nil, err
	}
	upProj, err := weights.Get(prefix + ".mlp.up_proj.weight")
	if err != nil {
		return nil, err
	}
	downProj, err := weights.Get(prefix + ".mlp.down_proj.weight")
	if err != nil {
		return nil, err
	}

	return &VLTextMLP{
		GateProj: mlx.Transpose(gateProj, 1, 0),
		UpProj:   mlx.Transpose(upProj, 1, 0),
		DownProj: mlx.Transpose(downProj, 1, 0),
	}, nil
}

// Forward applies the SwiGLU MLP
func (mlp *VLTextMLP) Forward(x *mlx.Array) *mlx.Array {
	gate := mlx.Linear(x, mlp.GateProj)
	gate = mlx.SiLU(gate)
	up := mlx.Linear(x, mlp.UpProj)
	h := mlx.Mul(gate, up)
	return mlx.Linear(h, mlp.DownProj)
}

// VisionPatchEmbed embeds image patches
type VisionPatchEmbed struct {
	ProjWeight *mlx.Array
	ProjBias   *mlx.Array
	PatchSize  int32
}

// newVisionPatchEmbed creates a vision patch embed layer
func newVisionPatchEmbed(weights *safetensors.ModelWeights, cfg *Qwen25VLConfig) (*VisionPatchEmbed, error) {
	projWeight, err := weights.Get("visual.patch_embed.proj.weight")
	if err != nil {
		return nil, err
	}
	projBias, _ := weights.Get("visual.patch_embed.proj.bias")

	return &VisionPatchEmbed{
		ProjWeight: projWeight,
		ProjBias:   projBias,
		PatchSize:  cfg.VisionPatchSize,
	}, nil
}

// Forward embeds patches from an image
// image: [B, C, H, W]
// Returns: [B, num_patches, hidden_size]
func (pe *VisionPatchEmbed) Forward(image *mlx.Array) *mlx.Array {
	// Qwen2.5-VL uses 3D conv for patch embedding to support video
	// Weight shape is [O, I, kT, kH, kW] e.g. [1280, 3, 2, 14, 14]
	// For single image, we duplicate the frame to match temporal_patch_size

	wShape := pe.ProjWeight.Shape()
	if len(wShape) == 5 {
		// 3D convolution case
		temporalPatchSize := wShape[2] // kT from weight shape

		// Add temporal dimension: [B, C, H, W] -> [B, C, 1, H, W]
		image = mlx.ExpandDims(image, 2)

		// Duplicate frame to match temporal_patch_size (Python does this for single images)
		// [B, C, 1, H, W] -> [B, C, T, H, W] where T = temporal_patch_size
		if temporalPatchSize > 1 {
			image = mlx.Tile(image, []int32{1, 1, temporalPatchSize, 1, 1})
		}

		// Convert to channels-last: [B, C, T, H, W] -> [B, T, H, W, C]
		image = mlx.Transpose(image, 0, 2, 3, 4, 1)

		// Weight is [O, I, kT, kH, kW] - keep as-is since patches are now in [I, kT, kH, kW] order
		// (extractPatches3DStrided transposes each patch to [C, T, H, W] to match Python)

		// Apply 3D conv using manual patch extraction
		// Strides: (temporal_patch_size, patch_size, patch_size)
		x := conv3DStrided(image, pe.ProjWeight, temporalPatchSize, pe.PatchSize, pe.PatchSize)

		if pe.ProjBias != nil {
			outC := pe.ProjBias.Dim(0)
			bias := mlx.Reshape(pe.ProjBias, 1, 1, 1, 1, outC)
			x = mlx.Add(x, bias)
		}

		// x is [B, T', H', W', C], squeeze T' and flatten spatial
		shape := x.Shape()
		// T' should be 1 for single image (since we used stride=temporal_patch_size)
		x = mlx.Reshape(x, shape[0], shape[2]*shape[3], shape[4])

		return x
	}

	// Original 2D case (fallback)
	// Convert to channels-last for Conv2d
	image = mlx.Transpose(image, 0, 2, 3, 1) // [B, H, W, C]

	// Apply conv with stride=patch_size using manual strided convolution
	weight := mlx.Transpose(pe.ProjWeight, 0, 2, 3, 1) // [O, I, kH, kW] -> [O, kH, kW, I]
	x := conv2DStrided(image, weight, pe.PatchSize)
	if pe.ProjBias != nil {
		bias := mlx.Reshape(pe.ProjBias, 1, 1, 1, pe.ProjBias.Dim(0))
		x = mlx.Add(x, bias)
	}

	// Flatten patches: [B, pH, pW, C] -> [B, pH*pW, C]
	shape := x.Shape()
	x = mlx.Reshape(x, shape[0], shape[1]*shape[2], shape[3])

	return x
}

// VisionBlock is a single vision transformer block
type VisionBlock struct {
	Norm1     *mlx.Array
	Norm2     *mlx.Array
	Attention *VisionAttention
	MLP       *VisionMLP
}

// newVisionBlock creates a vision block
func newVisionBlock(weights *safetensors.ModelWeights, layerIdx int, cfg *Qwen25VLConfig) (*VisionBlock, error) {
	prefix := fmt.Sprintf("visual.blocks.%d", layerIdx)

	norm1, err := weights.Get(prefix + ".norm1.weight")
	if err != nil {
		return nil, err
	}
	norm2, err := weights.Get(prefix + ".norm2.weight")
	if err != nil {
		return nil, err
	}

	attention, err := newVisionAttention(weights, prefix, cfg)
	if err != nil {
		return nil, err
	}

	mlpLayer, err := newVisionMLP(weights, prefix, cfg)
	if err != nil {
		return nil, err
	}

	return &VisionBlock{
		Norm1:     norm1,
		Norm2:     norm2,
		Attention: attention,
		MLP:       mlpLayer,
	}, nil
}

// Forward applies the vision block
// posEmb: [2]*mlx.Array containing (cos, sin) for RoPE, can be nil
// cuSeqlens: cumulative sequence lengths for window attention
func (vb *VisionBlock) Forward(x *mlx.Array, posEmb [2]*mlx.Array, cuSeqlens []int32) *mlx.Array {
	// Python uses RMSNorm, not LayerNorm!
	h := mlx.RMSNormNoWeight(x, 1e-6)
	h = mlx.Mul(h, vb.Norm1)
	attnOut := vb.Attention.Forward(h, posEmb, cuSeqlens)
	x = mlx.Add(x, attnOut)

	h = mlx.RMSNormNoWeight(x, 1e-6)
	h = mlx.Mul(h, vb.Norm2)
	mlpOut := vb.MLP.Forward(h)
	x = mlx.Add(x, mlpOut)

	return x
}

// VisionAttention implements vision attention
type VisionAttention struct {
	QKVProj *mlx.Array
	QKVBias *mlx.Array
	OutProj *mlx.Array
	OutBias *mlx.Array
	NHeads  int32
	HeadDim int32
	Scale   float32
}

// newVisionAttention creates a vision attention layer
func newVisionAttention(weights *safetensors.ModelWeights, prefix string, cfg *Qwen25VLConfig) (*VisionAttention, error) {
	qkvProj, err := weights.Get(prefix + ".attn.qkv.weight")
	if err != nil {
		return nil, err
	}
	qkvBias, _ := weights.Get(prefix + ".attn.qkv.bias")
	outProj, err := weights.Get(prefix + ".attn.proj.weight")
	if err != nil {
		return nil, err
	}
	outBias, _ := weights.Get(prefix + ".attn.proj.bias")

	headDim := cfg.VisionHiddenSize / cfg.VisionNumHeads

	return &VisionAttention{
		QKVProj: mlx.Transpose(qkvProj, 1, 0),
		QKVBias: qkvBias,
		OutProj: mlx.Transpose(outProj, 1, 0),
		OutBias: outBias,
		NHeads:  cfg.VisionNumHeads,
		HeadDim: headDim,
		Scale:   float32(1.0 / math.Sqrt(float64(headDim))),
	}, nil
}

// Forward applies vision attention with optional RoPE and window attention
// posEmb: [2]*mlx.Array containing (cos, sin) for RoPE, can be nil
// cuSeqlens: cumulative sequence lengths for window boundaries
func (attn *VisionAttention) Forward(x *mlx.Array, posEmb [2]*mlx.Array, cuSeqlens []int32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	D := shape[2]

	qkv := mlx.Linear(x, attn.QKVProj)
	if attn.QKVBias != nil {
		qkv = mlx.Add(qkv, attn.QKVBias)
	}

	// Split into Q, K, V
	qkv = mlx.Reshape(qkv, B, L, 3, attn.NHeads, attn.HeadDim)
	q := mlx.Slice(qkv, []int32{0, 0, 0, 0, 0}, []int32{B, L, 1, attn.NHeads, attn.HeadDim})
	k := mlx.Slice(qkv, []int32{0, 0, 1, 0, 0}, []int32{B, L, 2, attn.NHeads, attn.HeadDim})
	v := mlx.Slice(qkv, []int32{0, 0, 2, 0, 0}, []int32{B, L, 3, attn.NHeads, attn.HeadDim})

	q = mlx.Squeeze(q, 2) // [B, L, H, D]
	k = mlx.Squeeze(k, 2)
	v = mlx.Squeeze(v, 2)

	// Apply RoPE if position embeddings provided
	if posEmb[0] != nil && posEmb[1] != nil {
		q, k = applyVisionRoPE(q, k, posEmb[0], posEmb[1])
	}

	q = mlx.Transpose(q, 0, 2, 1, 3) // [B, H, L, D]
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	var out *mlx.Array

	// Check if we need window attention (more than 1 window)
	numWindows := len(cuSeqlens) - 1
	if numWindows <= 1 {
		// Full attention - single window covering entire sequence
		out = mlx.ScaledDotProductAttention(q, k, v, attn.Scale, false)
	} else {
		// Window attention - process each window separately
		attnOutputs := make([]*mlx.Array, numWindows)

		for w := 0; w < numWindows; w++ {
			start := cuSeqlens[w]
			end := cuSeqlens[w+1]

			// Slice Q, K, V for this window: [B, H, winLen, D]
			qWin := mlx.Slice(q, []int32{0, 0, start, 0}, []int32{B, attn.NHeads, end, attn.HeadDim})
			kWin := mlx.Slice(k, []int32{0, 0, start, 0}, []int32{B, attn.NHeads, end, attn.HeadDim})
			vWin := mlx.Slice(v, []int32{0, 0, start, 0}, []int32{B, attn.NHeads, end, attn.HeadDim})

			// Compute attention for this window
			attnWin := mlx.ScaledDotProductAttention(qWin, kWin, vWin, attn.Scale, false)
			attnOutputs[w] = attnWin
		}

		// Concatenate all window outputs along sequence dimension
		out = mlx.Concatenate(attnOutputs, 2)
	}

	out = mlx.Transpose(out, 0, 2, 1, 3) // [B, L, H, D]
	out = mlx.Reshape(out, B, L, D)

	out = mlx.Linear(out, attn.OutProj)
	if attn.OutBias != nil {
		out = mlx.Add(out, attn.OutBias)
	}

	return out
}

// applyVisionRoPE applies rotary position embedding to Q and K for vision
// q, k: [B, L, H, D], cos, sin: [L, D] (already doubled: D = head_dim)
// Returns: rotated q, k with same shape
// Note: Python does this computation in float32 for numerical stability
func applyVisionRoPE(q, k, cos, sin *mlx.Array) (*mlx.Array, *mlx.Array) {
	// Convert to float32 for numerical stability (matches Python)
	origDtype := q.Dtype()
	q = mlx.AsType(q, mlx.DtypeFloat32)
	k = mlx.AsType(k, mlx.DtypeFloat32)
	cos = mlx.AsType(cos, mlx.DtypeFloat32)
	sin = mlx.AsType(sin, mlx.DtypeFloat32)

	// Expand cos/sin to match q/k shape: [L, D] -> [1, L, 1, D]
	cos = mlx.ExpandDims(cos, 0)
	cos = mlx.ExpandDims(cos, 2)
	sin = mlx.ExpandDims(sin, 0)
	sin = mlx.ExpandDims(sin, 2)

	// rotate_half: split last dim in half and swap with negation
	// q_rot = q * cos + rotate_half(q) * sin
	qRotated := rotateHalf(q)
	kRotated := rotateHalf(k)

	qOut := mlx.Add(mlx.Mul(q, cos), mlx.Mul(qRotated, sin))
	kOut := mlx.Add(mlx.Mul(k, cos), mlx.Mul(kRotated, sin))

	// Convert back to original dtype
	qOut = mlx.AsType(qOut, origDtype)
	kOut = mlx.AsType(kOut, origDtype)

	return qOut, kOut
}

// rotateHalf rotates the last dimension by splitting in half and swapping with negation
// x: [..., D] -> split to [..., D/2] and [..., D/2], then concat(-x2, x1)
func rotateHalf(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	lastDim := shape[len(shape)-1]
	halfDim := lastDim / 2

	// Split into two halves
	x1 := mlx.Slice(x, []int32{0, 0, 0, 0}, []int32{shape[0], shape[1], shape[2], halfDim})
	x2 := mlx.Slice(x, []int32{0, 0, 0, halfDim}, []int32{shape[0], shape[1], shape[2], lastDim})

	// Negate x2 and concatenate
	x2Neg := mlx.MulScalar(x2, -1.0)
	return mlx.Concatenate([]*mlx.Array{x2Neg, x1}, 3)
}

// VisionMLP implements vision SwiGLU MLP
type VisionMLP struct {
	GateProj     *mlx.Array
	GateProjBias *mlx.Array
	UpProj       *mlx.Array
	UpProjBias   *mlx.Array
	DownProj     *mlx.Array
	DownProjBias *mlx.Array
}

// newVisionMLP creates a vision MLP layer
func newVisionMLP(weights *safetensors.ModelWeights, prefix string, cfg *Qwen25VLConfig) (*VisionMLP, error) {
	gateProj, err := weights.Get(prefix + ".mlp.gate_proj.weight")
	if err != nil {
		return nil, err
	}
	gateProjBias, _ := weights.Get(prefix + ".mlp.gate_proj.bias")
	upProj, err := weights.Get(prefix + ".mlp.up_proj.weight")
	if err != nil {
		return nil, err
	}
	upProjBias, _ := weights.Get(prefix + ".mlp.up_proj.bias")
	downProj, err := weights.Get(prefix + ".mlp.down_proj.weight")
	if err != nil {
		return nil, err
	}
	downProjBias, _ := weights.Get(prefix + ".mlp.down_proj.bias")

	return &VisionMLP{
		GateProj:     mlx.Transpose(gateProj, 1, 0),
		GateProjBias: gateProjBias,
		UpProj:       mlx.Transpose(upProj, 1, 0),
		UpProjBias:   upProjBias,
		DownProj:     mlx.Transpose(downProj, 1, 0),
		DownProjBias: downProjBias,
	}, nil
}

// Forward applies the vision SwiGLU MLP
func (m *VisionMLP) Forward(x *mlx.Array) *mlx.Array {
	gate := mlx.Linear(x, m.GateProj)
	if m.GateProjBias != nil {
		gate = mlx.Add(gate, m.GateProjBias)
	}
	gate = mlx.SiLU(gate)

	up := mlx.Linear(x, m.UpProj)
	if m.UpProjBias != nil {
		up = mlx.Add(up, m.UpProjBias)
	}

	h := mlx.Mul(gate, up)
	h = mlx.Linear(h, m.DownProj)
	if m.DownProjBias != nil {
		h = mlx.Add(h, m.DownProjBias)
	}
	return h
}

// VisionMerger merges spatial patches (2x2 -> 1)
type VisionMerger struct {
	MLP0Weight *mlx.Array
	MLP0Bias   *mlx.Array
	MLP2Weight *mlx.Array
	MLP2Bias   *mlx.Array
	LNWeight   *mlx.Array
}

// newVisionMerger creates a vision merger
func newVisionMerger(weights *safetensors.ModelWeights, cfg *Qwen25VLConfig) (*VisionMerger, error) {
	mlp0Weight, err := weights.Get("visual.merger.mlp.0.weight")
	if err != nil {
		return nil, err
	}
	mlp0Bias, _ := weights.Get("visual.merger.mlp.0.bias")
	mlp2Weight, err := weights.Get("visual.merger.mlp.2.weight")
	if err != nil {
		return nil, err
	}
	mlp2Bias, _ := weights.Get("visual.merger.mlp.2.bias")
	lnWeight, _ := weights.Get("visual.merger.ln_q.weight")

	return &VisionMerger{
		MLP0Weight: mlx.Transpose(mlp0Weight, 1, 0),
		MLP0Bias:   mlp0Bias,
		MLP2Weight: mlx.Transpose(mlp2Weight, 1, 0),
		MLP2Bias:   mlp2Bias,
		LNWeight:   lnWeight,
	}, nil
}

// Forward merges 2x2 patches into 1 (assumes square grid - use ForwardWithDims for non-square)
func (m *VisionMerger) Forward(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	L := shape[1]
	side := int32(math.Sqrt(float64(L)))
	return m.ForwardWithDims(x, side, side)
}

// ForwardWithDims merges 2x2 patches into 1 with explicit grid dimensions
// After window reordering, consecutive 4 patches form a 2x2 block, so we just
// reshape [B, L, D] -> [B, L/4, 4*D] without 2D spatial rearrangement.
func (m *VisionMerger) ForwardWithDims(x *mlx.Array, pH, pW int32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	D := shape[2]

	// RMSNorm BEFORE merge (applied to each token with D dimensions)
	// Python: ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
	if m.LNWeight != nil {
		x = mlx.RMSNormNoWeight(x, 1e-6)
		x = mlx.Mul(x, m.LNWeight)
	}

	// After window reordering, consecutive 4 patches belong to a 2x2 block
	// Just reshape to [B, L/4, 4*D] - no spatial rearrangement needed
	newL := L / 4
	x = mlx.Reshape(x, B, newL, 4*D)

	// MLP
	h := mlx.Linear(x, m.MLP0Weight)
	if m.MLP0Bias != nil {
		h = mlx.Add(h, m.MLP0Bias)
	}
	h = mlx.GELU(h)
	h = mlx.Linear(h, m.MLP2Weight)
	if m.MLP2Bias != nil {
		h = mlx.Add(h, m.MLP2Bias)
	}

	return h
}

// LoadQwen25VLFromPath loads the encoder from path
func LoadQwen25VLFromPath(path string) (*Qwen25VL, error) {
	m := &Qwen25VL{}
	if err := m.Load(filepath.Join(path, "text_encoder")); err != nil {
		return nil, err
	}
	return m, nil
}

// conv2DStrided applies conv with stride > 1 using manual patch extraction
// x: [B, H, W, C] (channels-last), weight: [O, kH, kW, I]
func conv2DStrided(x, weight *mlx.Array, stride int32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	H := shape[1]
	W := shape[2]

	wShape := weight.Shape()
	Cout := wShape[0]
	kH := wShape[1]
	kW := wShape[2]

	outH := (H - kH) / stride + 1
	outW := (W - kW) / stride + 1

	patches := extractPatches2DStrided(x, kH, kW, stride)
	wFlat := mlx.Reshape(weight, Cout, -1)
	patches = mlx.Reshape(patches, B*outH*outW, -1)
	out := mlx.Linear(patches, mlx.Transpose(wFlat, 1, 0))
	return mlx.Reshape(out, B, outH, outW, Cout)
}

// conv3DStrided applies 3D conv with strides using manual patch extraction
// x: [B, T, H, W, C] (channels-last), weight: [O, I, kT, kH, kW] (PyTorch format)
// strideT, strideH, strideW are the strides for each dimension
// Patches are extracted in [C, T, H, W] order to match Python's preprocessing
func conv3DStrided(x, weight *mlx.Array, strideT, strideH, strideW int32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	T := shape[1]
	H := shape[2]
	W := shape[3]
	C := shape[4]

	wShape := weight.Shape()
	Cout := wShape[0]
	// I := wShape[1]
	kT := wShape[2]
	kH := wShape[3]
	kW := wShape[4]

	// For temporal: if T < kT, we need to repeat frames temporally
	// For single image with T=1 and kT=2, we duplicate the frame to T=kT
	// Python Qwen2.5-VL duplicates the frame, not zero-pads
	if T < kT {
		// Tile along T dimension: [B, T, H, W, C] -> [B, kT, H, W, C]
		x = mlx.Tile(x, []int32{1, kT, 1, 1, 1})
		T = kT
	}

	outT := (T - kT) / strideT + 1
	outH := (H - kH) / strideH + 1
	outW := (W - kW) / strideW + 1

	// Extract 3D patches in [C, T, H, W] order to match Python
	patches := extractPatches3DStrided(x, kT, kH, kW, strideT, strideH, strideW)
	// patches shape: [B, outT, outH, outW, C*kT*kH*kW]

	// Weight is [O, I, kT, kH, kW] - flatten to [O, I*kT*kH*kW] to match patch order [C, T, H, W]
	wFlat := mlx.Reshape(weight, Cout, -1) // [Cout, I*kT*kH*kW]
	patches = mlx.Reshape(patches, B*outT*outH*outW, C*kT*kH*kW)
	out := mlx.Linear(patches, mlx.Transpose(wFlat, 1, 0))
	return mlx.Reshape(out, B, outT, outH, outW, Cout)
}

// extractPatches3DStrided extracts 3D patches with given strides
// Returns patches with values in [C, T, H, W] order to match Python's preprocessing
func extractPatches3DStrided(x *mlx.Array, kT, kH, kW, strideT, strideH, strideW int32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	T := shape[1]
	H := shape[2]
	W := shape[3]
	C := shape[4]

	outT := (T - kT) / strideT + 1
	outH := (H - kH) / strideH + 1
	outW := (W - kW) / strideW + 1

	numPatches := outT * outH * outW
	patches := make([]*mlx.Array, numPatches)
	idx := 0
	for t := int32(0); t < outT; t++ {
		for i := int32(0); i < outH; i++ {
			for j := int32(0); j < outW; j++ {
				startT := t * strideT
				startH := i * strideH
				startW := j * strideW
				// Extract patch: [B, kT, kH, kW, C]
				patch := mlx.Slice(x,
					[]int32{0, startT, startH, startW, 0},
					[]int32{B, startT + kT, startH + kH, startW + kW, C})
				// Transpose from [B, T, H, W, C] to [B, C, T, H, W] to match Python's order
				patch = mlx.Transpose(patch, 0, 4, 1, 2, 3)
				// Flatten to [B, C*T*H*W]
				patch = mlx.Reshape(patch, B, C*kT*kH*kW)
				patches[idx] = patch
				idx++
			}
		}
	}

	for i := range patches {
		patches[i] = mlx.ExpandDims(patches[i], 1)
	}
	stacked := mlx.Concatenate(patches, 1)
	return mlx.Reshape(stacked, B, outT, outH, outW, C*kT*kH*kW)
}

// extractPatches2DStrided extracts patches with given stride
func extractPatches2DStrided(x *mlx.Array, kH, kW, stride int32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	H := shape[1]
	W := shape[2]
	C := shape[3]

	outH := (H - kH) / stride + 1
	outW := (W - kW) / stride + 1

	patches := make([]*mlx.Array, outH*outW)
	idx := 0
	for i := int32(0); i < outH; i++ {
		for j := int32(0); j < outW; j++ {
			startH := i * stride
			startW := j * stride
			patch := mlx.Slice(x, []int32{0, startH, startW, 0}, []int32{B, startH + kH, startW + kW, C})
			patch = mlx.Reshape(patch, B, kH*kW*C)
			patches[idx] = patch
			idx++
		}
	}

	for i := range patches {
		patches[i] = mlx.ExpandDims(patches[i], 1)
	}
	stacked := mlx.Concatenate(patches, 1)
	return mlx.Reshape(stacked, B, outH, outW, kH*kW*C)
}

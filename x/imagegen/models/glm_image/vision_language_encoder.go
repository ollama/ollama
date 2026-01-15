//go:build mlx

package glm_image

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// VisionLanguageConfig holds GLM-Image AR generator configuration
type VisionLanguageConfig struct {
	// Text model config
	HiddenSize        int32   `json:"hidden_size"`         // 4096
	NumHiddenLayers   int32   `json:"num_hidden_layers"`   // 40
	IntermediateSize  int32   `json:"intermediate_size"`   // 13696
	NumAttentionHeads int32   `json:"num_attention_heads"` // 32
	NumKeyValueHeads  int32   `json:"num_key_value_heads"` // 2
	VocabSize         int32   `json:"vocab_size"`          // 168064
	RMSNormEps        float32 `json:"rms_norm_eps"`        // 1e-5

	// RoPE config
	RopeTheta           float32 `json:"rope_theta"`            // 10000
	PartialRotaryFactor float32 `json:"partial_rotary_factor"` // 0.5
	MRoPESection        []int32 `json:"mrope_section"`         // [8, 12, 12]

	// Visual token config
	VisionVocabSize   int32 `json:"vision_vocab_size"`    // 16512
	ImageStartTokenID int32 `json:"image_start_token_id"` // 16384
	ImageEndTokenID   int32 `json:"image_end_token_id"`   // 16385
	ImageTokenID      int32 `json:"image_token_id"`       // 167855

	// Computed
	HeadDim int32
}

// VisionLanguageEncoder is the 9B AR generator
type VisionLanguageEncoder struct {
	Config *VisionLanguageConfig

	// Embedding
	EmbedTokens *nn.Embedding `weight:"model.language_model.embed_tokens"`

	// Transformer layers
	Layers []*GLMBlock `weight:"model.language_model.layers"`

	// Final norm
	FinalNorm *nn.RMSNorm `weight:"model.language_model.norm"`

	// LM Head
	LMHead *mlx.Array `weight:"lm_head.weight"`
}

// GLMBlock is a single transformer block in GLM-4 style
type GLMBlock struct {
	// Pre-attention norm (GLM uses post-LN variant)
	InputLayerNorm    *nn.RMSNorm `weight:"input_layernorm"`
	PostSelfAttnNorm  *nn.RMSNorm `weight:"post_self_attn_layernorm"`
	PostAttnLayerNorm *nn.RMSNorm `weight:"post_attention_layernorm"`
	PostMLPLayerNorm  *nn.RMSNorm `weight:"post_mlp_layernorm"`

	// Attention
	SelfAttn *GLMAttention `weight:"self_attn"`

	// MLP (fused gate_up)
	MLP *GLMMLP `weight:"mlp"`
}

// GLMAttention implements GQA with partial rotary and MRoPE
type GLMAttention struct {
	QProj *mlx.Array `weight:"q_proj.weight"`
	KProj *mlx.Array `weight:"k_proj.weight"`
	VProj *mlx.Array `weight:"v_proj.weight"`
	OProj *mlx.Array `weight:"o_proj.weight"`

	// QKV have biases in GLM
	QBias *mlx.Array `weight:"q_proj.bias"`
	KBias *mlx.Array `weight:"k_proj.bias"`
	VBias *mlx.Array `weight:"v_proj.bias"`

	// Computed
	NHeads           int32
	NKVHeads         int32
	HeadDim          int32
	Scale            float32
	PartialRotary    float32   // Only rotate this fraction of head_dim
	RopeTheta        float32
	MRoPESection     []int32   // [8, 12, 12] - frequency pairs per dimension (temporal, height, width)
}

// ARCache holds KV caches for all layers using the shared cache implementation
type ARCache struct {
	Layers []cache.Cache
}

// NewARCache creates a new cache for the given number of layers
func NewARCache(numLayers int32) *ARCache {
	layers := make([]cache.Cache, numLayers)
	for i := range layers {
		layers[i] = cache.NewKVCache()
	}
	return &ARCache{Layers: layers}
}

// Free releases all cached tensors
func (c *ARCache) Free() {
	for _, layer := range c.Layers {
		for _, arr := range layer.State() {
			if arr != nil {
				arr.Free()
			}
		}
	}
}

// GLMMLP implements fused gate_up SwiGLU MLP
type GLMMLP struct {
	// GLM uses fused gate_up_proj: [hidden, 2*intermediate]
	GateUpProj *mlx.Array `weight:"gate_up_proj.weight"`
	DownProj   *mlx.Array `weight:"down_proj.weight"`
}

// Load loads the vision-language encoder from manifest
func (m *VisionLanguageEncoder) Load(manifest *imagegen.ModelManifest) error {
	fmt.Print("  Loading vision-language encoder... ")

	// Load config
	var rawCfg struct {
		TextConfig struct {
			HiddenSize        int32   `json:"hidden_size"`
			NumHiddenLayers   int32   `json:"num_hidden_layers"`
			IntermediateSize  int32   `json:"intermediate_size"`
			NumAttentionHeads int32   `json:"num_attention_heads"`
			NumKeyValueHeads  int32   `json:"num_key_value_heads"`
			VocabSize         int32   `json:"vocab_size"`
			RMSNormEps        float32 `json:"rms_norm_eps"`
			VisionVocabSize   int32   `json:"vision_vocab_size"`
			RopeParameters    struct {
				RopeTheta           float32 `json:"rope_theta"`
				PartialRotaryFactor float32 `json:"partial_rotary_factor"`
				MRoPESection        []int32 `json:"mrope_section"`
			} `json:"rope_parameters"`
		} `json:"text_config"`
		ImageStartTokenID int32 `json:"image_start_token_id"`
		ImageEndTokenID   int32 `json:"image_end_token_id"`
		ImageTokenID      int32 `json:"image_token_id"`
	}

	if err := manifest.ReadConfigJSON("vision_language_encoder/config.json", &rawCfg); err != nil {
		return fmt.Errorf("config: %w", err)
	}

	cfg := &VisionLanguageConfig{
		HiddenSize:          rawCfg.TextConfig.HiddenSize,
		NumHiddenLayers:     rawCfg.TextConfig.NumHiddenLayers,
		IntermediateSize:    rawCfg.TextConfig.IntermediateSize,
		NumAttentionHeads:   rawCfg.TextConfig.NumAttentionHeads,
		NumKeyValueHeads:    rawCfg.TextConfig.NumKeyValueHeads,
		VocabSize:           rawCfg.TextConfig.VocabSize,
		RMSNormEps:          rawCfg.TextConfig.RMSNormEps,
		VisionVocabSize:     rawCfg.TextConfig.VisionVocabSize,
		RopeTheta:           rawCfg.TextConfig.RopeParameters.RopeTheta,
		PartialRotaryFactor: rawCfg.TextConfig.RopeParameters.PartialRotaryFactor,
		MRoPESection:        rawCfg.TextConfig.RopeParameters.MRoPESection,
		ImageStartTokenID:   rawCfg.ImageStartTokenID,
		ImageEndTokenID:     rawCfg.ImageEndTokenID,
		ImageTokenID:        rawCfg.ImageTokenID,
	}

	cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	m.Config = cfg

	// Pre-allocate layers
	m.Layers = make([]*GLMBlock, cfg.NumHiddenLayers)

	// Load weights
	weights, err := imagegen.LoadWeightsFromManifest(manifest, "vision_language_encoder")
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

	m.initComputedFields()
	fmt.Printf("✓ [%d layers]\n", cfg.NumHiddenLayers)
	return nil
}

// LoadFromPath loads the vision-language encoder from a directory path
func (m *VisionLanguageEncoder) LoadFromPath(path string) error {
	fmt.Print("  Loading vision-language encoder... ")

	// Load config
	var rawCfg struct {
		TextConfig struct {
			HiddenSize        int32   `json:"hidden_size"`
			NumHiddenLayers   int32   `json:"num_hidden_layers"`
			IntermediateSize  int32   `json:"intermediate_size"`
			NumAttentionHeads int32   `json:"num_attention_heads"`
			NumKeyValueHeads  int32   `json:"num_key_value_heads"`
			VocabSize         int32   `json:"vocab_size"`
			RMSNormEps        float32 `json:"rms_norm_eps"`
			VisionVocabSize   int32   `json:"vision_vocab_size"`
			RopeParameters    struct {
				RopeTheta           float32 `json:"rope_theta"`
				PartialRotaryFactor float32 `json:"partial_rotary_factor"`
				MRoPESection        []int32 `json:"mrope_section"`
			} `json:"rope_parameters"`
		} `json:"text_config"`
		ImageStartTokenID int32 `json:"image_start_token_id"`
		ImageEndTokenID   int32 `json:"image_end_token_id"`
		ImageTokenID      int32 `json:"image_token_id"`
	}

	configPath := filepath.Join(path, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("read config: %w", err)
	}
	if err := json.Unmarshal(data, &rawCfg); err != nil {
		return fmt.Errorf("parse config: %w", err)
	}

	cfg := &VisionLanguageConfig{
		HiddenSize:          rawCfg.TextConfig.HiddenSize,
		NumHiddenLayers:     rawCfg.TextConfig.NumHiddenLayers,
		IntermediateSize:    rawCfg.TextConfig.IntermediateSize,
		NumAttentionHeads:   rawCfg.TextConfig.NumAttentionHeads,
		NumKeyValueHeads:    rawCfg.TextConfig.NumKeyValueHeads,
		VocabSize:           rawCfg.TextConfig.VocabSize,
		RMSNormEps:          rawCfg.TextConfig.RMSNormEps,
		VisionVocabSize:     rawCfg.TextConfig.VisionVocabSize,
		RopeTheta:           rawCfg.TextConfig.RopeParameters.RopeTheta,
		PartialRotaryFactor: rawCfg.TextConfig.RopeParameters.PartialRotaryFactor,
		MRoPESection:        rawCfg.TextConfig.RopeParameters.MRoPESection,
		ImageStartTokenID:   rawCfg.ImageStartTokenID,
		ImageEndTokenID:     rawCfg.ImageEndTokenID,
		ImageTokenID:        rawCfg.ImageTokenID,
	}

	cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	m.Config = cfg

	// Pre-allocate layers
	m.Layers = make([]*GLMBlock, cfg.NumHiddenLayers)

	// Load weights
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

	m.initComputedFields()
	fmt.Printf("✓ [%d layers]\n", cfg.NumHiddenLayers)
	return nil
}

func (m *VisionLanguageEncoder) initComputedFields() {
	cfg := m.Config
	for _, block := range m.Layers {
		block.SelfAttn.NHeads = cfg.NumAttentionHeads
		block.SelfAttn.NKVHeads = cfg.NumKeyValueHeads
		block.SelfAttn.HeadDim = cfg.HeadDim
		block.SelfAttn.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
		block.SelfAttn.PartialRotary = cfg.PartialRotaryFactor
		block.SelfAttn.RopeTheta = cfg.RopeTheta
		block.SelfAttn.MRoPESection = cfg.MRoPESection

		// Set norm eps
		block.InputLayerNorm.Eps = cfg.RMSNormEps
		block.PostSelfAttnNorm.Eps = cfg.RMSNormEps
		block.PostAttnLayerNorm.Eps = cfg.RMSNormEps
		block.PostMLPLayerNorm.Eps = cfg.RMSNormEps
	}
	m.FinalNorm.Eps = cfg.RMSNormEps
}

// Generate autoregressively generates visual tokens with KV caching
func (m *VisionLanguageEncoder) Generate(
	prompt string,
	tok *GLMTokenizer,
	maxTokens int32,
	temperature float32,
	topP float32,
	seed int64,
	targetHeight, targetWidth int32,
	progressFn func(int),
) *mlx.Array {
	cfg := m.Config

	// Encode prompt with grid tokens using GLM tokenizer
	// Format: {prompt}<sop>{h} {w}<eop><sop>{prev_h} {prev_w}<eop><|dit_token_16384|>
	tokens := tok.EncodeForGeneration(prompt, targetHeight, targetWidth)

	// Calculate grid dimensions for MRoPE position IDs
	factor := int32(32)
	tokenH := targetHeight / factor
	tokenW := targetWidth / factor
	ratio := float64(tokenH) / float64(tokenW)
	prevTokenH := int32(math.Sqrt(ratio) * 16)
	prevTokenW := int32(math.Sqrt(1.0/ratio) * 16)
	prevGridSize := prevTokenH * prevTokenW

	// Create KV cache for all layers
	cache := NewARCache(cfg.NumHiddenLayers)
	defer cache.Free()

	// ===== PREFILL PHASE =====
	// Process entire prompt at once, populate cache
	promptLen := int32(len(tokens))
	tokenArr := mlx.NewArrayInt32(tokens, []int32{1, promptLen})
	h := m.EmbedTokens.Forward(tokenArr)
	tokenArr.Free()

	mlx.Eval(h)

	// Compute position IDs for prefill (text tokens use same position for all dims)
	prefillPositions := make([][]int32, 3)
	for dim := 0; dim < 3; dim++ {
		prefillPositions[dim] = make([]int32, promptLen)
		for i := int32(0); i < promptLen; i++ {
			prefillPositions[dim][i] = i
		}
	}

	// Forward through layers (prefill)
	for i, layer := range m.Layers {
		oldH := h
		h = layer.ForwardWithCache(h, promptLen, 0, cfg.RMSNormEps, cache.Layers[i], prefillPositions)
		if i > 0 {
			oldH.Free()
		}
	}
	// Eval h and cache arrays together so cache is materialized
	evalArgs := []*mlx.Array{h}
	for _, lc := range cache.Layers {
		evalArgs = append(evalArgs, lc.State()...)
	}
	mlx.Eval(evalArgs...)

	// Final norm and get logits for last position
	preNormH := h
	h = m.FinalNorm.Forward(h, cfg.RMSNormEps)
	preNormH.Free()

	lastH := mlx.Slice(h, []int32{0, promptLen - 1, 0}, []int32{1, promptLen, cfg.HiddenSize})
	h.Free()
	lastH = mlx.Reshape(lastH, 1, cfg.HiddenSize)
	logits := mlx.Matmul(lastH, mlx.Transpose(m.LMHead, 1, 0))
	lastH.Free()

	// Sample first token
	var sampleCounter int64 = 0
	nextToken := sampleVisualToken(logits, temperature, topP, cfg, seed, &sampleCounter)
	logits.Free()

	// AR generation loop with caching
	// Visual tokens are stored as VQ codebook indices [0, 16383]
	// The LM head outputs indices [0, 16511] where:
	// - [0, 16383] are VQ codes
	// - 16384 is BOS
	// - 16385 is EOS
	visualTokens := make([]int32, 0, maxTokens)
	posOffset := promptLen
	visualTokenIdx := int32(0) // Index within visual token sequence for grid position calculation

	// Preallocate slice for old cache state to reuse
	oldCacheState := make([]*mlx.Array, 0, len(m.Layers)*2)

	for i := int32(0); i < maxTokens; i++ {
		if progressFn != nil {
			progressFn(int(i))
		}

		// Check for end token (EOS = 16385)
		if nextToken == cfg.ImageEndTokenID {
			break
		}

		// Skip BOS token (16384), only store actual VQ codes [0, 16383]
		if nextToken == cfg.ImageStartTokenID {
			// BOS token - skip storing but continue generation
		} else if nextToken < cfg.ImageStartTokenID {
			// This is an actual VQ code [0, 16383] - store it
			visualTokens = append(visualTokens, nextToken)
		}
		// Tokens >= 16386 are other special tokens, skip them

		// ===== DECODE PHASE =====
		// Save old cache state before forward (to free after eval)
		oldCacheState = oldCacheState[:0]
		for _, lc := range cache.Layers {
			oldCacheState = append(oldCacheState, lc.State()...)
		}

		// Only process the new token, use cached K,V
		tokenArr := mlx.NewArrayInt32([]int32{nextToken}, []int32{1, 1})
		h := m.EmbedTokens.Forward(tokenArr)
		tokenArr.Free()

		// Compute MRoPE position IDs for this visual token
		// Visual tokens are arranged in two grids: prev grid then target grid
		// Position dimensions: [temporal, height, width]
		decodePositions := computeVisualTokenPositions(
			visualTokenIdx, posOffset, promptLen,
			prevTokenH, prevTokenW, prevGridSize,
			tokenH, tokenW,
		)

		// Forward through layers (decode with cache)
		for j, layer := range m.Layers {
			oldH := h
			h = layer.ForwardWithCache(h, 1, posOffset, cfg.RMSNormEps, cache.Layers[j], decodePositions)
			if j > 0 { // Don't free the embedding on first layer
				oldH.Free()
			}
		}

		// Eval h and new cache state
		newCacheState := make([]*mlx.Array, 0, len(m.Layers)*2)
		for _, lc := range cache.Layers {
			newCacheState = append(newCacheState, lc.State()...)
		}
		mlx.Eval(append([]*mlx.Array{h}, newCacheState...)...)

		// Free old cache state (now that new state is evaluated)
		for _, arr := range oldCacheState {
			if arr != nil {
				arr.Free()
			}
		}

		// Final norm
		preNormH := h
		h = m.FinalNorm.Forward(h, cfg.RMSNormEps)
		preNormH.Free()

		// Get logits (h is already [1, 1, hidden_size])
		h = mlx.Reshape(h, 1, cfg.HiddenSize)
		logits := mlx.Matmul(h, mlx.Transpose(m.LMHead, 1, 0))
		h.Free()

		// Sample next token
		nextToken = sampleVisualToken(logits, temperature, topP, cfg, seed, &sampleCounter)
		logits.Free()

		posOffset++
		visualTokenIdx++

		// Periodically clear cache to release intermediate memory
		if i%256 == 0 {
			mlx.ClearCache()
		}
	}

	if len(visualTokens) == 0 {
		// Return at least one token to avoid empty tensor issues
		visualTokens = append(visualTokens, 0)
	}

	return mlx.NewArrayInt32(visualTokens, []int32{1, int32(len(visualTokens))})
}

// computeVisualTokenPositions computes MRoPE position IDs for a visual token
// Returns [3][1] position IDs for temporal, height, and width dimensions
//
// MRoPE position encoding for GLM-Image visual tokens:
// - temporal: CONSTANT within each grid (= decode_pos at grid start)
// - height: decode_pos + row index within grid
// - width: decode_pos + column index within grid
//
// Between grids, decode_pos advances by max(grid_h, grid_w) to ensure
// sufficient positional separation.
func computeVisualTokenPositions(
	visualIdx int32, absPos int32, promptLen int32,
	prevH, prevW, prevSize int32,
	targetH, targetW int32,
) [][]int32 {
	positions := make([][]int32, 3)
	for dim := 0; dim < 3; dim++ {
		positions[dim] = make([]int32, 1)
	}

	// First grid (prev grid) starts at decode_pos = promptLen
	prevGridDecodePos := promptLen

	// Second grid (target grid) starts after first grid
	// next_pos = prev_decode_pos + max(prevH, prevW)
	maxPrev := prevH
	if prevW > maxPrev {
		maxPrev = prevW
	}
	targetGridDecodePos := prevGridDecodePos + maxPrev

	// Compute position IDs based on which grid the token is in
	if visualIdx < prevSize {
		// Token is in the prev grid (prev_token_h × prev_token_w)
		row := visualIdx / prevW
		col := visualIdx % prevW

		// temporal is CONSTANT for all tokens in this grid
		positions[0][0] = prevGridDecodePos
		// height and width are relative to grid's decode_pos
		positions[1][0] = prevGridDecodePos + row
		positions[2][0] = prevGridDecodePos + col
	} else {
		// Token is in the target grid (token_h × token_w)
		targetIdx := visualIdx - prevSize
		row := targetIdx / targetW
		col := targetIdx % targetW

		// temporal is CONSTANT for all tokens in this grid
		positions[0][0] = targetGridDecodePos
		// height and width are relative to grid's decode_pos
		positions[1][0] = targetGridDecodePos + row
		positions[2][0] = targetGridDecodePos + col
	}

	_ = targetH // Used for documentation clarity
	_ = absPos  // No longer used - kept for API compatibility
	return positions
}

// sampleVisualToken samples from the visual vocabulary using top-p (nucleus) sampling
// Note: For GLM-Image, greedy decoding is not allowed as it may cause repetitive outputs
// Returns a visual token ID in range [0, 16511] which directly indexes into the embedding table
// sampleCounter is incremented for each call to ensure different random values
func sampleVisualToken(logits *mlx.Array, temperature float32, topP float32, cfg *VisionLanguageConfig, seed int64, sampleCounter *int64) int32 {
	// The LMHead outputs logits for visual tokens only (shape [1, 16512])
	// Output index directly corresponds to vocab ID [0, 16511]
	// No offset needed - the visual tokens are at vocab IDs [0, 16511]
	visualLogits := logits

	// Apply temperature
	if temperature != 1.0 && temperature > 0 {
		visualLogits = mlx.DivScalar(visualLogits, temperature)
	}

	// Apply softmax to get probabilities
	probs := mlx.Softmax(visualLogits, -1)
	mlx.Eval(probs)

	// Get the sampled index using top-p sampling
	// This directly gives us the vocab ID in [0, 16511]
	// Special tokens: 16384 = BOS, 16385 = EOS
	// Use seed + counter for reproducible but different random values
	effectiveSeed := seed + *sampleCounter
	*sampleCounter++
	return sampleTopP(probs, topP, effectiveSeed)
}

// sampleTopP implements nucleus (top-p) sampling
// probs: [1, vocab_size] probability distribution
// topP: cumulative probability threshold (e.g., 0.75)
// seed: random seed for reproducible sampling
func sampleTopP(probs *mlx.Array, topP float32, seed int64) int32 {
	// Negate probs for descending sort (Argsort only does ascending)
	negProbs := mlx.MulScalar(probs, -1)
	sortedIndices := mlx.Argsort(negProbs, -1)
	sortedProbs := mlx.TakeAlongAxis(probs, sortedIndices, -1)
	cumProbs := mlx.Cumsum(sortedProbs, -1)
	mlx.Eval(sortedIndices, sortedProbs, cumProbs)

	// Find cutoff index where cumulative probability exceeds topP
	probsData := sortedProbs.Data()
	cumProbsData := cumProbs.Data()
	indicesData := sortedIndices.DataInt32()

	// Calculate cutoff and renormalize
	var cutoffIdx int
	var totalProb float32
	for i, cp := range cumProbsData {
		totalProb += probsData[i]
		if cp >= topP {
			cutoffIdx = i + 1 // Include this token
			break
		}
	}
	if cutoffIdx == 0 {
		cutoffIdx = len(probsData) // Use all tokens if topP is very high
	}

	// Sample from the truncated distribution
	// Renormalize the truncated probabilities
	truncatedProbs := make([]float32, cutoffIdx)
	for i := 0; i < cutoffIdx; i++ {
		truncatedProbs[i] = probsData[i] / totalProb
	}

	// Sample using random number with provided seed for reproducibility
	r := mlx.RandomUniform([]int32{1}, uint64(seed))
	mlx.Eval(r)
	randVal := r.Data()[0]

	// Find the sampled token
	var cumulative float32
	for i := 0; i < cutoffIdx; i++ {
		cumulative += truncatedProbs[i]
		if randVal < cumulative {
			return indicesData[i]
		}
	}

	// Fallback to the last token in truncated set
	return indicesData[cutoffIdx-1]
}

// Forward for GLMBlock
func (b *GLMBlock) Forward(x *mlx.Array, seqLen int32, eps float32) *mlx.Array {
	return b.ForwardWithCache(x, seqLen, 0, eps, nil, nil)
}

// ForwardWithCache performs block forward with optional KV caching and MRoPE
// positionIDs: [3][L] - position indices for MRoPE (nil = use sequential positions)
func (b *GLMBlock) ForwardWithCache(x *mlx.Array, seqLen int32, posOffset int32, eps float32, kvcache cache.Cache, positionIDs [][]int32) *mlx.Array {
	// Pre-attention norm
	normed := b.InputLayerNorm.Forward(x, eps)

	// Self-attention with RoPE/MRoPE and cache
	attnOut := b.SelfAttn.ForwardWithCache(normed, seqLen, posOffset, kvcache, positionIDs)

	// Post-attention norm (GLM-4 style)
	attnOut = b.PostSelfAttnNorm.Forward(attnOut, eps)

	// Residual connection
	x = mlx.Add(x, attnOut)

	// Post-attention layer norm
	normed = b.PostAttnLayerNorm.Forward(x, eps)

	// MLP
	mlpOut := b.MLP.Forward(normed)

	// Post-MLP norm
	mlpOut = b.PostMLPLayerNorm.Forward(mlpOut, eps)

	// Residual connection
	x = mlx.Add(x, mlpOut)

	return x
}

// Forward for GLMAttention (without cache - used for prefill)
func (attn *GLMAttention) Forward(x *mlx.Array, seqLen int32) *mlx.Array {
	return attn.ForwardWithCache(x, seqLen, 0, nil, nil)
}

// ForwardWithCache performs attention with optional KV caching and MRoPE
// posOffset is the position offset for RoPE (0 for prefill, cached_len for decode)
// positionIDs: [3][L] - if nil, uses sequential positions for all dims (text mode)
// kvcache is updated in-place if provided
func (attn *GLMAttention) ForwardWithCache(x *mlx.Array, seqLen int32, posOffset int32, kvcache cache.Cache, positionIDs [][]int32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]

	// Q, K, V projections
	q := mlx.Matmul(x, mlx.Transpose(attn.QProj, 1, 0))
	k := mlx.Matmul(x, mlx.Transpose(attn.KProj, 1, 0))
	v := mlx.Matmul(x, mlx.Transpose(attn.VProj, 1, 0))

	// Add biases
	if attn.QBias != nil {
		q = mlx.Add(q, attn.QBias)
	}
	if attn.KBias != nil {
		k = mlx.Add(k, attn.KBias)
	}
	if attn.VBias != nil {
		v = mlx.Add(v, attn.VBias)
	}

	// Reshape to [B, L, nheads, head_dim]
	q = mlx.Reshape(q, B, L, attn.NHeads, attn.HeadDim)
	k = mlx.Reshape(k, B, L, attn.NKVHeads, attn.HeadDim)
	v = mlx.Reshape(v, B, L, attn.NKVHeads, attn.HeadDim)

	// Apply partial RoPE or MRoPE
	rotaryDim := int32(float32(attn.HeadDim) * attn.PartialRotary)
	if len(attn.MRoPESection) == 3 && positionIDs != nil {
		// Use MRoPE with explicit position IDs
		q = applyPartialMRoPE(q, positionIDs, rotaryDim, attn.RopeTheta, attn.MRoPESection)
		k = applyPartialMRoPE(k, positionIDs, rotaryDim, attn.RopeTheta, attn.MRoPESection)
	} else if len(attn.MRoPESection) == 3 {
		// Use MRoPE with sequential positions (same for all dims - text mode)
		seqPositions := make([][]int32, 3)
		for dim := 0; dim < 3; dim++ {
			seqPositions[dim] = make([]int32, L)
			for i := int32(0); i < L; i++ {
				seqPositions[dim][i] = i + posOffset
			}
		}
		q = applyPartialMRoPE(q, seqPositions, rotaryDim, attn.RopeTheta, attn.MRoPESection)
		k = applyPartialMRoPE(k, seqPositions, rotaryDim, attn.RopeTheta, attn.MRoPESection)
	} else {
		// Fallback to standard RoPE
		q = applyPartialRoPEWithOffset(q, L, posOffset, rotaryDim, attn.RopeTheta)
		k = applyPartialRoPEWithOffset(k, L, posOffset, rotaryDim, attn.RopeTheta)
	}

	// Transpose to [B, nheads, L, head_dim]
	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	// Update cache and get full K, V for attention
	if kvcache != nil {
		k, v = kvcache.Update(k, v, int(L))
	}

	// Repeat KV for GQA
	kExpanded := k
	vExpanded := v
	if attn.NKVHeads < attn.NHeads {
		repeats := attn.NHeads / attn.NKVHeads
		kExpanded = repeatKV(k, repeats)
		vExpanded = repeatKV(v, repeats)
	}

	// Scaled dot-product attention with causal mask
	out := mlx.ScaledDotProductAttention(q, kExpanded, vExpanded, attn.Scale, true)

	// Transpose back [B, nheads, L, head_dim] -> [B, L, nheads, head_dim]
	out = mlx.Transpose(out, 0, 2, 1, 3)
	// Reshape to [B, L, hidden_size]
	out = mlx.Reshape(out, B, L, attn.NHeads*attn.HeadDim)

	// Output projection
	out = mlx.Matmul(out, mlx.Transpose(attn.OProj, 1, 0))

	return out
}

// applyPartialRoPE applies RoPE to only the first rotaryDim dimensions
func applyPartialRoPE(x *mlx.Array, seqLen int32, rotaryDim int32, theta float32) *mlx.Array {
	return applyPartialRoPEWithOffset(x, seqLen, 0, rotaryDim, theta)
}

// applyPartialRoPEWithOffset applies RoPE with a position offset
func applyPartialRoPEWithOffset(x *mlx.Array, seqLen int32, posOffset int32, rotaryDim int32, theta float32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	H := shape[2]
	D := shape[3]

	if rotaryDim <= 0 || rotaryDim > D {
		rotaryDim = D
	}

	// Split into rotary and pass-through parts
	xRot := mlx.Slice(x, []int32{0, 0, 0, 0}, []int32{B, L, H, rotaryDim})
	xPass := mlx.Slice(x, []int32{0, 0, 0, rotaryDim}, []int32{B, L, H, D})

	// Apply RoPE to rotary part with position offset
	xRot = applyRoPEWithOffset(xRot, L, posOffset, theta)

	// Concatenate back
	return mlx.Concatenate([]*mlx.Array{xRot, xPass}, 3)
}

// applyPartialMRoPE applies Multi-dimensional RoPE (MRoPE) to the first rotaryDim dimensions
// positionIDs: [3, L] - position indices for each dimension (temporal, height, width)
// mrope_section: [8, 12, 12] - frequency pairs per dimension
// For text tokens: all 3 dimensions have the same sequential position
// For image tokens: temporal=seq_idx, height=row, width=col
func applyPartialMRoPE(x *mlx.Array, positionIDs [][]int32, rotaryDim int32, theta float32, mropeSection []int32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	H := shape[2]
	D := shape[3]

	if rotaryDim <= 0 || rotaryDim > D {
		rotaryDim = D
	}

	// Split into rotary and pass-through parts
	xRot := mlx.Slice(x, []int32{0, 0, 0, 0}, []int32{B, L, H, rotaryDim})
	xPass := mlx.Slice(x, []int32{0, 0, 0, rotaryDim}, []int32{B, L, H, D})

	// Apply MRoPE to rotary part
	xRot = applyMRoPE(xRot, positionIDs, theta, mropeSection)

	// Concatenate back
	return mlx.Concatenate([]*mlx.Array{xRot, xPass}, 3)
}

// applyMRoPE applies multi-dimensional rotary position embedding
// x: [B, L, H, D] where D is the rotary dimension
// positionIDs: [3][L] - positions for temporal, height, width dimensions
// mropeSection: [8, 12, 12] - frequency pairs per dimension
func applyMRoPE(x *mlx.Array, positionIDs [][]int32, theta float32, mropeSection []int32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	H := shape[2]
	D := shape[3]
	half := D / 2

	// Validate mrope_section sums to half (number of frequency pairs)
	var totalPairs int32
	for _, s := range mropeSection {
		totalPairs += s
	}
	if totalPairs != half {
		// Fallback to standard RoPE if section doesn't match
		return applyRoPEWithOffset(x, L, 0, theta)
	}

	// Build angles for each position dimension (matching Python's MRoPE approach)
	// Python: compute freqs for all dims, then apply_mrope selects freq ranges, then duplicate
	// Order: [temporal_8, height_12, width_12] -> duplicate -> [t8, h12, w12, t8, h12, w12]
	angleVals := make([]*mlx.Array, 3)

	freqOffset := int32(0)
	for dim := 0; dim < 3; dim++ {
		numPairs := mropeSection[dim]
		if numPairs == 0 {
			continue
		}

		// Compute inverse frequencies for this section
		// Each dimension uses DIFFERENT frequency ranges:
		// - Temporal: frequencies 0 to section[0]-1
		// - Height: frequencies section[0] to section[0]+section[1]-1
		// - Width: frequencies section[0]+section[1] to sum(section)-1
		freqsArr := make([]float32, numPairs)
		for i := int32(0); i < numPairs; i++ {
			globalIdx := freqOffset + i
			freqsArr[i] = float32(1.0 / math.Pow(float64(theta), float64(2*globalIdx)/float64(D)))
		}
		freqs := mlx.NewArray(freqsArr, []int32{numPairs})

		// Position indices for this dimension
		posArr := make([]float32, L)
		for i := int32(0); i < L; i++ {
			posArr[i] = float32(positionIDs[dim][i])
		}
		pos := mlx.NewArray(posArr, []int32{L})

		// Compute angles: [L, numPairs] = outer(pos, freqs)
		posExpanded := mlx.Reshape(pos, L, 1)
		freqsExpanded := mlx.Reshape(freqs, 1, numPairs)
		angleVals[dim] = mlx.Mul(posExpanded, freqsExpanded)

		freqOffset += numPairs
	}

	// Concatenate all sections: [L, half] = [L, 32]
	allAngles := mlx.Concatenate(angleVals, 1)

	// Duplicate AFTER concatenation: [L, D] = [L, 64]
	// This gives: [temporal_8, height_12, width_12, temporal_8, height_12, width_12]
	allAngles = mlx.Concatenate([]*mlx.Array{allAngles, allAngles}, 1)

	// Compute cos/sin
	allCos := mlx.Cos(allAngles)
	allSin := mlx.Sin(allAngles)

	// Reshape for broadcasting: [1, L, 1, D] to match x [B, L, H, D]
	allCos = mlx.Reshape(allCos, 1, L, 1, D)
	allSin = mlx.Reshape(allSin, 1, L, 1, D)

	// x_rotated = cat([-x_imag, x_real], dim=-1)
	x1 := mlx.Slice(x, []int32{0, 0, 0, 0}, []int32{B, L, H, half})  // x_real
	x2 := mlx.Slice(x, []int32{0, 0, 0, half}, []int32{B, L, H, D})  // x_imag
	x2Neg := mlx.MulScalar(x2, -1)                                   // -x_imag
	xRotated := mlx.Concatenate([]*mlx.Array{x2Neg, x1}, 3)          // [-x_imag, x_real]

	// out = x * cos + x_rotated * sin
	return mlx.Add(mlx.Mul(x, allCos), mlx.Mul(xRotated, allSin))
}

// applyRoPE applies rotary position embedding
func applyRoPE(x *mlx.Array, seqLen int32, theta float32) *mlx.Array {
	return applyRoPEWithOffset(x, seqLen, 0, theta)
}

// applyRoPEWithOffset applies rotary position embedding with position offset
// Uses the split-half approach (matches diffusers GLM-Image with use_real_unbind_dim=-2)
func applyRoPEWithOffset(x *mlx.Array, seqLen int32, posOffset int32, theta float32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	H := shape[2]
	D := shape[3]
	half := D / 2

	// Compute inverse frequencies: 1 / (theta^(2i/d))
	freqsArr := make([]float32, half)
	for i := int32(0); i < half; i++ {
		freqsArr[i] = float32(1.0 / math.Pow(float64(theta), float64(2*i)/float64(D)))
	}
	freqs := mlx.NewArray(freqsArr, []int32{half})

	// Position indices with offset
	posArr := make([]float32, L)
	for i := int32(0); i < L; i++ {
		posArr[i] = float32(i + posOffset)
	}
	pos := mlx.NewArray(posArr, []int32{L})

	// Compute angles: [L, half] = outer(pos, freqs)
	posExpanded := mlx.Reshape(pos, L, 1)
	freqsExpanded := mlx.Reshape(freqs, 1, half)
	angles := mlx.Mul(posExpanded, freqsExpanded)

	// Duplicate angles to match diffusers: cat([angles, angles], dim=-1) -> [L, D]
	anglesDup := mlx.Concatenate([]*mlx.Array{angles, angles}, 1)

	// Cos and sin: [L, 1, D] for broadcasting to [B, L, H, D]
	cosVals := mlx.Cos(anglesDup)
	sinVals := mlx.Sin(anglesDup)
	cosVals = mlx.Reshape(cosVals, L, 1, D)
	sinVals = mlx.Reshape(sinVals, L, 1, D)

	// x_rotated = cat([-x_imag, x_real], dim=-1) where x_real=x[..., :half], x_imag=x[..., half:]
	x1 := mlx.Slice(x, []int32{0, 0, 0, 0}, []int32{B, L, H, half})      // x_real
	x2 := mlx.Slice(x, []int32{0, 0, 0, half}, []int32{B, L, H, D})      // x_imag
	x2Neg := mlx.MulScalar(x2, -1)                                       // -x_imag
	xRotated := mlx.Concatenate([]*mlx.Array{x2Neg, x1}, 3)              // [-x_imag, x_real]

	// out = x * cos + x_rotated * sin
	return mlx.Add(mlx.Mul(x, cosVals), mlx.Mul(xRotated, sinVals))
}

// repeatKV repeats key/value heads for GQA
func repeatKV(x *mlx.Array, repeats int32) *mlx.Array {
	if repeats == 1 {
		return x
	}
	shape := x.Shape()
	// x: [B, nkvheads, L, head_dim]
	x = mlx.ExpandDims(x, 2)
	// x: [B, nkvheads, 1, L, head_dim]
	x = mlx.Tile(x, []int32{1, 1, repeats, 1, 1})
	// x: [B, nkvheads, repeats, L, head_dim]
	return mlx.Reshape(x, shape[0], shape[1]*repeats, shape[2], shape[3])
}

// Forward for GLMMLP (fused gate_up SwiGLU)
func (m *GLMMLP) Forward(x *mlx.Array) *mlx.Array {
	// gate_up_proj outputs [gate, up] concatenated
	gateUp := mlx.Matmul(x, mlx.Transpose(m.GateUpProj, 1, 0))

	shape := gateUp.Shape()
	halfDim := shape[len(shape)-1] / 2

	// Split into gate and up
	gate := mlx.Slice(gateUp, []int32{0, 0, 0}, []int32{shape[0], shape[1], halfDim})
	up := mlx.Slice(gateUp, []int32{0, 0, halfDim}, []int32{shape[0], shape[1], shape[2]})

	// SwiGLU: silu(gate) * up
	gate = mlx.SiLU(gate)
	h := mlx.Mul(gate, up)

	// Down projection
	return mlx.Matmul(h, mlx.Transpose(m.DownProj, 1, 0))
}

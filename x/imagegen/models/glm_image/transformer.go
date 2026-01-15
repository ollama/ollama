//go:build mlx

package glm_image

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

var debugOnce = true

// DiffusionLayerKVCache holds KV cache for a single diffusion layer
type DiffusionLayerKVCache struct {
	Keys   *mlx.Array
	Values *mlx.Array
	Mode   string // "write", "read", "skip", or ""
}

// Store adds K,V to the cache
func (c *DiffusionLayerKVCache) Store(k, v *mlx.Array) {
	if c.Keys == nil {
		c.Keys = k
		c.Values = v
	} else {
		oldK, oldV := c.Keys, c.Values
		c.Keys = mlx.Concatenate([]*mlx.Array{oldK, k}, 1)
		c.Values = mlx.Concatenate([]*mlx.Array{oldV, v}, 1)
		oldK.Free()
		oldV.Free()
	}
}

// Get returns cached K,V concatenated with new K,V
func (c *DiffusionLayerKVCache) Get(k, v *mlx.Array) (*mlx.Array, *mlx.Array) {
	// Expand cache if batch size differs
	kCache, vCache := c.Keys, c.Values
	if c.Keys.Shape()[0] != k.Shape()[0] {
		kCache = mlx.BroadcastTo(c.Keys, []int32{k.Shape()[0], c.Keys.Shape()[1], c.Keys.Shape()[2], c.Keys.Shape()[3]})
		vCache = mlx.BroadcastTo(c.Values, []int32{v.Shape()[0], c.Values.Shape()[1], c.Values.Shape()[2], c.Values.Shape()[3]})
	}
	return mlx.Concatenate([]*mlx.Array{kCache, k}, 1),
		mlx.Concatenate([]*mlx.Array{vCache, v}, 1)
}

// Clear releases cached tensors
func (c *DiffusionLayerKVCache) Clear() {
	if c.Keys != nil {
		c.Keys.Free()
		c.Keys = nil
	}
	if c.Values != nil {
		c.Values.Free()
		c.Values = nil
	}
	c.Mode = ""
}

// DiffusionKVCache holds KV caches for all diffusion layers
type DiffusionKVCache struct {
	Layers []*DiffusionLayerKVCache
}

// NewDiffusionKVCache creates a cache for the given number of layers
func NewDiffusionKVCache(numLayers int32) *DiffusionKVCache {
	layers := make([]*DiffusionLayerKVCache, numLayers)
	for i := range layers {
		layers[i] = &DiffusionLayerKVCache{}
	}
	return &DiffusionKVCache{Layers: layers}
}

// SetMode sets the cache mode for all layers
func (c *DiffusionKVCache) SetMode(mode string) {
	for _, layer := range c.Layers {
		layer.Mode = mode
	}
}

// Clear releases all cached tensors
func (c *DiffusionKVCache) Clear() {
	for _, layer := range c.Layers {
		layer.Clear()
	}
}

// DiffusionConfig holds diffusion transformer configuration
type DiffusionConfig struct {
	AttentionHeadDim        int32   `json:"attention_head_dim"`                  // 128
	NumAttentionHeads       int32   `json:"num_attention_heads"`                 // 32
	NumLayers               int32   `json:"num_layers"`                          // 30
	InChannels              int32   `json:"in_channels"`                         // 16
	OutChannels             int32   `json:"out_channels"`                        // 16
	PatchSize               int32   `json:"patch_size"`                          // 2
	TextEmbedDim            int32   `json:"text_embed_dim"`                      // 1472 (T5 output)
	TimeEmbedDim            int32   `json:"time_embed_dim"`                      // 512
	ConditionDim            int32   `json:"condition_dim"`                       // 256
	PriorVQCodebookSize     int32   `json:"prior_vq_quantizer_codebook_size"`    // 16384
	RopeTheta               float32 `json:"rope_theta"`                          // 10000.0

	// Computed
	HiddenDim int32 // num_heads * head_dim = 4096
}

// DiffusionTransformer is the 7B diffusion decoder
type DiffusionTransformer struct {
	Config *DiffusionConfig

	// Prior token embedding (VQ codebook)
	PriorTokenEmbedding *nn.Embedding `weight:"prior_token_embedding"`

	// Projectors
	PriorProjector  *DiTMLPSiLU `weight:"prior_projector"`
	ImageProjector  *mlx.Array `weight:"image_projector.proj.weight"`
	ImageProjectorBias *mlx.Array `weight:"image_projector.proj.bias"`
	GlyphProjector  *DiTMLP `weight:"glyph_projector"`

	// Time + condition embedding
	TimeProj        *mlx.Array `weight:"time_condition_embed.timestep_embedder.linear_1.weight"`
	TimeProjBias    *mlx.Array `weight:"time_condition_embed.timestep_embedder.linear_1.bias"`
	TimeProj2       *mlx.Array `weight:"time_condition_embed.timestep_embedder.linear_2.weight"`
	TimeProjBias2   *mlx.Array `weight:"time_condition_embed.timestep_embedder.linear_2.bias"`
	ConditionProj   *mlx.Array `weight:"time_condition_embed.condition_embedder.linear_1.weight"`
	ConditionProjBias *mlx.Array `weight:"time_condition_embed.condition_embedder.linear_1.bias"`
	ConditionProj2  *mlx.Array `weight:"time_condition_embed.condition_embedder.linear_2.weight"`
	ConditionProjBias2 *mlx.Array `weight:"time_condition_embed.condition_embedder.linear_2.bias"`

	// Transformer blocks (single-stream)
	Blocks []*DiTBlock `weight:"transformer_blocks"`

	// Output
	NormOutLinear *mlx.Array `weight:"norm_out.linear.weight"`
	NormOutLinearBias *mlx.Array `weight:"norm_out.linear.bias"`
	ProjOut       *mlx.Array `weight:"proj_out.weight"`
	ProjOutBias   *mlx.Array `weight:"proj_out.bias"`
}

// DiTMLP is a simple MLP with GELU activation (used for glyph_projector)
type DiTMLP struct {
	Linear1 *mlx.Array `weight:"net.0.proj.weight"`
	Bias1   *mlx.Array `weight:"net.0.proj.bias"`
	Linear2 *mlx.Array `weight:"net.2.weight"`
	Bias2   *mlx.Array `weight:"net.2.bias"`
}

// DiTMLPSiLU is an MLP with SiLU activation (used for prior_projector)
type DiTMLPSiLU struct {
	Linear1 *mlx.Array `weight:"net.0.proj.weight"`
	Bias1   *mlx.Array `weight:"net.0.proj.bias"`
	Linear2 *mlx.Array `weight:"net.2.weight"`
	Bias2   *mlx.Array `weight:"net.2.bias"`
}

// DiTBlock is a single-stream transformer block
type DiTBlock struct {
	// Single attention (no cross-attention)
	Attn1 *DiTAttention `weight:"attn1"`

	// FFN
	FF *DiTFeedForward `weight:"ff"`

	// AdaLN modulation
	Norm1Linear *mlx.Array `weight:"norm1.linear.weight"`
	Norm1LinearBias *mlx.Array `weight:"norm1.linear.bias"`
}

// DiTAttention implements self-attention for DiT
type DiTAttention struct {
	ToQ   *mlx.Array `weight:"to_q.weight"`
	ToK   *mlx.Array `weight:"to_k.weight"`
	ToV   *mlx.Array `weight:"to_v.weight"`
	ToOut *mlx.Array `weight:"to_out.0.weight"`
	ToOutBias *mlx.Array `weight:"to_out.0.bias"`

	// Have biases
	QBias *mlx.Array `weight:"to_q.bias"`
	KBias *mlx.Array `weight:"to_k.bias"`
	VBias *mlx.Array `weight:"to_v.bias"`

	NHeads  int32
	HeadDim int32
	Scale   float32
}

// DiTFeedForward is the FFN in DiT blocks
type DiTFeedForward struct {
	Linear1 *mlx.Array `weight:"net.0.proj.weight"`
	Bias1   *mlx.Array `weight:"net.0.proj.bias"`
	Linear2 *mlx.Array `weight:"net.2.weight"`
	Bias2   *mlx.Array `weight:"net.2.bias"`
}

// Load loads the diffusion transformer
func (m *DiffusionTransformer) Load(manifest *imagegen.ModelManifest) error {
	fmt.Print("  Loading diffusion transformer... ")

	// Load config
	var cfg DiffusionConfig
	if err := manifest.ReadConfigJSON("transformer/config.json", &cfg); err != nil {
		return fmt.Errorf("config: %w", err)
	}
	cfg.HiddenDim = cfg.NumAttentionHeads * cfg.AttentionHeadDim
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000.0 // Default value matching diffusers
	}
	m.Config = &cfg

	// Pre-allocate blocks
	m.Blocks = make([]*DiTBlock, cfg.NumLayers)

	// Load weights
	weights, err := imagegen.LoadWeightsFromManifest(manifest, "transformer")
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
	fmt.Printf("✓ [%d layers]\n", cfg.NumLayers)
	return nil
}

// LoadFromPath loads the diffusion transformer from a directory path
func (m *DiffusionTransformer) LoadFromPath(path string) error {
	fmt.Print("  Loading diffusion transformer... ")

	// Load config
	var cfg DiffusionConfig
	configPath := filepath.Join(path, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("read config: %w", err)
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return fmt.Errorf("parse config: %w", err)
	}
	cfg.HiddenDim = cfg.NumAttentionHeads * cfg.AttentionHeadDim
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000.0 // Default value matching diffusers
	}
	m.Config = &cfg

	// Pre-allocate blocks
	m.Blocks = make([]*DiTBlock, cfg.NumLayers)

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
	fmt.Printf("✓ [%d layers]\n", cfg.NumLayers)
	return nil
}

func (m *DiffusionTransformer) initComputedFields() {
	cfg := m.Config
	for _, block := range m.Blocks {
		block.Attn1.NHeads = cfg.NumAttentionHeads
		block.Attn1.HeadDim = cfg.AttentionHeadDim
		block.Attn1.Scale = float32(1.0 / math.Sqrt(float64(cfg.AttentionHeadDim)))
	}
}

// EmbedPriorTokens converts visual token IDs to embeddings
func (m *DiffusionTransformer) EmbedPriorTokens(tokens *mlx.Array) *mlx.Array {
	// Visual tokens from AR are already relative indices (0 to VisionVocabSize-1)
	// stored in VisionLanguageEncoder.Generate() as: nextToken - ImageStartTokenID
	// VQ codebook has PriorVQCodebookSize entries (16384), indices 0 to 16383
	// Note: VisionVocabSize (16512) > PriorVQCodebookSize (16384), so we clamp
	tokensInt := mlx.AsType(tokens, mlx.DtypeInt32)

	// Clamp to valid range [0, codebook_size-1] to avoid out-of-bounds
	// (VisionVocabSize may be larger than PriorVQCodebookSize)
	codebookIndices := mlx.ClipScalar(tokensInt, 0, float32(m.Config.PriorVQCodebookSize-1), true, true)
	codebookIndices = mlx.AsType(codebookIndices, mlx.DtypeInt32)

	// Lookup in VQ codebook
	embedded := m.PriorTokenEmbedding.Forward(codebookIndices)
	// Project to hidden dim via MLP (uses SiLU activation)
	return m.PriorProjector.Forward(embedded)
}

// ProjectTextEmbeddings projects T5 embeddings for conditioning
func (m *DiffusionTransformer) ProjectTextEmbeddings(textEmbed *mlx.Array) *mlx.Array {
	return m.GlyphProjector.Forward(textEmbed)
}

// Forward runs the diffusion transformer
// imgPatches: [B, L_img, C*p*p] patchified latents
// priorEmbed: [B, L_img, hidden_dim] visual token embeddings (same length as image patches!)
// textCond: [B, L_text, hidden_dim] text condition embeddings
// timestep: [B] timestep values (0-indexed, 0 to num_train_timesteps-1)
// targetSize: [B, 2] target height and width
// cropCoords: [B, 2] crop coordinates (top, left)
// pH, pW: patch grid dimensions
func (m *DiffusionTransformer) Forward(
	imgPatches, priorEmbed, textCond *mlx.Array,
	timestep *mlx.Array,
	targetSize, cropCoords *mlx.Array,
	pH, pW int32,
) *mlx.Array {
	return m.ForwardWithPriorDrop(imgPatches, priorEmbed, textCond, timestep, targetSize, cropCoords, pH, pW, false)
}

// ForwardWithPriorDrop runs the diffusion transformer with optional prior token dropping for CFG.
// priorTokenDrop: when true, zeros out prior embeddings (for unconditional CFG pass)
func (m *DiffusionTransformer) ForwardWithPriorDrop(
	imgPatches, priorEmbed, textCond *mlx.Array,
	timestep *mlx.Array,
	targetSize, cropCoords *mlx.Array,
	pH, pW int32,
	priorTokenDrop bool,
) *mlx.Array {
	cfg := m.Config

	// Project image patches to hidden dim
	imgH := mlx.Matmul(imgPatches, mlx.Transpose(m.ImageProjector, 1, 0))
	if m.ImageProjectorBias != nil {
		imgH = mlx.Add(imgH, m.ImageProjectorBias)
	}

	// Add prior embeddings to image patches (element-wise, NOT concatenation!)
	// This is the key architectural difference from a standard DiT
	// For CFG unconditional pass, zero out prior embeddings (matches diffusers prior_token_drop)
	if priorTokenDrop {
		// Don't add prior embeddings - effectively zeros them out
	} else {
		imgH = mlx.Add(imgH, priorEmbed)
	}

	// Compute timestep + condition embedding
	temb := m.computeTimestepEmbedding(timestep, targetSize, cropCoords)

	// Sequence for attention: [text, image]
	// Text = encoder_hidden_states (glyph embeddings)
	// Image = hidden_states (image patches + prior embeddings)
	textLen := textCond.Shape()[1]
	imgLen := imgH.Shape()[1]

	// Compute 2D RoPE for IMAGE tokens ONLY
	// Text tokens do NOT get RoPE
	rope := ComputeRoPE2D(pH, pW, cfg.AttentionHeadDim, cfg.RopeTheta)

	// Forward through transformer blocks
	// Note: textCond is encoder_hidden_states, imgH is hidden_states
	for _, block := range m.Blocks {
		imgH, textCond = block.ForwardMMDiT(imgH, textCond, temb, cfg.HiddenDim, rope.Cos, rope.Sin)
	}

	// Final norm with modulation (only on image hidden states)
	imgOut := m.applyOutputNorm(imgH, temb)

	// Project to output channels
	output := mlx.Matmul(imgOut, mlx.Transpose(m.ProjOut, 1, 0))
	if m.ProjOutBias != nil {
		output = mlx.Add(output, m.ProjOutBias)
	}

	_ = textLen
	_ = imgLen

	return output
}

// computeTimestepEmbedding computes the timestep + condition embedding
// targetSize: [B, 2] - target height and width
// cropCoords: [B, 2] - crop top and left coordinates
func (m *DiffusionTransformer) computeTimestepEmbedding(timestep, targetSize, cropCoords *mlx.Array) *mlx.Array {
	cfg := m.Config

	// Sinusoidal timestep embedding (flip_sin_to_cos=True, downscale_freq_shift=0)
	halfDim := cfg.TimeEmbedDim / 2
	freqs := make([]float32, halfDim)
	for i := int32(0); i < halfDim; i++ {
		freqs[i] = float32(math.Exp(-math.Log(10000.0) * float64(i) / float64(halfDim)))
	}
	freqsArr := mlx.NewArray(freqs, []int32{halfDim})

	// timestep: [B] -> [B, 1] * [1, halfDim] -> [B, halfDim]
	t := mlx.Reshape(timestep, -1, 1)
	freqsArr = mlx.Reshape(freqsArr, 1, halfDim)
	args := mlx.Mul(t, freqsArr)

	// flip_sin_to_cos: concatenate cos first, then sin
	cosEmb := mlx.Cos(args)
	sinEmb := mlx.Sin(args)
	temb := mlx.Concatenate([]*mlx.Array{cosEmb, sinEmb}, -1) // [B, TimeEmbedDim]

	// Project through TimestepEmbedding MLP: linear1 -> SiLU -> linear2
	temb = mlx.Matmul(temb, mlx.Transpose(m.TimeProj, 1, 0))
	if m.TimeProjBias != nil {
		temb = mlx.Add(temb, m.TimeProjBias)
	}
	temb = mlx.SiLU(temb)
	temb = mlx.Matmul(temb, mlx.Transpose(m.TimeProj2, 1, 0))
	if m.TimeProjBias2 != nil {
		temb = mlx.Add(temb, m.TimeProjBias2)
	}

	// Compute condition embedding from crop_coords and target_size
	// Each is [B, 2] -> sinusoidal embed each value -> [B, 2*condition_dim]
	condEmb := m.computeConditionEmbedding(cropCoords, targetSize)

	// Add condition embedding to timestep embedding
	temb = mlx.Add(temb, condEmb)

	// Apply final SiLU
	temb = mlx.SiLU(temb)

	return temb
}

// computeConditionEmbedding computes sinusoidal embeddings for condition values
func (m *DiffusionTransformer) computeConditionEmbedding(cropCoords, targetSize *mlx.Array) *mlx.Array {
	cfg := m.Config

	// Sinusoidal embedding for each condition value
	halfDim := cfg.ConditionDim / 2
	freqs := make([]float32, halfDim)
	for i := int32(0); i < halfDim; i++ {
		freqs[i] = float32(math.Exp(-math.Log(10000.0) * float64(i) / float64(halfDim)))
	}
	freqsArr := mlx.NewArray(freqs, []int32{halfDim})

	// Flatten crop_coords: [B, 2] -> [B*2]
	cropFlat := mlx.Reshape(cropCoords, -1)
	cropEmb := sinusoidalEmbed(cropFlat, freqsArr, halfDim)
	// Reshape back: [B*2, condDim] -> [B, 2*condDim]
	B := cropCoords.Shape()[0]
	cropEmb = mlx.Reshape(cropEmb, B, 2*cfg.ConditionDim)

	// Same for target_size
	targetFlat := mlx.Reshape(targetSize, -1)
	targetEmb := sinusoidalEmbed(targetFlat, freqsArr, halfDim)
	targetEmb = mlx.Reshape(targetEmb, B, 2*cfg.ConditionDim)

	// Concatenate: [B, 4*condDim] = pooled_projection_dim
	condProj := mlx.Concatenate([]*mlx.Array{cropEmb, targetEmb}, -1)

	// Project through condition embedder MLP: linear1 -> SiLU -> linear2
	condEmb := mlx.Matmul(condProj, mlx.Transpose(m.ConditionProj, 1, 0))
	if m.ConditionProjBias != nil {
		condEmb = mlx.Add(condEmb, m.ConditionProjBias)
	}
	condEmb = mlx.SiLU(condEmb)
	condEmb = mlx.Matmul(condEmb, mlx.Transpose(m.ConditionProj2, 1, 0))
	if m.ConditionProjBias2 != nil {
		condEmb = mlx.Add(condEmb, m.ConditionProjBias2)
	}

	return condEmb
}

// sinusoidalEmbed computes sinusoidal embeddings for a 1D array of values
func sinusoidalEmbed(x *mlx.Array, freqs *mlx.Array, halfDim int32) *mlx.Array {
	// x: [N] -> [N, 1]
	x = mlx.Reshape(x, -1, 1)
	// freqs: [halfDim] -> [1, halfDim]
	freqs = mlx.Reshape(freqs, 1, halfDim)
	// args: [N, halfDim]
	args := mlx.Mul(x, freqs)

	// flip_sin_to_cos: cos first, then sin
	cosEmb := mlx.Cos(args)
	sinEmb := mlx.Sin(args)
	return mlx.Concatenate([]*mlx.Array{cosEmb, sinEmb}, -1) // [N, 2*halfDim]
}

// applyOutputNorm applies the final norm with AdaLN modulation
func (m *DiffusionTransformer) applyOutputNorm(x *mlx.Array, temb *mlx.Array) *mlx.Array {
	// Compute modulation parameters from temb
	modParams := mlx.Matmul(temb, mlx.Transpose(m.NormOutLinear, 1, 0))
	if m.NormOutLinearBias != nil {
		modParams = mlx.Add(modParams, m.NormOutLinearBias)
	}

	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	D := shape[2]

	// Split into scale and shift (each is D-dimensional)
	// IMPORTANT: diffusers does "scale, shift = chunk(2)" so scale comes FIRST
	halfDim := D
	modParams = mlx.Reshape(modParams, B, 1, -1)
	// Assuming modParams has 2*D dimensions for scale and shift
	modDim := modParams.Shape()[2]
	if modDim >= 2*halfDim {
		scale := mlx.Slice(modParams, []int32{0, 0, 0}, []int32{B, 1, halfDim})
		shift := mlx.Slice(modParams, []int32{0, 0, halfDim}, []int32{B, 1, 2 * halfDim})

		// Apply LayerNorm then modulate
		// LN(x) * (1 + scale) + shift
		xNorm := layerNorm(x)
		xNorm = mlx.Mul(xNorm, mlx.AddScalar(scale, 1.0))
		xNorm = mlx.Add(xNorm, shift)
		return xNorm
	}

	// Fallback: just apply layer norm
	_ = L
	return layerNorm(x)
}

// layerNorm applies layer normalization
// Uses eps=1e-5 to match diffusers GlmImageAdaLayerNormZero
func layerNorm(x *mlx.Array) *mlx.Array {
	eps := float32(1e-5)
	mean := mlx.Mean(x, -1, true)
	x = mlx.Sub(x, mean)
	variance := mlx.Mean(mlx.Square(x), -1, true)
	return mlx.Div(x, mlx.Sqrt(mlx.AddScalar(variance, eps)))
}

// ForwardMMDiT implements the MMDiT-style transformer block for GLM-Image
// hiddenStates: image tokens [B, L_img, D]
// encoderHiddenStates: text tokens [B, L_text, D]
// RoPE is applied only to image tokens
// Returns updated (hiddenStates, encoderHiddenStates)
func (b *DiTBlock) ForwardMMDiT(
	hiddenStates, encoderHiddenStates *mlx.Array,
	temb *mlx.Array,
	hiddenDim int32,
	cos, sin *mlx.Array,
) (*mlx.Array, *mlx.Array) {
	shape := hiddenStates.Shape()
	B := shape[0]
	imgSeqLen := shape[1]
	textSeqLen := encoderHiddenStates.Shape()[1]

	// === 1. Timestep conditioning (AdaLN) ===
	// norm1 produces 12 modulation parameters
	modParams := mlx.Matmul(temb, mlx.Transpose(b.Norm1Linear, 1, 0))
	if b.Norm1LinearBias != nil {
		modParams = mlx.Add(modParams, b.Norm1LinearBias)
	}

	// Extract 12 modulation parameters (NO tanh on gates, per diffusers reference)
	// Order: shift_msa, c_shift_msa, scale_msa, c_scale_msa, gate_msa, c_gate_msa,
	//        shift_mlp, c_shift_mlp, scale_mlp, c_scale_mlp, gate_mlp, c_gate_mlp
	modParams = mlx.Reshape(modParams, B, -1)

	shiftMsa := mlx.Reshape(mlx.Slice(modParams, []int32{0, 0}, []int32{B, hiddenDim}), B, 1, hiddenDim)
	cShiftMsa := mlx.Reshape(mlx.Slice(modParams, []int32{0, hiddenDim}, []int32{B, 2 * hiddenDim}), B, 1, hiddenDim)
	scaleMsa := mlx.Reshape(mlx.Slice(modParams, []int32{0, 2 * hiddenDim}, []int32{B, 3 * hiddenDim}), B, 1, hiddenDim)
	cScaleMsa := mlx.Reshape(mlx.Slice(modParams, []int32{0, 3 * hiddenDim}, []int32{B, 4 * hiddenDim}), B, 1, hiddenDim)
	gateMsa := mlx.Reshape(mlx.Slice(modParams, []int32{0, 4 * hiddenDim}, []int32{B, 5 * hiddenDim}), B, 1, hiddenDim)
	cGateMsa := mlx.Reshape(mlx.Slice(modParams, []int32{0, 5 * hiddenDim}, []int32{B, 6 * hiddenDim}), B, 1, hiddenDim)

	shiftMlp := mlx.Reshape(mlx.Slice(modParams, []int32{0, 6 * hiddenDim}, []int32{B, 7 * hiddenDim}), B, 1, hiddenDim)
	cShiftMlp := mlx.Reshape(mlx.Slice(modParams, []int32{0, 7 * hiddenDim}, []int32{B, 8 * hiddenDim}), B, 1, hiddenDim)
	scaleMlp := mlx.Reshape(mlx.Slice(modParams, []int32{0, 8 * hiddenDim}, []int32{B, 9 * hiddenDim}), B, 1, hiddenDim)
	cScaleMlp := mlx.Reshape(mlx.Slice(modParams, []int32{0, 9 * hiddenDim}, []int32{B, 10 * hiddenDim}), B, 1, hiddenDim)
	gateMlp := mlx.Reshape(mlx.Slice(modParams, []int32{0, 10 * hiddenDim}, []int32{B, 11 * hiddenDim}), B, 1, hiddenDim)
	cGateMlp := mlx.Reshape(mlx.Slice(modParams, []int32{0, 11 * hiddenDim}, []int32{B, 12 * hiddenDim}), B, 1, hiddenDim)

	// === 2. Apply LayerNorm and modulation ===
	// Image tokens: LN(x) * (1 + scale) + shift
	normHiddenStates := layerNorm(hiddenStates)
	normHiddenStates = mlx.Mul(normHiddenStates, mlx.AddScalar(scaleMsa, 1.0))
	normHiddenStates = mlx.Add(normHiddenStates, shiftMsa)

	// Text tokens (encoder_hidden_states): LN(x) * (1 + c_scale) + c_shift
	normEncoderStates := layerNorm(encoderHiddenStates)
	normEncoderStates = mlx.Mul(normEncoderStates, mlx.AddScalar(cScaleMsa, 1.0))
	normEncoderStates = mlx.Add(normEncoderStates, cShiftMsa)

	// === 3. Self-attention (joint over text + image) ===
	// Concatenate for joint attention: [text, image]
	attnHiddenStates, attnEncoderStates := b.Attn1.ForwardMMDiT(
		normHiddenStates, normEncoderStates,
		cos, sin,
	)

	// Apply gated residual connection
	hiddenStates = mlx.Add(hiddenStates, mlx.Mul(attnHiddenStates, gateMsa))
	encoderHiddenStates = mlx.Add(encoderHiddenStates, mlx.Mul(attnEncoderStates, cGateMsa))

	// === 4. Feedforward ===
	// Apply norm and modulation
	normHiddenStates = layerNorm(hiddenStates)
	normHiddenStates = mlx.Mul(normHiddenStates, mlx.AddScalar(scaleMlp, 1.0))
	normHiddenStates = mlx.Add(normHiddenStates, shiftMlp)

	normEncoderStates = layerNorm(encoderHiddenStates)
	normEncoderStates = mlx.Mul(normEncoderStates, mlx.AddScalar(cScaleMlp, 1.0))
	normEncoderStates = mlx.Add(normEncoderStates, cShiftMlp)

	// FFN (same network for both)
	ffHiddenStates := b.FF.Forward(normHiddenStates)
	ffEncoderStates := b.FF.Forward(normEncoderStates)

	// Apply gated residual connection
	hiddenStates = mlx.Add(hiddenStates, mlx.Mul(ffHiddenStates, gateMlp))
	encoderHiddenStates = mlx.Add(encoderHiddenStates, mlx.Mul(ffEncoderStates, cGateMlp))

	_ = imgSeqLen
	_ = textSeqLen

	return hiddenStates, encoderHiddenStates
}

// ForwardMMDiT implements joint attention for MMDiT
// hiddenStates: image tokens [B, L_img, D] - gets RoPE
// encoderHiddenStates: text tokens [B, L_text, D] - no RoPE
func (attn *DiTAttention) ForwardMMDiT(
	hiddenStates, encoderHiddenStates *mlx.Array,
	cos, sin *mlx.Array,
) (*mlx.Array, *mlx.Array) {
	imgShape := hiddenStates.Shape()
	textShape := encoderHiddenStates.Shape()
	B := imgShape[0]
	imgSeqLen := imgShape[1]
	textSeqLen := textShape[1]

	// Concatenate: [text, image]
	combined := mlx.Concatenate([]*mlx.Array{encoderHiddenStates, hiddenStates}, 1)
	totalLen := textSeqLen + imgSeqLen

	// Q, K, V projections
	q := mlx.Matmul(combined, mlx.Transpose(attn.ToQ, 1, 0))
	if attn.QBias != nil {
		q = mlx.Add(q, attn.QBias)
	}
	k := mlx.Matmul(combined, mlx.Transpose(attn.ToK, 1, 0))
	if attn.KBias != nil {
		k = mlx.Add(k, attn.KBias)
	}
	v := mlx.Matmul(combined, mlx.Transpose(attn.ToV, 1, 0))
	if attn.VBias != nil {
		v = mlx.Add(v, attn.VBias)
	}

	// Reshape to [B, L, nheads, head_dim]
	q = mlx.Reshape(q, B, totalLen, attn.NHeads, attn.HeadDim)
	k = mlx.Reshape(k, B, totalLen, attn.NHeads, attn.HeadDim)
	v = mlx.Reshape(v, B, totalLen, attn.NHeads, attn.HeadDim)

	// Apply QK normalization if present (attn.norm_q, attn.norm_k)
	// GLM-Image uses LayerNorm on Q and K
	q = layerNormLastDim(q)
	k = layerNormLastDim(k)

	// Apply RoPE to image tokens ONLY (after text tokens)
	if cos != nil && sin != nil {
		// Extract image Q and K
		qImg := mlx.Slice(q, []int32{0, textSeqLen, 0, 0}, []int32{B, totalLen, attn.NHeads, attn.HeadDim})
		kImg := mlx.Slice(k, []int32{0, textSeqLen, 0, 0}, []int32{B, totalLen, attn.NHeads, attn.HeadDim})

		// Apply RoPE
		qImg = applyRoPE2D(qImg, cos, sin)
		kImg = applyRoPE2D(kImg, cos, sin)

		// Reconstruct full Q and K
		qText := mlx.Slice(q, []int32{0, 0, 0, 0}, []int32{B, textSeqLen, attn.NHeads, attn.HeadDim})
		kText := mlx.Slice(k, []int32{0, 0, 0, 0}, []int32{B, textSeqLen, attn.NHeads, attn.HeadDim})

		q = mlx.Concatenate([]*mlx.Array{qText, qImg}, 1)
		k = mlx.Concatenate([]*mlx.Array{kText, kImg}, 1)
	}

	// Transpose to [B, nheads, L, head_dim]
	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	// SDPA (no causal mask for diffusion - all tokens attend to all)
	out := mlx.ScaledDotProductAttention(q, k, v, attn.Scale, false)

	// Transpose back and reshape
	out = mlx.Transpose(out, 0, 2, 1, 3)
	out = mlx.Reshape(out, B, totalLen, attn.NHeads*attn.HeadDim)

	// Output projection
	out = mlx.Matmul(out, mlx.Transpose(attn.ToOut, 1, 0))
	if attn.ToOutBias != nil {
		out = mlx.Add(out, attn.ToOutBias)
	}

	// Split back into text and image
	encoderOut := mlx.Slice(out, []int32{0, 0, 0}, []int32{B, textSeqLen, attn.NHeads * attn.HeadDim})
	hiddenOut := mlx.Slice(out, []int32{0, textSeqLen, 0}, []int32{B, totalLen, attn.NHeads * attn.HeadDim})

	return hiddenOut, encoderOut
}

// layerNormLastDim applies layer normalization on the last dimension
func layerNormLastDim(x *mlx.Array) *mlx.Array {
	eps := float32(1e-5)
	mean := mlx.Mean(x, -1, true)
	x = mlx.Sub(x, mean)
	variance := mlx.Mean(mlx.Square(x), -1, true)
	return mlx.Div(x, mlx.Sqrt(mlx.AddScalar(variance, eps)))
}

// Forward for DiTBlock (no RoPE, for compatibility)
func (b *DiTBlock) Forward(x *mlx.Array, temb *mlx.Array, hiddenDim int32, contextLen int32) *mlx.Array {
	return b.ForwardWithRoPE(x, temb, hiddenDim, nil, nil, contextLen)
}

// ForwardWithRoPE applies the block with optional RoPE (legacy interface)
// contextLen is the number of context tokens (prior + text) at the start of the sequence
func (b *DiTBlock) ForwardWithRoPE(x *mlx.Array, temb *mlx.Array, hiddenDim int32, cos, sin *mlx.Array, contextLen int32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]

	// AdaLN modulation: norm1 produces 12 parameters per hidden dim
	modParams := mlx.Matmul(temb, mlx.Transpose(b.Norm1Linear, 1, 0))
	if b.Norm1LinearBias != nil {
		modParams = mlx.Add(modParams, b.Norm1LinearBias)
	}
	modParams = mlx.Reshape(modParams, B, 1, -1)

	modDim := modParams.Shape()[2]

	// Debug: print modDim vs expected once
	if debugOnce {
		fmt.Printf("    [DEBUG] modDim=%d, 12*hiddenDim=%d\n", modDim, 12*hiddenDim)
		debugOnce = false
	}

	if modDim >= 12*hiddenDim {
		// Extract 12 modulation parameters (NO tanh on gates)
		shiftMsa := mlx.Slice(modParams, []int32{0, 0, 0}, []int32{B, 1, hiddenDim})
		cShiftMsa := mlx.Slice(modParams, []int32{0, 0, hiddenDim}, []int32{B, 1, 2 * hiddenDim})
		scaleMsa := mlx.Slice(modParams, []int32{0, 0, 2 * hiddenDim}, []int32{B, 1, 3 * hiddenDim})
		cScaleMsa := mlx.Slice(modParams, []int32{0, 0, 3 * hiddenDim}, []int32{B, 1, 4 * hiddenDim})
		gateMsa := mlx.Slice(modParams, []int32{0, 0, 4 * hiddenDim}, []int32{B, 1, 5 * hiddenDim})
		cGateMsa := mlx.Slice(modParams, []int32{0, 0, 5 * hiddenDim}, []int32{B, 1, 6 * hiddenDim})

		shiftMlp := mlx.Slice(modParams, []int32{0, 0, 6 * hiddenDim}, []int32{B, 1, 7 * hiddenDim})
		cShiftMlp := mlx.Slice(modParams, []int32{0, 0, 7 * hiddenDim}, []int32{B, 1, 8 * hiddenDim})
		scaleMlp := mlx.Slice(modParams, []int32{0, 0, 8 * hiddenDim}, []int32{B, 1, 9 * hiddenDim})
		cScaleMlp := mlx.Slice(modParams, []int32{0, 0, 9 * hiddenDim}, []int32{B, 1, 10 * hiddenDim})
		gateMlp := mlx.Slice(modParams, []int32{0, 0, 10 * hiddenDim}, []int32{B, 1, 11 * hiddenDim})
		cGateMlp := mlx.Slice(modParams, []int32{0, 0, 11 * hiddenDim}, []int32{B, 1, 12 * hiddenDim})

		// Apply LayerNorm
		xNorm := layerNorm(x)

		// Split context (prior + text) and image tokens
		imgLen := L - contextLen

		// Apply different modulation to context vs image tokens
		// Image tokens: use regular parameters
		var xMod *mlx.Array
		if contextLen > 0 && imgLen > 0 {
			contextNorm := mlx.Slice(xNorm, []int32{0, 0, 0}, []int32{B, contextLen, hiddenDim})
			imgNorm := mlx.Slice(xNorm, []int32{0, contextLen, 0}, []int32{B, L, hiddenDim})

			// Modulate context: (1 + c_scale) * x + c_shift
			contextMod := mlx.Mul(contextNorm, mlx.AddScalar(cScaleMsa, 1.0))
			contextMod = mlx.Add(contextMod, cShiftMsa)

			// Modulate image: (1 + scale) * x + shift
			imgMod := mlx.Mul(imgNorm, mlx.AddScalar(scaleMsa, 1.0))
			imgMod = mlx.Add(imgMod, shiftMsa)

			xMod = mlx.Concatenate([]*mlx.Array{contextMod, imgMod}, 1)
		} else {
			// All tokens treated the same (shouldn't happen normally)
			xMod = mlx.Mul(xNorm, mlx.AddScalar(scaleMsa, 1.0))
			xMod = mlx.Add(xMod, shiftMsa)
		}

		// Self-attention with RoPE
		attnOut := b.Attn1.ForwardWithRoPE(xMod, cos, sin)

		// Apply different gates to context vs image
		if contextLen > 0 && imgLen > 0 {
			contextAttn := mlx.Slice(attnOut, []int32{0, 0, 0}, []int32{B, contextLen, hiddenDim})
			imgAttn := mlx.Slice(attnOut, []int32{0, contextLen, 0}, []int32{B, L, hiddenDim})

			contextAttn = mlx.Mul(contextAttn, cGateMsa)
			imgAttn = mlx.Mul(imgAttn, gateMsa)

			attnOut = mlx.Concatenate([]*mlx.Array{contextAttn, imgAttn}, 1)
		} else {
			attnOut = mlx.Mul(attnOut, gateMsa)
		}

		x = mlx.Add(x, attnOut)

		// FFN with modulation
		xNorm = layerNorm(x)

		if contextLen > 0 && imgLen > 0 {
			contextNorm := mlx.Slice(xNorm, []int32{0, 0, 0}, []int32{B, contextLen, hiddenDim})
			imgNorm := mlx.Slice(xNorm, []int32{0, contextLen, 0}, []int32{B, L, hiddenDim})

			// Modulate context
			contextMod := mlx.Mul(contextNorm, mlx.AddScalar(cScaleMlp, 1.0))
			contextMod = mlx.Add(contextMod, cShiftMlp)

			// Modulate image
			imgMod := mlx.Mul(imgNorm, mlx.AddScalar(scaleMlp, 1.0))
			imgMod = mlx.Add(imgMod, shiftMlp)

			xMod = mlx.Concatenate([]*mlx.Array{contextMod, imgMod}, 1)
		} else {
			xMod = mlx.Mul(xNorm, mlx.AddScalar(scaleMlp, 1.0))
			xMod = mlx.Add(xMod, shiftMlp)
		}

		ffOut := b.FF.Forward(xMod)

		// Apply gates
		if contextLen > 0 && imgLen > 0 {
			contextFF := mlx.Slice(ffOut, []int32{0, 0, 0}, []int32{B, contextLen, hiddenDim})
			imgFF := mlx.Slice(ffOut, []int32{0, contextLen, 0}, []int32{B, L, hiddenDim})

			contextFF = mlx.Mul(contextFF, cGateMlp)
			imgFF = mlx.Mul(imgFF, gateMlp)

			ffOut = mlx.Concatenate([]*mlx.Array{contextFF, imgFF}, 1)
		} else {
			ffOut = mlx.Mul(ffOut, gateMlp)
		}

		x = mlx.Add(x, ffOut)
	} else {
		// Fallback path without full modulation (shouldn't happen for GLM-Image)
		xNorm := layerNorm(x)
		attnOut := b.Attn1.ForwardWithRoPE(xNorm, cos, sin)
		x = mlx.Add(x, attnOut)

		xNorm = layerNorm(x)
		ffOut := b.FF.Forward(xNorm)
		x = mlx.Add(x, ffOut)
	}

	return x
}

// Forward for DiTAttention with optional RoPE
func (attn *DiTAttention) Forward(x *mlx.Array) *mlx.Array {
	return attn.ForwardWithRoPE(x, nil, nil)
}

// ForwardWithRoPE applies attention with rotary position embeddings
// RoPE is applied ONLY to image tokens (after contextLen positions)
// cos, sin have shape [1, imgLen, 1, headDim] for image tokens only
func (attn *DiTAttention) ForwardWithRoPE(x *mlx.Array, cos, sin *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	D := shape[2]

	// Q, K, V projections
	q := mlx.Matmul(x, mlx.Transpose(attn.ToQ, 1, 0))
	if attn.QBias != nil {
		q = mlx.Add(q, attn.QBias)
	}
	k := mlx.Matmul(x, mlx.Transpose(attn.ToK, 1, 0))
	if attn.KBias != nil {
		k = mlx.Add(k, attn.KBias)
	}
	v := mlx.Matmul(x, mlx.Transpose(attn.ToV, 1, 0))
	if attn.VBias != nil {
		v = mlx.Add(v, attn.VBias)
	}

	// Reshape to [B, L, nheads, head_dim]
	q = mlx.Reshape(q, B, L, attn.NHeads, attn.HeadDim)
	k = mlx.Reshape(k, B, L, attn.NHeads, attn.HeadDim)
	v = mlx.Reshape(v, B, L, attn.NHeads, attn.HeadDim)

	// Transpose to [B, nheads, L, head_dim] (before RoPE for easier slicing)
	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	// Apply RoPE to image tokens only (like CogView4)
	// cos, sin are for image tokens only [1, imgLen, 1, headDim]
	if cos != nil && sin != nil {
		imgLen := cos.Shape()[1]
		contextLen := L - imgLen

		if contextLen >= 0 && imgLen > 0 {
			// Split Q and K into context and image parts
			qContext := mlx.Slice(q, []int32{0, 0, 0, 0}, []int32{B, attn.NHeads, contextLen, attn.HeadDim})
			qImg := mlx.Slice(q, []int32{0, 0, contextLen, 0}, []int32{B, attn.NHeads, L, attn.HeadDim})

			kContext := mlx.Slice(k, []int32{0, 0, 0, 0}, []int32{B, attn.NHeads, contextLen, attn.HeadDim})
			kImg := mlx.Slice(k, []int32{0, 0, contextLen, 0}, []int32{B, attn.NHeads, L, attn.HeadDim})

			// Apply RoPE only to image tokens
			// cos, sin: [1, imgLen, 1, headDim] -> need to transpose for [B, nheads, imgLen, headDim]
			cosT := mlx.Transpose(cos, 0, 2, 1, 3) // [1, 1, imgLen, headDim]
			sinT := mlx.Transpose(sin, 0, 2, 1, 3)

			qImgRope := applyRoPETransposed(qImg, cosT, sinT)
			kImgRope := applyRoPETransposed(kImg, cosT, sinT)

			// Reconstruct full Q and K
			q = mlx.Concatenate([]*mlx.Array{qContext, qImgRope}, 2)
			k = mlx.Concatenate([]*mlx.Array{kContext, kImgRope}, 2)
		}
	}

	// SDPA (no causal mask for diffusion - all tokens attend to all)
	out := mlx.ScaledDotProductAttention(q, k, v, attn.Scale, false)

	// Transpose back and reshape
	out = mlx.Transpose(out, 0, 2, 1, 3)
	out = mlx.Reshape(out, B, L, D)

	// Output projection
	out = mlx.Matmul(out, mlx.Transpose(attn.ToOut, 1, 0))
	if attn.ToOutBias != nil {
		out = mlx.Add(out, attn.ToOutBias)
	}

	return out
}

// applyRoPETransposed applies RoPE when Q/K are in [B, nheads, L, headDim] format
// Uses split-half approach (use_real_unbind_dim=-2) to match diffusers GLM-Image
func applyRoPETransposed(x *mlx.Array, cos, sin *mlx.Array) *mlx.Array {
	// x: [B, nheads, L, headDim]
	// cos, sin: [1, 1, L, headDim] (first half == second half, duplicated)
	shape := x.Shape()
	B := shape[0]
	nHeads := shape[1]
	L := shape[2]
	headDim := shape[3]
	halfDim := headDim / 2

	// Split x into first and second half
	x1 := mlx.Slice(x, []int32{0, 0, 0, 0}, []int32{B, nHeads, L, halfDim})
	x2 := mlx.Slice(x, []int32{0, 0, 0, halfDim}, []int32{B, nHeads, L, headDim})

	// Get first half of cos/sin (they're duplicated, so first half == second half)
	cosHalf := mlx.Slice(cos, []int32{0, 0, 0, 0}, []int32{1, 1, L, halfDim})
	sinHalf := mlx.Slice(sin, []int32{0, 0, 0, 0}, []int32{1, 1, L, halfDim})

	// Apply rotation: out1 = x1*cos - x2*sin, out2 = x2*cos + x1*sin
	out1 := mlx.Sub(mlx.Mul(x1, cosHalf), mlx.Mul(x2, sinHalf))
	out2 := mlx.Add(mlx.Mul(x2, cosHalf), mlx.Mul(x1, sinHalf))

	// Concatenate back to full dimension
	return mlx.Concatenate([]*mlx.Array{out1, out2}, 3)
}

// geluApproximate implements GELU with tanh approximation (matches diffusers "gelu-approximate")
// Formula: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
func geluApproximate(x *mlx.Array) *mlx.Array {
	// Constants
	sqrt2OverPi := float32(0.7978845608) // sqrt(2/π)
	coeff := float32(0.044715)

	// x³
	x3 := mlx.Mul(mlx.Mul(x, x), x)

	// inner = sqrt(2/π) * (x + 0.044715 * x³)
	inner := mlx.MulScalar(mlx.Add(x, mlx.MulScalar(x3, coeff)), sqrt2OverPi)

	// 0.5 * x * (1 + tanh(inner))
	return mlx.Mul(mlx.MulScalar(x, 0.5), mlx.AddScalar(mlx.Tanh(inner), 1.0))
}

// Forward for DiTFeedForward
func (ff *DiTFeedForward) Forward(x *mlx.Array) *mlx.Array {
	// GELU approximate MLP (matches diffusers activation_fn="gelu-approximate")
	h := mlx.Matmul(x, mlx.Transpose(ff.Linear1, 1, 0))
	if ff.Bias1 != nil {
		h = mlx.Add(h, ff.Bias1)
	}
	h = geluApproximate(h)

	h = mlx.Matmul(h, mlx.Transpose(ff.Linear2, 1, 0))
	if ff.Bias2 != nil {
		h = mlx.Add(h, ff.Bias2)
	}

	return h
}

// Forward for DiTMLP (projector MLPs with GELU - used for glyph_projector)
func (m *DiTMLP) Forward(x *mlx.Array) *mlx.Array {
	h := mlx.Matmul(x, mlx.Transpose(m.Linear1, 1, 0))
	if m.Bias1 != nil {
		h = mlx.Add(h, m.Bias1)
	}
	h = mlx.GELU(h)

	h = mlx.Matmul(h, mlx.Transpose(m.Linear2, 1, 0))
	if m.Bias2 != nil {
		h = mlx.Add(h, m.Bias2)
	}

	return h
}

// Forward for DiTMLPSiLU (projector MLPs with SiLU - used for prior_projector)
func (m *DiTMLPSiLU) Forward(x *mlx.Array) *mlx.Array {
	h := mlx.Matmul(x, mlx.Transpose(m.Linear1, 1, 0))
	if m.Bias1 != nil {
		h = mlx.Add(h, m.Bias1)
	}
	h = mlx.SiLU(h) // SiLU activation for prior_projector (matches diffusers "linear-silu")

	h = mlx.Matmul(h, mlx.Transpose(m.Linear2, 1, 0))
	if m.Bias2 != nil {
		h = mlx.Add(h, m.Bias2)
	}

	return h
}

// RoPE2DCache holds precomputed RoPE values for 2D image positions
type RoPE2DCache struct {
	Cos *mlx.Array // [1, L, 1, head_dim]
	Sin *mlx.Array // [1, L, 1, head_dim]
}

// ComputeUnifiedRoPE computes RoPE for the full unified sequence (prior + text + image)
// Prior and text tokens get sequential 1D positions (h=0, w=index)
// Image tokens get 2D grid positions (h, w) from patch grid
func ComputeUnifiedRoPE(priorLen, textLen, pH, pW, headDim int32, theta float32) *RoPE2DCache {
	imgLen := pH * pW
	totalLen := priorLen + textLen + imgLen

	// Split head_dim between h and w dimensions
	dimH := headDim / 2
	dimW := headDim / 2

	// Compute inverse frequencies
	hFreqs := make([]float32, dimH/2)
	for i := int32(0); i < dimH/2; i++ {
		hFreqs[i] = float32(1.0 / math.Pow(float64(theta), float64(2*i)/float64(dimH)))
	}

	wFreqs := make([]float32, dimW/2)
	for i := int32(0); i < dimW/2; i++ {
		wFreqs[i] = float32(1.0 / math.Pow(float64(theta), float64(2*i)/float64(dimW)))
	}

	cosVals := make([]float32, totalLen*headDim)
	sinVals := make([]float32, totalLen*headDim)

	// Prior tokens: h=0, w=idx (sequential positions on w axis)
	for idx := int32(0); idx < priorLen; idx++ {
		offset := idx * headDim
		h := float32(0)
		w := float32(idx)

		for i := int32(0); i < dimH/2; i++ {
			angle := h * hFreqs[i]
			cosVals[offset+2*i] = float32(math.Cos(float64(angle)))
			cosVals[offset+2*i+1] = float32(math.Cos(float64(angle)))
			sinVals[offset+2*i] = float32(math.Sin(float64(angle)))
			sinVals[offset+2*i+1] = float32(math.Sin(float64(angle)))
		}
		for i := int32(0); i < dimW/2; i++ {
			angle := w * wFreqs[i]
			idx2 := dimH + 2*i
			cosVals[offset+idx2] = float32(math.Cos(float64(angle)))
			cosVals[offset+idx2+1] = float32(math.Cos(float64(angle)))
			sinVals[offset+idx2] = float32(math.Sin(float64(angle)))
			sinVals[offset+idx2+1] = float32(math.Sin(float64(angle)))
		}
	}

	// Text tokens: h=0, w=priorLen+idx (continue sequential positions)
	for idx := int32(0); idx < textLen; idx++ {
		offset := (priorLen + idx) * headDim
		h := float32(0)
		w := float32(priorLen + idx)

		for i := int32(0); i < dimH/2; i++ {
			angle := h * hFreqs[i]
			cosVals[offset+2*i] = float32(math.Cos(float64(angle)))
			cosVals[offset+2*i+1] = float32(math.Cos(float64(angle)))
			sinVals[offset+2*i] = float32(math.Sin(float64(angle)))
			sinVals[offset+2*i+1] = float32(math.Sin(float64(angle)))
		}
		for i := int32(0); i < dimW/2; i++ {
			angle := w * wFreqs[i]
			idx2 := dimH + 2*i
			cosVals[offset+idx2] = float32(math.Cos(float64(angle)))
			cosVals[offset+idx2+1] = float32(math.Cos(float64(angle)))
			sinVals[offset+idx2] = float32(math.Sin(float64(angle)))
			sinVals[offset+idx2+1] = float32(math.Sin(float64(angle)))
		}
	}

	// Image tokens: 2D grid positions (h, w)
	for hPos := int32(0); hPos < pH; hPos++ {
		for wPos := int32(0); wPos < pW; wPos++ {
			patchIdx := hPos*pW + wPos
			offset := (priorLen + textLen + patchIdx) * headDim
			h := float32(hPos)
			w := float32(wPos)

			for i := int32(0); i < dimH/2; i++ {
				angle := h * hFreqs[i]
				cosVals[offset+2*i] = float32(math.Cos(float64(angle)))
				cosVals[offset+2*i+1] = float32(math.Cos(float64(angle)))
				sinVals[offset+2*i] = float32(math.Sin(float64(angle)))
				sinVals[offset+2*i+1] = float32(math.Sin(float64(angle)))
			}
			for i := int32(0); i < dimW/2; i++ {
				angle := w * wFreqs[i]
				idx2 := dimH + 2*i
				cosVals[offset+idx2] = float32(math.Cos(float64(angle)))
				cosVals[offset+idx2+1] = float32(math.Cos(float64(angle)))
				sinVals[offset+idx2] = float32(math.Sin(float64(angle)))
				sinVals[offset+idx2+1] = float32(math.Sin(float64(angle)))
			}
		}
	}

	cos := mlx.NewArray(cosVals, []int32{1, totalLen, 1, headDim})
	sin := mlx.NewArray(sinVals, []int32{1, totalLen, 1, headDim})

	cos = mlx.ToBFloat16(cos)
	sin = mlx.ToBFloat16(sin)

	return &RoPE2DCache{Cos: cos, Sin: sin}
}

// ComputeRoPE2D computes 2D rotary position embeddings for image patches
// Matches the diffusers GlmImageRotaryPosEmbed implementation exactly.
// pH, pW: patch grid dimensions (height, width in patches)
// headDim: attention head dimension (40 for GLM-Image)
// theta: RoPE theta (10000.0 for GLM-Image)
func ComputeRoPE2D(pH, pW, headDim int32, theta float32) *RoPE2DCache {
	// Split head_dim between h and w dimensions
	// For headDim=40: dimH = dimW = 20
	dimH := headDim / 2
	dimW := headDim / 2

	// Compute inverse frequencies matching diffusers GlmImageRotaryPosEmbed:
	// h_inv_freq = 1.0 / (theta ** (arange(0, dim_h, 2)[:dim_h // 2] / dim_h))
	// For dim_h=20: arange(0, 20, 2) = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
	// [:10] keeps all 10 values, so numHFreqs = dim_h / 2 = 10
	numHFreqs := dimH / 2
	hFreqs := make([]float32, numHFreqs)
	for i := int32(0); i < numHFreqs; i++ {
		// exponent = (2*i) / dim_h
		hFreqs[i] = float32(1.0 / math.Pow(float64(theta), float64(2*i)/float64(dimH)))
	}

	numWFreqs := dimW / 2
	wFreqs := make([]float32, numWFreqs)
	for i := int32(0); i < numWFreqs; i++ {
		wFreqs[i] = float32(1.0 / math.Pow(float64(theta), float64(2*i)/float64(dimW)))
	}

	// Build the full frequency tensor
	numPatches := pH * pW
	halfDim := headDim / 2 // dim_h/2 + dim_w/2 = headDim/2
	cosVals := make([]float32, numPatches*headDim)
	sinVals := make([]float32, numPatches*headDim)

	for h := int32(0); h < pH; h++ {
		for w := int32(0); w < pW; w++ {
			patchIdx := h*pW + w
			offset := patchIdx * headDim

			// Compute freqs for this position
			// freqs = [freqs_h, freqs_w, freqs_h, freqs_w] (duplicated)
			// First half: [freqs_h, freqs_w]
			// Second half: same as first half

			// Height frequencies (first dim_h/2 values)
			for i := int32(0); i < numHFreqs; i++ {
				angle := float32(h) * hFreqs[i]
				cosVals[offset+i] = float32(math.Cos(float64(angle)))
				sinVals[offset+i] = float32(math.Sin(float64(angle)))
			}

			// Width frequencies (next dim_w/2 values)
			for i := int32(0); i < numWFreqs; i++ {
				angle := float32(w) * wFreqs[i]
				idx := numHFreqs + i
				cosVals[offset+idx] = float32(math.Cos(float64(angle)))
				sinVals[offset+idx] = float32(math.Sin(float64(angle)))
			}

			// Duplicate for second half (freqs = cat([freqs, freqs], -1))
			for i := int32(0); i < halfDim; i++ {
				cosVals[offset+halfDim+i] = cosVals[offset+i]
				sinVals[offset+halfDim+i] = sinVals[offset+i]
			}
		}
	}

	cos := mlx.NewArray(cosVals, []int32{1, numPatches, 1, headDim})
	sin := mlx.NewArray(sinVals, []int32{1, numPatches, 1, headDim})

	cos = mlx.ToBFloat16(cos)
	sin = mlx.ToBFloat16(sin)

	return &RoPE2DCache{Cos: cos, Sin: sin}
}

// applyRoPE2D applies 2D rotary position embeddings to Q or K
// Uses split-half approach (use_real_unbind_dim=-2) to match diffusers GLM-Image
// x: [B, L, nheads, head_dim]
// cos, sin: [1, L, 1, head_dim] (first half == second half, duplicated)
func applyRoPE2D(x *mlx.Array, cos, sin *mlx.Array) *mlx.Array {
	// Split-half RoPE (use_real_unbind_dim=-2):
	// x1 = x[..., :head_dim/2], x2 = x[..., head_dim/2:]
	// output = cat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1)
	// Since cos/sin are duplicated (first half == second half), we use half values

	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	nHeads := shape[2]
	headDim := shape[3]
	halfDim := headDim / 2

	// Split x into first and second half
	x1 := mlx.Slice(x, []int32{0, 0, 0, 0}, []int32{B, L, nHeads, halfDim})
	x2 := mlx.Slice(x, []int32{0, 0, 0, halfDim}, []int32{B, L, nHeads, headDim})

	// Get first half of cos/sin (they're duplicated, so first half == second half)
	cosHalf := mlx.Slice(cos, []int32{0, 0, 0, 0}, []int32{1, L, 1, halfDim})
	sinHalf := mlx.Slice(sin, []int32{0, 0, 0, 0}, []int32{1, L, 1, halfDim})

	// Apply rotation: out1 = x1*cos - x2*sin, out2 = x2*cos + x1*sin
	out1 := mlx.Sub(mlx.Mul(x1, cosHalf), mlx.Mul(x2, sinHalf))
	out2 := mlx.Add(mlx.Mul(x2, cosHalf), mlx.Mul(x1, sinHalf))

	// Concatenate back to full dimension
	return mlx.Concatenate([]*mlx.Array{out1, out2}, 3)
}

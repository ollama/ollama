//go:build mlx

package gemma3

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
	"github.com/ollama/ollama/x/imagegen/safetensors"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
)

// TextConfig holds configuration for the text model
type TextConfig struct {
	HiddenSize            int32   `json:"hidden_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	IntermediateSize      int32   `json:"intermediate_size"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads"`
	HeadDim               int32   `json:"head_dim"`
	VocabSize             int32   `json:"vocab_size"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	RopeLocalBaseFreq     float32 `json:"rope_local_base_freq"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`
	SlidingWindow         int32   `json:"sliding_window"`
	SlidingWindowPattern  int32   `json:"sliding_window_pattern"`

	// Computed fields
	Scale float32 `json:"-"`
}

// TextModel is the Gemma 3 text-only model
type TextModel struct {
	EmbedTokens *nn.Embedding   `weight:"model.embed_tokens"`
	Layers      []*DecoderLayer `weight:"model.layers"`
	Norm        *nn.RMSNorm     `weight:"model.norm"`
	Output      *nn.Linear      `weight:"-"` // Tied to EmbedTokens, set manually

	// Precomputed (1 + weight) for Gemma-style RMSNorm to avoid allocation per forward
	NormScaled *mlx.Array `weight:"-"`

	tok *tokenizer.Tokenizer
	*TextConfig
}

// DecoderLayer is a single transformer block
type DecoderLayer struct {
	InputNorm    *nn.RMSNorm `weight:"input_layernorm"`
	Attention    *Attention
	PostAttnNorm *nn.RMSNorm `weight:"post_attention_layernorm"`
	PreFFNorm    *nn.RMSNorm `weight:"pre_feedforward_layernorm"`
	MLP          *MLP
	PostFFNorm   *nn.RMSNorm `weight:"post_feedforward_layernorm"`

	// Precomputed (1 + weight) for Gemma-style RMSNorm
	InputNormScaled    *mlx.Array `weight:"-"`
	PostAttnNormScaled *mlx.Array `weight:"-"`
	PreFFNormScaled    *mlx.Array `weight:"-"`
	PostFFNormScaled   *mlx.Array `weight:"-"`

	// Whether this layer uses sliding window attention
	IsSliding bool
	LayerIdx  int32
}

// Attention implements Gemma 3 attention with Q/K normalization
type Attention struct {
	QProj *nn.Linear  `weight:"self_attn.q_proj"`
	KProj *nn.Linear  `weight:"self_attn.k_proj"`
	VProj *nn.Linear  `weight:"self_attn.v_proj"`
	OProj *nn.Linear  `weight:"self_attn.o_proj"`
	QNorm *nn.RMSNorm `weight:"self_attn.q_norm"`
	KNorm *nn.RMSNorm `weight:"self_attn.k_norm"`

	// Precomputed (1 + weight) for Gemma-style RMSNorm
	QNormScaled *mlx.Array `weight:"-"`
	KNormScaled *mlx.Array `weight:"-"`
}

// MLP is the feed-forward network with GELU activation
type MLP struct {
	GateProj *nn.Linear `weight:"mlp.gate_proj"`
	UpProj   *nn.Linear `weight:"mlp.up_proj"`
	DownProj *nn.Linear `weight:"mlp.down_proj"`
}

// LoadText loads the text-only Gemma 3 model
func LoadText(modelPath string) (*TextModel, error) {
	data, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	var cfg TextConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	// Compute scale
	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))

	// Set defaults if not specified
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 1000000
	}
	if cfg.RopeLocalBaseFreq == 0 {
		cfg.RopeLocalBaseFreq = 10000
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}

	weights, err := safetensors.LoadModelWeights(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load weights: %w", err)
	}

	tok, err := tokenizer.Load(filepath.Join(modelPath, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	m := &TextModel{
		Layers:     make([]*DecoderLayer, cfg.NumHiddenLayers),
		TextConfig: &cfg,
		tok:        tok,
	}

	// Initialize layer metadata
	for i := range m.Layers {
		m.Layers[i] = &DecoderLayer{
			LayerIdx:  int32(i),
			IsSliding: isLayerSliding(int32(i), cfg.SlidingWindowPattern),
		}
	}

	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return nil, err
	}

	// Tied embeddings for output
	m.Output = nn.NewLinear(m.EmbedTokens.Weight, nil)

	mlx.Eval(mlx.Collect(m)...)
	weights.ReleaseAll()

	// Precompute (1 + weight) for Gemma-style RMSNorm to avoid per-forward allocation
	precomputeGemmaScaledWeights(m)

	return m, nil
}

// precomputeGemmaScaledWeights computes (1 + weight) for all RMSNorm layers
// This avoids creating temporary arrays on every forward pass
func precomputeGemmaScaledWeights(m *TextModel) {
	m.NormScaled = mlx.AddScalar(m.Norm.Weight, 1.0)

	for _, layer := range m.Layers {
		layer.InputNormScaled = mlx.AddScalar(layer.InputNorm.Weight, 1.0)
		layer.PostAttnNormScaled = mlx.AddScalar(layer.PostAttnNorm.Weight, 1.0)
		layer.PreFFNormScaled = mlx.AddScalar(layer.PreFFNorm.Weight, 1.0)
		layer.PostFFNormScaled = mlx.AddScalar(layer.PostFFNorm.Weight, 1.0)

		layer.Attention.QNormScaled = mlx.AddScalar(layer.Attention.QNorm.Weight, 1.0)
		layer.Attention.KNormScaled = mlx.AddScalar(layer.Attention.KNorm.Weight, 1.0)
	}

	// Eval all the precomputed weights
	var scaled []*mlx.Array
	scaled = append(scaled, m.NormScaled)
	for _, layer := range m.Layers {
		scaled = append(scaled, layer.InputNormScaled, layer.PostAttnNormScaled,
			layer.PreFFNormScaled, layer.PostFFNormScaled,
			layer.Attention.QNormScaled, layer.Attention.KNormScaled)
	}
	mlx.Eval(scaled...)
}

// isLayerSliding determines if a layer uses sliding window attention
// Pattern N means: layers 0 to N-1 sliding, N full, N+1 to 2N-1 sliding, 2N full, etc.
func isLayerSliding(layerIdx, pattern int32) bool {
	if pattern <= 0 {
		return false // No sliding window
	}
	// Layer is full attention if (layerIdx + 1) % pattern == 0
	return (layerIdx+1)%pattern != 0
}

// Forward runs the text model forward pass
func (m *TextModel) Forward(tokens *mlx.Array, caches []cache.Cache) *mlx.Array {
	B, L := tokens.Shape()[0], tokens.Shape()[1]

	// Get embeddings and scale by sqrt(hidden_size)
	h := m.EmbedTokens.Forward(tokens)
	h = mlx.MulScalar(h, float32(math.Sqrt(float64(m.HiddenSize))))

	for i, layer := range m.Layers {
		h = layer.Forward(h, caches[i], B, L, m.TextConfig)
	}

	// Final norm and output projection
	return m.Output.Forward(mlx.RMSNorm(h, m.NormScaled, m.RMSNormEps))
}

// Forward runs a decoder layer
func (l *DecoderLayer) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *TextConfig) *mlx.Array {
	// Pre-attention norm (use precomputed scaled weight)
	normed := mlx.RMSNorm(x, l.InputNormScaled, cfg.RMSNormEps)

	// Attention
	attnOut := l.Attention.Forward(normed, c, B, L, l.IsSliding, cfg)

	// Post-attention norm and residual
	attnOut = mlx.RMSNorm(attnOut, l.PostAttnNormScaled, cfg.RMSNormEps)
	h := mlx.Add(x, attnOut)

	// Pre-FFN norm
	normed = mlx.RMSNorm(h, l.PreFFNormScaled, cfg.RMSNormEps)

	// MLP
	mlpOut := l.MLP.Forward(normed)

	// Post-FFN norm and residual
	mlpOut = mlx.RMSNorm(mlpOut, l.PostFFNormScaled, cfg.RMSNormEps)
	return mlx.Add(h, mlpOut)
}

// Forward runs attention with Q/K normalization
func (a *Attention) Forward(x *mlx.Array, c cache.Cache, B, L int32, isSliding bool, cfg *TextConfig) *mlx.Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	// Reshape to [B, num_heads, L, head_dim]
	q = mlx.AsStrided(q, []int32{B, cfg.NumAttentionHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumAttentionHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumAttentionHeads * cfg.HeadDim), 1}, 0)
	k = mlx.AsStrided(k, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)
	v = mlx.AsStrided(v, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)

	// Q/K normalization after reshaping (use precomputed scaled weight)
	q = mlx.RMSNorm(q, a.QNormScaled, cfg.RMSNormEps)
	k = mlx.RMSNorm(k, a.KNormScaled, cfg.RMSNormEps)

	// Apply RoPE with appropriate theta
	ropeTheta := cfg.RopeTheta
	if isSliding {
		ropeTheta = cfg.RopeLocalBaseFreq
	}
	q = mlx.RoPE(q, int(cfg.HeadDim), false, ropeTheta, 1.0, c.Offset())
	k = mlx.RoPE(k, int(cfg.HeadDim), false, ropeTheta, 1.0, c.Offset())

	// Update cache
	k, v = c.Update(k, v, int(L))

	// Repeat K/V for GQA if needed
	repeatFactor := cfg.NumAttentionHeads / cfg.NumKeyValueHeads
	if repeatFactor > 1 {
		k = nn.RepeatKV(k, repeatFactor)
		v = nn.RepeatKV(v, repeatFactor)
	}

	// Attention
	out := mlx.ScaledDotProductAttention(q, k, v, cfg.Scale, L > 1)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

// compiledGeluApprox is a singleton compiled GELU function shared across all layers
var compiledGeluApprox *mlx.CompiledFunc

// getCompiledGeluApprox returns the compiled GELU function, creating it once if needed
func getCompiledGeluApprox() *mlx.CompiledFunc {
	if compiledGeluApprox == nil {
		compiledGeluApprox = mlx.CompileShapeless(func(inputs []*mlx.Array) []*mlx.Array {
			return []*mlx.Array{geluApproxImpl(inputs[0])}
		}, true)
	}
	return compiledGeluApprox
}

// Forward runs the MLP with GELU approximation (tanh variant)
func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	gate := getCompiledGeluApprox().Call(m.GateProj.Forward(x))[0]
	return m.DownProj.Forward(mlx.Mul(gate, m.UpProj.Forward(x)))
}

// geluApproxImpl computes GELU using the tanh approximation (gelu_pytorch_tanh):
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func geluApproxImpl(x *mlx.Array) *mlx.Array {
	// Constants
	const sqrt2OverPi = 0.7978845608028654 // sqrt(2/pi)
	const coeff = 0.044715

	// x^3
	x3 := mlx.Mul(mlx.Mul(x, x), x)
	// x + 0.044715 * x^3
	inner := mlx.Add(x, mlx.MulScalar(x3, coeff))
	// sqrt(2/pi) * (x + 0.044715 * x^3)
	scaled := mlx.MulScalar(inner, sqrt2OverPi)
	// tanh(...)
	tanh := mlx.Tanh(scaled)
	// 1 + tanh(...)
	onePlusTanh := mlx.AddScalar(tanh, 1.0)
	// 0.5 * x * (1 + tanh(...))
	return mlx.Mul(mlx.MulScalar(x, 0.5), onePlusTanh)
}

// gemmaRMSNorm applies Gemma-style RMS normalization: x * rsqrt(mean(x^2) + eps) * (1 + weight)
// Uses mlx.RMSNorm fast kernel with pre-computed (1 + weight)
func gemmaRMSNorm(x, weight *mlx.Array, eps float32) *mlx.Array {
	// Gemma uses (1 + weight) instead of weight
	scaledWeight := mlx.AddScalar(weight, 1.0)
	return mlx.RMSNorm(x, scaledWeight, eps)
}

// Interface methods
func (m *TextModel) NumLayers() int          { return len(m.Layers) }
func (m *TextModel) MaxContextLength() int32 { return m.MaxPositionEmbeddings }
func (m *TextModel) VocabSize() int32        { return m.TextConfig.VocabSize }

// Tokenizer returns the tokenizer wrapped to add BOS and apply chat template
func (m *TextModel) Tokenizer() *tokenizer.Tokenizer {
	return m.tok
}

// FormatPrompt applies the Gemma 3 chat template to a prompt
func (m *TextModel) FormatPrompt(prompt string) string {
	// Gemma 3 chat format: <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
	return fmt.Sprintf("<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", prompt)
}

func (m *TextModel) NewCache(maxSeqLen int32) []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i := range caches {
		if m.Layers[i].IsSliding {
			// Use rotating cache for sliding window layers
			caches[i] = cache.NewRotatingKVCache(int(m.SlidingWindow))
		} else {
			// Use regular cache for global attention layers
			caches[i] = cache.NewKVCache()
		}
	}
	return caches
}

// Config holds config for the full multimodal model
type Config struct {
	TextConfig   TextConfig   `json:"text_config"`
	VisionConfig VisionConfig `json:"vision_config"`

	// Image token config (from config.json)
	BOITokenIndex     int32 `json:"boi_token_index"`     // <start_of_image> = 255999
	EOITokenIndex     int32 `json:"eoi_token_index"`     // <end_of_image> = 256000
	ImageTokenIndex   int32 `json:"image_token_index"`   // <image_soft_token> = 262144
	MMTokensPerImage  int32 `json:"mm_tokens_per_image"` // 256
}

// Model is the full Gemma 3 multimodal model
type Model struct {
	VisionTower *VisionTower         `weight:"vision_tower"`
	Projector   *MultiModalProjector `weight:"multi_modal_projector"`
	TextModel   *TextModel           `weight:"language_model"`
	Config      *Config
	tok         *tokenizer.Tokenizer
}

// Load loads the full multimodal Gemma 3 model
func Load(modelPath string) (*Model, error) {
	data, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	// Set defaults for text config (multimodal config often has incomplete text_config)
	// These defaults match transformers.Gemma3TextConfig defaults
	tc := &cfg.TextConfig
	if tc.HeadDim == 0 {
		tc.HeadDim = 256 // Gemma 3 uses head_dim=256
	}
	if tc.NumAttentionHeads == 0 {
		// Gemma 3 4B uses 8 attention heads (cannot infer from hidden_size/head_dim)
		tc.NumAttentionHeads = 8
	}
	if tc.NumKeyValueHeads == 0 {
		// Gemma 3 4B uses 4 KV heads (GQA with 2:1 ratio)
		tc.NumKeyValueHeads = 4
	}
	if tc.VocabSize == 0 {
		tc.VocabSize = 262208 // Gemma 3 vocab size (not 262144!)
	}
	if tc.RopeTheta == 0 {
		tc.RopeTheta = 1000000
	}
	if tc.RopeLocalBaseFreq == 0 {
		tc.RopeLocalBaseFreq = 10000
	}
	if tc.RMSNormEps == 0 {
		tc.RMSNormEps = 1e-6
	}
	if tc.SlidingWindowPattern == 0 {
		tc.SlidingWindowPattern = 6
	}
	if tc.MaxPositionEmbeddings == 0 {
		tc.MaxPositionEmbeddings = 131072 // Gemma 3 4B default
	}

	// Compute text model scale
	tc.Scale = float32(1.0 / math.Sqrt(float64(tc.HeadDim)))

	// Set defaults for image token config
	if cfg.BOITokenIndex == 0 {
		cfg.BOITokenIndex = 255999 // <start_of_image>
	}
	if cfg.EOITokenIndex == 0 {
		cfg.EOITokenIndex = 256000 // <end_of_image>
	}
	if cfg.ImageTokenIndex == 0 {
		cfg.ImageTokenIndex = 262144 // <image_soft_token>
	}
	if cfg.MMTokensPerImage == 0 {
		cfg.MMTokensPerImage = 256
	}

	weights, err := safetensors.LoadModelWeights(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load weights: %w", err)
	}

	tok, err := tokenizer.Load(filepath.Join(modelPath, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	m := &Model{
		VisionTower: &VisionTower{
			Embeddings: &VisionEmbeddings{},
			Encoder:    make([]*VisionEncoderLayer, cfg.VisionConfig.NumHiddenLayers),
			Config:     &cfg.VisionConfig,
		},
		Projector: &MultiModalProjector{},
		TextModel: &TextModel{
			Layers:     make([]*DecoderLayer, cfg.TextConfig.NumHiddenLayers),
			TextConfig: &cfg.TextConfig,
		},
		Config: &cfg,
		tok:    tok,
	}

	// Initialize text layer metadata
	for i := range m.TextModel.Layers {
		m.TextModel.Layers[i] = &DecoderLayer{
			LayerIdx:  int32(i),
			IsSliding: isLayerSliding(int32(i), cfg.TextConfig.SlidingWindowPattern),
		}
	}

	// Initialize vision encoder layers
	for i := range m.VisionTower.Encoder {
		m.VisionTower.Encoder[i] = &VisionEncoderLayer{}
	}

	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return nil, err
	}

	// Tied embeddings for text output
	m.TextModel.Output = nn.NewLinear(m.TextModel.EmbedTokens.Weight, nil)
	m.TextModel.tok = tok

	mlx.Eval(mlx.Collect(m)...)
	weights.ReleaseAll()

	// Precompute (1 + weight) for Gemma-style RMSNorm
	precomputeGemmaScaledWeights(m.TextModel)

	// Precompute projector's scaled weight
	m.Projector.SoftEmbNormScaled = mlx.AddScalar(m.Projector.SoftEmbNorm.Weight, 1.0)
	mlx.Eval(m.Projector.SoftEmbNormScaled)

	return m, nil
}

// Forward runs the text-only forward pass
func (m *Model) Forward(tokens *mlx.Array, caches []cache.Cache) *mlx.Array {
	return m.TextModel.Forward(tokens, caches)
}

// ForwardWithImage runs the multimodal forward pass
// tokens: [B, L] input token IDs (with image placeholder tokens)
// image: [B, H, W, C] preprocessed image tensor
func (m *Model) ForwardWithImage(tokens *mlx.Array, image *mlx.Array, caches []cache.Cache) *mlx.Array {
	B, L := tokens.Shape()[0], tokens.Shape()[1]
	cfg := m.Config.TextConfig

	// Find image token position FIRST before any eval that might free tokens
	imageStartPos := int32(-1)
	if image != nil && B == 1 {
		tokenData := tokens.DataInt32() // This evals tokens
		for i, t := range tokenData {
			if t == m.Config.ImageTokenIndex {
				imageStartPos = int32(i)
				break
			}
		}
	}

	// Get text embeddings and scale
	h := m.TextModel.EmbedTokens.Forward(tokens)
	h = mlx.MulScalar(h, float32(math.Sqrt(float64(cfg.HiddenSize))))

	// Process image if provided
	if image != nil && imageStartPos >= 0 {
		// Vision tower: [B, H, W, C] -> [B, num_patches, vision_hidden]
		visionFeatures := m.VisionTower.Forward(image)

		// Project to text space: [B, num_patches, vision_hidden] -> [B, 256, text_hidden]
		imageEmbeds := m.Projector.Forward(visionFeatures, cfg.RMSNormEps)

		// Eval h and imageEmbeds together so neither gets freed
		mlx.Eval(h, imageEmbeds)

		// Cast imageEmbeds to match text embeddings dtype (bf16)
		if imageEmbeds.Dtype() != h.Dtype() {
			imageEmbeds = mlx.AsType(imageEmbeds, h.Dtype())
			mlx.Eval(imageEmbeds)
		}

		// Insert image embeddings at the known position
		h = m.insertImageEmbeddingsAt(h, imageEmbeds, imageStartPos)
	}

	// Run through text model layers
	for i, layer := range m.TextModel.Layers {
		h = layer.Forward(h, caches[i], B, L, m.TextModel.TextConfig)
	}

	// Final norm and output projection
	return m.TextModel.Output.Forward(mlx.RMSNorm(h, m.TextModel.NormScaled, cfg.RMSNormEps))
}

// insertImageEmbeddingsAt replaces image placeholder tokens with actual image embeddings
// at a known position (to avoid re-scanning tokens after eval)
// textEmbeds: [B, L, hidden_size] text embeddings
// imageEmbeds: [B, 256, hidden_size] image embeddings from projector
// startPos: starting position of image tokens in the sequence
func (m *Model) insertImageEmbeddingsAt(textEmbeds, imageEmbeds *mlx.Array, startPos int32) *mlx.Array {
	numImageTokens := imageEmbeds.Shape()[1]
	L := textEmbeds.Shape()[1]

	// Split text embeddings: [0:startPos] + imageEmbeds + [startPos+256:L]
	afterStart := startPos + numImageTokens

	// Slice before image tokens: textEmbeds[:, 0:startPos, :]
	before := mlx.SliceAxis(textEmbeds, 1, 0, startPos)

	// Slice after image tokens: textEmbeds[:, startPos+256:L, :]
	after := mlx.SliceAxis(textEmbeds, 1, afterStart, L)

	// Concatenate: before + imageEmbeds + after along axis 1
	return mlx.Concatenate([]*mlx.Array{before, imageEmbeds, after}, 1)
}

// Interface methods for Model
func (m *Model) NumLayers() int                         { return len(m.TextModel.Layers) }
func (m *Model) MaxContextLength() int32                { return m.Config.TextConfig.MaxPositionEmbeddings }
func (m *Model) VocabSize() int32                       { return m.Config.TextConfig.VocabSize }
func (m *Model) Tokenizer() *tokenizer.Tokenizer     { return m.tok }
func (m *Model) NewCache(maxSeqLen int32) []cache.Cache { return m.TextModel.NewCache(maxSeqLen) }
func (m *Model) ImageSize() int32                       { return m.Config.VisionConfig.ImageSize }

// FormatPrompt applies the Gemma 3 multimodal chat template
func (m *Model) FormatPrompt(prompt string) string {
	return fmt.Sprintf("<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", prompt)
}

// FormatPromptWithImage applies the Gemma 3 multimodal chat template with image
func (m *Model) FormatPromptWithImage(prompt string) string {
	return fmt.Sprintf("<start_of_turn>user\n<start_of_image>%s<end_of_turn>\n<start_of_turn>model\n", prompt)
}

// ExpandImageTokens expands <start_of_image> into 256 image placeholder tokens
// Input tokens containing boi_token (255999) are expanded to:
// boi_token + 256 * image_token + eoi_token
func (m *Model) ExpandImageTokens(tokens []int32) []int32 {
	result := make([]int32, 0, len(tokens)+int(m.Config.MMTokensPerImage)+1)

	for _, t := range tokens {
		if t == m.Config.BOITokenIndex {
			// Expand: boi + 256 * image_token + eoi
			result = append(result, m.Config.BOITokenIndex)
			for i := int32(0); i < m.Config.MMTokensPerImage; i++ {
				result = append(result, m.Config.ImageTokenIndex)
			}
			result = append(result, m.Config.EOITokenIndex)
		} else {
			result = append(result, t)
		}
	}

	return result
}

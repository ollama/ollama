//go:build mlx

package glm_image

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// T5Config holds T5 encoder configuration
type T5Config struct {
	DModel      int32   `json:"d_model"`               // 1472
	DFF         int32   `json:"d_ff"`                  // 3584
	DKV         int32   `json:"d_kv"`                  // 64
	NumHeads    int32   `json:"num_heads"`             // 6
	NumLayers   int32   `json:"num_layers"`            // 12
	VocabSize   int32   `json:"vocab_size"`            // 384 (byte-level)
	LayerNormEps float32 `json:"layer_norm_epsilon"`   // 1e-6
	IsGatedAct  bool    `json:"is_gated_act"`          // true (gated-gelu)

	// Relative position bias
	RelativeAttentionNumBuckets  int32 `json:"relative_attention_num_buckets"`  // 32
	RelativeAttentionMaxDistance int32 `json:"relative_attention_max_distance"` // 128
}

// T5TextEncoder is the T5 encoder for text conditioning
type T5TextEncoder struct {
	Config *T5Config

	// Embedding (shared for ByT5)
	SharedEmbed *nn.Embedding `weight:"shared"`

	// Encoder layers
	Layers []*T5Block `weight:"encoder.block"`

	// Final layer norm
	FinalNorm *T5LayerNorm `weight:"encoder.final_layer_norm"`

	// Relative position bias (from first layer, shared across all)
	RelativeAttentionBias *mlx.Array `weight:"encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"`
}

// T5Block is a single T5 encoder block
type T5Block struct {
	// Self attention
	Layer0 *T5LayerSelfAttention `weight:"layer.0"`
	// FFN
	Layer1 *T5LayerFF `weight:"layer.1"`
}

// T5LayerSelfAttention is T5's self-attention layer
type T5LayerSelfAttention struct {
	SelfAttention *T5Attention `weight:"SelfAttention"`
	LayerNorm     *T5LayerNorm `weight:"layer_norm"`
}

// T5Attention implements T5's relative attention
type T5Attention struct {
	Q *mlx.Array `weight:"q.weight"` // No bias in T5
	K *mlx.Array `weight:"k.weight"`
	V *mlx.Array `weight:"v.weight"`
	O *mlx.Array `weight:"o.weight"`

	NHeads int32
	DKV    int32
	Scale  float32
}

// T5LayerFF is T5's feedforward layer with gated-gelu
type T5LayerFF struct {
	DenseReluDense *T5DenseGatedGelu `weight:"DenseReluDense"`
	LayerNorm      *T5LayerNorm      `weight:"layer_norm"`
}

// T5DenseGatedGelu is T5's gated-gelu FFN
type T5DenseGatedGelu struct {
	Wi0 *mlx.Array `weight:"wi_0.weight"` // gate projection
	Wi1 *mlx.Array `weight:"wi_1.weight"` // up projection
	Wo  *mlx.Array `weight:"wo.weight"`   // down projection
}

// T5LayerNorm is T5's RMSNorm variant (no bias, no mean subtraction)
type T5LayerNorm struct {
	Weight *mlx.Array `weight:"weight"`
	Eps    float32
}

// Load loads the T5 text encoder from manifest
func (m *T5TextEncoder) Load(manifest *imagegen.ModelManifest) error {
	fmt.Print("  Loading T5 text encoder... ")

	// Load config
	var cfg T5Config
	if err := manifest.ReadConfigJSON("text_encoder/config.json", &cfg); err != nil {
		return fmt.Errorf("config: %w", err)
	}
	m.Config = &cfg

	// Pre-allocate layers
	m.Layers = make([]*T5Block, cfg.NumLayers)

	// Load weights
	weights, err := imagegen.LoadWeightsFromManifest(manifest, "text_encoder")
	if err != nil {
		return fmt.Errorf("weights: %w", err)
	}
	if err := weights.Load(0); err != nil {
		return fmt.Errorf("load weights: %w", err)
	}
	defer weights.ReleaseAll()

	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return fmt.Errorf("load module: %w", err)
	}

	m.initComputedFields()
	fmt.Println("✓")
	return nil
}

// LoadFromPath loads the T5 text encoder from a directory path
func (m *T5TextEncoder) LoadFromPath(path string) error {
	fmt.Print("  Loading T5 text encoder... ")

	// Load config
	var cfg T5Config
	configPath := filepath.Join(path, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("read config: %w", err)
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return fmt.Errorf("parse config: %w", err)
	}
	m.Config = &cfg

	// Pre-allocate layers
	m.Layers = make([]*T5Block, cfg.NumLayers)

	// Load weights from safetensors files
	weights, err := safetensors.LoadModelWeights(path)
	if err != nil {
		return fmt.Errorf("weights: %w", err)
	}
	if err := weights.Load(0); err != nil {
		return fmt.Errorf("load weights: %w", err)
	}
	defer weights.ReleaseAll()

	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return fmt.Errorf("load module: %w", err)
	}

	m.initComputedFields()
	fmt.Println("✓")
	return nil
}

func (m *T5TextEncoder) initComputedFields() {
	cfg := m.Config
	m.FinalNorm.Eps = cfg.LayerNormEps
	for _, block := range m.Layers {
		attn := block.Layer0.SelfAttention
		attn.NHeads = cfg.NumHeads
		attn.DKV = cfg.DKV
		attn.Scale = float32(1.0 / math.Sqrt(float64(cfg.DKV)))

		block.Layer0.LayerNorm.Eps = cfg.LayerNormEps
		block.Layer1.LayerNorm.Eps = cfg.LayerNormEps
	}
}

// Forward encodes text tokens
func (m *T5TextEncoder) Forward(tokens *mlx.Array) *mlx.Array {
	cfg := m.Config

	// Get embeddings
	h := m.SharedEmbed.Forward(tokens)

	// Compute relative position bias once
	seqLen := tokens.Shape()[1]
	posBias := m.computeRelativePositionBias(seqLen)

	// Forward through layers
	for _, block := range m.Layers {
		h = block.Forward(h, posBias, cfg.LayerNormEps)
	}

	// Final norm
	h = m.FinalNorm.Forward(h)

	return h
}

// extractGlyphTexts extracts quoted text (glyphs) from the prompt
// This matches diffusers' get_glyph_texts from pipeline_glm_image.py
// Glyph texts are used for text rendering guidance in the generated image
func extractGlyphTexts(prompt string) []string {
	var glyphTexts []string

	// Extract text in single quotes: 'text'
	re1 := regexp.MustCompile(`'([^']*)'`)
	for _, match := range re1.FindAllStringSubmatch(prompt, -1) {
		if len(match) > 1 {
			glyphTexts = append(glyphTexts, match[1])
		}
	}

	// Extract text in Unicode curly double quotes: "text"
	re2 := regexp.MustCompile(`"([^""]*)"`)
	for _, match := range re2.FindAllStringSubmatch(prompt, -1) {
		if len(match) > 1 {
			glyphTexts = append(glyphTexts, match[1])
		}
	}

	// Extract text in ASCII double quotes: "text"
	re3 := regexp.MustCompile(`"([^"]*)"`)
	for _, match := range re3.FindAllStringSubmatch(prompt, -1) {
		if len(match) > 1 {
			glyphTexts = append(glyphTexts, match[1])
		}
	}

	// Extract text in Japanese quotes: 「text」
	re4 := regexp.MustCompile(`「([^「」]*)」`)
	for _, match := range re4.FindAllStringSubmatch(prompt, -1) {
		if len(match) > 1 {
			glyphTexts = append(glyphTexts, match[1])
		}
	}

	return glyphTexts
}

// EncodePrompt encodes the prompt text using the ByT5 tokenizer and encoder
// This provides text conditioning for the diffusion transformer via the glyph projector
//
// IMPORTANT: This encodes only the GLYPH TEXTS (quoted strings in the prompt), not the
// full prompt. Glyph texts are used for text rendering guidance in the generated image.
// Multiple glyph texts are encoded and concatenated to form the conditioning signal.
// This matches diffusers' _get_glyph_embeds() behavior.
func (m *T5TextEncoder) EncodePrompt(tok *ByT5Tokenizer, prompt string) *mlx.Array {
	// Extract glyph texts from prompt (text in quotes)
	glyphTexts := extractGlyphTexts(prompt)

	// If no glyph texts found, encode empty string (matches diffusers: [""] fallback)
	if len(glyphTexts) == 0 {
		glyphTexts = []string{""}
	}

	// Encode each glyph text and collect token sequences
	// Matching diffusers' _get_glyph_embeds() which batches all glyph texts
	var allTokenSeqs [][]int32

	for _, glyphText := range glyphTexts {
		// ByT5 uses byte-level encoding: each byte (0-255) -> token (3-258)
		tokens := tok.Encode(glyphText)

		// Add EOS token (1) at the end to match HuggingFace tokenizer behavior
		tokens = append(tokens, tok.EOSTokenID)

		allTokenSeqs = append(allTokenSeqs, tokens)
	}

	// Process each glyph text through the encoder
	var allEmbeddings []*mlx.Array
	for _, tokens := range allTokenSeqs {
		tokenLen := len(tokens)
		if tokenLen == 0 {
			continue
		}

		// Create token array [1, L]
		tokensArr := mlx.NewArrayInt32(tokens, []int32{1, int32(tokenLen)})

		// Forward through encoder
		output := m.Forward(tokensArr)
		mlx.Eval(output)

		allEmbeddings = append(allEmbeddings, output)
	}

	// Concatenate all glyph embeddings along sequence dimension
	var output *mlx.Array
	if len(allEmbeddings) == 0 {
		// Fallback: return single zero embedding
		output = mlx.Zeros([]int32{1, 1, m.Config.DModel}, mlx.DtypeBFloat16)
	} else if len(allEmbeddings) == 1 {
		output = allEmbeddings[0]
	} else {
		output = mlx.Concatenate(allEmbeddings, 1)
	}
	mlx.Eval(output)

	return output
}

// computeRelativePositionBias computes T5's relative position encoding
func (m *T5TextEncoder) computeRelativePositionBias(seqLen int32) *mlx.Array {
	cfg := m.Config

	// Create relative position matrix
	// For each (query_pos, key_pos) pair, compute bucketed relative position
	numBuckets := cfg.RelativeAttentionNumBuckets
	maxDistance := cfg.RelativeAttentionMaxDistance

	// Create position indices
	contextPos := make([]int32, seqLen*seqLen)
	memoryPos := make([]int32, seqLen*seqLen)
	for i := int32(0); i < seqLen; i++ {
		for j := int32(0); j < seqLen; j++ {
			contextPos[i*seqLen+j] = i
			memoryPos[i*seqLen+j] = j
		}
	}

	// Compute relative positions and bucket them
	buckets := make([]int32, seqLen*seqLen)
	for i := int32(0); i < seqLen*seqLen; i++ {
		relPos := memoryPos[i] - contextPos[i]
		buckets[i] = relativePosistionBucket(relPos, numBuckets, maxDistance, false)
	}

	// Create bucket indices array
	bucketsArr := mlx.NewArrayInt32(buckets, []int32{seqLen, seqLen})

	// Look up bias: RelativeAttentionBias shape is [numBuckets, numHeads] = [32, 6]
	// Take along axis 0 (buckets dimension) -> [seqLen, seqLen, numHeads]
	bias := mlx.Take(m.RelativeAttentionBias, bucketsArr, 0) // [seqLen, seqLen, numHeads]

	// Transpose to [numHeads, seqLen, seqLen]
	bias = mlx.Transpose(bias, 2, 0, 1) // [numHeads, seqLen, seqLen]
	bias = mlx.ExpandDims(bias, 0)      // [1, numHeads, seqLen, seqLen]

	return bias
}

// relativePosistionBucket computes the bucket for a relative position
func relativePosistionBucket(relativePosition int32, numBuckets int32, maxDistance int32, bidirectional bool) int32 {
	var bucket int32 = 0
	var n int32 = -relativePosition

	if bidirectional {
		numBuckets /= 2
		if n < 0 {
			bucket += numBuckets
			n = -n
		}
	} else {
		if n < 0 {
			n = 0
		}
	}

	// Half buckets are for exact positions, half are for log-spaced
	maxExact := numBuckets / 2
	if n < maxExact {
		bucket += n
	} else {
		// Log-spaced buckets
		logVal := math.Log(float64(n)/float64(maxExact)) / math.Log(float64(maxDistance)/float64(maxExact))
		bucket += maxExact + int32(logVal*float64(numBuckets-maxExact))
		if bucket > numBuckets-1 {
			bucket = numBuckets - 1
		}
	}

	return bucket
}

// Forward for T5Block
func (b *T5Block) Forward(x *mlx.Array, posBias *mlx.Array, eps float32) *mlx.Array {
	// Self attention with residual
	h := b.Layer0.Forward(x, posBias, eps)

	// FFN with residual
	h = b.Layer1.Forward(h, eps)

	return h
}

// Forward for T5LayerSelfAttention
func (l *T5LayerSelfAttention) Forward(x *mlx.Array, posBias *mlx.Array, eps float32) *mlx.Array {
	// Pre-norm
	normed := l.LayerNorm.Forward(x)

	// Attention
	attnOut := l.SelfAttention.Forward(normed, posBias)

	// Residual
	return mlx.Add(x, attnOut)
}

// Forward for T5Attention
func (attn *T5Attention) Forward(x *mlx.Array, posBias *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	D := shape[2]

	// Q, K, V projections (no bias)
	// Weights are [out_features, in_features], so we use matmul with transpose
	q := mlx.Matmul(x, mlx.Transpose(attn.Q, 1, 0))
	k := mlx.Matmul(x, mlx.Transpose(attn.K, 1, 0))
	v := mlx.Matmul(x, mlx.Transpose(attn.V, 1, 0))

	// Reshape to [B, L, nheads, d_kv]
	q = mlx.Reshape(q, B, L, attn.NHeads, attn.DKV)
	k = mlx.Reshape(k, B, L, attn.NHeads, attn.DKV)
	v = mlx.Reshape(v, B, L, attn.NHeads, attn.DKV)

	// Transpose to [B, nheads, L, d_kv]
	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	// Attention scores with relative position bias
	// T5 uses UNSCALED dot-product attention: scores = q @ k.T + pos_bias
	// (no 1/sqrt(d_k) scale factor like in standard transformers)
	scores := mlx.Matmul(q, mlx.Transpose(k, 0, 1, 3, 2))
	scores = mlx.Add(scores, posBias)

	// Softmax
	attnWeights := mlx.Softmax(scores, -1)

	// Attend to values
	out := mlx.Matmul(attnWeights, v)

	// Transpose back [B, nheads, L, d_kv] -> [B, L, nheads, d_kv]
	out = mlx.Transpose(out, 0, 2, 1, 3)
	// Reshape to [B, L, D]
	out = mlx.Reshape(out, B, L, attn.NHeads*attn.DKV)

	// Output projection
	out = mlx.Matmul(out, mlx.Transpose(attn.O, 1, 0))

	_ = D // Silence unused warning
	return out
}

// Forward for T5LayerFF
func (l *T5LayerFF) Forward(x *mlx.Array, eps float32) *mlx.Array {
	// Pre-norm
	normed := l.LayerNorm.Forward(x)

	// FFN
	ffOut := l.DenseReluDense.Forward(normed)

	// Residual
	return mlx.Add(x, ffOut)
}

// geluNew implements the GELU activation with tanh approximation (gelu_new)
// This matches HuggingFace transformers' gelu_new/OpenAI GPT implementation
// Formula: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
func geluNew(x *mlx.Array) *mlx.Array {
	sqrt2OverPi := float32(0.7978845608) // sqrt(2/π)
	coeff := float32(0.044715)

	x3 := mlx.Mul(mlx.Mul(x, x), x)
	inner := mlx.MulScalar(mlx.Add(x, mlx.MulScalar(x3, coeff)), sqrt2OverPi)
	return mlx.Mul(mlx.MulScalar(x, 0.5), mlx.AddScalar(mlx.Tanh(inner), 1.0))
}

// Forward for T5DenseGatedGelu (gated-gelu activation)
func (d *T5DenseGatedGelu) Forward(x *mlx.Array) *mlx.Array {
	// Gate projection with GELU activation (T5 v1.1/ByT5 uses gelu_new)
	gate := mlx.Matmul(x, mlx.Transpose(d.Wi0, 1, 0))
	gate = geluNew(gate)

	// Up projection
	up := mlx.Matmul(x, mlx.Transpose(d.Wi1, 1, 0))

	// Gated output
	h := mlx.Mul(gate, up)

	// Down projection
	return mlx.Matmul(h, mlx.Transpose(d.Wo, 1, 0))
}

// Forward for T5LayerNorm (RMSNorm variant)
func (ln *T5LayerNorm) Forward(x *mlx.Array) *mlx.Array {
	// T5 uses RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
	variance := mlx.Mean(mlx.Square(x), -1, true)
	x = mlx.Mul(x, mlx.RSqrt(mlx.AddScalar(variance, ln.Eps)))
	return mlx.Mul(x, ln.Weight)
}

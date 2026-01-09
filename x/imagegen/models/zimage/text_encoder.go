//go:build mlx

package zimage

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
	"github.com/ollama/ollama/x/imagegen/safetensors"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
)

// Qwen3Config holds Qwen3 text encoder configuration
type Qwen3Config struct {
	HiddenSize        int32   `json:"hidden_size"`
	NumHiddenLayers   int32   `json:"num_hidden_layers"`
	IntermediateSize  int32   `json:"intermediate_size"`
	NumAttentionHeads int32   `json:"num_attention_heads"`
	NumKeyValueHeads  int32   `json:"num_key_value_heads"`
	VocabSize         int32   `json:"vocab_size"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	RopeTheta         float32 `json:"rope_theta"`
	HeadDim           int32   `json:"head_dim"`
}

// loadQwen3Config loads text encoder config from a JSON file
func loadQwen3Config(path string) (*Qwen3Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var cfg Qwen3Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	return &cfg, nil
}

// Qwen3Attention implements Qwen3 attention with QK norms
type Qwen3Attention struct {
	QProj *nn.Linear  `weight:"q_proj"`
	KProj *nn.Linear  `weight:"k_proj"`
	VProj *nn.Linear  `weight:"v_proj"`
	OProj *nn.Linear  `weight:"o_proj"`
	QNorm *nn.RMSNorm `weight:"q_norm"`
	KNorm *nn.RMSNorm `weight:"k_norm"`
	// Computed fields
	NHeads    int32
	NKVHeads  int32
	HeadDim   int32
	Scale     float32
	RopeTheta float32
}

// applyRoPEQwen3 applies the custom RoPE for Qwen3 text encoder
func applyRoPEQwen3(x *mlx.Array, seqLen int32, theta float32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	H := shape[2]
	D := shape[3]
	half := D / 2

	freqsArr := make([]float32, half)
	logTheta := float32(math.Log(float64(theta)))
	for i := int32(0); i < half; i++ {
		freqsArr[i] = float32(math.Exp(float64(-logTheta * float32(i) / float32(half))))
	}
	freqs := mlx.NewArray(freqsArr, []int32{half})

	posArr := make([]float32, seqLen)
	for i := int32(0); i < seqLen; i++ {
		posArr[i] = float32(i)
	}
	pos := mlx.NewArray(posArr, []int32{seqLen})

	posExpanded := mlx.Reshape(pos, seqLen, 1)
	freqsExpanded := mlx.Reshape(freqs, 1, half)
	args := mlx.Mul(posExpanded, freqsExpanded)

	cosVals := mlx.Cos(args)
	sinVals := mlx.Sin(args)
	cosVals = mlx.Reshape(cosVals, seqLen, 1, half)
	sinVals = mlx.Reshape(sinVals, seqLen, 1, half)

	x1 := mlx.Slice(x, []int32{0, 0, 0, 0}, []int32{B, L, H, half})
	x2 := mlx.Slice(x, []int32{0, 0, 0, half}, []int32{B, L, H, D})

	part1 := mlx.Sub(mlx.Mul(x1, cosVals), mlx.Mul(x2, sinVals))
	part2 := mlx.Add(mlx.Mul(x1, sinVals), mlx.Mul(x2, cosVals))

	return mlx.Concatenate([]*mlx.Array{part1, part2}, 3)
}

// Forward computes attention with causal masking
func (attn *Qwen3Attention) Forward(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]

	q := attn.QProj.Forward(x)
	k := attn.KProj.Forward(x)
	v := attn.VProj.Forward(x)

	q = mlx.Reshape(q, B, L, attn.NHeads, attn.HeadDim)
	k = mlx.Reshape(k, B, L, attn.NKVHeads, attn.HeadDim)
	v = mlx.Reshape(v, B, L, attn.NKVHeads, attn.HeadDim)

	// QK norm uses 1e-6 hardcoded (Qwen3 specific)
	q = attn.QNorm.Forward(q, 1e-6)
	k = attn.KNorm.Forward(k, 1e-6)

	q = applyRoPEQwen3(q, L, attn.RopeTheta)
	k = applyRoPEQwen3(k, L, attn.RopeTheta)

	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	if attn.NKVHeads < attn.NHeads {
		repeats := attn.NHeads / attn.NKVHeads
		k = repeatKV(k, repeats)
		v = repeatKV(v, repeats)
	}

	out := mlx.ScaledDotProductAttention(q, k, v, attn.Scale, true)

	out = mlx.Transpose(out, 0, 2, 1, 3)
	out = mlx.Reshape(out, B, L, attn.NHeads*attn.HeadDim)

	out = attn.OProj.Forward(out)

	return out
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

// Qwen3MLP implements Qwen3 SwiGLU MLP
type Qwen3MLP struct {
	GateProj *nn.Linear `weight:"gate_proj"`
	UpProj   *nn.Linear `weight:"up_proj"`
	DownProj *nn.Linear `weight:"down_proj"`
}

// Forward applies the MLP
func (m *Qwen3MLP) Forward(x *mlx.Array) *mlx.Array {
	gate := m.GateProj.Forward(x)
	gate = mlx.SiLU(gate)
	up := m.UpProj.Forward(x)
	h := mlx.Mul(gate, up)
	return m.DownProj.Forward(h)
}

// Qwen3Block represents a single Qwen3 transformer block
type Qwen3Block struct {
	Attention         *Qwen3Attention `weight:"self_attn"`
	MLP               *Qwen3MLP       `weight:"mlp"`
	InputLayerNorm    *nn.RMSNorm     `weight:"input_layernorm"`
	PostAttnLayerNorm *nn.RMSNorm     `weight:"post_attention_layernorm"`
}

// Forward applies the Qwen3 block
func (qb *Qwen3Block) Forward(x *mlx.Array, eps float32) *mlx.Array {
	h := qb.InputLayerNorm.Forward(x, eps)
	attnOut := qb.Attention.Forward(h)
	x = mlx.Add(x, attnOut)

	h = qb.PostAttnLayerNorm.Forward(x, eps)
	mlpOut := qb.MLP.Forward(h)
	x = mlx.Add(x, mlpOut)

	return x
}

// Qwen3TextEncoder is the full Qwen3 encoder for Z-Image
type Qwen3TextEncoder struct {
	EmbedTokens *nn.Embedding   `weight:"model.embed_tokens"`
	Layers      []*Qwen3Block   `weight:"model.layers"`
	FinalNorm   *nn.RMSNorm     `weight:"model.norm"`
	*Qwen3Config
}

// Load loads the Qwen3 text encoder from a directory
func (m *Qwen3TextEncoder) Load(path string) error {
	fmt.Println("Loading Qwen3 text encoder...")

	// Load config
	cfg, err := loadQwen3Config(filepath.Join(path, "config.json"))
	if err != nil {
		return fmt.Errorf("config: %w", err)
	}
	m.Qwen3Config = cfg

	// Pre-allocate layers slice
	m.Layers = make([]*Qwen3Block, cfg.NumHiddenLayers)

	// Load weights
	weights, err := safetensors.LoadModelWeights(path)
	if err != nil {
		return fmt.Errorf("weights: %w", err)
	}

	fmt.Print("  Loading weights via struct tags... ")
	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return fmt.Errorf("load module: %w", err)
	}
	fmt.Println("✓")

	// Initialize computed fields
	m.FinalNorm.Eps = cfg.RMSNormEps
	for _, block := range m.Layers {
		// Attention
		block.Attention.NHeads = cfg.NumAttentionHeads
		block.Attention.NKVHeads = cfg.NumKeyValueHeads
		block.Attention.HeadDim = cfg.HeadDim
		block.Attention.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
		block.Attention.RopeTheta = cfg.RopeTheta
		block.Attention.QNorm.Eps = cfg.RMSNormEps
		block.Attention.KNorm.Eps = cfg.RMSNormEps
		// Block norms
		block.InputLayerNorm.Eps = cfg.RMSNormEps
		block.PostAttnLayerNorm.Eps = cfg.RMSNormEps
	}

	weights.ReleaseAll()
	return nil
}

// Forward encodes text tokens
func (te *Qwen3TextEncoder) Forward(tokens *mlx.Array) *mlx.Array {
	h := te.EmbedTokens.Forward(tokens)
	eps := te.RMSNormEps

	for _, layer := range te.Layers {
		h = layer.Forward(h, eps)
	}

	// Apply final RMS norm
	h = te.FinalNorm.Forward(h, eps)

	return h
}

// ApplyChatTemplate wraps prompt in Qwen3 chat format
func ApplyChatTemplate(prompt string) string {
	return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
}

// EncodePrompt encodes a text prompt using the tokenizer and encoder
func (te *Qwen3TextEncoder) EncodePrompt(tok *tokenizer.Tokenizer, prompt string, maxLen int) (*mlx.Array, *mlx.Array) {
	formattedPrompt := ApplyChatTemplate(prompt)

	tokens := tok.Encode(formattedPrompt, false)

	if len(tokens) > maxLen {
		tokens = tokens[:maxLen]
	}

	maskData := make([]float32, maxLen)
	for i := 0; i < len(tokens); i++ {
		maskData[i] = 1.0
	}

	// Get PAD token (different from EOS for Qwen3)
	padToken := tok.PAD()
	if padToken < 0 {
		padToken = tok.EOS() // fallback
	}

	paddedTokens := make([]int32, maxLen)
	copy(paddedTokens, tokens)
	for i := len(tokens); i < maxLen; i++ {
		paddedTokens[i] = padToken
	}

	tokensArr := mlx.NewArrayInt32(paddedTokens, []int32{1, int32(maxLen)})
	maskArr := mlx.NewArray(maskData, []int32{1, int32(maxLen)})

	embeddings := te.Forward(tokensArr)

	return embeddings, maskArr
}

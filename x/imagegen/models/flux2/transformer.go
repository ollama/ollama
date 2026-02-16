//go:build mlx

package flux2

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// TransformerConfig holds Flux2 transformer configuration
type TransformerConfig struct {
	AttentionHeadDim         int32   `json:"attention_head_dim"`          // 128
	AxesDimsRoPE             []int32 `json:"axes_dims_rope"`              // [32, 32, 32, 32]
	Eps                      float32 `json:"eps"`                         // 1e-6
	GuidanceEmbeds           bool    `json:"guidance_embeds"`             // false for Klein
	InChannels               int32   `json:"in_channels"`                 // 128
	JointAttentionDim        int32   `json:"joint_attention_dim"`         // 7680
	MLPRatio                 float32 `json:"mlp_ratio"`                   // 3.0
	NumAttentionHeads        int32   `json:"num_attention_heads"`         // 24
	NumLayers                int32   `json:"num_layers"`                  // 5
	NumSingleLayers          int32   `json:"num_single_layers"`           // 20
	PatchSize                int32   `json:"patch_size"`                  // 1
	RopeTheta                int32   `json:"rope_theta"`                  // 2000
	TimestepGuidanceChannels int32   `json:"timestep_guidance_channels"`  // 256
}

// Computed dimensions
func (c *TransformerConfig) InnerDim() int32 {
	return c.NumAttentionHeads * c.AttentionHeadDim // 24 * 128 = 3072
}

func (c *TransformerConfig) MLPHiddenDim() int32 {
	return int32(float32(c.InnerDim()) * c.MLPRatio) // 3072 * 3.0 = 9216
}

// TimestepEmbedder creates timestep embeddings
// Weight names: time_guidance_embed.timestep_embedder.linear_1.weight, linear_2.weight
type TimestepEmbedder struct {
	Linear1  nn.LinearLayer `weight:"linear_1"`
	Linear2  nn.LinearLayer `weight:"linear_2"`
	EmbedDim int32          // 256
}

// Forward creates sinusoidal embeddings and projects them
func (t *TimestepEmbedder) Forward(timesteps *mlx.Array) *mlx.Array {
	half := t.EmbedDim / 2
	freqs := make([]float32, half)
	for i := int32(0); i < half; i++ {
		freqs[i] = float32(math.Exp(-math.Log(10000.0) * float64(i) / float64(half)))
	}
	freqsArr := mlx.NewArray(freqs, []int32{1, half})

	// timesteps: [B] -> [B, 1]
	tExpanded := mlx.ExpandDims(timesteps, 1)
	// args: [B, half]
	args := mlx.Mul(tExpanded, freqsArr)

	// [cos(args), sin(args)] -> [B, embed_dim]
	sinEmbed := mlx.Concatenate([]*mlx.Array{mlx.Cos(args), mlx.Sin(args)}, 1)

	// MLP: linear_1 -> silu -> linear_2
	h := t.Linear1.Forward(sinEmbed)
	h = mlx.SiLU(h)
	return t.Linear2.Forward(h)
}

// TimeGuidanceEmbed wraps the timestep embedder
// Weight names: time_guidance_embed.timestep_embedder.*
type TimeGuidanceEmbed struct {
	TimestepEmbedder *TimestepEmbedder `weight:"timestep_embedder"`
}

// Forward computes timestep embeddings
func (t *TimeGuidanceEmbed) Forward(timesteps *mlx.Array) *mlx.Array {
	return t.TimestepEmbedder.Forward(timesteps)
}

// Modulation computes adaptive modulation parameters
// Weight names: double_stream_modulation_img.linear.weight, etc.
type Modulation struct {
	Linear nn.LinearLayer `weight:"linear"`
}

// Forward computes modulation parameters
func (m *Modulation) Forward(temb *mlx.Array) *mlx.Array {
	h := mlx.SiLU(temb)
	return m.Linear.Forward(h)
}

// TransformerBlockAttn implements dual-stream attention
// Weight names: transformer_blocks.N.attn.*
type TransformerBlockAttn struct {
	// Image stream (separate Q, K, V projections)
	ToQ nn.LinearLayer `weight:"to_q"`
	ToK nn.LinearLayer `weight:"to_k"`
	ToV nn.LinearLayer `weight:"to_v"`
	// Note: to_out has .0 suffix in weights, handled specially
	ToOut0 nn.LinearLayer `weight:"to_out.0"`

	// Text stream (add_ projections)
	AddQProj nn.LinearLayer `weight:"add_q_proj"`
	AddKProj nn.LinearLayer `weight:"add_k_proj"`
	AddVProj nn.LinearLayer `weight:"add_v_proj"`
	ToAddOut nn.LinearLayer `weight:"to_add_out"`

	// QK norms for image stream
	NormQ *mlx.Array `weight:"norm_q.weight"`
	NormK *mlx.Array `weight:"norm_k.weight"`

	// QK norms for text stream (added)
	NormAddedQ *mlx.Array `weight:"norm_added_q.weight"`
	NormAddedK *mlx.Array `weight:"norm_added_k.weight"`
}

// FeedForward implements SwiGLU MLP
// Weight names: transformer_blocks.N.ff.linear_in.weight, linear_out.weight
type FeedForward struct {
	LinearIn  nn.LinearLayer `weight:"linear_in"`
	LinearOut nn.LinearLayer `weight:"linear_out"`
}

// Forward applies SwiGLU MLP
func (ff *FeedForward) Forward(x *mlx.Array) *mlx.Array {
	// LinearIn outputs 2x hidden dim for SwiGLU
	h := ff.LinearIn.Forward(x)
	shape := h.Shape()
	half := shape[len(shape)-1] / 2

	// Split into gate and up
	gate := mlx.Slice(h, []int32{0, 0, 0}, []int32{shape[0], shape[1], half})
	up := mlx.Slice(h, []int32{0, 0, half}, []int32{shape[0], shape[1], shape[2]})

	// SwiGLU: silu(gate) * up
	h = mlx.Mul(mlx.SiLU(gate), up)
	return ff.LinearOut.Forward(h)
}

// TransformerBlock implements a dual-stream transformer block
// Weight names: transformer_blocks.N.*
type TransformerBlock struct {
	Attn      *TransformerBlockAttn `weight:"attn"`
	FF        *FeedForward          `weight:"ff"`
	FFContext *FeedForward          `weight:"ff_context"`

	// Config (set after loading)
	NHeads  int32
	HeadDim int32
	Scale   float32
}

// Forward applies the dual-stream block
// imgHidden: [B, imgLen, dim]
// txtHidden: [B, txtLen, dim]
// imgMod, txtMod: modulation params [B, 6*dim] each
// cos, sin: RoPE values
func (block *TransformerBlock) Forward(imgHidden, txtHidden *mlx.Array, imgMod, txtMod *mlx.Array, cos, sin *mlx.Array) (*mlx.Array, *mlx.Array) {
	imgShape := imgHidden.Shape()
	B := imgShape[0]
	imgLen := imgShape[1]
	dim := imgShape[2]
	txtLen := txtHidden.Shape()[1]

	// Parse modulation: 6 params each (shift1, scale1, gate1, shift2, scale2, gate2)
	imgShift1, imgScale1, imgGate1 := parseModulation3(imgMod, dim, 0)
	imgShift2, imgScale2, imgGate2 := parseModulation3(imgMod, dim, 3)
	txtShift1, txtScale1, txtGate1 := parseModulation3(txtMod, dim, 0)
	txtShift2, txtScale2, txtGate2 := parseModulation3(txtMod, dim, 3)

	// === Attention branch ===
	// Modulate inputs
	imgNorm := modulateLayerNorm(imgHidden, imgShift1, imgScale1)
	txtNorm := modulateLayerNorm(txtHidden, txtShift1, txtScale1)

	// Compute Q, K, V for image stream (separate projections)
	imgQ := block.Attn.ToQ.Forward(imgNorm)
	imgK := block.Attn.ToK.Forward(imgNorm)
	imgV := block.Attn.ToV.Forward(imgNorm)

	// Compute Q, K, V for text stream (add_ projections)
	txtQ := block.Attn.AddQProj.Forward(txtNorm)
	txtK := block.Attn.AddKProj.Forward(txtNorm)
	txtV := block.Attn.AddVProj.Forward(txtNorm)

	// Reshape for attention: [B, L, dim] -> [B, L, nheads, headDim]
	imgQ = mlx.Reshape(imgQ, B, imgLen, block.NHeads, block.HeadDim)
	imgK = mlx.Reshape(imgK, B, imgLen, block.NHeads, block.HeadDim)
	imgV = mlx.Reshape(imgV, B, imgLen, block.NHeads, block.HeadDim)
	txtQ = mlx.Reshape(txtQ, B, txtLen, block.NHeads, block.HeadDim)
	txtK = mlx.Reshape(txtK, B, txtLen, block.NHeads, block.HeadDim)
	txtV = mlx.Reshape(txtV, B, txtLen, block.NHeads, block.HeadDim)

	// Apply QK norm (RMSNorm with learned scale)
	imgQ = applyQKNorm(imgQ, block.Attn.NormQ)
	imgK = applyQKNorm(imgK, block.Attn.NormK)
	txtQ = applyQKNorm(txtQ, block.Attn.NormAddedQ)
	txtK = applyQKNorm(txtK, block.Attn.NormAddedK)

	// Concatenate for joint attention: text first, then image
	q := mlx.Concatenate([]*mlx.Array{txtQ, imgQ}, 1)
	k := mlx.Concatenate([]*mlx.Array{txtK, imgK}, 1)
	v := mlx.Concatenate([]*mlx.Array{txtV, imgV}, 1)

	// Apply RoPE
	q = ApplyRoPE4D(q, cos, sin)
	k = ApplyRoPE4D(k, cos, sin)

	// Transpose for SDPA: [B, nheads, L, headDim]
	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	// Scaled dot-product attention
	out := mlx.ScaledDotProductAttention(q, k, v, block.Scale, false)

	// Transpose back: [B, L, nheads, headDim]
	out = mlx.Transpose(out, 0, 2, 1, 3)

	// Split back into txt and img
	totalLen := txtLen + imgLen
	txtOut := mlx.Slice(out, []int32{0, 0, 0, 0}, []int32{B, txtLen, block.NHeads, block.HeadDim})
	imgOut := mlx.Slice(out, []int32{0, txtLen, 0, 0}, []int32{B, totalLen, block.NHeads, block.HeadDim})

	// Reshape and project
	txtOut = mlx.Reshape(txtOut, B, txtLen, dim)
	imgOut = mlx.Reshape(imgOut, B, imgLen, dim)
	txtOut = block.Attn.ToAddOut.Forward(txtOut)
	imgOut = block.Attn.ToOut0.Forward(imgOut)

	// Apply gates and residual
	imgHidden = mlx.Add(imgHidden, mlx.Mul(imgGate1, imgOut))
	txtHidden = mlx.Add(txtHidden, mlx.Mul(txtGate1, txtOut))

	// === MLP branch ===
	imgNorm = modulateLayerNorm(imgHidden, imgShift2, imgScale2)
	txtNorm = modulateLayerNorm(txtHidden, txtShift2, txtScale2)

	imgFFOut := block.FF.Forward(imgNorm)
	txtFFOut := block.FFContext.Forward(txtNorm)

	imgHidden = mlx.Add(imgHidden, mlx.Mul(imgGate2, imgFFOut))
	txtHidden = mlx.Add(txtHidden, mlx.Mul(txtGate2, txtFFOut))

	return imgHidden, txtHidden
}

// SingleTransformerBlockAttn implements attention for single-stream blocks
// Weight names: single_transformer_blocks.N.attn.*
type SingleTransformerBlockAttn struct {
	ToQKVMlpProj nn.LinearLayer `weight:"to_qkv_mlp_proj"` // Fused QKV + MLP input
	ToOut        nn.LinearLayer `weight:"to_out"`          // Fused attn_out + MLP out
	NormQ        *mlx.Array     `weight:"norm_q.weight"`
	NormK        *mlx.Array     `weight:"norm_k.weight"`
}

// SingleTransformerBlock implements a single-stream transformer block
// Weight names: single_transformer_blocks.N.*
type SingleTransformerBlock struct {
	Attn *SingleTransformerBlockAttn `weight:"attn"`

	// Config
	NHeads    int32
	HeadDim   int32
	InnerDim  int32
	MLPHidDim int32
	Scale     float32
}

// Forward applies the single-stream block
// x: [B, L, dim] concatenated text+image
// mod: modulation [B, 3*dim]
func (block *SingleTransformerBlock) Forward(x *mlx.Array, mod *mlx.Array, cos, sin *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	dim := shape[2]

	// Parse modulation: (shift, scale, gate)
	shift, scale, gate := parseModulation3(mod, dim, 0)

	// Modulate input
	h := modulateLayerNorm(x, shift, scale)

	// Fused projection: QKV + MLP gate/up
	// linear1 outputs: [q, k, v, mlp_gate, mlp_up] = [dim, dim, dim, mlpHid, mlpHid]
	qkvMlp := block.Attn.ToQKVMlpProj.Forward(h)

	// Split: first 3*dim is QKV, rest is MLP
	qkvDim := 3 * block.InnerDim
	qkv := mlx.Slice(qkvMlp, []int32{0, 0, 0}, []int32{B, L, qkvDim})
	mlpIn := mlx.Slice(qkvMlp, []int32{0, 0, qkvDim}, []int32{B, L, qkvMlp.Shape()[2]})

	// Split QKV
	q, k, v := splitQKV(qkv, B, L, block.InnerDim)

	// Reshape for attention
	q = mlx.Reshape(q, B, L, block.NHeads, block.HeadDim)
	k = mlx.Reshape(k, B, L, block.NHeads, block.HeadDim)
	v = mlx.Reshape(v, B, L, block.NHeads, block.HeadDim)

	// QK norm
	q = applyQKNorm(q, block.Attn.NormQ)
	k = applyQKNorm(k, block.Attn.NormK)

	// Apply RoPE
	q = ApplyRoPE4D(q, cos, sin)
	k = ApplyRoPE4D(k, cos, sin)

	// Transpose for SDPA
	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	// SDPA
	attnOut := mlx.ScaledDotProductAttention(q, k, v, block.Scale, false)

	// Transpose back and reshape
	attnOut = mlx.Transpose(attnOut, 0, 2, 1, 3)
	attnOut = mlx.Reshape(attnOut, B, L, block.InnerDim)

	// MLP: SwiGLU
	mlpShape := mlpIn.Shape()
	half := mlpShape[2] / 2
	mlpGate := mlx.Slice(mlpIn, []int32{0, 0, 0}, []int32{B, L, half})
	mlpUp := mlx.Slice(mlpIn, []int32{0, 0, half}, []int32{B, L, mlpShape[2]})
	mlpOut := mlx.Mul(mlx.SiLU(mlpGate), mlpUp)

	// Concatenate attention and MLP for fused output
	combined := mlx.Concatenate([]*mlx.Array{attnOut, mlpOut}, 2)

	// Output projection
	out := block.Attn.ToOut.Forward(combined)

	// Apply gate and residual
	return mlx.Add(x, mlx.Mul(gate, out))
}

// NormOut implements the output normalization with modulation
// Weight names: norm_out.linear.weight
type NormOut struct {
	Linear nn.LinearLayer `weight:"linear"`
}

// Forward computes final modulated output
func (n *NormOut) Forward(x *mlx.Array, temb *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	dim := shape[2]

	// Modulation: temb -> silu -> linear -> [shift, scale]
	mod := mlx.SiLU(temb)
	mod = n.Linear.Forward(mod)

	// Split into scale and shift (diffusers order: scale first, shift second)
	scale := mlx.Slice(mod, []int32{0, 0}, []int32{B, dim})
	shift := mlx.Slice(mod, []int32{0, dim}, []int32{B, 2 * dim})
	shift = mlx.ExpandDims(shift, 1)
	scale = mlx.ExpandDims(scale, 1)

	// Modulate with RMSNorm
	return modulateLayerNorm(x, shift, scale)
}

// Flux2Transformer2DModel is the main Flux2 transformer
// Weight names at top level: time_guidance_embed.*, double_stream_modulation_*.*, etc.
type Flux2Transformer2DModel struct {
	// Timestep embedding
	TimeGuidanceEmbed *TimeGuidanceEmbed `weight:"time_guidance_embed"`

	// Shared modulation
	DoubleStreamModulationImg *Modulation `weight:"double_stream_modulation_img"`
	DoubleStreamModulationTxt *Modulation `weight:"double_stream_modulation_txt"`
	SingleStreamModulation    *Modulation `weight:"single_stream_modulation"`

	// Embedders
	XEmbedder       nn.LinearLayer `weight:"x_embedder"`
	ContextEmbedder nn.LinearLayer `weight:"context_embedder"`

	// Transformer blocks
	TransformerBlocks       []*TransformerBlock       `weight:"transformer_blocks"`
	SingleTransformerBlocks []*SingleTransformerBlock `weight:"single_transformer_blocks"`

	// Output
	NormOut *NormOut       `weight:"norm_out"`
	ProjOut nn.LinearLayer `weight:"proj_out"`

	*TransformerConfig
}

// Load loads the Flux2 transformer from ollama blob storage.
func (m *Flux2Transformer2DModel) Load(manifest *imagegen.ModelManifest) error {
	fmt.Print("  Loading transformer... ")

	// Load config from blob
	var cfg TransformerConfig
	if err := manifest.ReadConfigJSON("transformer/config.json", &cfg); err != nil {
		return fmt.Errorf("config: %w", err)
	}
	m.TransformerConfig = &cfg

	// Initialize slices
	m.TransformerBlocks = make([]*TransformerBlock, cfg.NumLayers)
	m.SingleTransformerBlocks = make([]*SingleTransformerBlock, cfg.NumSingleLayers)

	// Initialize TimeGuidanceEmbed with embed dim
	m.TimeGuidanceEmbed = &TimeGuidanceEmbed{
		TimestepEmbedder: &TimestepEmbedder{EmbedDim: cfg.TimestepGuidanceChannels},
	}

	// Load weights from tensor blobs
	weights, err := imagegen.LoadWeightsFromManifest(manifest, "transformer")
	if err != nil {
		return fmt.Errorf("weights: %w", err)
	}
	if err := weights.Load(0); err != nil {
		return fmt.Errorf("load weights: %w", err)
	}
	defer weights.ReleaseAll()

	return m.loadWeights(weights)
}

// loadWeights loads weights from any WeightSource into the model
func (m *Flux2Transformer2DModel) loadWeights(weights safetensors.WeightSource) error {
	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return fmt.Errorf("load module: %w", err)
	}
	m.initComputedFields()
	fmt.Println("✓")
	return nil
}

// initComputedFields initializes computed fields after loading weights
func (m *Flux2Transformer2DModel) initComputedFields() {
	cfg := m.TransformerConfig
	innerDim := cfg.InnerDim()
	scale := float32(1.0 / math.Sqrt(float64(cfg.AttentionHeadDim)))

	// Initialize transformer blocks
	for _, block := range m.TransformerBlocks {
		block.NHeads = cfg.NumAttentionHeads
		block.HeadDim = cfg.AttentionHeadDim
		block.Scale = scale
	}

	// Initialize single transformer blocks
	for _, block := range m.SingleTransformerBlocks {
		block.NHeads = cfg.NumAttentionHeads
		block.HeadDim = cfg.AttentionHeadDim
		block.InnerDim = innerDim
		block.MLPHidDim = cfg.MLPHiddenDim()
		block.Scale = scale
	}
}

// Forward runs the Flux2 transformer
func (m *Flux2Transformer2DModel) Forward(patches, txtEmbeds *mlx.Array, timesteps *mlx.Array, rope *RoPECache) *mlx.Array {
	patchShape := patches.Shape()
	B := patchShape[0]
	imgLen := patchShape[1]
	txtLen := txtEmbeds.Shape()[1]

	// Scale timestep to 0-1000 range (diffusers multiplies by 1000)
	scaledTimesteps := mlx.MulScalar(timesteps, 1000.0)

	// Compute timestep embedding
	temb := m.TimeGuidanceEmbed.Forward(scaledTimesteps)

	// Embed patches and text
	imgHidden := m.XEmbedder.Forward(patches)
	txtHidden := m.ContextEmbedder.Forward(txtEmbeds)

	// Compute shared modulation
	imgMod := m.DoubleStreamModulationImg.Forward(temb)
	txtMod := m.DoubleStreamModulationTxt.Forward(temb)
	singleMod := m.SingleStreamModulation.Forward(temb)

	// Double (dual-stream) blocks
	for _, block := range m.TransformerBlocks {
		imgHidden, txtHidden = block.Forward(imgHidden, txtHidden, imgMod, txtMod, rope.Cos, rope.Sin)
	}

	// Concatenate for single-stream: text first, then image
	hidden := mlx.Concatenate([]*mlx.Array{txtHidden, imgHidden}, 1)

	// Single-stream blocks
	for _, block := range m.SingleTransformerBlocks {
		hidden = block.Forward(hidden, singleMod, rope.Cos, rope.Sin)
	}

	// Extract image portion
	totalLen := txtLen + imgLen
	imgOut := mlx.Slice(hidden, []int32{0, txtLen, 0}, []int32{B, totalLen, hidden.Shape()[2]})

	// Final norm and projection
	imgOut = m.NormOut.Forward(imgOut, temb)
	return m.ProjOut.Forward(imgOut)
}

// Note: QK normalization uses mlx.RMSNorm (the fast version) directly
// See applyQKNorm function below

// compiledSwiGLU fuses: silu(gate) * up
// Called 30x per step (10 in dual-stream + 20 in single-stream blocks)
var compiledSwiGLU *mlx.CompiledFunc

func getCompiledSwiGLU() *mlx.CompiledFunc {
	if compiledSwiGLU == nil {
		compiledSwiGLU = mlx.CompileShapeless(func(inputs []*mlx.Array) []*mlx.Array {
			gate, up := inputs[0], inputs[1]
			return []*mlx.Array{mlx.Mul(mlx.SiLU(gate), up)}
		}, true)
	}
	return compiledSwiGLU
}

// Helper functions

// parseModulation3 extracts 3 modulation params (shift, scale, gate) starting at offset
func parseModulation3(mod *mlx.Array, dim int32, offset int32) (*mlx.Array, *mlx.Array, *mlx.Array) {
	B := mod.Shape()[0]
	start := offset * dim
	shift := mlx.Slice(mod, []int32{0, start}, []int32{B, start + dim})
	scale := mlx.Slice(mod, []int32{0, start + dim}, []int32{B, start + 2*dim})
	gate := mlx.Slice(mod, []int32{0, start + 2*dim}, []int32{B, start + 3*dim})

	// Expand for broadcasting [B, dim] -> [B, 1, dim]
	shift = mlx.ExpandDims(shift, 1)
	scale = mlx.ExpandDims(scale, 1)
	gate = mlx.ExpandDims(gate, 1)

	return shift, scale, gate
}

// modulateLayerNorm applies LayerNorm then shift/scale modulation
// Diffusers uses LayerNorm(elementwise_affine=False) which centers the data
func modulateLayerNorm(x *mlx.Array, shift, scale *mlx.Array) *mlx.Array {
	// Fast LayerNorm without learnable params
	x = mlx.LayerNorm(x, 1e-6)

	// Modulate: x * (1 + scale) + shift
	x = mlx.Mul(x, mlx.AddScalar(scale, 1.0))
	return mlx.Add(x, shift)
}

// splitQKV splits a fused QKV tensor into Q, K, V
func splitQKV(qkv *mlx.Array, B, L, dim int32) (*mlx.Array, *mlx.Array, *mlx.Array) {
	q := mlx.Slice(qkv, []int32{0, 0, 0}, []int32{B, L, dim})
	k := mlx.Slice(qkv, []int32{0, 0, dim}, []int32{B, L, 2 * dim})
	v := mlx.Slice(qkv, []int32{0, 0, 2 * dim}, []int32{B, L, 3 * dim})
	return q, k, v
}

// applyQKNorm applies RMSNorm with learned scale (no bias)
// Uses the optimized mlx_fast_rms_norm
func applyQKNorm(x *mlx.Array, scale *mlx.Array) *mlx.Array {
	return mlx.RMSNorm(x, scale, 1e-6)
}

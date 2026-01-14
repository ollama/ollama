//go:build mlx

package qwen_image

import (
	"fmt"
	"math"
	"path/filepath"

	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// TransformerConfig holds Qwen-Image transformer configuration
type TransformerConfig struct {
	HiddenDim         int32   `json:"hidden_dim"`          // 3072 (24 * 128)
	NHeads            int32   `json:"num_attention_heads"` // 24
	HeadDim           int32   `json:"attention_head_dim"`  // 128
	NLayers           int32   `json:"num_layers"`          // 60
	InChannels        int32   `json:"in_channels"`         // 64
	OutChannels       int32   `json:"out_channels"`        // 16
	PatchSize         int32   `json:"patch_size"`          // 2
	JointAttentionDim int32   `json:"joint_attention_dim"` // 3584 (text encoder dim)
	NormEps           float32 `json:"norm_eps"`            // 1e-6
	AxesDimsRope      []int32 `json:"axes_dims_rope"`      // [16, 56, 56]
	GuidanceEmbeds    bool    `json:"guidance_embeds"`     // false
}

// defaultTransformerConfig returns config for Qwen-Image transformer
func defaultTransformerConfig() *TransformerConfig {
	return &TransformerConfig{
		HiddenDim:         3072, // 24 * 128
		NHeads:            24,
		HeadDim:           128,
		NLayers:           60,
		InChannels:        64,
		OutChannels:       16,
		PatchSize:         2,
		JointAttentionDim: 3584,
		NormEps:           1e-6,
		AxesDimsRope:      []int32{16, 56, 56},
		GuidanceEmbeds:    false,
	}
}

// TimestepEmbedder creates timestep embeddings
type TimestepEmbedder struct {
	Linear1Weight *mlx.Array // [256, hidden_dim]
	Linear1Bias   *mlx.Array
	Linear2Weight *mlx.Array // [hidden_dim, hidden_dim]
	Linear2Bias   *mlx.Array
}

// newTimestepEmbedder creates a timestep embedder from weights
func newTimestepEmbedder(weights *safetensors.ModelWeights) (*TimestepEmbedder, error) {
	linear1Weight, err := weights.Get("time_text_embed.timestep_embedder.linear_1.weight")
	if err != nil {
		return nil, err
	}
	linear1Bias, err := weights.Get("time_text_embed.timestep_embedder.linear_1.bias")
	if err != nil {
		return nil, err
	}
	linear2Weight, err := weights.Get("time_text_embed.timestep_embedder.linear_2.weight")
	if err != nil {
		return nil, err
	}
	linear2Bias, err := weights.Get("time_text_embed.timestep_embedder.linear_2.bias")
	if err != nil {
		return nil, err
	}

	return &TimestepEmbedder{
		Linear1Weight: mlx.Transpose(linear1Weight, 1, 0),
		Linear1Bias:   linear1Bias,
		Linear2Weight: mlx.Transpose(linear2Weight, 1, 0),
		Linear2Bias:   linear2Bias,
	}, nil
}

// Forward computes timestep embeddings
// t: [B] timesteps (normalized 0-1, will be scaled by 1000 internally)
func (te *TimestepEmbedder) Forward(t *mlx.Array) *mlx.Array {
	half := int32(128) // embedding_dim / 2

	// Sinusoidal embedding with flip_sin_to_cos=True, scale=1000
	freqs := make([]float32, half)
	for i := int32(0); i < half; i++ {
		freqs[i] = float32(math.Exp(-math.Log(10000.0) * float64(i) / float64(half)))
	}
	freqsArr := mlx.NewArray(freqs, []int32{1, half})

	tExpanded := mlx.ExpandDims(t, 1)
	args := mlx.Mul(tExpanded, freqsArr)
	args = mlx.MulScalar(args, 1000.0) // scale

	// [cos, sin] (flip_sin_to_cos=True)
	sinArgs := mlx.Sin(args)
	cosArgs := mlx.Cos(args)
	embedding := mlx.Concatenate([]*mlx.Array{cosArgs, sinArgs}, 1) // [B, 256]

	// MLP: linear1 -> silu -> linear2
	h := mlx.Linear(embedding, te.Linear1Weight)
	h = mlx.Add(h, te.Linear1Bias)
	h = mlx.SiLU(h)
	h = mlx.Linear(h, te.Linear2Weight)
	h = mlx.Add(h, te.Linear2Bias)

	return h
}

// JointAttention implements dual-stream joint attention
type JointAttention struct {
	// Image projections
	ToQ    *mlx.Array
	ToQB   *mlx.Array
	ToK    *mlx.Array
	ToKB   *mlx.Array
	ToV    *mlx.Array
	ToVB   *mlx.Array
	ToOut  *mlx.Array
	ToOutB *mlx.Array
	NormQ  *mlx.Array
	NormK  *mlx.Array

	// Text (added) projections
	AddQProj  *mlx.Array
	AddQProjB *mlx.Array
	AddKProj  *mlx.Array
	AddKProjB *mlx.Array
	AddVProj  *mlx.Array
	AddVProjB *mlx.Array
	ToAddOut  *mlx.Array
	ToAddOutB *mlx.Array
	NormAddQ  *mlx.Array
	NormAddK  *mlx.Array

	NHeads  int32
	HeadDim int32
	Scale   float32
}

// newJointAttention creates a joint attention layer
func newJointAttention(weights *safetensors.ModelWeights, prefix string, cfg *TransformerConfig) (*JointAttention, error) {
	toQ, _ := weights.Get(prefix + ".attn.to_q.weight")
	toQB, _ := weights.Get(prefix + ".attn.to_q.bias")
	toK, _ := weights.Get(prefix + ".attn.to_k.weight")
	toKB, _ := weights.Get(prefix + ".attn.to_k.bias")
	toV, _ := weights.Get(prefix + ".attn.to_v.weight")
	toVB, _ := weights.Get(prefix + ".attn.to_v.bias")
	toOut, _ := weights.Get(prefix + ".attn.to_out.0.weight")
	toOutB, _ := weights.Get(prefix + ".attn.to_out.0.bias")
	normQ, _ := weights.Get(prefix + ".attn.norm_q.weight")
	normK, _ := weights.Get(prefix + ".attn.norm_k.weight")

	addQProj, _ := weights.Get(prefix + ".attn.add_q_proj.weight")
	addQProjB, _ := weights.Get(prefix + ".attn.add_q_proj.bias")
	addKProj, _ := weights.Get(prefix + ".attn.add_k_proj.weight")
	addKProjB, _ := weights.Get(prefix + ".attn.add_k_proj.bias")
	addVProj, _ := weights.Get(prefix + ".attn.add_v_proj.weight")
	addVProjB, _ := weights.Get(prefix + ".attn.add_v_proj.bias")
	toAddOut, _ := weights.Get(prefix + ".attn.to_add_out.weight")
	toAddOutB, _ := weights.Get(prefix + ".attn.to_add_out.bias")
	normAddQ, _ := weights.Get(prefix + ".attn.norm_added_q.weight")
	normAddK, _ := weights.Get(prefix + ".attn.norm_added_k.weight")

	return &JointAttention{
		ToQ:       mlx.Transpose(toQ, 1, 0),
		ToQB:      toQB,
		ToK:       mlx.Transpose(toK, 1, 0),
		ToKB:      toKB,
		ToV:       mlx.Transpose(toV, 1, 0),
		ToVB:      toVB,
		ToOut:     mlx.Transpose(toOut, 1, 0),
		ToOutB:    toOutB,
		NormQ:     normQ,
		NormK:     normK,
		AddQProj:  mlx.Transpose(addQProj, 1, 0),
		AddQProjB: addQProjB,
		AddKProj:  mlx.Transpose(addKProj, 1, 0),
		AddKProjB: addKProjB,
		AddVProj:  mlx.Transpose(addVProj, 1, 0),
		AddVProjB: addVProjB,
		ToAddOut:  mlx.Transpose(toAddOut, 1, 0),
		ToAddOutB: toAddOutB,
		NormAddQ:  normAddQ,
		NormAddK:  normAddK,
		NHeads:    cfg.NHeads,
		HeadDim:   cfg.HeadDim,
		Scale:     float32(1.0 / math.Sqrt(float64(cfg.HeadDim))),
	}, nil
}

// Forward computes joint attention
// img: [B, L_img, D], txt: [B, L_txt, D]
// imgFreqs, txtFreqs: complex RoPE frequencies [L, head_dim/2] as interleaved real/imag
func (attn *JointAttention) Forward(img, txt *mlx.Array, imgFreqs, txtFreqs *mlx.Array) (*mlx.Array, *mlx.Array) {
	imgShape := img.Shape()
	B := imgShape[0]
	Limg := imgShape[1]
	D := imgShape[2]

	txtShape := txt.Shape()
	Ltxt := txtShape[1]

	// === Image Q/K/V ===
	imgFlat := mlx.Reshape(img, B*Limg, D)
	qImg := mlx.Add(mlx.Linear(imgFlat, attn.ToQ), attn.ToQB)
	kImg := mlx.Add(mlx.Linear(imgFlat, attn.ToK), attn.ToKB)
	vImg := mlx.Add(mlx.Linear(imgFlat, attn.ToV), attn.ToVB)

	qImg = mlx.Reshape(qImg, B, Limg, attn.NHeads, attn.HeadDim)
	kImg = mlx.Reshape(kImg, B, Limg, attn.NHeads, attn.HeadDim)
	vImg = mlx.Reshape(vImg, B, Limg, attn.NHeads, attn.HeadDim)

	// QK norm (RMSNorm per head)
	qImg = mlx.RMSNorm(qImg, attn.NormQ, 1e-6)
	kImg = mlx.RMSNorm(kImg, attn.NormK, 1e-6)

	// Apply RoPE
	if imgFreqs != nil {
		qImg = applyRoPE(qImg, imgFreqs)
		kImg = applyRoPE(kImg, imgFreqs)
	}

	// === Text Q/K/V ===
	txtFlat := mlx.Reshape(txt, B*Ltxt, D)
	qTxt := mlx.Add(mlx.Linear(txtFlat, attn.AddQProj), attn.AddQProjB)
	kTxt := mlx.Add(mlx.Linear(txtFlat, attn.AddKProj), attn.AddKProjB)
	vTxt := mlx.Add(mlx.Linear(txtFlat, attn.AddVProj), attn.AddVProjB)

	qTxt = mlx.Reshape(qTxt, B, Ltxt, attn.NHeads, attn.HeadDim)
	kTxt = mlx.Reshape(kTxt, B, Ltxt, attn.NHeads, attn.HeadDim)
	vTxt = mlx.Reshape(vTxt, B, Ltxt, attn.NHeads, attn.HeadDim)

	qTxt = mlx.RMSNorm(qTxt, attn.NormAddQ, 1e-6)
	kTxt = mlx.RMSNorm(kTxt, attn.NormAddK, 1e-6)

	if txtFreqs != nil {
		qTxt = applyRoPE(qTxt, txtFreqs)
		kTxt = applyRoPE(kTxt, txtFreqs)
	}

	// Concatenate for joint attention: [txt, img] order
	qJoint := mlx.Concatenate([]*mlx.Array{qTxt, qImg}, 1)
	kJoint := mlx.Concatenate([]*mlx.Array{kTxt, kImg}, 1)
	vJoint := mlx.Concatenate([]*mlx.Array{vTxt, vImg}, 1)

	// Transpose to [B, nheads, L, head_dim]
	qJoint = mlx.Transpose(qJoint, 0, 2, 1, 3)
	kJoint = mlx.Transpose(kJoint, 0, 2, 1, 3)
	vJoint = mlx.Transpose(vJoint, 0, 2, 1, 3)

	// SDPA
	outJoint := mlx.ScaledDotProductAttention(qJoint, kJoint, vJoint, attn.Scale, false)

	// Transpose back and split
	outJoint = mlx.Transpose(outJoint, 0, 2, 1, 3) // [B, L, nheads, head_dim]
	outJoint = mlx.Reshape(outJoint, B, Ltxt+Limg, D)

	outTxt := mlx.Slice(outJoint, []int32{0, 0, 0}, []int32{B, Ltxt, D})
	outImg := mlx.Slice(outJoint, []int32{0, Ltxt, 0}, []int32{B, Ltxt + Limg, D})

	// Output projections
	outImg = mlx.Reshape(outImg, B*Limg, D)
	outImg = mlx.Add(mlx.Linear(outImg, attn.ToOut), attn.ToOutB)
	outImg = mlx.Reshape(outImg, B, Limg, D)

	outTxt = mlx.Reshape(outTxt, B*Ltxt, D)
	outTxt = mlx.Add(mlx.Linear(outTxt, attn.ToAddOut), attn.ToAddOutB)
	outTxt = mlx.Reshape(outTxt, B, Ltxt, D)

	return outImg, outTxt
}

// applyRoPE applies rotary embeddings using complex multiplication
// x: [B, L, nheads, head_dim]
// freqs: [L, head_dim] as complex (interleaved real/imag pairs)
func applyRoPE(x *mlx.Array, freqs *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	nheads := shape[2]
	headDim := shape[3]
	halfDim := headDim / 2

	// Reshape x to pairs: [B, L, nheads, half, 2]
	xPairs := mlx.Reshape(x, B, L, nheads, halfDim, 2)

	// freqs: [L, head_dim] -> [1, L, 1, half, 2]
	freqsExp := mlx.Reshape(freqs, 1, L, 1, halfDim, 2)

	// Extract real/imag parts
	xReal := mlx.SliceStride(xPairs, []int32{0, 0, 0, 0, 0}, []int32{B, L, nheads, halfDim, 1}, []int32{1, 1, 1, 1, 1})
	xImag := mlx.SliceStride(xPairs, []int32{0, 0, 0, 0, 1}, []int32{B, L, nheads, halfDim, 2}, []int32{1, 1, 1, 1, 1})
	xReal = mlx.Squeeze(xReal, 4)
	xImag = mlx.Squeeze(xImag, 4)

	freqReal := mlx.SliceStride(freqsExp, []int32{0, 0, 0, 0, 0}, []int32{1, L, 1, halfDim, 1}, []int32{1, 1, 1, 1, 1})
	freqImag := mlx.SliceStride(freqsExp, []int32{0, 0, 0, 0, 1}, []int32{1, L, 1, halfDim, 2}, []int32{1, 1, 1, 1, 1})
	freqReal = mlx.Squeeze(freqReal, 4)
	freqImag = mlx.Squeeze(freqImag, 4)

	// Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
	outReal := mlx.Sub(mlx.Mul(xReal, freqReal), mlx.Mul(xImag, freqImag))
	outImag := mlx.Add(mlx.Mul(xReal, freqImag), mlx.Mul(xImag, freqReal))

	// Interleave back
	outReal = mlx.ExpandDims(outReal, 4)
	outImag = mlx.ExpandDims(outImag, 4)
	out := mlx.Concatenate([]*mlx.Array{outReal, outImag}, 4)

	return mlx.Reshape(out, B, L, nheads, headDim)
}

// MLP implements GELU MLP (not GEGLU)
type MLP struct {
	ProjWeight *mlx.Array
	ProjBias   *mlx.Array
	OutWeight  *mlx.Array
	OutBias    *mlx.Array
}

// newMLP creates a GELU MLP
func newMLP(weights *safetensors.ModelWeights, prefix string) (*MLP, error) {
	projWeight, _ := weights.Get(prefix + ".net.0.proj.weight")
	projBias, _ := weights.Get(prefix + ".net.0.proj.bias")
	outWeight, _ := weights.Get(prefix + ".net.2.weight")
	outBias, _ := weights.Get(prefix + ".net.2.bias")

	return &MLP{
		ProjWeight: mlx.Transpose(projWeight, 1, 0),
		ProjBias:   projBias,
		OutWeight:  mlx.Transpose(outWeight, 1, 0),
		OutBias:    outBias,
	}, nil
}

// Forward applies GELU MLP
func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	D := shape[2]

	xFlat := mlx.Reshape(x, B*L, D)
	h := mlx.Add(mlx.Linear(xFlat, m.ProjWeight), m.ProjBias)
	h = geluApprox(h)
	h = mlx.Add(mlx.Linear(h, m.OutWeight), m.OutBias)
	return mlx.Reshape(h, B, L, m.OutBias.Dim(0))
}

// geluApprox implements approximate GELU
func geluApprox(x *mlx.Array) *mlx.Array {
	sqrt2OverPi := float32(math.Sqrt(2.0 / math.Pi))
	x3 := mlx.Mul(mlx.Mul(x, x), x)
	inner := mlx.Add(x, mlx.MulScalar(x3, 0.044715))
	inner = mlx.MulScalar(inner, sqrt2OverPi)
	return mlx.Mul(mlx.MulScalar(x, 0.5), mlx.AddScalar(mlx.Tanh(inner), 1.0))
}

// TransformerBlock is a single dual-stream transformer block
type TransformerBlock struct {
	Attention *JointAttention
	ImgMLP    *MLP
	TxtMLP    *MLP

	ImgModWeight *mlx.Array
	ImgModBias   *mlx.Array
	TxtModWeight *mlx.Array
	TxtModBias   *mlx.Array

	HiddenDim int32
	NormEps   float32
}

// newTransformerBlock creates a transformer block
func newTransformerBlock(weights *safetensors.ModelWeights, prefix string, cfg *TransformerConfig) (*TransformerBlock, error) {
	attn, err := newJointAttention(weights, prefix, cfg)
	if err != nil {
		return nil, err
	}

	imgMLP, _ := newMLP(weights, prefix+".img_mlp")
	txtMLP, _ := newMLP(weights, prefix+".txt_mlp")

	imgModWeight, _ := weights.Get(prefix + ".img_mod.1.weight")
	imgModBias, _ := weights.Get(prefix + ".img_mod.1.bias")
	txtModWeight, _ := weights.Get(prefix + ".txt_mod.1.weight")
	txtModBias, _ := weights.Get(prefix + ".txt_mod.1.bias")

	return &TransformerBlock{
		Attention:    attn,
		ImgMLP:       imgMLP,
		TxtMLP:       txtMLP,
		ImgModWeight: mlx.Transpose(imgModWeight, 1, 0),
		ImgModBias:   imgModBias,
		TxtModWeight: mlx.Transpose(txtModWeight, 1, 0),
		TxtModBias:   txtModBias,
		HiddenDim:    cfg.HiddenDim,
		NormEps:      cfg.NormEps,
	}, nil
}

// Forward applies the transformer block
func (tb *TransformerBlock) Forward(img, txt, temb *mlx.Array, imgFreqs, txtFreqs *mlx.Array) (*mlx.Array, *mlx.Array) {
	// Compute modulation: silu(temb) -> linear -> [B, 6*D]
	siluT := mlx.SiLU(temb)
	imgMod := mlx.Add(mlx.Linear(siluT, tb.ImgModWeight), tb.ImgModBias)
	txtMod := mlx.Add(mlx.Linear(siluT, tb.TxtModWeight), tb.TxtModBias)

	// Split into 6 parts: shift1, scale1, gate1, shift2, scale2, gate2
	imgModParts := splitMod6(imgMod, tb.HiddenDim)
	txtModParts := splitMod6(txtMod, tb.HiddenDim)

	// Pre-attention: norm + modulate
	imgNorm := layerNormNoAffine(img, tb.NormEps)
	imgNorm = mlx.Add(mlx.Mul(imgNorm, mlx.AddScalar(imgModParts[1], 1.0)), imgModParts[0])

	txtNorm := layerNormNoAffine(txt, tb.NormEps)
	txtNorm = mlx.Add(mlx.Mul(txtNorm, mlx.AddScalar(txtModParts[1], 1.0)), txtModParts[0])

	// Joint attention
	attnImg, attnTxt := tb.Attention.Forward(imgNorm, txtNorm, imgFreqs, txtFreqs)

	// Residual with gate
	img = mlx.Add(img, mlx.Mul(imgModParts[2], attnImg))
	txt = mlx.Add(txt, mlx.Mul(txtModParts[2], attnTxt))

	// Pre-MLP: norm + modulate
	imgNorm2 := layerNormNoAffine(img, tb.NormEps)
	imgNorm2 = mlx.Add(mlx.Mul(imgNorm2, mlx.AddScalar(imgModParts[4], 1.0)), imgModParts[3])

	txtNorm2 := layerNormNoAffine(txt, tb.NormEps)
	txtNorm2 = mlx.Add(mlx.Mul(txtNorm2, mlx.AddScalar(txtModParts[4], 1.0)), txtModParts[3])

	// MLP
	mlpImg := tb.ImgMLP.Forward(imgNorm2)
	mlpTxt := tb.TxtMLP.Forward(txtNorm2)

	// Residual with gate
	img = mlx.Add(img, mlx.Mul(imgModParts[5], mlpImg))
	txt = mlx.Add(txt, mlx.Mul(txtModParts[5], mlpTxt))

	return img, txt
}

// splitMod6 splits modulation into 6 parts each [B, 1, D]
func splitMod6(mod *mlx.Array, hiddenDim int32) []*mlx.Array {
	shape := mod.Shape()
	B := shape[0]
	parts := make([]*mlx.Array, 6)
	for i := int32(0); i < 6; i++ {
		part := mlx.Slice(mod, []int32{0, i * hiddenDim}, []int32{B, (i + 1) * hiddenDim})
		parts[i] = mlx.ExpandDims(part, 1)
	}
	return parts
}

// layerNormNoAffine applies layer norm without learnable parameters
func layerNormNoAffine(x *mlx.Array, eps float32) *mlx.Array {
	ndim := x.Ndim()
	lastAxis := ndim - 1
	mean := mlx.Mean(x, lastAxis, true)
	xCentered := mlx.Sub(x, mean)
	variance := mlx.Mean(mlx.Square(xCentered), lastAxis, true)
	return mlx.Div(xCentered, mlx.Sqrt(mlx.AddScalar(variance, eps)))
}

// Transformer is the full Qwen-Image transformer model
type Transformer struct {
	Config *TransformerConfig

	ImgIn     *mlx.Array
	ImgInBias *mlx.Array
	TxtIn     *mlx.Array
	TxtInBias *mlx.Array
	TxtNorm   *mlx.Array

	TEmbed *TimestepEmbedder
	Layers []*TransformerBlock

	NormOutWeight *mlx.Array
	NormOutBias   *mlx.Array
	ProjOut       *mlx.Array
	ProjOutBias   *mlx.Array
}

// Load loads the transformer from a directory
func (m *Transformer) Load(path string) error {
	fmt.Println("Loading Qwen-Image transformer...")

	cfg := defaultTransformerConfig()
	m.Config = cfg

	weights, err := safetensors.LoadModelWeights(path)
	if err != nil {
		return fmt.Errorf("weights: %w", err)
	}

	// Bulk load all weights as bf16
	fmt.Print("  Loading weights as bf16... ")
	if err := weights.Load(mlx.DtypeBFloat16); err != nil {
		return fmt.Errorf("load weights: %w", err)
	}
	fmt.Printf("✓ (%.1f GB)\n", float64(mlx.MetalGetActiveMemory())/(1024*1024*1024))

	fmt.Print("  Loading input projections... ")
	imgIn, _ := weights.Get("img_in.weight")
	imgInBias, _ := weights.Get("img_in.bias")
	txtIn, _ := weights.Get("txt_in.weight")
	txtInBias, _ := weights.Get("txt_in.bias")
	txtNorm, _ := weights.Get("txt_norm.weight")
	m.ImgIn = mlx.Transpose(imgIn, 1, 0)
	m.ImgInBias = imgInBias
	m.TxtIn = mlx.Transpose(txtIn, 1, 0)
	m.TxtInBias = txtInBias
	m.TxtNorm = txtNorm
	fmt.Println("✓")

	fmt.Print("  Loading timestep embedder... ")
	m.TEmbed, err = newTimestepEmbedder(weights)
	if err != nil {
		return fmt.Errorf("timestep embedder: %w", err)
	}
	fmt.Println("✓")

	m.Layers = make([]*TransformerBlock, cfg.NLayers)
	for i := int32(0); i < cfg.NLayers; i++ {
		fmt.Printf("\r  Loading transformer layers... %d/%d", i+1, cfg.NLayers)
		prefix := fmt.Sprintf("transformer_blocks.%d", i)
		m.Layers[i], err = newTransformerBlock(weights, prefix, cfg)
		if err != nil {
			return fmt.Errorf("layer %d: %w", i, err)
		}
	}
	fmt.Printf("\r  Loading transformer layers... ✓ [%d blocks]          \n", cfg.NLayers)

	fmt.Print("  Loading output layers... ")
	normOutWeight, _ := weights.Get("norm_out.linear.weight")
	normOutBias, _ := weights.Get("norm_out.linear.bias")
	projOut, _ := weights.Get("proj_out.weight")
	projOutBias, _ := weights.Get("proj_out.bias")
	m.NormOutWeight = mlx.Transpose(normOutWeight, 1, 0)
	m.NormOutBias = normOutBias
	m.ProjOut = mlx.Transpose(projOut, 1, 0)
	m.ProjOutBias = projOutBias
	fmt.Println("✓")

	weights.ReleaseAll()
	return nil
}

// LoadFromPath is a convenience function to load transformer from path
func LoadTransformerFromPath(path string) (*Transformer, error) {
	m := &Transformer{}
	if err := m.Load(filepath.Join(path, "transformer")); err != nil {
		return nil, err
	}
	return m, nil
}

// Forward runs the transformer
// img: [B, L_img, in_channels] patchified latents
// txt: [B, L_txt, joint_attention_dim] text embeddings
// t: [B] timesteps (0-1)
// imgFreqs, txtFreqs: RoPE frequencies
func (tr *Transformer) Forward(img, txt, t *mlx.Array, imgFreqs, txtFreqs *mlx.Array) *mlx.Array {
	imgShape := img.Shape()
	B := imgShape[0]
	Limg := imgShape[1]

	txtShape := txt.Shape()
	Ltxt := txtShape[1]

	// Timestep embedding
	temb := tr.TEmbed.Forward(t)

	// Project image: [B, L, in_channels] -> [B, L, hidden_dim]
	imgFlat := mlx.Reshape(img, B*Limg, tr.Config.InChannels)
	imgH := mlx.Add(mlx.Linear(imgFlat, tr.ImgIn), tr.ImgInBias)
	imgH = mlx.Reshape(imgH, B, Limg, tr.Config.HiddenDim)

	// Project text: RMSNorm then linear
	txtFlat := mlx.Reshape(txt, B*Ltxt, tr.Config.JointAttentionDim)
	txtNormed := mlx.RMSNorm(txtFlat, tr.TxtNorm, 1e-6)
	txtH := mlx.Add(mlx.Linear(txtNormed, tr.TxtIn), tr.TxtInBias)
	txtH = mlx.Reshape(txtH, B, Ltxt, tr.Config.HiddenDim)

	for _, layer := range tr.Layers {
		imgH, txtH = layer.Forward(imgH, txtH, temb, imgFreqs, txtFreqs)
	}

	// Final norm with modulation (AdaLayerNormContinuous)
	// Python: scale, shift = torch.chunk(emb, 2, dim=1)
	finalMod := mlx.Add(mlx.Linear(mlx.SiLU(temb), tr.NormOutWeight), tr.NormOutBias)
	modShape := finalMod.Shape()
	halfDim := modShape[1] / 2
	scale := mlx.ExpandDims(mlx.Slice(finalMod, []int32{0, 0}, []int32{B, halfDim}), 1)
	shift := mlx.ExpandDims(mlx.Slice(finalMod, []int32{0, halfDim}, []int32{B, modShape[1]}), 1)

	imgH = layerNormNoAffine(imgH, tr.Config.NormEps)
	imgH = mlx.Add(mlx.Mul(imgH, mlx.AddScalar(scale, 1.0)), shift)

	// Final projection: [B, L, hidden_dim] -> [B, L, patch_size^2 * out_channels]
	imgFlat = mlx.Reshape(imgH, B*Limg, tr.Config.HiddenDim)
	out := mlx.Add(mlx.Linear(imgFlat, tr.ProjOut), tr.ProjOutBias)

	outChannels := tr.Config.PatchSize * tr.Config.PatchSize * tr.Config.OutChannels
	return mlx.Reshape(out, B, Limg, outChannels)
}

// ForwardWithCache runs the transformer with layer caching for speedup.
// Based on DeepCache (CVPR 2024) / Learning-to-Cache (NeurIPS 2024):
// shallow layers change little between denoising steps, so we cache their
// outputs and reuse them on non-refresh steps.
//
// stepCache: cache for layer outputs (use cache.NewStepCache(cacheLayers))
// step: current denoising step (0-indexed)
// cacheInterval: refresh cache every N steps (e.g., 3)
// cacheLayers: number of shallow layers to cache (e.g., 15)
func (tr *Transformer) ForwardWithCache(
	img, txt, t *mlx.Array,
	imgFreqs, txtFreqs *mlx.Array,
	stepCache *cache.StepCache,
	step, cacheInterval, cacheLayers int,
) *mlx.Array {
	imgShape := img.Shape()
	B := imgShape[0]
	Limg := imgShape[1]

	txtShape := txt.Shape()
	Ltxt := txtShape[1]

	// Timestep embedding
	temb := tr.TEmbed.Forward(t)

	// Project image: [B, L, in_channels] -> [B, L, hidden_dim]
	imgFlat := mlx.Reshape(img, B*Limg, tr.Config.InChannels)
	imgH := mlx.Add(mlx.Linear(imgFlat, tr.ImgIn), tr.ImgInBias)
	imgH = mlx.Reshape(imgH, B, Limg, tr.Config.HiddenDim)

	// Project text: RMSNorm then linear
	txtFlat := mlx.Reshape(txt, B*Ltxt, tr.Config.JointAttentionDim)
	txtNormed := mlx.RMSNorm(txtFlat, tr.TxtNorm, 1e-6)
	txtH := mlx.Add(mlx.Linear(txtNormed, tr.TxtIn), tr.TxtInBias)
	txtH = mlx.Reshape(txtH, B, Ltxt, tr.Config.HiddenDim)

	// Check if we should refresh the cache
	refreshCache := stepCache.ShouldRefresh(step, cacheInterval)

	for i, layer := range tr.Layers {
		if i < cacheLayers && !refreshCache && stepCache.Get(i) != nil {
			// Use cached outputs for shallow layers
			imgH = stepCache.Get(i)
			txtH = stepCache.Get2(i)
		} else {
			// Compute layer
			imgH, txtH = layer.Forward(imgH, txtH, temb, imgFreqs, txtFreqs)
			// Cache shallow layers on refresh steps
			if i < cacheLayers && refreshCache {
				stepCache.Set(i, imgH)
				stepCache.Set2(i, txtH)
			}
		}
	}

	// Final norm with modulation (AdaLayerNormContinuous)
	finalMod := mlx.Add(mlx.Linear(mlx.SiLU(temb), tr.NormOutWeight), tr.NormOutBias)
	modShape := finalMod.Shape()
	halfDim := modShape[1] / 2
	scale := mlx.ExpandDims(mlx.Slice(finalMod, []int32{0, 0}, []int32{B, halfDim}), 1)
	shift := mlx.ExpandDims(mlx.Slice(finalMod, []int32{0, halfDim}, []int32{B, modShape[1]}), 1)

	imgH = layerNormNoAffine(imgH, tr.Config.NormEps)
	imgH = mlx.Add(mlx.Mul(imgH, mlx.AddScalar(scale, 1.0)), shift)

	// Final projection: [B, L, hidden_dim] -> [B, L, patch_size^2 * out_channels]
	imgFlat = mlx.Reshape(imgH, B*Limg, tr.Config.HiddenDim)
	out := mlx.Add(mlx.Linear(imgFlat, tr.ProjOut), tr.ProjOutBias)

	outChannels := tr.Config.PatchSize * tr.Config.PatchSize * tr.Config.OutChannels
	return mlx.Reshape(out, B, Limg, outChannels)
}

// RoPECache holds precomputed RoPE frequencies
type RoPECache struct {
	ImgFreqs *mlx.Array // [L_img, head_dim]
	TxtFreqs *mlx.Array // [L_txt, head_dim]
}

// PrepareRoPE computes RoPE for image and text sequences
// This matches Python's QwenEmbedRope with scale_rope=True
func PrepareRoPE(imgH, imgW int32, txtLen int32, axesDims []int32) *RoPECache {
	theta := float64(10000)
	maxIdx := int32(4096)

	// Compute base frequencies for each axis dimension
	freqsT := ComputeAxisFreqs(axesDims[0], theta)
	freqsH := ComputeAxisFreqs(axesDims[1], theta)
	freqsW := ComputeAxisFreqs(axesDims[2], theta)

	// Build frequency lookup tables
	posFreqsT := MakeFreqTable(maxIdx, freqsT, false)
	posFreqsH := MakeFreqTable(maxIdx, freqsH, false)
	posFreqsW := MakeFreqTable(maxIdx, freqsW, false)
	negFreqsH := MakeFreqTable(maxIdx, freqsH, true)
	negFreqsW := MakeFreqTable(maxIdx, freqsW, true)

	// Image frequencies with scale_rope=True
	imgLen := imgH * imgW
	headDim := int32(len(freqsT)+len(freqsH)+len(freqsW)) * 2
	imgFreqsData := make([]float32, imgLen*headDim)

	hHalf := imgH / 2
	wHalf := imgW / 2

	idx := int32(0)
	for y := int32(0); y < imgH; y++ {
		for x := int32(0); x < imgW; x++ {
			// Frame = 0
			for i := 0; i < len(freqsT)*2; i++ {
				imgFreqsData[idx+int32(i)] = posFreqsT[0][i]
			}
			idx += int32(len(freqsT) * 2)

			// Height: scale_rope pattern
			hNegCount := imgH - hHalf
			if y < hNegCount {
				negTableIdx := maxIdx - hNegCount + y
				for i := 0; i < len(freqsH)*2; i++ {
					imgFreqsData[idx+int32(i)] = negFreqsH[negTableIdx][i]
				}
			} else {
				posIdx := y - hNegCount
				for i := 0; i < len(freqsH)*2; i++ {
					imgFreqsData[idx+int32(i)] = posFreqsH[posIdx][i]
				}
			}
			idx += int32(len(freqsH) * 2)

			// Width: scale_rope pattern
			wNegCount := imgW - wHalf
			if x < wNegCount {
				negTableIdx := maxIdx - wNegCount + x
				for i := 0; i < len(freqsW)*2; i++ {
					imgFreqsData[idx+int32(i)] = negFreqsW[negTableIdx][i]
				}
			} else {
				posIdx := x - wNegCount
				for i := 0; i < len(freqsW)*2; i++ {
					imgFreqsData[idx+int32(i)] = posFreqsW[posIdx][i]
				}
			}
			idx += int32(len(freqsW) * 2)
		}
	}

	imgFreqs := mlx.NewArray(imgFreqsData, []int32{imgLen, headDim})
	imgFreqs = mlx.ToBFloat16(imgFreqs)

	// Text frequencies
	maxVidIdx := max(hHalf, wHalf)
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

	return &RoPECache{
		ImgFreqs: imgFreqs,
		TxtFreqs: txtFreqs,
	}
}

// ComputeAxisFreqs computes RoPE base frequencies for a given dimension.
func ComputeAxisFreqs(dim int32, theta float64) []float64 {
	halfDim := dim / 2
	freqs := make([]float64, halfDim)
	for i := int32(0); i < halfDim; i++ {
		freqs[i] = 1.0 / math.Pow(theta, float64(i)/float64(halfDim))
	}
	return freqs
}

// MakeFreqTable builds a table of cos/sin values for RoPE positions.
func MakeFreqTable(maxIdx int32, baseFreqs []float64, negative bool) [][]float32 {
	table := make([][]float32, maxIdx)
	for idx := int32(0); idx < maxIdx; idx++ {
		var pos float64
		if negative {
			pos = float64(-maxIdx + int32(idx))
		} else {
			pos = float64(idx)
		}

		row := make([]float32, len(baseFreqs)*2)
		for i, f := range baseFreqs {
			angle := pos * f
			row[i*2] = float32(math.Cos(angle))
			row[i*2+1] = float32(math.Sin(angle))
		}
		table[idx] = row
	}
	return table
}

func max(a, b int32) int32 {
	if a > b {
		return a
	}
	return b
}

// PackLatents converts [B, C, H, W] to [B, L, C*4] patches
func PackLatents(latents *mlx.Array, patchSize int32) *mlx.Array {
	shape := latents.Shape()
	B := shape[0]
	C := shape[1]
	H := shape[2]
	W := shape[3]

	pH := H / patchSize
	pW := W / patchSize

	// [B, C, H, W] -> [B, C, pH, 2, pW, 2]
	x := mlx.Reshape(latents, B, C, pH, patchSize, pW, patchSize)
	// -> [B, pH, pW, C, 2, 2]
	x = mlx.Transpose(x, 0, 2, 4, 1, 3, 5)
	// -> [B, pH*pW, C*4]
	return mlx.Reshape(x, B, pH*pW, C*patchSize*patchSize)
}

// UnpackLatents converts [B, L, C*4] back to [B, C, 1, H, W] (5D for VAE)
func UnpackLatents(patches *mlx.Array, H, W, patchSize int32) *mlx.Array {
	shape := patches.Shape()
	B := shape[0]
	channels := shape[2] / (patchSize * patchSize)

	pH := H / patchSize
	pW := W / patchSize

	// [B, L, C*4] -> [B, pH, pW, C, 2, 2]
	x := mlx.Reshape(patches, B, pH, pW, channels, patchSize, patchSize)
	// -> [B, C, pH, 2, pW, 2]
	x = mlx.Transpose(x, 0, 3, 1, 4, 2, 5)
	// -> [B, C, H, W]
	x = mlx.Reshape(x, B, channels, pH*patchSize, pW*patchSize)
	// Add temporal dimension for VAE: [B, C, 1, H, W]
	return mlx.ExpandDims(x, 2)
}

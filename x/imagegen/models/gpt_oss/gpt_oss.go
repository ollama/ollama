//go:build mlx

package gpt_oss

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

// RopeScaling holds YaRN or other RoPE scaling configuration
type RopeScaling struct {
	RopeType                      string  `json:"rope_type"`
	Factor                        float32 `json:"factor"`
	OriginalMaxPositionEmbeddings int32   `json:"original_max_position_embeddings"`
	BetaFast                      float32 `json:"beta_fast"`
	BetaSlow                      float32 `json:"beta_slow"`
}

type Config struct {
	HiddenSize        int32        `json:"hidden_size"`
	NumHiddenLayers   int32        `json:"num_hidden_layers"`
	IntermediateSize  int32        `json:"intermediate_size"`
	NumAttentionHeads int32        `json:"num_attention_heads"`
	NumKeyValueHeads  int32        `json:"num_key_value_heads"`
	VocabSize         int32        `json:"vocab_size"`
	RMSNormEps        float32      `json:"rms_norm_eps"`
	RopeTheta         float32      `json:"rope_theta"`
	HeadDim           int32        `json:"head_dim"`
	SlidingWindow     int32        `json:"sliding_window"`
	NumLocalExperts   int32        `json:"num_local_experts"`
	NumExpertsPerTok  int32        `json:"num_experts_per_tok"`
	LayerTypes        []string     `json:"layer_types"`
	SwiGLULimit       float32      `json:"swiglu_limit"`
	RopeScaling       *RopeScaling `json:"rope_scaling"`
	Scale             float32      `json:"-"` // computed: 1/sqrt(HeadDim)
}

type Attention struct {
	QProj      *nn.Linear `weight:"self_attn.q_proj"`
	KProj      *nn.Linear `weight:"self_attn.k_proj"`
	VProj      *nn.Linear `weight:"self_attn.v_proj"`
	OProj      *nn.Linear `weight:"self_attn.o_proj"`
	Sinks      *mlx.Array `weight:"self_attn.sinks,optional"`
	YarnFreqs  *mlx.Array // computed
	YarnMscale float32
}

// swiGLU applies the GPT-OSS custom SwiGLU activation.
// Formula: (gate * sigmoid(alpha * gate)) * (up + 1)
// with clipping: gate to [None, limit], up to [-limit, limit]
func swiGLU(gate, up *mlx.Array, alpha, limit float32) *mlx.Array {
	// Clip gate to [None, limit]
	gateClipped := mlx.ClipScalar(gate, 0, limit, false, true)

	// Clip up to [-limit, limit]
	upClipped := mlx.ClipScalar(up, -limit, limit, true, true)

	// glu_scaled = alpha * gate_clipped
	gluScaled := mlx.MulScalar(gateClipped, alpha)

	// sig = sigmoid(glu_scaled)
	sig := mlx.Sigmoid(gluScaled)

	// out_glu = gate_clipped * sig
	outGlu := mlx.Mul(gateClipped, sig)

	// result = out_glu * (up_clipped + 1)
	return mlx.Mul(outGlu, mlx.AddScalar(upClipped, 1.0))
}

// compiledSwiGLU is a singleton compiled SwiGLU function shared across all layers
var compiledSwiGLU *mlx.CompiledFunc

// getCompiledSwiGLU returns the compiled SwiGLU function, creating it once if needed
func getCompiledSwiGLU() *mlx.CompiledFunc {
	if compiledSwiGLU == nil {
		const alpha float32 = 1.702
		const limit float32 = 7.0
		compiledSwiGLU = mlx.CompileShapeless(func(inputs []*mlx.Array) []*mlx.Array {
			return []*mlx.Array{swiGLU(inputs[0], inputs[1], alpha, limit)}
		}, true) // shapeless=true so it works for any input size
	}
	return compiledSwiGLU
}

// ComputeYarnFreqs computes YaRN-modified RoPE frequencies
// Based on mlx-lm's YarnRoPE implementation
func ComputeYarnFreqs(dims int32, base, scalingFactor float32, origMaxPos int32, betaFast, betaSlow float32) (*mlx.Array, float32) {
	// yarn_find_correction_dim
	yarnFindCorrectionDim := func(numRotations float64) float64 {
		return float64(dims) * math.Log(float64(origMaxPos)/(numRotations*2*math.Pi)) / (2 * math.Log(float64(base)))
	}

	// yarn_find_correction_range
	low := int(math.Floor(yarnFindCorrectionDim(float64(betaFast))))
	high := int(math.Ceil(yarnFindCorrectionDim(float64(betaSlow))))
	if low < 0 {
		low = 0
	}
	if high > int(dims)-1 {
		high = int(dims) - 1
	}

	// yarn_get_mscale
	yarnGetMscale := func(scale, mscale float64) float64 {
		if scale <= 1 {
			return 1.0
		}
		return 0.1*mscale*math.Log(scale) + 1.0
	}
	mscale := float32(yarnGetMscale(float64(scalingFactor), 1.0) / yarnGetMscale(float64(scalingFactor), 0.0))

	// Compute frequencies
	// freq_extra = base ** (arange(0, dims, 2) / dims)
	// freq_inter = scaling_factor * freq_extra
	halfDims := dims / 2
	freqData := make([]float32, halfDims)
	for i := int32(0); i < halfDims; i++ {
		exp := float64(2*i) / float64(dims)
		freqExtra := math.Pow(float64(base), exp)
		freqInter := float64(scalingFactor) * freqExtra

		// linear ramp mask
		var freqMask float64
		if low == high {
			freqMask = 0.0
		} else {
			t := (float64(i) - float64(low)) / float64(high-low)
			if t < 0 {
				t = 0
			}
			if t > 1 {
				t = 1
			}
			freqMask = 1.0 - t
		}

		// Combined frequency: (inter * extra) / (inter * mask + extra * (1 - mask))
		freqData[i] = float32((freqInter * freqExtra) / (freqInter*freqMask + freqExtra*(1-freqMask)))
	}

	return mlx.NewArray(freqData, []int32{halfDims}), mscale
}

// initYarn initializes YaRN RoPE if configured
func (a *Attention) initYarn(cfg *Config) {
	a.YarnMscale = 1.0
	if cfg.RopeScaling != nil && cfg.RopeScaling.RopeType == "yarn" {
		a.YarnFreqs, a.YarnMscale = ComputeYarnFreqs(
			cfg.HeadDim,
			cfg.RopeTheta,
			cfg.RopeScaling.Factor,
			cfg.RopeScaling.OriginalMaxPositionEmbeddings,
			cfg.RopeScaling.BetaFast,
			cfg.RopeScaling.BetaSlow,
		)
	}
}

func (a *Attention) Forward(x *mlx.Array, c cache.Cache, B, L int32, mask *mlx.Array, maskMode string, cfg *Config) *mlx.Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	// Reshape via AsStrided: [B, L, n_heads * head_dim] -> [B, n_heads, L, head_dim]
	q = mlx.AsStrided(q, []int32{B, cfg.NumAttentionHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumAttentionHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumAttentionHeads * cfg.HeadDim), 1}, 0)
	k = mlx.AsStrided(k, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)
	v = mlx.AsStrided(v, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)

	offset := 0
	if c != nil {
		offset = c.Offset()
	}
	if a.YarnFreqs != nil {
		if a.YarnMscale != 1.0 {
			q = mlx.MulScalar(q, a.YarnMscale)
		}
		q = mlx.RoPEWithFreqs(q, a.YarnFreqs, int(cfg.HeadDim), false, 1.0, offset)
		k = mlx.RoPEWithFreqs(k, a.YarnFreqs, int(cfg.HeadDim), false, 1.0, offset)
	} else {
		q = mlx.RoPE(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, offset)
		k = mlx.RoPE(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, offset)
	}

	if c != nil {
		k, v = c.Update(k, v, int(L))
	}

	out := mlx.ScaledDotProductAttentionWithSinks(q, k, v, cfg.Scale, maskMode, mask, a.Sinks)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

// CreateSlidingWindowMask creates a causal mask with sliding window
// Mirrors mlx-lm's create_causal_mask with window_size
func CreateSlidingWindowMask(seqLen, queryStart, keyStart, keyLen, windowSize int) *mlx.Array {
	// Build mask aligned to actual cache length (may be rotated)
	// rinds covers existing keys: [keyStart, keyStart+keyLen)
	// linds covers new queries: [queryStart, queryStart+seqLen)
	rinds := mlx.Arange(float32(keyStart), float32(keyStart+keyLen), 1)     // [keyLen]
	linds := mlx.Arange(float32(queryStart), float32(queryStart+seqLen), 1) // [seqLen]

	linds = mlx.ExpandDims(linds, 1) // [seqLen, 1]
	rinds = mlx.ExpandDims(rinds, 0) // [1, keyLen]

	causalMask := mlx.GreaterEqual(linds, rinds) // [seqLen, keyLen]
	windowLimit := mlx.AddScalar(rinds, float32(windowSize))
	windowMask := mlx.LessArray(linds, windowLimit) // [seqLen, keyLen]

	return mlx.LogicalAnd(causalMask, windowMask)
}

// MoE represents the Mixture of Experts SwiGLU layer with quantized experts.
type MoE struct {
	Router     *nn.Linear `weight:"mlp.router"`
	TopK       int32
	HiddenSize int32
	GroupSize  int
	Bits       int
	// Expert weights (loaded manually via sanitizeExpertWeights)
	GateBlocks, GateScales, GateBias *mlx.Array
	UpBlocks, UpScales, UpBias       *mlx.Array
	DownBlocks, DownScales, DownBias *mlx.Array
}

func (moe *MoE) Forward(x *mlx.Array, B, L int32) *mlx.Array {
	logits := moe.Router.Forward(x)
	neg := mlx.Neg(logits)
	part := mlx.Argpartition(neg, int(moe.TopK)-1, -1)
	topKIdx := mlx.Slice(part, []int32{0, 0, 0}, []int32{B, L, moe.TopK})
	topKVal := mlx.TakeAlongAxis(logits, topKIdx, -1)
	weights := mlx.Softmax(topKVal, -1)

	xFlat := mlx.Reshape(x, B*L, 1, 1, moe.HiddenSize)
	idxFlat := mlx.Reshape(topKIdx, B*L, moe.TopK)

	doSort := B*L >= 64
	var invOrder *mlx.Array
	sorted := false
	n := B * L * moe.TopK

	if doSort {
		idxAll := mlx.Flatten(idxFlat)
		order := mlx.Argsort(idxAll, 0)
		invOrder = mlx.Argsort(order, 0)
		xFlat = mlx.ExpandDims(mlx.Take(mlx.Squeeze(xFlat, 1), mlx.FloorDivideScalar(order, moe.TopK), 0), 1)
		idxFlat = mlx.Reshape(mlx.Take(idxAll, order, 0), n, 1)
		sorted = true
	}

	gate := mlx.GatherQMM(xFlat, moe.GateBlocks, moe.GateScales, nil, nil, idxFlat, true, moe.GroupSize, moe.Bits, "mxfp4", sorted)
	up := mlx.GatherQMM(xFlat, moe.UpBlocks, moe.UpScales, nil, nil, idxFlat, true, moe.GroupSize, moe.Bits, "mxfp4", sorted)

	if moe.GateBias != nil {
		gate = mlx.Add(gate, mlx.ExpandDims(mlx.Take(moe.GateBias, idxFlat, 0), 2))
	}
	if moe.UpBias != nil {
		up = mlx.Add(up, mlx.ExpandDims(mlx.Take(moe.UpBias, idxFlat, 0), 2))
	}

	hidden := getCompiledSwiGLU().Call(gate, up)[0]

	down := mlx.GatherQMM(hidden, moe.DownBlocks, moe.DownScales, nil, nil, idxFlat, true, moe.GroupSize, moe.Bits, "mxfp4", sorted)
	if moe.DownBias != nil {
		down = mlx.Add(down, mlx.ExpandDims(mlx.Take(moe.DownBias, idxFlat, 0), 2))
	}

	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), invOrder, 0), B*L, moe.TopK, moe.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}

	ewFlat := mlx.Reshape(weights, B*L, moe.TopK, 1)
	return mlx.Reshape(mlx.Sum(mlx.Mul(down, ewFlat), 1, false), B, L, moe.HiddenSize)
}

type Block struct {
	Attention    *Attention
	MLP          *MoE
	InputNorm    *nn.RMSNorm `weight:"input_layernorm"`
	PostAttnNorm *nn.RMSNorm `weight:"post_attention_layernorm"`
	LayerType    string      // "sliding_attention" or "full_attention"
}

func (b *Block) Forward(x *mlx.Array, c cache.Cache, B, L int32, mask *mlx.Array, maskMode string, cfg *Config) *mlx.Array {
	h := mlx.Add(x, b.Attention.Forward(b.InputNorm.Forward(x, cfg.RMSNormEps), c, B, L, mask, maskMode, cfg))
	return mlx.Add(h, b.MLP.Forward(b.PostAttnNorm.Forward(h, cfg.RMSNormEps), B, L))
}

type Model struct {
	EmbedTokens *nn.Embedding `weight:"model.embed_tokens"`
	Layers      []*Block      `weight:"-"` // loaded manually due to MoE sanitization
	Norm        *nn.RMSNorm   `weight:"model.norm"`
	LMHead      *nn.Linear    `weight:"lm_head"`

	tok *tokenizer.Tokenizer
	*Config
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }
func (m *Model) NumLayers() int                     { return len(m.Layers) }
func (m *Model) VocabSize() int32                   { return m.Config.VocabSize }

func (m *Model) NewCache(int32) []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i, layer := range m.Layers {
		if layer.LayerType == "sliding_attention" && m.SlidingWindow > 0 {
			caches[i] = cache.NewRotatingKVCache(int(m.SlidingWindow))
		} else {
			caches[i] = cache.NewKVCache()
		}
	}
	return caches
}

func (m *Model) Forward(tokens *mlx.Array, caches []cache.Cache) *mlx.Array {
	B, L := tokens.Shape()[0], tokens.Shape()[1]
	x := m.EmbedTokens.Forward(tokens)

	// Find representative cache indices for sliding window attention
	var swaIdx int = -1
	for i, layer := range m.Layers {
		if layer.LayerType == "sliding_attention" {
			swaIdx = i
			break
		}
	}

	// Create masks once at model level
	var fullMask, swaMask *mlx.Array
	var fullMaskMode, swaMaskMode string

	if L > 1 {
		fullMaskMode = "causal"
		if swaIdx >= 0 && m.SlidingWindow > 0 && caches != nil {
			c := caches[swaIdx]
			offset := c.Offset()
			windowSize := int(m.SlidingWindow)
			cacheLen := min(int(L), windowSize)
			if offset > 0 {
				cacheLen = min(c.Len()+int(L), windowSize)
			}
			if int(L) > windowSize {
				swaMask = CreateSlidingWindowMask(int(L), offset, offset+int(L)-cacheLen, cacheLen, windowSize)
			} else {
				swaMaskMode = "causal"
			}
		} else {
			swaMaskMode = "causal"
		}
	}

	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil {
			c = caches[i]
		}
		mask, maskMode := fullMask, fullMaskMode
		if layer.LayerType == "sliding_attention" {
			mask, maskMode = swaMask, swaMaskMode
		}
		x = layer.Forward(x, c, B, L, mask, maskMode, m.Config)
	}

	return m.LMHead.Forward(m.Norm.Forward(x, m.RMSNormEps))
}

// sanitizeExpertWeights splits merged gate_up weights into separate gate/up arrays.
// MXFP4 quantized weights require contiguous memory - strided views give wrong results.
func sanitizeExpertWeights(weights *safetensors.ModelWeights, prefix string) (moe *MoE) {
	gateUpBlocks, _ := weights.GetTensor(prefix + ".mlp.experts.gate_up_proj_blocks")
	gateUpScales, _ := weights.GetTensor(prefix + ".mlp.experts.gate_up_proj_scales")
	gateUpBias, _ := weights.GetTensor(prefix + ".mlp.experts.gate_up_proj_bias")
	downBlocks, _ := weights.GetTensor(prefix + ".mlp.experts.down_proj_blocks")
	downScales, _ := weights.GetTensor(prefix + ".mlp.experts.down_proj_scales")
	downBias, _ := weights.GetTensor(prefix + ".mlp.experts.down_proj_bias")

	moe = &MoE{GroupSize: 32, Bits: 4, DownScales: downScales, DownBias: downBias}

	if gateUpBlocks != nil {
		gub := mlx.FlattenRange(mlx.View(gateUpBlocks, int(mlx.DtypeUint32)), -2, -1)
		s := gub.Shape()
		moe.GateBlocks = mlx.Contiguous(mlx.SliceStride(gub, []int32{0, 0, 0}, []int32{s[0], s[1], s[2]}, []int32{1, 2, 1}))
		moe.UpBlocks = mlx.Contiguous(mlx.SliceStride(gub, []int32{0, 1, 0}, []int32{s[0], s[1], s[2]}, []int32{1, 2, 1}))
	}
	if gateUpScales != nil {
		s := gateUpScales.Shape()
		moe.GateScales = mlx.Contiguous(mlx.SliceStride(gateUpScales, []int32{0, 0, 0}, []int32{s[0], s[1], s[2]}, []int32{1, 2, 1}))
		moe.UpScales = mlx.Contiguous(mlx.SliceStride(gateUpScales, []int32{0, 1, 0}, []int32{s[0], s[1], s[2]}, []int32{1, 2, 1}))
	}
	if gateUpBias != nil {
		s := gateUpBias.Shape()
		moe.GateBias = mlx.Contiguous(mlx.SliceStride(gateUpBias, []int32{0, 0}, []int32{s[0], s[1]}, []int32{1, 2}))
		moe.UpBias = mlx.Contiguous(mlx.SliceStride(gateUpBias, []int32{0, 1}, []int32{s[0], s[1]}, []int32{1, 2}))
	}
	if downBlocks != nil {
		moe.DownBlocks = mlx.FlattenRange(mlx.View(downBlocks, int(mlx.DtypeUint32)), -2, -1)
	}
	return moe
}

func Load(modelPath string) (*Model, error) {
	data, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))

	weights, err := safetensors.LoadModelWeights(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load weights: %w", err)
	}

	tok, err := tokenizer.Load(filepath.Join(modelPath, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	m := &Model{
		Layers: make([]*Block, cfg.NumHiddenLayers),
		Config: &cfg,
		tok:    tok,
	}

	// Load simple weights via struct tags
	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return nil, err
	}

	// Load layers with custom MoE handling
	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d", i)
		layer := &Block{}
		if err := safetensors.LoadModule(layer, weights, prefix); err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}

		// Initialize attention YaRN
		layer.Attention.initYarn(&cfg)

		// Load MoE with weight sanitization
		moe := sanitizeExpertWeights(weights, prefix)
		moe.Router = layer.MLP.Router // Router was loaded by LoadModule
		moe.TopK = cfg.NumExpertsPerTok
		moe.HiddenSize = cfg.HiddenSize
		layer.MLP = moe

		// Set layer type
		layer.LayerType = "full_attention"
		if int(i) < len(cfg.LayerTypes) {
			layer.LayerType = cfg.LayerTypes[i]
		}

		m.Layers[i] = layer
	}

	// Release safetensors BEFORE eval - lazy arrays have captured data,
	// this reduces peak memory by freeing mmap during materialization
	weights.ReleaseAll()
	mlx.Eval(mlx.Collect(m)...)

	return m, nil
}

func (m *Model) MaxContextLength() int32 {
	if m.RopeScaling != nil && m.RopeScaling.OriginalMaxPositionEmbeddings > 0 {
		return m.RopeScaling.OriginalMaxPositionEmbeddings
	}
	return 131072
}

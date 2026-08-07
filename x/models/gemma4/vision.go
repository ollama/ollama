package gemma4

// Vision support for Gemma 4 MLX checkpoints, ported from mlx_vlm (main
// branch) models/gemma4/vision.py and models/gemma4_unified. Two lineages
// share this file:
//
//   - gemma4_unified_vision (12b): an encoder-free embedder — LayerNorm →
//     Linear over flat 48px patches → LayerNorm → learned per-axis position
//     embeddings → LayerNorm.
//   - gemma4_vision (26b/31b): a 27-layer encoder over 16px patches with 2D
//     RoPE attention (scale 1.0), 3×3 average pooling and standardization.
//
// Both project into the text hidden size through embed_vision: a weightless
// RMSNorm followed by a linear. Image sizing follows ADR 0008's budget-fill
// ladder via llm.BudgetFillSize — grids are always exact 48-multiples, so the
// padding/masking paths of the reference are deliberately not ported.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"math"

	xdraw "golang.org/x/image/draw"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
)

// VisionConfig holds the vision_config block of a multimodal checkpoint.
type VisionConfig struct {
	ModelType string `json:"model_type"`

	// Unified embedder (12b).
	MMEmbedDim     int32 `json:"mm_embed_dim"`
	MMPosembSize   int32 `json:"mm_posemb_size"`
	ModelPatchSize int32 `json:"model_patch_size"`
	OutputProjDims int32 `json:"output_proj_dims"`

	// Full tower (26b/31b).
	HiddenSize            int32 `json:"hidden_size"`
	IntermediateSize      int32 `json:"intermediate_size"`
	NumAttentionHeads     int32 `json:"num_attention_heads"`
	HeadDim               int32 `json:"head_dim"`
	NumHiddenLayers       int32 `json:"num_hidden_layers"`
	PositionEmbeddingSize int32 `json:"position_embedding_size"`
	Standardize           bool  `json:"standardize"`
	RopeParameters        struct {
		RopeTheta float32 `json:"rope_theta"`
	} `json:"rope_parameters"`

	// Shared.
	PatchSize         int32   `json:"patch_size"`
	PoolingKernelSize int32   `json:"pooling_kernel_size"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
}

// IsUnified reports the encoder-free 12b lineage.
func (c *VisionConfig) IsUnified() bool { return c.ModelType == "gemma4_unified_vision" }

// multimodalTokens are the top-level special token ids bracketing one
// image's soft tokens in the prompt.
type multimodalTokens struct {
	BOI   int32 `json:"boi_token_id"`
	EOI   int32 `json:"eoi_token_id"`
	Image int32 `json:"image_token_id"`
}

// parseVisionConfig reads vision_config and the top-level multimodal token
// ids from the raw config bytes. It is a separate unmarshal on purpose:
// parseTextConfig replaces its result wholesale with the nested text_config,
// which would silently discard these top-level keys.
func parseVisionConfig(configData []byte) (*VisionConfig, multimodalTokens, error) {
	var wrapped struct {
		multimodalTokens
		VisionConfig *VisionConfig `json:"vision_config"`
	}
	if err := json.Unmarshal(configData, &wrapped); err != nil {
		return nil, multimodalTokens{}, fmt.Errorf("parse vision config: %w", err)
	}
	cfg := wrapped.VisionConfig
	if cfg == nil {
		return nil, wrapped.multimodalTokens, nil
	}
	if cfg.PatchSize == 0 {
		cfg.PatchSize = 16
	}
	if cfg.PoolingKernelSize == 0 {
		cfg.PoolingKernelSize = 3
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.RopeParameters.RopeTheta == 0 {
		cfg.RopeParameters.RopeTheta = 100
	}
	if cfg.IsUnified() && cfg.ModelPatchSize == 0 {
		cfg.ModelPatchSize = cfg.PatchSize * cfg.PoolingKernelSize
	}
	return cfg, wrapped.multimodalTokens, nil
}

// VisionEmbedder is the encoder-free unified path (12b).
type VisionEmbedder struct {
	PatchLN1   *nn.LayerNorm
	PatchDense nn.LinearLayer
	PatchLN2   *nn.LayerNorm
	PosEmbX    *mlx.Array // [MMPosembSize, MMEmbedDim] — pos_embedding[:, 0, :]
	PosEmbY    *mlx.Array // [MMPosembSize, MMEmbedDim] — pos_embedding[:, 1, :]
	PosNorm    *nn.LayerNorm
}

// VisionAttention is one tower block's attention: per-head q/k RMSNorm with
// weights, weightless v RMSNorm, 2D RoPE, SDPA at scale 1.0.
type VisionAttention struct {
	QProj, KProj, VProj, OProj nn.LinearLayer
	QNormWeight, KNormWeight   *mlx.Array // [HeadDim]
}

// VisionLayer is one sandwich-norm tower block. Norm weights apply directly
// (no +1 shift, matching the reference's plain rms_norm).
type VisionLayer struct {
	InputNorm, PostAttnNorm, PreFFNorm, PostFFNorm *mlx.Array
	Attn                                           *VisionAttention
	GateProj, UpProj, DownProj                     nn.LinearLayer
}

// VisionTower is the shared 27-layer encoder (26b/31b).
type VisionTower struct {
	InputProj  nn.LinearLayer
	PosTableX  *mlx.Array // [PositionEmbeddingSize, HiddenSize] — table[0]
	PosTableY  *mlx.Array // [PositionEmbeddingSize, HiddenSize] — table[1]
	Layers     []*VisionLayer
	NegStdBias *mlx.Array // -std_bias, precomputed; nil unless Standardize
	StdScale   *mlx.Array
}

// resolveVisionPrefix probes the spellings vision tensors ship under.
func resolveVisionPrefix(tensors map[string]*mlx.Array, marker string) (string, bool) {
	for _, prefix := range []string{"model.", ""} {
		if tensors[prefix+marker] != nil {
			return prefix, true
		}
	}
	return "", false
}

// materialize detaches a lazy view (slice/negate of a loaded tensor) into
// its own buffer so the parent blob can be released.
func materialize(a *mlx.Array) *mlx.Array {
	c := a.Clone()
	mlx.Eval(c)
	return c
}

func visionLayerNorm(tensors map[string]*mlx.Array, path string) (*nn.LayerNorm, error) {
	w := tensors[path+".weight"]
	b := tensors[path+".bias"]
	if w == nil || b == nil {
		return nil, fmt.Errorf("missing vision layer norm: %s", path)
	}
	// Eps 0 defers to nn.LayerNorm's 1e-5 default, matching the reference's
	// torch/mlx nn.LayerNorm defaults (distinct from the 1e-6 rms_norm_eps).
	return &nn.LayerNorm{Weight: w, Bias: b}, nil
}

// loadVisionWeights binds the vision tensors for whichever lineage the
// checkpoint ships. Called from LoadWeights when the config carries a
// vision_config.
func (m *Model) loadVisionWeights(tensors map[string]*mlx.Array, linears model.LinearFactory) error {
	cfg := m.VisionCfg

	projPrefix, ok := resolveVisionPrefix(tensors, "embed_vision.embedding_projection.weight")
	if !ok {
		return fmt.Errorf("missing vision projection: embed_vision.embedding_projection.weight")
	}
	m.EmbedVisionProj = linears.Make(projPrefix + "embed_vision.embedding_projection")
	if m.EmbedVisionProj == nil {
		return fmt.Errorf("missing vision projection: %sembed_vision.embedding_projection.weight", projPrefix)
	}

	if cfg.IsUnified() {
		prefix, ok := resolveVisionPrefix(tensors, "vision_embedder.pos_embedding")
		if !ok {
			return fmt.Errorf("missing vision embedder: vision_embedder.pos_embedding")
		}
		p := prefix + "vision_embedder."

		e := &VisionEmbedder{}
		var err error
		if e.PatchLN1, err = visionLayerNorm(tensors, p+"patch_ln1"); err != nil {
			return err
		}
		if e.PatchDense = linears.Make(p + "patch_dense"); e.PatchDense == nil {
			return fmt.Errorf("missing vision embedder weight: %spatch_dense.weight", p)
		}
		if e.PatchLN2, err = visionLayerNorm(tensors, p+"patch_ln2"); err != nil {
			return err
		}
		if e.PosNorm, err = visionLayerNorm(tensors, p+"pos_norm"); err != nil {
			return err
		}
		pos := tensors[p+"pos_embedding"] // [MMPosembSize, 2, MMEmbedDim]
		if pos == nil {
			return fmt.Errorf("missing vision embedder weight: %spos_embedding", p)
		}
		e.PosEmbX = materialize(pos.Slice(mlx.Slice(), mlx.Slice(0, 1), mlx.Slice()).Squeeze(1))
		e.PosEmbY = materialize(pos.Slice(mlx.Slice(), mlx.Slice(1, 2), mlx.Slice()).Squeeze(1))
		m.VisionEmbedder = e
		return nil
	}

	prefix, ok := resolveVisionPrefix(tensors, "vision_tower.patch_embedder.input_proj.weight")
	if !ok {
		return fmt.Errorf("missing vision tower: vision_tower.patch_embedder.input_proj.weight")
	}
	p := prefix + "vision_tower."

	t := &VisionTower{}
	if t.InputProj = linears.Make(p + "patch_embedder.input_proj"); t.InputProj == nil {
		return fmt.Errorf("missing vision tower weight: %spatch_embedder.input_proj.weight", p)
	}
	table := tensors[p+"patch_embedder.position_embedding_table"] // [2, S, HiddenSize]
	if table == nil {
		return fmt.Errorf("missing vision tower weight: %spatch_embedder.position_embedding_table", p)
	}
	t.PosTableX = materialize(table.Slice(mlx.Slice(0, 1), mlx.Slice(), mlx.Slice()).Squeeze(0))
	t.PosTableY = materialize(table.Slice(mlx.Slice(1, 2), mlx.Slice(), mlx.Slice()).Squeeze(0))

	if cfg.Standardize {
		stdBias := tensors[p+"std_bias"]
		t.StdScale = tensors[p+"std_scale"]
		if stdBias == nil || t.StdScale == nil {
			return fmt.Errorf("missing vision tower standardization: %sstd_bias / %sstd_scale", p, p)
		}
		t.NegStdBias = materialize(mlx.MulScalar(stdBias, -1))
	}

	t.Layers = make([]*VisionLayer, cfg.NumHiddenLayers)
	for i := range t.Layers {
		lp := fmt.Sprintf("%sencoder.layers.%d.", p, i)
		l := &VisionLayer{Attn: &VisionAttention{}}

		for _, norm := range []struct {
			dst  **mlx.Array
			name string
		}{
			{&l.InputNorm, "input_layernorm"},
			{&l.PostAttnNorm, "post_attention_layernorm"},
			{&l.PreFFNorm, "pre_feedforward_layernorm"},
			{&l.PostFFNorm, "post_feedforward_layernorm"},
			{&l.Attn.QNormWeight, "self_attn.q_norm"},
			{&l.Attn.KNormWeight, "self_attn.k_norm"},
		} {
			w := tensors[lp+norm.name+".weight"]
			if w == nil {
				return fmt.Errorf("missing vision tower norm: %s%s.weight", lp, norm.name)
			}
			*norm.dst = w
		}

		// ClippableLinear wraps the projection as `.linear` in the checkpoint;
		// use_clipped_linears is false so no clip bounds ship alongside.
		for _, proj := range []struct {
			dst  *nn.LinearLayer
			name string
		}{
			{&l.Attn.QProj, "self_attn.q_proj.linear"},
			{&l.Attn.KProj, "self_attn.k_proj.linear"},
			{&l.Attn.VProj, "self_attn.v_proj.linear"},
			{&l.Attn.OProj, "self_attn.o_proj.linear"},
			{&l.GateProj, "mlp.gate_proj.linear"},
			{&l.UpProj, "mlp.up_proj.linear"},
			{&l.DownProj, "mlp.down_proj.linear"},
		} {
			if *proj.dst = linears.Make(lp + proj.name); *proj.dst == nil {
				return fmt.Errorf("missing vision tower weight: %s%s.weight", lp, proj.name)
			}
		}
		t.Layers[i] = l
	}
	m.VisionTower = t
	return nil
}

// visionRopeTables builds duplicated-half cos/sin tables for one spatial axis
// of the reference's multidimensional RoPE: coords is each patch's coordinate
// on that axis, halfDim the per-axis channel count (head_dim/2), theta the
// base frequency. Frequencies land on the first half and repeat on the
// second, matching mx.concatenate([cos, cos]). Layout [1, L, 1, halfDim].
func visionRopeTables(coords []int32, halfDim int, theta float64, dtype mlx.DType) (cos, sin *mlx.Array) {
	quarter := halfDim / 2
	c := make([]float32, len(coords)*halfDim)
	s := make([]float32, len(coords)*halfDim)
	for i, p := range coords {
		for j := 0; j < quarter; j++ {
			ts := math.Pow(theta, 2*float64(j)/float64(halfDim))
			angle := float64(p) / ts
			cv, sv := float32(math.Cos(angle)), float32(math.Sin(angle))
			c[i*halfDim+j], c[i*halfDim+quarter+j] = cv, cv
			s[i*halfDim+j], s[i*halfDim+quarter+j] = sv, sv
		}
	}
	cos = mlx.FromValues(c, 1, len(coords), 1, halfDim).AsType(dtype)
	sin = mlx.FromValues(s, 1, len(coords), 1, halfDim).AsType(dtype)
	return cos, sin
}

// visionRope2D rotates x [B, L, H, D]: the head dim splits into an x-axis
// and a y-axis half, each rotated independently — rotate_half must not mix
// features across spatial axes.
func visionRope2D(x, cosX, sinX, cosY, sinY *mlx.Array) *mlx.Array {
	d := x.Dim(3)
	half := d / 2
	rot := func(p *mlx.Array) *mlx.Array {
		lo := p.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, half/2))
		hi := p.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(half/2, half))
		return mlx.Concatenate([]*mlx.Array{mlx.MulScalar(hi, -1), lo}, 3)
	}
	xX := x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, half))
	xY := x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(half, d))
	outX := mlx.Add(mlx.Mul(xX, cosX), mlx.Mul(rot(xX), sinX))
	outY := mlx.Add(mlx.Mul(xY, cosY), mlx.Mul(rot(xY), sinY))
	return mlx.Concatenate([]*mlx.Array{outX, outY}, 3)
}

// visionAvgPool averages k×k patch blocks over an exact row-major grid:
// [1, gridH*gridW, D] → [1, (gridH/k)*(gridW/k), D]. Runs in float32 like
// the reference's einsum pooler, returning the input dtype.
func visionAvgPool(h *mlx.Array, gridW, gridH, k int32) *mlx.Array {
	dt := h.DType()
	D := int32(h.Dim(2))
	rB, cB := gridH/k, gridW/k
	p := mlx.Reshape(h.AsType(mlx.DTypeFloat32), 1, rB, k, cB, k, D)
	p = mlx.Mean(p, 4, false) // [1, rB, k, cB, D]
	p = mlx.Mean(p, 2, false) // [1, rB, cB, D]
	return mlx.Reshape(p, 1, rB*cB, D).AsType(dt)
}

// Forward runs one tower block's attention: per-head q/k RMSNorm (learned),
// weightless v RMSNorm, 2D RoPE, and SDPA at the reference's scale of 1.0.
// Grids are exact, so there is never a padding mask.
func (a *VisionAttention) Forward(x, cosX, sinX, cosY, sinY *mlx.Array, cfg *VisionConfig) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	H, D := cfg.NumAttentionHeads, cfg.HeadDim

	q := mlx.Reshape(a.QProj.Forward(x), B, L, H, D)
	k := mlx.Reshape(a.KProj.Forward(x), B, L, H, D)
	v := mlx.Reshape(a.VProj.Forward(x), B, L, H, D)

	q = mlx.RMSNormFn(q, a.QNormWeight, cfg.RMSNormEps)
	k = mlx.RMSNormFn(k, a.KNormWeight, cfg.RMSNormEps)
	v = mlx.RMSNormFn(v, nil, cfg.RMSNormEps)

	q = visionRope2D(q, cosX, sinX, cosY, sinY)
	k = visionRope2D(k, cosX, sinX, cosY, sinY)

	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)
	out := mlx.FastScaledDotProductAttention(q, k, v, 1.0, "", nil)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, H*D)
	if !mlx.MetalIsAvailable() {
		out = mlx.Contiguous(out, false)
	}
	return a.OProj.Forward(out)
}

// Forward encodes 16px patches [1, L, 768] (values already 2x−1) through the
// 27-layer tower: input projection, per-axis position embeddings, sandwich-
// norm blocks, 3×3 pooling ×√hidden, and standardization.
func (t *VisionTower) Forward(patches *mlx.Array, xs, ys []int32, gridW, gridH int32, cfg *VisionConfig) *mlx.Array {
	h := t.InputProj.Forward(patches)

	xi := mlx.FromValues(xs, len(xs))
	yi := mlx.FromValues(ys, len(ys))
	pos := mlx.Add(mlx.Take(t.PosTableX, xi, 0), mlx.Take(t.PosTableY, yi, 0))
	h = mlx.Add(h, mlx.ExpandDims(pos, 0))

	half := int(cfg.HeadDim) / 2
	theta := float64(cfg.RopeParameters.RopeTheta)
	cosX, sinX := visionRopeTables(xs, half, theta, h.DType())
	cosY, sinY := visionRopeTables(ys, half, theta, h.DType())

	for _, l := range t.Layers {
		n := mlx.RMSNormFn(h, l.InputNorm, cfg.RMSNormEps)
		a := l.Attn.Forward(n, cosX, sinX, cosY, sinY, cfg)
		a = mlx.RMSNormFn(a, l.PostAttnNorm, cfg.RMSNormEps)
		h = mlx.Add(h, a)

		n = mlx.RMSNormFn(h, l.PreFFNorm, cfg.RMSNormEps)
		f := l.DownProj.Forward(mlx.GeGLU(l.GateProj.Forward(n), l.UpProj.Forward(n)))
		f = mlx.RMSNormFn(f, l.PostFFNorm, cfg.RMSNormEps)
		h = mlx.Add(h, f)
	}

	h = visionAvgPool(h, gridW, gridH, cfg.PoolingKernelSize)
	h = mlx.MulScalar(h, float32(math.Sqrt(float64(cfg.HiddenSize))))

	if t.NegStdBias != nil {
		h = mlx.Mul(mlx.Add(h, t.NegStdBias), t.StdScale)
	}
	return h
}

// Forward embeds 48px patches [1, L, 6912] (values in [0,1]) through the
// encoder-free unified path. One patch is one soft token; positions index
// the learned per-axis tables at soft-token granularity.
func (e *VisionEmbedder) Forward(patches *mlx.Array, xs, ys []int32) *mlx.Array {
	h := e.PatchLN1.Forward(patches)
	h = e.PatchDense.Forward(h)
	h = e.PatchLN2.Forward(h)

	xi := mlx.FromValues(xs, len(xs))
	yi := mlx.FromValues(ys, len(ys))
	pos := mlx.Add(mlx.Take(e.PosEmbX, xi, 0), mlx.Take(e.PosEmbY, yi, 0))
	h = mlx.Add(h, mlx.ExpandDims(pos, 0))

	return e.PosNorm.Forward(h)
}

// projectVision maps soft tokens into the text hidden size through
// embed_vision: a weightless RMSNorm then the linear projection.
func (m *Model) projectVision(h *mlx.Array) *mlx.Array {
	eps := float32(1e-6)
	if m.VisionCfg != nil && m.VisionCfg.RMSNormEps != 0 {
		eps = m.VisionCfg.RMSNormEps
	}
	return m.EmbedVisionProj.Forward(mlx.RMSNormFn(h, nil, eps))
}

// visionInput is one preprocessed image: budget-fill-resized, rescaled, and
// patchified at the lineage's patch granularity (48px unified, 16px tower).
type visionInput struct {
	patches  []float32 // [n, patchDim] flattened, channel fastest
	n        int32     // patches at the lineage's granularity
	patchDim int32
	xs, ys   []int32 // per-patch grid coordinates (x=col, y=row)
	gridW    int32   // patch-level grid dims
	gridH    int32
	soft     int // soft tokens = (targetW/48)·(targetH/48)
}

func (v *visionInput) SoftTokens() int { return v.soft }

// SupportsVision reports whether this checkpoint shipped a vision path; the
// gemma4 package also registers text-only architectures.
func (m *Model) SupportsVision() bool {
	return m.VisionCfg != nil && (m.VisionEmbedder != nil || m.VisionTower != nil)
}

// VisionTokens returns the ids bracketing one image's soft tokens.
func (m *Model) VisionTokens() (int32, int32, int32) {
	return m.MMTokens.BOI, m.MMTokens.Image, m.MMTokens.EOI
}

// NewVisionInput decodes and preprocesses one image per the reference
// pipeline pinned to ADR 0008 sizing: convert to RGB, bicubic-resize to the
// budget-fill target (always a 48-multiple on both axes), rescale by 1/255
// with no mean/std normalization, and patchify channel-fastest. The tower
// lineage additionally maps values to 2x−1, which its reference applies
// inside _patchify. Pure Go — safe off the MLX thread.
func (m *Model) NewVisionInput(data []byte, opts api.Options) (base.VisionInput, error) {
	if m.VisionCfg == nil {
		return nil, errors.New("model has no vision configuration")
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	bounds := img.Bounds()
	_, maxTok := llm.Gemma4ImageBudget(opts)
	tw, th := llm.BudgetFillSize(bounds.Dx(), bounds.Dy(), llm.Gemma4ImageAlign, maxTok)

	// CatmullRom is the Keys a=-0.5 cubic — the same family as PIL's BICUBIC
	// resample (processor_config resample=3).
	dst := image.NewRGBA(image.Rect(0, 0, tw, th))
	xdraw.CatmullRom.Scale(dst, dst.Bounds(), img, bounds, xdraw.Src, nil)

	p := int(m.VisionCfg.PatchSize) * int(m.VisionCfg.PoolingKernelSize)
	scale, shift := float32(1)/255, float32(0)
	if !m.VisionCfg.IsUnified() {
		p = int(m.VisionCfg.PatchSize)
		scale, shift = 2.0/255, -1
	}
	pW, pH := tw/p, th/p
	patchDim := p * p * 3
	patches := make([]float32, pW*pH*patchDim)
	xs, ys := make([]int32, pW*pH), make([]int32, pW*pH)
	for py := 0; py < pH; py++ {
		for px := 0; px < pW; px++ {
			i := py*pW + px
			xs[i], ys[i] = int32(px), int32(py)
			for dy := 0; dy < p; dy++ {
				o := dst.PixOffset(px*p, (py*p)+dy)
				row := dst.Pix[o : o+p*4]
				for dx := 0; dx < p; dx++ {
					d := i*patchDim + (dy*p+dx)*3
					patches[d+0] = float32(row[dx*4+0])*scale + shift
					patches[d+1] = float32(row[dx*4+1])*scale + shift
					patches[d+2] = float32(row[dx*4+2])*scale + shift
				}
			}
		}
	}
	return &visionInput{
		patches: patches, n: int32(pW * pH), patchDim: int32(patchDim),
		xs: xs, ys: ys, gridW: int32(pW), gridH: int32(pH),
		soft: (tw / llm.Gemma4ImageAlign) * (th / llm.Gemma4ImageAlign),
	}, nil
}

// EncodeVision embeds one preprocessed image into text hidden space,
// returning [1, SoftTokens, hidden]. MLX thread only.
func (m *Model) EncodeVision(in base.VisionInput) *mlx.Array {
	vi := in.(*visionInput)
	x := mlx.FromValues(vi.patches, 1, int(vi.n), int(vi.patchDim)).AsType(mlx.DTypeBFloat16)
	var h *mlx.Array
	if m.VisionEmbedder != nil {
		h = m.VisionEmbedder.Forward(x, vi.xs, vi.ys)
	} else {
		h = m.VisionTower.Forward(x, vi.xs, vi.ys, vi.gridW, vi.gridH, m.VisionCfg)
	}
	return m.projectVision(h)
}

// MergedEmbeddings returns the embed-scaled token embeddings for inputIDs
// with vision features spliced over spans — the reference's masked_scatter,
// specialized to the runner's known half-open span layout. Features replace
// the placeholder embeddings unscaled, matching get_input_embeddings.
func (m *Model) MergedEmbeddings(inputIDs *mlx.Array, features []*mlx.Array, spans [][2]int32) *mlx.Array {
	h := mlx.MulScalar(m.EmbedTokens.Forward(inputIDs), m.EmbedScale)
	for i, f := range features {
		h = h.SliceUpdate(f.AsType(h.DType()), mlx.Slice(), mlx.Slice(int(spans[i][0]), int(spans[i][1])), mlx.Slice())
	}
	return h
}

var _ base.VisionModel = (*Model)(nil)

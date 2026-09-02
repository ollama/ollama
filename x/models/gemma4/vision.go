package gemma4

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
)

// VisionConfig holds configuration for the Gemma 4 vision path: the
// gemma4_vision transformer tower, or the gemma4_unified_vision encoder-free
// embedder (one dense projection over merged 48px patches).
type VisionConfig struct {
	ModelType         string      `json:"model_type"`
	HiddenSize        int32       `json:"hidden_size"`
	NumHiddenLayers   int32       `json:"num_hidden_layers"`
	NumAttentionHeads int32       `json:"num_attention_heads"`
	HeadDim           int32       `json:"head_dim"`
	PatchSize         int32       `json:"patch_size"`
	PoolingKernelSize int32       `json:"pooling_kernel_size"`
	DefaultOutputLen  int32       `json:"default_output_length"`
	RMSNormEps        float32     `json:"rms_norm_eps"`
	Standardize       bool        `json:"standardize"`
	RopeParameters    *RopeParams `json:"rope_parameters"`

	// Unified-embedder field.
	NumSoftTokens int32 `json:"num_soft_tokens"`

	RopeTheta float32 `json:"-"`
}

func (v *VisionConfig) unified() bool {
	return v.ModelType == "gemma4_unified_vision"
}

// visionSoftTokenBudgets are the soft-token counts the reference processor
// supports; the budget fixes the patch budget as budget*pooling².
var visionSoftTokenBudgets = []int32{70, 140, 280, 560, 1120}

// visionSoftTokenBudget returns the per-image soft-token budget the checkpoint
// requests.
func (m *Model) visionSoftTokenBudget() int32 {
	if m.MM.VisionSoftTokensPerImage > 0 {
		return m.MM.VisionSoftTokensPerImage
	}
	if m.Vision.unified() {
		return m.Vision.NumSoftTokens
	}
	return m.Vision.DefaultOutputLen
}

// VisionTower is the Gemma 4 image encoder: a bidirectional transformer over
// patch embeddings with 2D positions, pooled 3x3 into soft tokens.
type VisionTower struct {
	PatchEmbedder *PatchEmbedder
	Layers        []*VisionLayer
	StdBias       *mlx.Array
	StdScale      *mlx.Array
}

type PatchEmbedder struct {
	InputProj              nn.LinearLayer
	PositionEmbeddingTable *mlx.Array // [2, positions, hidden]; x and y rows summed
}

type VisionLayer struct {
	InputNorm    *mlx.Array
	Attention    *VisionAttention
	PostAttnNorm *mlx.Array
	PreFFNorm    *mlx.Array
	MLP          *MLP
	PostFFNorm   *mlx.Array
}

type VisionAttention struct {
	QProj, KProj, VProj, OProj nn.LinearLayer
	QNorm, KNorm               *mlx.Array // per-head; v-norm has no weight
}

// UnifiedVisionEmbedder is the encoder-free vision path: one dense
// projection over merged model patches with factorized 2D positions.
type UnifiedVisionEmbedder struct {
	PatchLN1     *nn.LayerNorm
	PatchDense   nn.LinearLayer
	PatchLN2     *nn.LayerNorm
	PosEmbedding *mlx.Array // [positions, 2, mm_embed_dim]; x and y rows summed
	PosNorm      *nn.LayerNorm
}

// visionWeightRoot locates the tower's tensor prefix by probing for the
// patch embedder projection.
func visionWeightRoot(tensors map[string]*mlx.Array) (string, error) {
	for _, root := range []string{"model.", ""} {
		if tensors[root+"vision_tower.patch_embedder.input_proj.weight"] != nil {
			return root, nil
		}
	}
	return "", fmt.Errorf("config declares a gemma4_vision tower but vision_tower weights are missing from the manifest")
}

func (m *Model) loadVisionWeights(tensors map[string]*mlx.Array, linears model.LinearFactory) error {
	if m.Vision.unified() {
		return m.loadUnifiedVisionWeights(tensors, linears)
	}
	root, err := visionWeightRoot(tensors)
	if err != nil {
		return err
	}
	vt := root + "vision_tower."
	v := m.Vision

	requiredNorm := func(name string) (*mlx.Array, error) {
		w := tensors[name]
		if w == nil {
			return nil, fmt.Errorf("missing vision weight: %s", name)
		}
		return w, nil
	}

	inputProj, err := makeClippableLinear(linears, tensors, vt+"patch_embedder.input_proj")
	if err != nil {
		return err
	}
	table := tensors[vt+"patch_embedder.position_embedding_table"]
	if table == nil {
		return fmt.Errorf("missing vision weight: %spatch_embedder.position_embedding_table", vt)
	}
	tower := &VisionTower{
		PatchEmbedder: &PatchEmbedder{InputProj: inputProj, PositionEmbeddingTable: table},
		Layers:        make([]*VisionLayer, v.NumHiddenLayers),
	}

	for i := range tower.Layers {
		lp := fmt.Sprintf("%sencoder.layers.%d.", vt, i)
		layer := &VisionLayer{Attention: &VisionAttention{}, MLP: &MLP{}}

		if layer.InputNorm, err = requiredNorm(lp + "input_layernorm.weight"); err != nil {
			return err
		}
		if layer.PostAttnNorm, err = requiredNorm(lp + "post_attention_layernorm.weight"); err != nil {
			return err
		}
		if layer.PreFFNorm, err = requiredNorm(lp + "pre_feedforward_layernorm.weight"); err != nil {
			return err
		}
		if layer.PostFFNorm, err = requiredNorm(lp + "post_feedforward_layernorm.weight"); err != nil {
			return err
		}
		if layer.Attention.QNorm, err = requiredNorm(lp + "self_attn.q_norm.weight"); err != nil {
			return err
		}
		if layer.Attention.KNorm, err = requiredNorm(lp + "self_attn.k_norm.weight"); err != nil {
			return err
		}
		if layer.Attention.QProj, err = makeClippableLinear(linears, tensors, lp+"self_attn.q_proj"); err != nil {
			return err
		}
		if layer.Attention.KProj, err = makeClippableLinear(linears, tensors, lp+"self_attn.k_proj"); err != nil {
			return err
		}
		if layer.Attention.VProj, err = makeClippableLinear(linears, tensors, lp+"self_attn.v_proj"); err != nil {
			return err
		}
		if layer.Attention.OProj, err = makeClippableLinear(linears, tensors, lp+"self_attn.o_proj"); err != nil {
			return err
		}
		if layer.MLP.GateProj, err = makeClippableLinear(linears, tensors, lp+"mlp.gate_proj"); err != nil {
			return err
		}
		if layer.MLP.UpProj, err = makeClippableLinear(linears, tensors, lp+"mlp.up_proj"); err != nil {
			return err
		}
		if layer.MLP.DownProj, err = makeClippableLinear(linears, tensors, lp+"mlp.down_proj"); err != nil {
			return err
		}
		tower.Layers[i] = layer
	}

	if v.Standardize {
		if tower.StdBias, err = requiredNorm(vt + "std_bias"); err != nil {
			return err
		}
		if tower.StdScale, err = requiredNorm(vt + "std_scale"); err != nil {
			return err
		}
	}

	projection, err := makeClippableLinear(linears, tensors, root+"embed_vision.embedding_projection")
	if err != nil {
		return err
	}

	m.VisionTower = tower
	m.EmbedVision = &MultimodalEmbedder{Projection: projection}
	return nil
}

// loadUnifiedVisionWeights loads the encoder-free embedder. Its projections
// carry plain names (no .linear infix) and no clamps.
func (m *Model) loadUnifiedVisionWeights(tensors map[string]*mlx.Array, linears model.LinearFactory) error {
	var root string
	for _, r := range []string{"model.", ""} {
		if tensors[r+"vision_embedder.patch_dense.weight"] != nil {
			root = r + "vision_embedder."
			break
		}
	}
	if root == "" {
		return fmt.Errorf("config declares a gemma4_unified_vision embedder but vision_embedder weights are missing from the manifest")
	}

	norm := func(name string) (*nn.LayerNorm, error) {
		w, b := tensors[root+name+".weight"], tensors[root+name+".bias"]
		if w == nil || b == nil {
			return nil, fmt.Errorf("missing vision weight: %s%s", root, name)
		}
		// The reference uses torch LayerNorm defaults; rms_norm_eps feeds
		// only the multimodal embedder's RMSNorm.
		return &nn.LayerNorm{Weight: w, Bias: b, Eps: 1e-5}, nil
	}

	e := &UnifiedVisionEmbedder{}
	var err error
	if e.PatchLN1, err = norm("patch_ln1"); err != nil {
		return err
	}
	if e.PatchDense = linears.Make(root + "patch_dense"); e.PatchDense == nil {
		return fmt.Errorf("missing vision weight: %spatch_dense", root)
	}
	if e.PatchLN2, err = norm("patch_ln2"); err != nil {
		return err
	}
	if e.PosEmbedding = tensors[root+"pos_embedding"]; e.PosEmbedding == nil {
		return fmt.Errorf("missing vision weight: %spos_embedding", root)
	}
	if e.PosNorm, err = norm("pos_norm"); err != nil {
		return err
	}

	projRoot := strings.TrimSuffix(root, "vision_embedder.")
	projection, err := makeClippableLinear(linears, tensors, projRoot+"embed_vision.embedding_projection")
	if err != nil {
		return err
	}

	m.UnifiedEmbedder = e
	m.EmbedVision = &MultimodalEmbedder{Projection: projection}
	return nil
}

// encodeUnifiedImage embeds one image's merged patches: LN, dense, LN, factorized
// 2D positions, LN, then the shared multimodal embedder.
func (m *Model) encodeUnifiedImage(patches *mlx.Array, positions []int32, geom ImageGeometry) *mlx.Array {
	e := m.UnifiedEmbedder
	n := geom.PatchesH * geom.PatchesW

	h := e.PatchLN1.Forward(patches.AsType(mlx.DTypeBFloat16))
	h = e.PatchDense.Forward(h)
	h = e.PatchLN2.Forward(h)

	for axis := range 2 {
		pos := make([]int32, n)
		for i := range int(n) {
			pos[i] = positions[2*i+axis]
		}
		axisTable := mlx.Squeeze(e.PosEmbedding.Slice(mlx.Slice(), mlx.Slice(axis, axis+1), mlx.Slice()), 1)
		h = mlx.Add(h, mlx.Take(axisTable, mlx.FromValues(pos, int(n)), 0).AsType(h.DType()))
	}
	h = e.PosNorm.Forward(h)

	h = mlx.RMSNormFn(h, nil, m.Vision.RMSNormEps)
	return m.EmbedVision.Projection.Forward(h)
}

func (m *Model) visionLoaded() bool {
	return m.VisionTower != nil || m.UnifiedEmbedder != nil
}

// visionRopeTables builds the per-axis rotation tables, shared by every
// layer: positions x inverse frequencies in f32, duplicated across the
// half's two rotation quarters, shaped to broadcast over heads.
func visionRopeTables(positions []int32, axis int, n, headDim int32, theta float32) (cos, sin *mlx.Array) {
	spatial := int(headDim) / 2
	nFreq := spatial / 2

	pos := make([]float32, n)
	for i := range int(n) {
		pos[i] = float32(positions[2*i+axis])
	}
	invFreq := make([]float32, nFreq)
	for i := range invFreq {
		invFreq[i] = float32(1 / math.Pow(float64(theta), float64(2*i)/float64(spatial)))
	}

	f := mlx.Matmul(mlx.FromValues(pos, int(n), 1), mlx.FromValues(invFreq, 1, nFreq))
	emb := mlx.Concatenate([]*mlx.Array{f, f}, -1)
	cos = mlx.Reshape(mlx.Cos(emb), 1, n, 1, int32(spatial)).AsType(mlx.DTypeBFloat16)
	sin = mlx.Reshape(mlx.Sin(emb), 1, n, 1, int32(spatial)).AsType(mlx.DTypeBFloat16)
	return cos, sin
}

// applyVisionRoPE rotates the head dim's two halves with the x and y tables;
// rotation happens within each half independently.
func applyVisionRoPE(t, cosX, sinX, cosY, sinY *mlx.Array, headDim int32) *mlx.Array {
	half := int(headDim) / 2
	rot := func(p *mlx.Array) *mlx.Array {
		quarter := half / 2
		p1 := p.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, quarter))
		p2 := p.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(quarter, half))
		return mlx.Concatenate([]*mlx.Array{mlx.Neg(p2), p1}, -1)
	}

	tx := t.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, half))
	ty := t.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(half, int(headDim)))
	tx = mlx.Add(mlx.Mul(tx, cosX), mlx.Mul(rot(tx), sinX))
	ty = mlx.Add(mlx.Mul(ty, cosY), mlx.Mul(rot(ty), sinY))
	return mlx.Concatenate([]*mlx.Array{tx, ty}, -1)
}

func (a *VisionAttention) Forward(x *mlx.Array, cosX, sinX, cosY, sinY *mlx.Array, n int32, v *VisionConfig) *mlx.Array {
	heads, headDim := v.NumAttentionHeads, v.HeadDim

	q := mlx.Reshape(a.QProj.Forward(x), 1, n, heads, headDim)
	k := mlx.Reshape(a.KProj.Forward(x), 1, n, heads, headDim)
	val := mlx.Reshape(a.VProj.Forward(x), 1, n, heads, headDim)

	q = mlx.RMSNormFn(q, a.QNorm, v.RMSNormEps)
	k = mlx.RMSNormFn(k, a.KNorm, v.RMSNormEps)
	val = mlx.RMSNormFn(val, nil, v.RMSNormEps)

	q = applyVisionRoPE(q, cosX, sinX, cosY, sinY, headDim)
	k = applyVisionRoPE(k, cosX, sinX, cosY, sinY, headDim)

	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	val = mlx.Transpose(val, 0, 2, 1, 3)

	// Fully bidirectional, and scale 1.0: the Q/K norms control magnitude.
	out := mlx.FastScaledDotProductAttention(q, k, val, 1.0, "", nil)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), 1, n, heads*headDim)
	return a.OProj.Forward(out)
}

// encodeImage runs the tower over one whole image. One image per call, no
// padding; the graph is lazy and evaluates with the consuming forward.
func (m *Model) encodeImage(pixels *mlx.Array, positions []int32, geom ImageGeometry) *mlx.Array {
	v := m.Vision
	n := geom.PatchesH * geom.PatchesW

	// The processor rescales to [0,1]; the reference folds the [-1,1]
	// normalization into the patch embedder, in f32 before the cast.
	x := mlx.MulScalar(mlx.AddScalar(pixels, -0.5), 2).AsType(mlx.DTypeBFloat16)
	h := m.VisionTower.PatchEmbedder.InputProj.Forward(x)

	table := m.VisionTower.PatchEmbedder.PositionEmbeddingTable
	for axis := range 2 {
		pos := make([]int32, n)
		for i := range int(n) {
			pos[i] = positions[2*i+axis]
		}
		axisTable := mlx.Squeeze(table.Slice(mlx.Slice(axis, axis+1), mlx.Slice(), mlx.Slice()), 0)
		h = mlx.Add(h, mlx.Take(axisTable, mlx.FromValues(pos, int(n)), 0).AsType(h.DType()))
	}
	h = mlx.Reshape(h, 1, n, v.HiddenSize)

	cosX, sinX := visionRopeTables(positions, 0, n, v.HeadDim, v.RopeTheta)
	cosY, sinY := visionRopeTables(positions, 1, n, v.HeadDim, v.RopeTheta)

	for _, l := range m.VisionTower.Layers {
		x1 := mlx.RMSNormFn(h, l.InputNorm, v.RMSNormEps)
		attn := l.Attention.Forward(x1, cosX, sinX, cosY, sinY, n, v)
		h = mlx.Add(h, mlx.RMSNormFn(attn, l.PostAttnNorm, v.RMSNormEps))

		x2 := mlx.RMSNormFn(h, l.PreFFNorm, v.RMSNormEps)
		h = mlx.Add(h, mlx.RMSNormFn(l.MLP.Forward(x2), l.PostFFNorm, v.RMSNormEps))
	}

	// Pool 3x3 over the patch grid, scale by sqrt(hidden), and standardize —
	// in f32 with the reference's dtype round-trip (the sqrt scaling can
	// exceed fp16 range, and the reference keeps this section in f32).
	pool := v.PoolingKernelSize
	f := mlx.Reshape(h.AsType(mlx.DTypeFloat32), geom.PatchesH/pool, pool, geom.PatchesW/pool, pool, v.HiddenSize)
	f = mlx.Mean(f, 3, false)
	f = mlx.Mean(f, 1, false)
	f = mlx.Reshape(f, n/(pool*pool), v.HiddenSize)
	f = f.AsType(mlx.DTypeBFloat16).AsType(mlx.DTypeFloat32)
	f = mlx.MulScalar(f, float32(math.Sqrt(float64(v.HiddenSize))))
	if m.VisionTower.StdBias != nil {
		f = mlx.Mul(mlx.Sub(f, m.VisionTower.StdBias.AsType(mlx.DTypeFloat32)), m.VisionTower.StdScale.AsType(mlx.DTypeFloat32))
	}
	f = f.AsType(mlx.DTypeBFloat16)

	f = mlx.RMSNormFn(f, nil, v.RMSNormEps)
	return m.EmbedVision.Projection.Forward(f)
}

func validateVisionSoftTokenBudget(budget int32) error {
	if !slices.Contains(visionSoftTokenBudgets, budget) {
		return fmt.Errorf("unsupported vision soft token budget %d", budget)
	}
	return nil
}

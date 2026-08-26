package qwen3_5

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
)

// VisionConfig holds the qwen3.5 vision tower configuration.
type VisionConfig struct {
	Depth                 int32 `json:"depth"`
	HiddenSize            int32 `json:"hidden_size"`
	NumHeads              int32 `json:"num_heads"`
	InChannels            int32 `json:"in_channels"`
	PatchSize             int32 `json:"patch_size"`
	SpatialMergeSize      int32 `json:"spatial_merge_size"`
	TemporalPatchSize     int32 `json:"temporal_patch_size"`
	NumPositionEmbeddings int32 `json:"num_position_embeddings"`

	// DeepstackVisualIndexes is rejected at load when non-empty: the
	// family ships it empty, and serving it would need per-layer feature
	// injection the tower does not implement.
	DeepstackVisualIndexes []int32 `json:"deepstack_visual_indexes"`
}

// multimodalConfig carries the top-level fields the text_config unwrap
// discards.
type multimodalConfig struct {
	VisionConfig       *VisionConfig `json:"vision_config"`
	ImageTokenID       int32         `json:"image_token_id"`
	VisionStartTokenID int32         `json:"vision_start_token_id"`
	VisionEndTokenID   int32         `json:"vision_end_token_id"`
}

func parseMultimodalConfig(configData []byte) (multimodalConfig, error) {
	var mm multimodalConfig
	if err := json.Unmarshal(configData, &mm); err != nil {
		return multimodalConfig{}, fmt.Errorf("parse multimodal config: %w", err)
	}
	if v := mm.VisionConfig; v != nil {
		if len(v.DeepstackVisualIndexes) > 0 {
			return multimodalConfig{}, fmt.Errorf("unsupported deepstack_visual_indexes %v", v.DeepstackVisualIndexes)
		}
		if v.PatchSize == 0 {
			v.PatchSize = 16
		}
		if v.SpatialMergeSize == 0 {
			v.SpatialMergeSize = 2
		}
		if v.TemporalPatchSize == 0 {
			v.TemporalPatchSize = 2
		}
		if v.NumHeads == 0 {
			v.NumHeads = 16
		}
	}
	return mm, nil
}

// VisionTower is the qwen3.5 image encoder: a bidirectional transformer over
// patch embeddings with interpolated learned positions and 2D rotary
// attention, merged 2x2 into LM-space tokens.
type VisionTower struct {
	PatchEmbed nn.LinearLayer
	PosEmbed   *mlx.Array // [num_position_embeddings, hidden]
	Blocks     []*VisionBlock
	MergerNorm *nn.LayerNorm
	MergerFC1  nn.LinearLayer
	MergerFC2  nn.LinearLayer
}

type VisionBlock struct {
	Norm1 *nn.LayerNorm
	Norm2 *nn.LayerNorm
	QKV   nn.LinearLayer
	Proj  nn.LinearLayer
	FC1   nn.LinearLayer
	FC2   nn.LinearLayer
}

// visionLayout is the request's Batch.Layout value: the prompt's 3-channel
// rope positions and the delta that continues them past the prompt.
type visionLayout struct {
	positions []int32 // flattened [3][promptLen]
	promptLen int32
	delta     int32
}

// VisionAdapter exposes the shared Qwen3.5/Qwen 4 vision tower without
// coupling another text architecture to this package's decoder internals.
// The publisher uses the same tensor layout, image preprocessing, and MRoPE
// layout for both families.
type VisionAdapter struct {
	// Model is exported so mlx.Collect traverses and pins every tower weight.
	// An unexported wrapper field is invisible to the reflection collector.
	Model *Model
}

type VisionAdapterConfig struct {
	HiddenSize   int32
	RopeDim      int32
	RopeTheta    float32
	MropeSection []int32
	QuantGroup   int
	QuantBits    int
	QuantMode    string
	TensorQuant  map[string]*model.TensorQuantInfo
}

// NewVisionAdapter returns nil when the config has no vision tower.
func NewVisionAdapter(configData []byte, tensors map[string]*mlx.Array, cfg VisionAdapterConfig) (*VisionAdapter, error) {
	mm, err := parseMultimodalConfig(configData)
	if err != nil {
		return nil, err
	}
	if mm.VisionConfig == nil {
		return nil, nil
	}
	text := &Config{
		HiddenSize:     cfg.HiddenSize,
		RopeDim:        cfg.RopeDim,
		RopeTheta:      cfg.RopeTheta,
		MropeSection:   cfg.MropeSection,
		QuantGroupSize: cfg.QuantGroup,
		QuantBits:      cfg.QuantBits,
		QuantMode:      cfg.QuantMode,
		TensorQuant:    cfg.TensorQuant,
	}
	m := &Model{Config: text, MM: mm, Vision: mm.VisionConfig}
	linears := model.NewLinearFactory(tensors, cfg.QuantGroup, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)
	if err := m.loadVisionWeights(tensors, linears); err != nil {
		return nil, err
	}
	return &VisionAdapter{Model: m}, nil
}

func (a *VisionAdapter) PrepareMedia(segments []base.Segment) (*base.PreparedRequest, error) {
	return a.Model.PrepareMedia(segments)
}

func (a *VisionAdapter) EncodeMedia(item *base.PreparedItem, data *mlx.Array) *mlx.Array {
	return a.Model.EncodeMedia(item, data)
}

func (a *VisionAdapter) ScatterMedia(hidden *mlx.Array, b *batch.Batch, delta int) *mlx.Array {
	return a.Model.scatterMedia(hidden, b, delta)
}

func (a *VisionAdapter) RopePositions(b *batch.Batch, length int32) []int32 {
	return ropePositions(b, length)
}

// visionWeightRoot locates the tower's tensor prefix: the import pipeline
// renames the reference's model.visual to vision_tower.
func visionWeightRoot(tensors map[string]*mlx.Array) string {
	for _, root := range []string{"vision_tower.", "model.visual.", "visual."} {
		if tensors[root+"patch_embed.proj.bias"] != nil {
			return root
		}
	}
	return ""
}

func (m *Model) loadVisionWeights(tensors map[string]*mlx.Array, linears model.LinearFactory) error {
	root := visionWeightRoot(tensors)
	v := m.Vision
	if root == "" {
		if v != nil {
			return fmt.Errorf("config declares a vision tower but vision weights are missing from the manifest")
		}
		return nil
	}
	if v == nil {
		return fmt.Errorf("vision tower tensors present but config has no vision_config")
	}

	required := func(name string) (*mlx.Array, error) {
		w := tensors[root+name]
		if w == nil {
			return nil, fmt.Errorf("missing vision weight: %s%s", root, name)
		}
		return w, nil
	}
	requiredLinear := func(name string) (nn.LinearLayer, error) {
		l := linears.Make(root + name)
		if l == nil {
			return nil, fmt.Errorf("missing vision weight: %s%s", root, name)
		}
		return l, nil
	}
	layerNorm := func(name string) (*nn.LayerNorm, error) {
		w, err := required(name + ".weight")
		if err != nil {
			return nil, err
		}
		b, err := required(name + ".bias")
		if err != nil {
			return nil, err
		}
		return &nn.LayerNorm{Weight: w, Bias: b, Eps: 1e-6}, nil
	}

	tower := &VisionTower{Blocks: make([]*VisionBlock, v.Depth)}

	// The reference patch embed is a Conv3d whose stride equals its kernel:
	// a linear over the flattened patch row.
	convW, err := required("patch_embed.proj.weight")
	if err != nil {
		return err
	}
	convB, err := required("patch_embed.proj.bias")
	if err != nil {
		return err
	}
	rowLen := v.InChannels * v.TemporalPatchSize * v.PatchSize * v.PatchSize
	tower.PatchEmbed = nn.NewLinear(mlx.Reshape(convW, v.HiddenSize, rowLen), convB)

	if tower.PosEmbed, err = required("pos_embed.weight"); err != nil {
		return err
	}

	for i := range tower.Blocks {
		bp := fmt.Sprintf("blocks.%d.", i)
		blk := &VisionBlock{}
		if blk.Norm1, err = layerNorm(bp + "norm1"); err != nil {
			return err
		}
		if blk.Norm2, err = layerNorm(bp + "norm2"); err != nil {
			return err
		}
		if blk.QKV, err = requiredLinear(bp + "attn.qkv"); err != nil {
			return err
		}
		if blk.Proj, err = requiredLinear(bp + "attn.proj"); err != nil {
			return err
		}
		if blk.FC1, err = requiredLinear(bp + "mlp.linear_fc1"); err != nil {
			return err
		}
		if blk.FC2, err = requiredLinear(bp + "mlp.linear_fc2"); err != nil {
			return err
		}
		tower.Blocks[i] = blk
	}

	if tower.MergerNorm, err = layerNorm("merger.norm"); err != nil {
		return err
	}
	if tower.MergerFC1, err = requiredLinear("merger.linear_fc1"); err != nil {
		return err
	}
	if tower.MergerFC2, err = requiredLinear("merger.linear_fc2"); err != nil {
		return err
	}

	m.VisionTower = tower
	return nil
}

func (m *Model) visionLoaded() bool { return m.VisionTower != nil }

// PrepareMedia implements base.MediaModel: preprocess each image segment,
// splice its vision_start + pads + vision_end expansion, and precompute the
// request's 3-channel rope positions.
func (m *Model) PrepareMedia(segments []base.Segment) (*base.PreparedRequest, error) {
	prepared := &base.PreparedRequest{}

	// pos walks the reference's multimodal position rule: text advances one
	// per token on all channels; an image spans a merged grid anchored at
	// the current position and advances it by the grid's longer side.
	var positions []int32
	cur := int32(0)
	maxPos := int32(-1)
	text := func(n int) {
		for range n {
			positions = append(positions, cur, cur, cur)
			maxPos = max(maxPos, cur)
			cur++
		}
	}

	for s, seg := range segments {
		if seg.Data == nil {
			prepared.Tokens = append(prepared.Tokens, seg.Tokens...)
			text(len(seg.Tokens))
			continue
		}
		if !m.visionLoaded() {
			return nil, fmt.Errorf("this model does not support %s input", seg.Kind)
		}
		if seg.Kind != "image" {
			return nil, fmt.Errorf("qwen3.5 does not support %s input", seg.Kind)
		}

		pixels, prep, err := m.preprocessImage(seg.Data)
		if err != nil {
			return nil, err
		}

		start := len(prepared.Tokens)
		prepared.Tokens = append(prepared.Tokens, m.MM.VisionStartTokenID)
		text(1)
		merge := m.Vision.SpatialMergeSize
		mh, mw := prep.gridH/merge, prep.gridW/merge
		for bh := range mh {
			for bw := range mw {
				prepared.Tokens = append(prepared.Tokens, m.MM.ImageTokenID)
				positions = append(positions, cur, cur+bh, cur+bw)
				maxPos = max(maxPos, max(cur+bh, cur+bw))
			}
		}
		cur += max(mh, mw)
		prepared.Tokens = append(prepared.Tokens, m.MM.VisionEndTokenID)
		text(1)

		n := int(prep.numPatches())
		prepared.Items = append(prepared.Items, base.PreparedItem{
			Range:     [2]int{start, len(prepared.Tokens)},
			Source:    s,
			MediaData: pixels,
			Dims:      []int{n, len(pixels) / n},
			Opaque:    prep,
			Causal:    true,
		})
	}

	if len(prepared.Items) > 0 {
		// positions accumulated as (t,h,w) triples per token; the layout
		// stores three channel rows.
		L := int32(len(prepared.Tokens))
		table := make([]int32, 3*L)
		for i := range L {
			table[i] = positions[3*i]
			table[L+i] = positions[3*i+1]
			table[2*L+i] = positions[3*i+2]
		}
		prepared.Layout = &visionLayout{positions: table, promptLen: L, delta: maxPos + 1 - L}
	}
	return prepared, nil
}

// EncodeMedia implements base.MediaModel: run the tower over one whole
// image, returning the lazy [numTokens, hidden] features.
func (m *Model) EncodeMedia(item *base.PreparedItem, data *mlx.Array) *mlx.Array {
	prep := item.Opaque.(preparedImage)
	v := m.Vision
	t := m.VisionTower
	n := prep.numPatches()

	h := t.PatchEmbed.Forward(data.AsType(t.PosEmbed.DType()))

	// Bilinear position interpolation: four corner rows summed with their
	// weights, in fp32 like the reference.
	pos := mlx.Zeros(mlx.DTypeFloat32, int(n), int(v.HiddenSize))
	for c := range 4 {
		idx := make([]int32, n)
		wt := make([]float32, n)
		for i := range int(n) {
			idx[i] = prep.posIdx[4*i+c]
			wt[i] = prep.posWt[4*i+c]
		}
		rows := mlx.Take(t.PosEmbed, mlx.FromValues(idx, int(n)), 0).AsType(mlx.DTypeFloat32)
		pos = mlx.Add(pos, mlx.Mul(rows, mlx.FromValues(wt, int(n), 1)))
	}
	h = mlx.Add(h, pos.AsType(h.DType()))
	h = mlx.Reshape(h, 1, n, v.HiddenSize)

	headDim := v.HiddenSize / v.NumHeads
	cos, sin := visionRopeTables(prep.ropePos, n, headDim)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	for _, blk := range t.Blocks {
		x := blk.Norm1.Forward(h)
		qkv := mlx.Reshape(blk.QKV.Forward(x), 1, n, 3, v.NumHeads, headDim)
		split := func(i int32) *mlx.Array {
			s := mlx.SliceStartStop(qkv, []int32{0, 0, i, 0, 0}, []int32{1, n, i + 1, v.NumHeads, headDim})
			return mlx.Transpose(mlx.Squeeze(s, 2), 0, 2, 1, 3)
		}
		q, k, val := split(0), split(1), split(2)
		q = applyVisionRoPE(q, cos, sin)
		k = applyVisionRoPE(k, cos, sin)

		// Fully bidirectional over the single image.
		out := mlx.FastScaledDotProductAttention(q, k, val, scale, "", nil)
		out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), 1, n, v.HiddenSize)
		h = mlx.Add(h, blk.Proj.Forward(out))

		x = blk.Norm2.Forward(h)
		h = mlx.Add(h, blk.FC2.Forward(mlx.GELUApprox(blk.FC1.Forward(x))))
	}

	// Block-major token order makes the merge reshape a plain grouping of
	// consecutive rows; the activation is the exact erf GELU in fp32, like
	// the reference.
	merged := t.MergerNorm.Forward(mlx.Squeeze(h, 0))
	unit := v.SpatialMergeSize * v.SpatialMergeSize
	merged = mlx.Reshape(merged, n/unit, v.HiddenSize*unit)
	f := t.MergerFC1.Forward(merged).AsType(mlx.DTypeFloat32)
	f = mlx.Mul(mlx.MulScalar(f, 0.5),
		mlx.AddScalar(mlx.Erf(mlx.MulScalar(f, float32(1/math.Sqrt2))), 1))
	out := t.MergerFC2.Forward(f.AsType(merged.DType()))
	return out
}

// visionRopeTables builds the tower's rotary tables: 2D positions, half the
// head dim per axis, in fp32.
func visionRopeTables(ropePos []int32, n, headDim int32) (cos, sin *mlx.Array) {
	half := int(headDim) / 2
	nFreq := half / 2
	invFreq := make([]float32, nFreq)
	for i := range invFreq {
		invFreq[i] = float32(1 / math.Pow(10000, float64(2*i)/float64(half)))
	}

	// Per token: h-axis freqs then w-axis freqs, duplicated across the two
	// rotation halves.
	emb := make([]float32, int(n)*int(headDim))
	for i := range n {
		hp, wp := float32(ropePos[2*i]), float32(ropePos[2*i+1])
		row := emb[int(i)*int(headDim):]
		for j := range nFreq {
			row[j] = hp * invFreq[j]
			row[nFreq+j] = wp * invFreq[j]
		}
		copy(row[half:], row[:half])
	}
	e := mlx.FromValues(emb, 1, int(n), 1, int(headDim))
	return mlx.Cos(e), mlx.Sin(e)
}

// applyVisionRoPE applies full-dim rotate-half rotary in fp32, matching the
// reference's float application.
func applyVisionRoPE(t, cos, sin *mlx.Array) *mlx.Array {
	dt := t.DType()
	tf := t.AsType(mlx.DTypeFloat32)
	dims := tf.Dims()
	headDim := int32(dims[3])
	half := headDim / 2
	t1 := tf.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, int(half)))
	t2 := tf.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(int(half), int(headDim)))
	rot := mlx.Concatenate([]*mlx.Array{mlx.Neg(t2), t1}, -1)
	// cos/sin are [1, n, 1, headDim]; broadcast over heads on axis 1 needs
	// the transpose-matched layout [1, 1, n, headDim].
	c := mlx.Transpose(cos, 0, 2, 1, 3)
	s := mlx.Transpose(sin, 0, 2, 1, 3)
	return mlx.Add(mlx.Mul(tf, c), mlx.Mul(rot, s)).AsType(dt)
}

// softRun returns an item's image-token range: the expansion is
// vision_start + pads + vision_end.
func softRun(item batch.MediaItem, merge int32) (start, end int) {
	prep := item.Opaque.(preparedImage)
	return item.Pos + 1, item.Pos + 1 + int(prep.numTokens(merge))
}

// scatterMedia overwrites the image-token rows this chunk covers with the
// item's encoded features. delta shifts each row's offset to the sequence
// position of batch column 0.
func (m *Model) scatterMedia(h *mlx.Array, b *batch.Batch, delta int) *mlx.Array {
	merge := m.Vision.SpatialMergeSize
	for _, item := range b.Media {
		if item.Features == nil {
			continue
		}
		start, end := softRun(item, merge)
		base := int(b.SeqOffsets[item.Seq]) + delta
		qLo := max(start, base)
		qHi := min(end, base+int(b.SeqQueryLens[item.Seq]))
		if qHi <= qLo {
			continue
		}
		feat := item.Features.Slice(mlx.Slice(qLo-start, qHi-start), mlx.Slice())
		feat = mlx.Reshape(feat.AsType(h.DType()), 1, int32(qHi-qLo), m.HiddenSize)
		h = h.SliceUpdate(feat, mlx.Slice(item.Seq, item.Seq+1), mlx.Slice(qLo-base, qHi-base), mlx.Slice())
	}
	return h
}

// ropePositions returns each row's 3-channel positions, flattened [B][3][L],
// from its layout: table rows inside the prompt, the uniform text
// continuation past it. Rows without a layout advance uniformly on all
// channels. Nil when no row carries a layout (plain 1D positions).
func ropePositions(b *batch.Batch, L int32) []int32 {
	carried := false
	for _, l := range b.Layout {
		carried = carried || l != nil
	}
	if !carried {
		return nil
	}
	out := make([]int32, 3*L*int32(len(b.SeqOffsets)))
	for r, base := range b.SeqOffsets {
		row := out[r*3*int(L):]
		var layout *visionLayout
		if b.Layout[r] != nil {
			layout = b.Layout[r].(*visionLayout)
		}
		for i := range L {
			p := base + i
			switch {
			case layout == nil:
				row[i], row[L+i], row[2*L+i] = p, p, p
			case p < layout.promptLen:
				row[i] = layout.positions[p]
				row[L+i] = layout.positions[layout.promptLen+p]
				row[2*L+i] = layout.positions[2*layout.promptLen+p]
			default:
				row[i], row[L+i], row[2*L+i] = p+layout.delta, p+layout.delta, p+layout.delta
			}
		}
	}
	return out
}

// mropeCosSin builds the interleaved 3-channel rotary tables for one chunk:
// each channel's frequencies laid out round-robin T,H,W across the rotary
// half-dims, matching the reference's interleaved M-RoPE. positions is
// ropePositions' per-row [3][L] blocks.
func mropeCosSin(cfg *Config, positions []int32, L int32) (cos, sin *mlx.Array) {
	half := int(cfg.RopeDim) / 2
	invFreq := make([]float32, half)
	for i := range invFreq {
		invFreq[i] = float32(1 / math.Pow(float64(cfg.RopeTheta), float64(2*i)/float64(cfg.RopeDim)))
	}

	// Interleave: start from T, then overwrite H at 1,4,7,... and W at
	// 2,5,8,..., each bounded by its section's 3x length like the reference.
	sel := make([]int32, half)
	for c := int32(1); c <= 2; c++ {
		limit := min(int(cfg.MropeSection[c])*3, half)
		for j := int(c); j < limit; j += 3 {
			sel[j] = c
		}
	}

	B := len(positions) / (3 * int(L))
	emb := make([]float32, B*int(L)*int(cfg.RopeDim))
	for r := range B {
		rowPos := positions[r*3*int(L):]
		for i := range int(L) {
			row := emb[(r*int(L)+i)*int(cfg.RopeDim):]
			for j, c := range sel {
				row[j] = float32(rowPos[int(c)*int(L)+i]) * invFreq[j]
			}
			copy(row[half:], row[:half])
		}
	}
	e := mlx.FromValues(emb, B, 1, int(L), int(cfg.RopeDim))
	return mlx.Cos(e), mlx.Sin(e)
}

// applyMRoPE rotates the leading RopeDim dims with the chunk's tables,
// passing the rest through, in the tensor's own dtype like the reference.
func applyMRoPE(t, cos, sin *mlx.Array, ropeDim int32) *mlx.Array {
	dims := t.Dims()
	headDim := int32(dims[3])
	rot := t.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, int(ropeDim)))
	half := ropeDim / 2
	r1 := rot.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, int(half)))
	r2 := rot.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(int(half), int(ropeDim)))
	rotated := mlx.Concatenate([]*mlx.Array{mlx.Neg(r2), r1}, -1)
	c := cos.AsType(t.DType())
	s := sin.AsType(t.DType())
	out := mlx.Add(mlx.Mul(rot, c), mlx.Mul(rotated, s))
	if ropeDim == headDim {
		return out
	}
	pass := t.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(int(ropeDim), int(headDim)))
	return mlx.Concatenate([]*mlx.Array{out, pass}, -1)
}

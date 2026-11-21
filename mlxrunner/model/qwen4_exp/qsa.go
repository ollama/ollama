package qwen4_exp

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type fullAttention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer
	QNorm *nn.RMSNorm
	KNorm *nn.RMSNorm

	Indexer *attentionIndexer
}

type attentionIndexer struct {
	QKProj nn.LinearLayer
	QNorm  *nn.RMSNorm
	KNorm  *nn.RMSNorm
}

func loadFullAttention(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string, eps float32) (*fullAttention, error) {
	a := &fullAttention{
		QProj: linears.Make(prefix + ".q_proj"),
		KProj: linears.Make(prefix + ".k_proj"),
		VProj: linears.Make(prefix + ".v_proj"),
		OProj: linears.Make(prefix + ".o_proj"),
	}
	var err error
	if a.QProj == nil || a.KProj == nil || a.VProj == nil || a.OProj == nil {
		return nil, fmt.Errorf("missing attention projection at %s", prefix)
	}
	if a.QNorm, err = loadNorm(tensors, prefix+".q_norm", eps); err != nil {
		return nil, err
	}
	if a.KNorm, err = loadNorm(tensors, prefix+".k_norm", eps); err != nil {
		return nil, err
	}

	idx := &attentionIndexer{}
	if idx.QKProj, err = requiredLinear(linears, prefix+".indexer.index_qk_proj"); err != nil {
		return nil, err
	}
	if idx.QNorm, err = loadNorm(tensors, prefix+".indexer.q_layernorm", eps); err != nil {
		return nil, err
	}
	if idx.KNorm, err = loadNorm(tensors, prefix+".indexer.k_layernorm", eps); err != nil {
		return nil, err
	}
	a.Indexer = idx
	return a, nil
}

func (a *fullAttention) Forward(x *mlx.Array, b *batch.Batch, c, side cache.Layer, ropePositions *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	qg := mlx.Reshape(a.QProj.Forward(x), B, L, cfg.NumAttentionHeads, cfg.HeadDim*2)
	q := qg.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, int(cfg.HeadDim)))
	gate := qg.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(int(cfg.HeadDim), mlx.End))
	gate = mlx.Reshape(gate, B, L, cfg.NumAttentionHeads*cfg.HeadDim)

	k := mlx.Reshape(a.KProj.Forward(x), B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v := mlx.Reshape(a.VProj.Forward(x), B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	q = a.QNorm.Forward(q, cfg.RMSNormEps)
	k = a.KNorm.Forward(k, cfg.RMSNormEps)
	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)
	cos, sin := qsaRopeCosSin(cfg, ropePositions)
	q = applyQwenRoPE(q, cos, sin, cfg.RopeDim)
	k = applyQwenRoPE(k, cos, sin, cfg.RopeDim)

	indexQ, rawIndexK := a.Indexer.Forward(x, cos, sin, cfg)

	var mainHistory, indexHistory *nn.KVHistory
	if c != nil {
		mainHistory = c.(cache.Attention).Update(b, k, v)
		if side == nil {
			panic("QSA side cache is missing")
		}
		indexHistory = side.(cache.Attention).Update(b, rawIndexK, ropePositions)
	}

	var out *mlx.Array
	if mainHistory == nil {
		out = nn.ScaledDotProductAttention(b, q, cfg.Scale,
			nn.WithKV(k, v, b.SeqQueryLens), nn.WithMask(nn.CausalMask()))
	} else {
		out = a.attend(b, q, indexQ, mainHistory, indexHistory, cfg)
	}
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	out = mlx.Mul(out, mlx.Sigmoid(gate))
	return a.OProj.Forward(out)
}

// attend uses each request's own sparse-selection budget and history length.
// The projections remain batched; attention does not pad or copy KV histories.
func (a *fullAttention) attend(b *batch.Batch, q, indexQ *mlx.Array, main, index *nn.KVHistory, cfg *Config) *mlx.Array {
	if rows := main.Rows(); rows != nil {
		outputs := make([]*mlx.Array, len(rows))
		for i, row := range rows {
			outputs[i] = a.attend(b.Row(i),
				q.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice(), mlx.Slice()),
				indexQ.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice(), mlx.Slice()),
				row, index.Rows()[i], cfg)
		}
		return mlx.Concatenate(outputs, 0)
	}
	keyLength := int32(q.Dim(2))
	for _, offset := range b.SeqOffsets {
		keyLength = max(keyLength, offset+int32(q.Dim(2)))
	}
	if keyLength <= cfg.IndexerBudget {
		return nn.ScaledDotProductAttention(b, q, cfg.Scale, nn.WithKVHistory(main), nn.WithMask(nn.CausalMask()))
	}
	compressed := qsaCompressedKeys(index.K(), index.V(), a.Indexer, cfg)
	scores := mlx.Matmul(indexQ.AsType(mlx.DTypeFloat32), mlx.Transpose(compressed.AsType(mlx.DTypeFloat32), 0, 1, 3, 2))
	scores = mlx.DivScalar(mlx.Sum(mlx.ReLU(scores), 1, false), float32(math.Sqrt(float64(cfg.IndexerHeadDim))))
	blocks, queryEnds := qsaLogicalBlocks(scores, b, keyLength, cfg)
	return qsaSparseAttention(q, main, blocks, queryEnds, cfg)
}

func (a *attentionIndexer) Forward(x, cos, sin *mlx.Array, cfg *Config) (*mlx.Array, *mlx.Array) {
	B, L := int32(x.Dim(0)), int32(x.Dim(1))
	projected := a.QKProj.Forward(x)
	split := int(cfg.IndexerNumHeads * cfg.IndexerHeadDim)
	q := projected.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, split))
	k := projected.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(split, int((cfg.IndexerNumHeads+cfg.IndexerKVHeads)*cfg.IndexerHeadDim)))
	q = mlx.Reshape(q, B, L, cfg.IndexerNumHeads, cfg.IndexerHeadDim)
	q = a.QNorm.Forward(q, cfg.RMSNormEps)
	q = applyQwenRoPE(mlx.Transpose(q, 0, 2, 1, 3), cos, sin, cfg.RopeDim)
	// Keys must stay raw in the cache: compression averages in FP32 before
	// normalization and applies RoPE at each complete block's first position.
	k = mlx.Transpose(mlx.Reshape(k, B, L, cfg.IndexerKVHeads, cfg.IndexerHeadDim), 0, 2, 1, 3)
	return q, k
}

// canonicalRopePositionRows returns exact [B, 1, L, 3] position rows. MRoPE
// input is the channel-major [B][3][L] layout used by the vision path.
func canonicalRopePositionRows(b *batch.Batch, L int32, mrope []int32) *mlx.Array {
	B := len(b.SeqOffsets)
	rows := make([]int32, B*int(L)*3)
	for r, base := range b.SeqOffsets {
		for i := range int(L) {
			row := (r*int(L) + i) * 3
			if mrope == nil {
				p := base + int32(i)
				rows[row], rows[row+1], rows[row+2] = p, p, p
				continue
			}
			channel := mrope[r*3*int(L):]
			rows[row] = channel[i]
			rows[row+1] = channel[int(L)+i]
			rows[row+2] = channel[2*int(L)+i]
		}
	}
	return mlx.FromValues(rows, B, 1, int(L), 3)
}

// qsaRopeCosSin builds the main attention's exact 1D/MRoPE tables from
// arbitrary cached positions. positions is [B, 1, L, 3].
func qsaRopeCosSin(cfg *Config, positions *mlx.Array) (cos, sin *mlx.Array) {
	half := int(cfg.RopeDim) / 2
	selector := make([]int32, half)
	sections := cfg.RopeParameters.MRoPESection
	if len(sections) == 3 && cfg.RopeParameters.MRoPEInterleaved {
		for channel := int32(1); channel <= 2; channel++ {
			limit := min(int(sections[channel])*3, half)
			for i := int(channel); i < limit; i += 3 {
				selector[i] = channel
			}
		}
	}

	invFreq := make([]float32, half)
	for i := range invFreq {
		invFreq[i] = float32(1 / math.Pow(float64(cfg.RopeParameters.RopeTheta), float64(2*i)/float64(cfg.RopeDim)))
	}
	selected := mlx.Take(positions, mlx.FromValues(selector, half), 3).AsType(mlx.DTypeFloat32)
	angles := mlx.Mul(selected, mlx.FromValues(invFreq, 1, 1, 1, half))
	embedding := mlx.Concatenate([]*mlx.Array{angles, angles}, -1)
	return mlx.Cos(embedding), mlx.Sin(embedding)
}

func applyQwenRoPE(x, cos, sin *mlx.Array, ropeDim int32) *mlx.Array {
	headDim := int32(x.Dim(3))
	rot := x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, int(ropeDim)))
	half := int(ropeDim / 2)
	left := rot.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, half))
	right := rot.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(half, mlx.End))
	rotated := mlx.Concatenate([]*mlx.Array{mlx.Neg(right), left}, -1)
	out := mlx.Add(
		mlx.Mul(rot, cos.AsType(x.DType())),
		mlx.Mul(rotated, sin.AsType(x.DType())),
	)
	if ropeDim == headDim {
		return out
	}
	pass := x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(int(ropeDim), mlx.End))
	return mlx.Concatenate([]*mlx.Array{out, pass}, -1)
}

func qsaCompressedKeys(rawKeys, positions *mlx.Array, indexer *attentionIndexer, cfg *Config) *mlx.Array {
	ratio := cfg.IndexerCompressRatio
	groups := int32(rawKeys.Dim(2)) / ratio
	complete := groups * ratio
	rawKeys = rawKeys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, int(complete)), mlx.Slice())
	rawKeys = mlx.Reshape(rawKeys, int32(rawKeys.Dim(0)), int32(rawKeys.Dim(1)), groups, ratio, cfg.IndexerHeadDim)
	pooled := mlx.Mean(rawKeys.AsType(mlx.DTypeFloat32), 3, false).AsType(rawKeys.DType())
	pooled = indexer.KNorm.Forward(pooled, cfg.RMSNormEps)

	positions = positions.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, int(complete)), mlx.Slice())
	positions = mlx.Reshape(positions, int32(positions.Dim(0)), int32(positions.Dim(1)), groups, ratio, 3)
	positions = mlx.Squeeze(positions.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0), mlx.Slice()), 3)
	cos, sin := qsaRopeCosSin(cfg, positions)
	return applyQwenRoPE(pooled, cos, sin, cfg.RopeDim)
}

// qsaLogicalBlocks reproduces the reference's compressed-block top-k.
// Blocks outside a query's causal history are represented by -1; queryEnds
// lets attention append the incomplete causal block without expanding the
// selected blocks into per-token indices.
func qsaLogicalBlocks(scores *mlx.Array, b *batch.Batch, keyLength int32, cfg *Config) (selected, queryEnds *mlx.Array) {
	B, L := int32(scores.Dim(0)), int32(scores.Dim(1))
	ratio := cfg.IndexerCompressRatio
	blocks := keyLength / ratio
	blockIDs := mlx.Reshape(mlx.Arange(0, float64(blocks), 1, mlx.DTypeInt32), 1, 1, blocks)

	visibleValues := make([]int32, int(B*L))
	for r, offset := range b.SeqOffsets {
		for i := range int(L) {
			visibleValues[r*int(L)+i] = offset + int32(i) + 1
		}
	}
	visibleTokens := mlx.FromValues(visibleValues, int(B), int(L), 1)
	visibleBlocks := mlx.FloorDivideScalar(visibleTokens, ratio)
	blockValid := blockIDs.Less(visibleBlocks)

	selectedCount := min(blocks, cfg.IndexerBudget/ratio)
	if blocks > selectedCount {
		fill := mlx.AddScalar(mlx.Zeros(scores.DType(), int(B), int(L), int(blocks)), -float32(math.MaxFloat32))
		masked := mlx.Where(blockValid, scores, fill)
		partitioned := mlx.Argpartition(mlx.Neg(masked), int(selectedCount)-1, -1)
		selected = partitioned.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, int(selectedCount)))
	} else {
		selected = mlx.Tile(blockIDs, []int32{B, L, 1})
	}
	blockValid = selected.Less(visibleBlocks)
	invalid := mlx.AddScalar(mlx.Zeros(mlx.DTypeInt32, int(B), int(L), int(selectedCount)), -1)
	selected = mlx.Where(blockValid, selected, invalid).AsType(mlx.DTypeInt32)
	return selected, mlx.Reshape(visibleTokens, B, L).AsType(mlx.DTypeInt32)
}

func qsaSparseAttention(q *mlx.Array, history *nn.KVHistory, blocks, queryEnds *mlx.Array, cfg *Config) *mlx.Array {
	return mlx.IndexedBlockScaledDotProductAttention(
		q, history.K(), history.V(), blocks, queryEnds, int(cfg.IndexerCompressRatio), cfg.Scale,
	)
}

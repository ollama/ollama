package qwen4_exp

import (
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/nn"
)

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
	rot := mlx.SliceStartStop(x,
		[]int32{0, 0, 0, 0},
		[]int32{int32(x.Dim(0)), int32(x.Dim(1)), int32(x.Dim(2)), ropeDim},
	)
	half := ropeDim / 2
	left := mlx.SliceStartStop(rot,
		[]int32{0, 0, 0, 0},
		[]int32{int32(rot.Dim(0)), int32(rot.Dim(1)), int32(rot.Dim(2)), half},
	)
	right := mlx.SliceStartStop(rot,
		[]int32{0, 0, 0, half},
		[]int32{int32(rot.Dim(0)), int32(rot.Dim(1)), int32(rot.Dim(2)), ropeDim},
	)
	rotated := mlx.Concatenate([]*mlx.Array{mlx.Neg(right), left}, -1)
	out := mlx.Add(
		mlx.Mul(rot, cos.AsType(x.DType())),
		mlx.Mul(rotated, sin.AsType(x.DType())),
	)
	if ropeDim == headDim {
		return out
	}
	pass := mlx.SliceStartStop(x,
		[]int32{0, 0, 0, ropeDim},
		[]int32{int32(x.Dim(0)), int32(x.Dim(1)), int32(x.Dim(2)), headDim},
	)
	return mlx.Concatenate([]*mlx.Array{out, pass}, -1)
}

func qsaCompressedKeys(rawKeys, positions *mlx.Array, indexer *attentionIndexer, cfg *Config) *mlx.Array {
	ratio := cfg.IndexerCompressRatio
	groups := int32(rawKeys.Dim(2)) / ratio
	complete := groups * ratio
	rawKeys = mlx.SliceStartStop(rawKeys,
		[]int32{0, 0, 0, 0},
		[]int32{int32(rawKeys.Dim(0)), int32(rawKeys.Dim(1)), complete, int32(rawKeys.Dim(3))},
	)
	rawKeys = mlx.Reshape(rawKeys, int32(rawKeys.Dim(0)), int32(rawKeys.Dim(1)), groups, ratio, cfg.IndexerHeadDim)
	pooled := mlx.Mean(rawKeys.AsType(mlx.DTypeFloat32), 3, false).AsType(rawKeys.DType())
	pooled = indexer.KNorm.Forward(pooled, cfg.RMSNormEps)

	positions = mlx.SliceStartStop(positions,
		[]int32{0, 0, 0, 0},
		[]int32{int32(positions.Dim(0)), int32(positions.Dim(1)), complete, int32(positions.Dim(3))},
	)
	positions = mlx.Reshape(positions, int32(positions.Dim(0)), int32(positions.Dim(1)), groups, ratio, 3)
	positions = mlx.Squeeze(mlx.SliceStartStop(positions,
		[]int32{0, 0, 0, 0, 0},
		[]int32{int32(positions.Dim(0)), int32(positions.Dim(1)), groups, 1, 3},
	), 3)
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
		selected = mlx.SliceStartStop(partitioned,
			[]int32{0, 0, 0},
			[]int32{B, L, selectedCount},
		)
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

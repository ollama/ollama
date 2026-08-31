package qwen4_exp

import (
	"math"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
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

// qsaLogicalIndices reproduces the reference's compressed-block top-k and
// incomplete-group tail. Invalid rows are kept as zero indices with a mask.
func qsaLogicalIndices(scores *mlx.Array, b *batch.Batch, keyLength int32, cfg *Config) (indices, valid *mlx.Array) {
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
		selected := mlx.Argpartition(mlx.Neg(masked), int(selectedCount)-1, -1)
		indices = mlx.SliceStartStop(selected,
			[]int32{0, 0, 0},
			[]int32{B, L, selectedCount},
		)
	} else {
		indices = mlx.Tile(blockIDs, []int32{B, L, 1})
	}
	blockValid = indices.Less(visibleBlocks)

	offsets := mlx.Reshape(mlx.Arange(0, float64(ratio), 1, mlx.DTypeInt32), 1, 1, 1, ratio)
	expanded := mlx.Add(mlx.MulScalar(mlx.ExpandDims(indices, -1), float32(ratio)), offsets)
	expandedValid := mlx.Mul(
		mlx.ExpandDims(blockValid.AsType(mlx.DTypeInt32), -1),
		expanded.Less(mlx.ExpandDims(visibleTokens, -1)).AsType(mlx.DTypeInt32),
	)
	expanded = mlx.Reshape(expanded, B, L, selectedCount*ratio)
	expandedValid = mlx.Reshape(expandedValid, B, L, selectedCount*ratio)

	// The current partial compression group is always attended causally.
	tailWidth := ratio - 1
	tailOffsets := mlx.Reshape(mlx.Arange(0, float64(tailWidth), 1, mlx.DTypeInt32), 1, 1, tailWidth)
	tailStart := mlx.MulScalar(mlx.FloorDivideScalar(visibleTokens, ratio), float32(ratio))
	tail := mlx.Add(tailStart, tailOffsets)
	tailCount := mlx.Sub(visibleTokens, tailStart)
	tailValid := tailOffsets.Less(tailCount).AsType(mlx.DTypeInt32)

	indices = mlx.Concatenate([]*mlx.Array{expanded, tail}, -1)
	validInt := mlx.Concatenate([]*mlx.Array{expandedValid, tailValid}, -1)
	valid = validInt.Greater(mlx.Zeros(mlx.DTypeInt32, int(B), int(L), validInt.Dim(2)))
	indices = mlx.Where(valid, indices, mlx.Zeros(mlx.DTypeInt32, int(B), int(L), indices.Dim(2)))
	return indices, valid
}

func qsaSparseAttention(q *mlx.Array, history *nn.KVHistory, indices, valid *mlx.Array, cfg *Config) *mlx.Array {
	outputType := q.DType()
	B, queryHeads, L, D := int32(q.Dim(0)), int32(q.Dim(1)), int32(q.Dim(2)), int32(q.Dim(3))
	kvHeads := cfg.NumKeyValueHeads
	repeats := queryHeads / kvHeads

	k := qsaGatherHistory(history.K(), indices)
	v := qsaGatherHistory(history.V(), indices)

	qr := mlx.Reshape(q, B, kvHeads, repeats, L, 1, D)
	kr := mlx.Transpose(mlx.ExpandDims(k, 2), 0, 1, 2, 3, 5, 4)
	scores := mlx.Squeeze(mlx.Matmul(qr.AsType(mlx.DTypeFloat32), kr.AsType(mlx.DTypeFloat32)), 4)
	scores = mlx.MulScalar(scores, cfg.Scale)
	mask := mlx.ExpandDims(mlx.ExpandDims(valid, 1), 1)
	fill := mlx.AddScalar(mlx.Zeros(scores.DType(), scores.Dims()...), -float32(math.MaxFloat32))
	scores = mlx.Where(mask, scores, fill)
	probs := mlx.SoftmaxAxis(scores, -1, true)

	vr := mlx.ExpandDims(v.AsType(mlx.DTypeFloat32), 2)
	out := mlx.Matmul(mlx.ExpandDims(probs, 4), vr)
	out = mlx.Squeeze(out, 4)
	return mlx.Reshape(out, B, queryHeads, L, D).AsType(outputType)
}

// qsaGatherHistory selects each batch row's logical token indices from a
// [B, H, K, D] cache history and returns [B, H, L, S, D].
func qsaGatherHistory(history, indices *mlx.Array) *mlx.Array {
	B, H, K, D := int32(history.Dim(0)), int32(history.Dim(1)), int32(history.Dim(2)), int32(history.Dim(3))
	offsets := make([]int32, B)
	for i := range offsets {
		offsets[i] = int32(i) * K
	}
	logical := mlx.Add(indices, mlx.FromValues(offsets, int(B), 1, 1))
	flattened := mlx.Reshape(mlx.Transpose(history, 1, 0, 2, 3), H, B*K, D)
	return mlx.Transpose(mlx.Take(flattened, logical, 1), 1, 0, 2, 3, 4)
}

package hy4_flash

import (
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type indexer struct {
	Query, HeadWeight, TokenKey, PoolValue, PoolGate nn.LinearLayer
	TokenNorm, GroupNorm                             *nn.RMSNorm
	SlotBias                                         *mlx.Array
}

func loadIndexer(w weights, prefix string) *indexer {
	// The group RMS weight loads as BF16 while the pool projections stay FP32.
	groupNorm := w.norm(prefix + ".key_norm")
	groupNorm.Weight = groupNorm.Weight.AsType(mlx.DTypeBFloat16)
	return &indexer{
		Query: w.linear(prefix + ".query_projection"), HeadWeight: w.linear(prefix + ".head_weight_projection"),
		TokenKey: w.linear(prefix + ".token_key_projection"), TokenNorm: w.norm(prefix + ".token_key_norm"),
		PoolValue: w.linear(prefix + ".key_value_projection"), PoolGate: w.linear(prefix + ".key_gate_projection"),
		GroupNorm: groupNorm, SlotBias: w.tensor(prefix + ".intra_group_ape"),
	}
}

type indexHistory struct {
	tokens int
	groups *mlx.Array
}

func (i *indexer) histories(x *mlx.Array, b *batch.Batch, slot cache.Attention, shape attentionShape, c *textConfig) []indexHistory {
	keys := i.TokenNorm.Forward(i.TokenKey.Forward(x), 0).ExpandDims(1)
	B, L, ratio := x.Dim(0), x.Dim(1), c.IndexRatio
	result := make([]indexHistory, B)
	if slot == nil {
		for row := range result {
			result[row].tokens = L
			if L > ratio*c.IndexTopK {
				raw := keys.Slice(mlx.Slice(row, row+1), mlx.Slice(), mlx.Slice(0, L/ratio*ratio), mlx.Slice())
				result[row].groups = i.pool(raw, b.Row(row), shape, c, 0)
			}
		}
		return result
	}

	// Store each pooled key at the token that completed its group. Both
	// channels use token offsets, so the regular KV cache can snapshot or
	// rewind across a partial group without preserving any future result.
	var previous *nn.KVHistory
	for _, offset := range b.SeqOffsets {
		if rem := int(offset) % ratio; rem != 0 && rem+L >= ratio {
			previous = slot.View(b)
			break
		}
	}
	completed := make([]*mlx.Array, B)
	for row, offset := range b.SeqOffsets {
		start := int(offset)
		first := start / ratio * ratio
		end := (start + L) / ratio * ratio
		values := mlx.Zeros(keys.DType(), 1, 1, L, c.IndexDim)
		if end > start {
			raw := keys.Slice(mlx.Slice(row, row+1), mlx.Slice(), mlx.Slice(0, end-start), mlx.Slice())
			if first < start {
				old := previous
				if rows := old.Rows(); rows != nil {
					old = rows[row]
				}
				head := old.K().Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(first, start), mlx.Slice())
				raw = mlx.Concatenate([]*mlx.Array{head, raw}, 2)
			}
			pooled := i.pool(raw, b.Row(row), shape, c, first)
			values = values.SliceUpdate(pooled, mlx.Slice(), mlx.Slice(), mlx.Slice(first+ratio-1-start, L, ratio), mlx.Slice())
		}
		completed[row] = values
	}
	values := completed[0]
	if B > 1 {
		values = mlx.Concatenate(completed, 0)
	}
	history := slot.Update(b, keys, values)
	for row := range result {
		h := history
		if rows := history.Rows(); rows != nil {
			h = rows[row]
		}
		n := h.V().Dim(2)
		result[row].tokens = n
		if n > ratio*c.IndexTopK {
			result[row].groups = h.V().Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(ratio-1, n, ratio), mlx.Slice())
		}
	}
	return result
}

func (i *indexer) pool(keys *mlx.Array, b *batch.Batch, shape attentionShape, c *textConfig, start int) *mlx.Array {
	groups := keys.Dim(2) / c.IndexRatio
	grouped := keys.Reshape(groups, c.IndexRatio, c.IndexDim).AsType(mlx.DTypeFloat32)
	gates := mlx.Add(i.PoolGate.Forward(grouped), i.SlotBias)
	pooled := mlx.Sum(mlx.Mul(i.PoolValue.Forward(grouped), mlx.SoftmaxAxis(gates, 1, true)), 1, false)
	pooled = i.GroupNorm.Forward(pooled, 0).AsType(keys.DType()).Reshape(1, 1, groups, c.IndexDim)
	return rotateTail(pooled, rotaryPositions{batch: b, groupStart: start, stride: c.IndexRatio}, shape)
}

// selectTokens recalls complete groups and adds the query's own group.
func (i *indexer) selectTokens(x, q *mlx.Array, history indexHistory, b *batch.Batch, shape attentionShape, c *textConfig) *mlx.Array {
	if history.groups == nil {
		return nil
	}
	L, groups := x.Dim(1), history.groups.Dim(2)
	queries := i.Query.Forward(q).Reshape(1, L, c.IndexHeads, c.IndexDim).Transpose(0, 2, 1, 3)
	queryPos := mlx.Arange(float64(b.SeqOffsets[0]), float64(int(b.SeqOffsets[0])+L), 1, mlx.DTypeInt32)
	queries = rotateTail(queries, rotaryPositions{batch: b}, shape)
	dots := mlx.Matmul(queries.AsType(mlx.DTypeFloat32), history.groups.AsType(mlx.DTypeFloat32).Transpose(0, 1, 3, 2))
	headScale := float32(1 / math.Sqrt(float64(c.IndexHeads*c.IndexDim)))
	weights := mlx.MulScalar(i.HeadWeight.Forward(x).AsType(mlx.DTypeFloat32), headScale).Transpose(0, 2, 1).ExpandDims(-1)
	scores := mlx.Sum(mlx.Mul(mlx.ReLU(dots), weights), 1, false)

	current := queryPos.FloorDivide(mlx.FromValue(c.IndexRatio)).Reshape(1, L, 1)
	groupIDs := mlx.Arange(0, float64(groups), 1, mlx.DTypeInt32)
	negInf := mlx.FromValue(float32(math.Inf(-1)))
	scores = mlx.Where(groupIDs.Less(current), scores, negInf)
	recalled := mlx.Concatenate([]*mlx.Array{mlx.TopKIndices(scores, c.IndexTopK-1), current}, -1)
	offset := mlx.Arange(0, float64(c.IndexRatio), 1, mlx.DTypeInt32)
	return mlx.Add(mlx.Mul(recalled.ExpandDims(-1), mlx.FromValue(c.IndexRatio)), offset).Reshape(1, L, c.IndexTopK*c.IndexRatio)
}

func selectionMask(tokens *mlx.Array, n int) nn.AttentionMask {
	if tokens == nil {
		return nn.AttentionMask{}
	}
	L := tokens.Dim(1)
	tokens = mlx.Minimum(tokens, mlx.FromValue(n-1))
	mask := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, L, n), float32(math.Inf(-1)))
	mask = mask.PutAlongAxis(tokens, mlx.Zeros(mlx.DTypeFloat32, 1, L, tokens.Dim(2)), -1)
	return nn.ArrayMask(mask.ExpandDims(1))
}

func rotateTail(x *mlx.Array, positions rotaryPositions, shape attentionShape) *mlx.Array {
	width := shape.rotary
	end := x.Dim(-1)
	plain := x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, end-width))
	rotary := x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(end-width, end))
	rotary = positions.apply(rotary, shape)
	return mlx.Concatenate([]*mlx.Array{plain, rotary}, -1)
}

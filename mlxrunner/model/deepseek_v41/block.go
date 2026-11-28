package deepseek_v41

import (
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func (l *transformerLayer) forward(streams, preMix, positions *mlx.Array, b *batch.Batch, state cache.Layer, engramRows []int64, sparseRows []*sparseState, imageMask *mlx.Array) (*mlx.Array, *mlx.Array) {
	if l.Engram != nil {
		var err error
		var textMask *mlx.Array
		if imageMask != nil {
			textMask = mlx.Where(imageMask, mlx.FromValue(false), mlx.FromValue(true))
		}
		streams, err = l.Engram.forward(streams, engramRows, textMask)
		if err != nil {
			panic(err)
		}
	}

	attnPre, attnPost, attnResidual := l.AttentionMix.coefficients(streams)
	input := l.AttentionNorm.Forward(collapseConnection(streams, preMix), 0)
	attended := l.attend(input, positions, b, state, sparseRows)
	streams = injectConnection(attended, streams, attnPost, attnResidual)

	ffnPre, ffnPost, ffnResidual := l.FFNMix.coefficients(streams)
	input = l.FFNNorm.Forward(collapseConnection(streams, attnPre), 0)
	result := l.FFN.forward(input, imageMask)
	return injectConnection(result, streams, ffnPost, ffnResidual), ffnPre
}

// Layer projections and the feed-forward pass run across all rows. Attention
// uses each row's cache length, compression boundary, and sparse candidates.
func (l *transformerLayer) attend(input, positions *mlx.Array, b *batch.Batch, state cache.Layer, sparse []*sparseState) *mlx.Array {
	query, kv, queryLow := l.Attention.project(input, positions)
	row := func(a *mlx.Array, i int) *mlx.Array {
		if a.Dim(0) == 1 {
			return a
		}
		return a.TakeAxis(mlx.FromValues([]int32{int32(i)}, 1), 0)
	}
	var previous []*nn.KVHistory
	if state != nil && l.Attention.Compressor != nil {
		view := state.(cache.Attention).View(b)
		previous = view.Rows()
		if previous == nil {
			previous = []*nn.KVHistory{view}
		}
	}
	keys := kv
	if c := l.Attention.Compressor; c != nil {
		keyRows := make([]*mlx.Array, input.Dim(0))
		for i := range keyRows {
			var prior *mlx.Array
			if state != nil && b.SeqOffsets[i] > 0 {
				prior = previous[i].K()
			}
			pooled, unrotated, values, scores := c.encode(row(input, i), prior, int(b.SeqOffsets[i]), l.Attention.Frequencies)
			channels := []*mlx.Array{row(kv, i).AsType(mlx.DTypeFloat32), mlx.ExpandDims(pooled, 2), mlx.ExpandDims(values, 2), mlx.ExpandDims(scores, 2)}
			if idx := l.Attention.Indexer; idx != nil && idx.KeyNorm != nil {
				index := idx.encodeKeys(unrotated, int(b.SeqOffsets[i]), c.Ratio, l.Attention.Frequencies)
				if padding := kv.Dim(3) - index.Dim(2); padding > 0 {
					index = mlx.Concatenate([]*mlx.Array{index, mlx.Zeros(mlx.DTypeFloat32, index.Dim(0), index.Dim(1), padding)}, -1)
				}
				channels = append(channels, mlx.ExpandDims(index, 2))
			}
			keyRows[i] = mlx.Concatenate(channels, 2)
		}
		if len(keyRows) == 1 {
			keys = keyRows[0]
		} else {
			keys = mlx.Concatenate(keyRows, 0)
		}
	}
	var histories []*nn.KVHistory
	if state != nil {
		stored := mlx.Transpose(keys, 0, 2, 1, 3)
		history := state.(cache.Attention).Update(b, stored, stored)
		histories = history.Rows()
		if histories == nil {
			histories = []*nn.KVHistory{history}
		}
	}
	outputs := make([]*mlx.Array, input.Dim(0))
	for i := range outputs {
		keyHistory := row(kv, i)
		keyStart := int(b.SeqOffsets[i])
		var source *mlx.Array
		if state != nil {
			source = histories[i].K()
			raw := source.Slice(mlx.Slice(), mlx.Slice(0, 1), mlx.Slice(), mlx.Slice())
			keyHistory = mlx.Transpose(raw, 0, 2, 1, 3)
			keyStart = 0
		} else {
			source = mlx.Transpose(row(keys, i), 0, 2, 1, 3)
		}
		if c := l.Attention.Compressor; c != nil {
			sparse[i].compressed, sparse[i].ratio = sourceKeys(source, 1, c.Ratio), c.Ratio
			sparse[i].indexKeys = sourceKeys(source, 4, c.Ratio)
			if sparse[i].indexKeys != nil {
				sparse[i].indexKeys = sparse[i].indexKeys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, l.Attention.Indexer.Width))
			}
		}
		if idx := l.Attention.Indexer; idx != nil && sparse[i].indexKeys != nil {
			var candidates *mlx.Array
			sparse[i].selection, candidates = idx.selectKeys(row(input, i), row(queryLow, i), sparse[i].indexKeys, sparse[i].candidates, int(b.SeqOffsets[i]), sparse[i].ratio, l.Attention.Frequencies)
			if idx.CandidateSource {
				sparse[i].candidates = candidates
			}
		}
		outputs[i] = l.Attention.finish(row(query, i), keyHistory, sparse[i].compressed, sparse[i].selection, row(positions, i), int(b.SeqOffsets[i]), keyStart, sparse[i].ratio)
	}
	if len(outputs) == 1 {
		return outputs[0]
	}
	return mlx.Concatenate(outputs, 0)
}

type connectionMix struct {
	Projection  *mlx.Array
	Base        *mlx.Array
	Scale       *mlx.Array
	Copies      int32
	Iterations  int
	NormEpsilon float32
	MixEpsilon  float32
}

// coefficients computes the next branch's input mix, this branch's output
// mix, and the doubly normalized residual mixing matrix. The statistic spans
// all copies and channels for one token.
func (h connectionMix) coefficients(streams *mlx.Array) (pre, post, residual *mlx.Array) {
	dims := streams.Dims()
	flat := mlx.Reshape(streams.AsType(mlx.DTypeFloat32), int32(dims[0]), int32(dims[1]), h.Copies*int32(dims[3]))
	variance := mlx.Mean(mlx.Mul(flat, flat), -1, true)
	projected := mlx.Matmul(flat, mlx.Transpose(h.Projection, 1, 0))
	return mlx.ConnectionMix(projected, variance, h.Scale, h.Base, int(h.Copies), h.Iterations, h.NormEpsilon, h.MixEpsilon)
}

func collapseConnection(streams, pre *mlx.Array) *mlx.Array {
	return mlx.Sum(mlx.Mul(streams.AsType(mlx.DTypeFloat32), mlx.ExpandDims(pre, -1)), -2, false).AsType(streams.DType())
}

func injectConnection(branch, streams, post, residual *mlx.Array) *mlx.Array {
	mixed := mlx.Sum(mlx.Mul(mlx.ExpandDims(residual, -1), mlx.ExpandDims(streams.AsType(mlx.DTypeFloat32), -2)), -3, false)
	injected := mlx.Mul(mlx.ExpandDims(post, -1), mlx.ExpandDims(branch.AsType(mlx.DTypeFloat32), -2))
	return mlx.Add(mixed, injected).AsType(streams.DType())
}

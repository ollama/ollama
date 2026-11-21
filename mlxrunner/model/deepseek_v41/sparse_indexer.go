package deepseek_v41

import (
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type sparseIndexer struct {
	Query              nn.LinearLayer
	Weights            nn.LinearLayer
	Key                nn.LinearLayer
	KeyNorm            *nn.RMSNorm
	Heads              int
	Width              int
	TopK               int
	CandidateSource    bool
	UsesCandidates     bool
	CandidateBlocks    int
	CandidateBlockSize int
}

// sparseState carries the current KV source and index decisions through the
// decoder's layer loop. It is rebuilt for each forward; cache history remains
// owned by the source layer's regular KV cache.
type sparseState struct {
	compressed *mlx.Array
	indexKeys  *mlx.Array
	selection  *mlx.Array
	candidates *mlx.Array
	ratio      int
}

// encodeKeys places each completed group's index key at its final raw-token
// position. The same source cache therefore owns raw KV, compressed KV, and
// index keys with identical snapshot offsets.
func (s *sparseIndexer) encodeKeys(latent *mlx.Array, start, ratio int, frequencies []float32) *mlx.Array {
	key := s.KeyNorm.Forward(s.Key.Forward(latent), 0)
	batch, length := key.Dim(0), key.Dim(1)
	positions := make([]int32, length)
	for i := range positions {
		if (start+i+1)%ratio == 0 {
			positions[i] = int32(start + i + 1 - ratio)
		}
	}
	rotated := mlx.RotateTail(mlx.ExpandDims(key, 2), mlx.FromValues(positions, batch, length), frequencies, false)
	return mlx.Squeeze(rotated, 2).AsType(mlx.DTypeFloat32)
}

// indexScoreBytes bounds each FP32 score matrix across batch rows, attention
// heads, queries, and compressed keys. Each query's selection is independent.
const indexScoreBytes = 384 << 20

// selectKeys ranks compressed positions once more history exists than the
// published top-k budget. Before then every completed position is selected.
// Candidate-block pruning precedes this ranking for index sources after the
// candidate source layer.
func (s *sparseIndexer) selectKeys(x, queryLow, keys, candidates *mlx.Array, start, ratio int, frequencies []float32) (selection, candidateMask *mlx.Array) {
	rows := max(1, indexScoreBytes/(4*max(x.Dim(0), 1)*s.Heads*max(keys.Dim(1), 1)))
	return s.selectKeysInRows(x, queryLow, keys, candidates, start, ratio, frequencies, rows)
}

func (s *sparseIndexer) selectKeysInRows(x, queryLow, keys, candidates *mlx.Array, start, ratio int, frequencies []float32, rows int) (selection, candidateMask *mlx.Array) {
	count := keys.Dim(1)
	if count <= s.TopK {
		return nil, nil
	}
	batch, length := x.Dim(0), x.Dim(1)
	query := s.Query.Forward(queryLow)
	query = mlx.Reshape(query, int32(batch), int32(length), int32(s.Heads), int32(s.Width))
	positions := make([]int32, length)
	for i := range positions {
		positions[i] = int32(start + i)
	}
	query = mlx.RotateTail(query, mlx.FromValues(positions, batch, length), frequencies, false)
	queries := mlx.Transpose(query.AsType(mlx.DTypeFloat32), 0, 2, 1, 3)
	keyMatrix := mlx.ExpandDims(mlx.Transpose(keys.AsType(mlx.DTypeFloat32), 0, 2, 1), 1)
	weights := s.Weights.Forward(x).AsType(mlx.DTypeFloat32)
	weights = mlx.MulScalar(weights, 1/float32(math.Sqrt(float64(s.Width*s.Heads))))
	var selections, masks []*mlx.Array
	for first := 0; first < length; first += rows {
		n := min(rows, length-first)
		block := mlx.Slice(first, first+n)
		var blockCandidates *mlx.Array
		if candidates != nil {
			blockCandidates = candidates.Slice(mlx.Slice(), block, mlx.Slice())
		}
		sel, mask := s.selectRows(queries.Slice(mlx.Slice(), mlx.Slice(), block, mlx.Slice()), keyMatrix,
			weights.Slice(mlx.Slice(), block, mlx.Slice()), blockCandidates, start+first, n, count, ratio)
		selections = append(selections, sel)
		if mask != nil {
			masks = append(masks, mask)
		}
	}
	selection = selections[0]
	if len(selections) > 1 {
		selection = mlx.Concatenate(selections, 1)
	}
	if len(masks) > 0 {
		candidateMask = masks[0]
		if len(masks) > 1 {
			candidateMask = mlx.Concatenate(masks, 1)
		}
	}
	return selection, candidateMask
}

func (s *sparseIndexer) selectRows(queries, keyMatrix, weights, candidates *mlx.Array, start, length, count, ratio int) (selection, candidateMask *mlx.Array) {
	scores := mlx.Transpose(mlx.Matmul(queries, keyMatrix), 0, 2, 1, 3)
	scores = mlx.Maximum(scores, mlx.FromValue(float32(0)))
	scores = mlx.Sum(mlx.Mul(scores, mlx.ExpandDims(weights, -1)), 2, false)

	visible := make([]bool, length*count)
	limits := make([]int32, length)
	for token := range length {
		limit := (start + token + 1) / ratio
		limits[token] = int32(limit)
		for key := range count {
			visible[token*count+key] = key < limit
		}
	}
	mask := mlx.FromValues(visible, 1, length, count)
	scores = mlx.Where(mask, scores, mlx.FromValue(float32(-math.MaxFloat32)))
	if s.CandidateSource {
		candidateMask = selectCandidateBlocks(scores, limits, s.CandidateBlocks, s.CandidateBlockSize)
	} else if s.UsesCandidates && candidates != nil {
		scores = mlx.Where(candidates, scores, mlx.FromValue(float32(-math.MaxFloat32)))
	}
	indices := mlx.Argpartition(mlx.Neg(scores), s.TopK-1, -1)
	indices = indices.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, s.TopK))
	indices = mlx.TakeAlongAxis(indices, mlx.Argsort(indices, -1), -1).AsType(mlx.DTypeInt32)
	limit := mlx.FromValues(limits, 1, length, 1)
	return mlx.Where(indices.Less(limit), indices, mlx.FromValue(-1)), candidateMask
}

// selectCandidateBlocks keeps the strongest compressed-key blocks for each
// query. The newest causal block is pinned because its partial score can be
// lower than a completed older block. A nil result means every block fits.
func selectCandidateBlocks(scores *mlx.Array, limits []int32, budget, blockSize int) *mlx.Array {
	batch, length, width := scores.Dim(0), scores.Dim(1), scores.Dim(2)
	blocks := (width + blockSize - 1) / blockSize
	if blocks <= budget {
		return nil
	}
	pad := blocks*blockSize - width
	padded := scores
	if pad != 0 {
		padded = mlx.Pad(scores, []int{2}, []int{0}, []int{pad}, mlx.FromValue(float32(-math.MaxFloat32)), "constant")
	}
	blockScores := mlx.Reshape(padded, int32(batch), int32(length), int32(blocks), int32(blockSize)).MaxAxis(-1, false)
	ids := make([]int32, blocks)
	for i := range ids {
		ids[i] = int32(i)
	}
	latest := make([]int32, length)
	for i, limit := range limits {
		latest[i] = (limit - 1) / int32(blockSize)
		if limit == 0 {
			latest[i] = -1
		}
	}
	pinned := mlx.FromValues(ids, 1, 1, blocks).Equal(mlx.FromValues(latest, 1, length, 1))
	blockScores = mlx.Where(pinned, mlx.FromValue(float32(math.MaxFloat32)), blockScores)
	top := mlx.Argpartition(mlx.Neg(blockScores), budget-1, -1)
	top = top.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, budget))
	valid := mlx.TakeAlongAxis(blockScores, top, -1).Greater(mlx.FromValue(float32(-math.MaxFloat32)))
	kept := mlx.Zeros(mlx.DTypeBool, batch, length, blocks).PutAlongAxis(top, valid, -1)
	keyBlocks := make([]int32, width)
	for i := range keyBlocks {
		keyBlocks[i] = int32(i / blockSize)
	}
	indices := mlx.BroadcastTo(mlx.FromValues(keyBlocks, 1, 1, width), int32(batch), int32(length), int32(width))
	return mlx.TakeAlongAxis(kept, indices, 2)
}

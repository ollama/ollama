package deepseek_v41

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type latentAttention struct {
	QueryLow    nn.LinearLayer
	QueryHigh   nn.LinearLayer
	KeyValue    nn.LinearLayer
	OutputLow   *groupedOutput
	OutputHigh  nn.LinearLayer
	QueryNorm   *nn.RMSNorm
	KeyNorm     *nn.RMSNorm
	Sink        *mlx.Array
	Frequencies []float32
	Heads       int32
	HeadWidth   int32
	Window      int
	Compressor  *compressor
	Indexer     *sparseIndexer
}

func (a *latentAttention) project(x, positions *mlx.Array) (query, keyValue, queryLow *mlx.Array) {
	batch, length := int32(x.Dim(0)), int32(x.Dim(1))
	queryLow = a.QueryNorm.Forward(a.QueryLow.Forward(x), 0)
	query = mlx.Reshape(a.QueryHigh.Forward(queryLow), batch, length, a.Heads, a.HeadWidth)
	keyValue = a.KeyNorm.Forward(a.KeyValue.Forward(x), 0)
	keyValue = mlx.Reshape(keyValue, batch, length, 1, a.HeadWidth)
	return mlx.RotateTail(query, positions, a.Frequencies, false), mlx.RotateTail(keyValue, positions, a.Frequencies, false), queryLow
}

func (a *latentAttention) finish(query, keyHistory, compressed, selected, positions *mlx.Array, queryStart, keyStart, ratio int) *mlx.Array {
	keys := mlx.Squeeze(keyHistory, 2)
	out := attendWindow(query, keys, compressed, selected, a.Sink, queryStart, keyStart, a.Window, ratio, 1/float32(math.Sqrt(float64(a.HeadWidth))))
	out = mlx.RotateTail(out, positions, a.Frequencies, true)
	out = a.OutputLow.forward(out)
	return a.OutputHigh.Forward(out)
}

func rotaryFrequencies(cfg *Config, compressed bool) []float32 {
	dim := cfg.QKRoPEHeadDim
	base := cfg.RoPETheta
	if compressed {
		base = cfg.CompressRoPETheta
	}
	freq := make([]float32, dim/2)
	for i := range freq {
		frequency := 1 / math.Pow(base, float64(2*i)/float64(dim))
		if compressed {
			scaling := cfg.RoPEScaling
			corrected := func(rotations float64) float64 {
				return float64(dim) * math.Log(float64(scaling.OriginalMaxPositionEmbeddings)/(rotations*2*math.Pi)) / (2 * math.Log(base))
			}
			low := max(math.Floor(corrected(scaling.BetaFast)), 0)
			high := min(math.Ceil(corrected(scaling.BetaSlow)), float64(dim-1))
			ramp := min(max((float64(i)-low)/max(high-low, 1e-3), 0), 1)
			frequency *= (1 - ramp) + ramp/scaling.Factor
		}
		freq[i] = float32(frequency)
	}
	return freq
}

// attendWindow includes each head's learned sink in the softmax denominator.
// Queries and keys have absolute positions, while the supplied raw KV begins
// at keyStart. Compressed key j becomes visible after raw token (j+1)*ratio-1.
func attendWindow(q, kv, compressed, selection, sink *mlx.Array, queryStart, keyStart, window, ratio int, scale float32) *mlx.Array {
	if compressed != nil && selection != nil {
		first := max(queryStart-window+1, keyStart)
		last := min(queryStart+q.Dim(1), keyStart+kv.Dim(1))
		if first < last {
			raw := kv.Slice(mlx.Slice(), mlx.Slice(first-keyStart, last-keyStart), mlx.Slice())
			if out, ok := mlx.SparseWindowAttention(q, raw, compressed, selection, sink, queryStart, first, window, scale); ok {
				return out
			}
		}
	}
	return attendWindowGraph(q, kv, compressed, selection, sink, queryStart, keyStart, window, ratio, scale)
}

func attendWindowGraph(q, kv, compressed, selection, sink *mlx.Array, queryStart, keyStart, window, ratio int, scale float32) *mlx.Array {
	batch, length, heads, width := q.Dim(0), q.Dim(1), q.Dim(2), q.Dim(3)
	keyCount := kv.Dim(1)
	indices := make([]int32, length*window)
	valid := make([]bool, length*window)
	for token := range length {
		queryPosition := queryStart + token
		begin := max(queryPosition-window+1, keyStart)
		end := min(queryPosition+1, keyStart+keyCount)
		for slot := range window {
			position := begin + slot
			if position < end {
				indices[token*window+slot] = int32(position - keyStart)
				valid[token*window+slot] = true
			}
		}
	}
	selected := mlx.Take(kv, mlx.FromValues(indices, length, window), 1).AsType(mlx.DTypeFloat32)
	queries := mlx.Reshape(q.AsType(mlx.DTypeFloat32), int32(batch), int32(length), int32(heads), 1, int32(width))
	keys := mlx.Transpose(mlx.ExpandDims(selected, 2), 0, 1, 2, 4, 3)
	scores := mlx.Squeeze(mlx.Matmul(queries, keys), -2)
	scores = mlx.MulScalar(scores, scale)
	mask := mlx.Reshape(mlx.FromValues(valid, length, window), 1, int32(length), 1, int32(window))
	scores = mlx.Where(mask, scores, mlx.FromValue(float32(-math.MaxFloat32)))
	allScores := []*mlx.Array{scores}
	compressedCount := 0
	var chosen *mlx.Array
	if compressed != nil {
		compressedCount = compressed.Dim(1)
		var compressedScores, mask *mlx.Array
		if selection != nil {
			compressedCount = selection.Dim(2)
			safe := mlx.Maximum(selection, mlx.FromValue(0))
			chosen = mlx.Take(compressed, mlx.Squeeze(safe, 0), 1).AsType(mlx.DTypeFloat32)
			compressedKeys := mlx.Transpose(mlx.ExpandDims(chosen, 2), 0, 1, 2, 4, 3)
			compressedScores = mlx.Squeeze(mlx.Matmul(queries, compressedKeys), -2)
			mask = mlx.ExpandDims(selection.Greater(mlx.FromValue(-1)), 2)
		} else {
			allQueries := mlx.Transpose(q.AsType(mlx.DTypeFloat32), 0, 2, 1, 3)
			allKeys := mlx.ExpandDims(mlx.Transpose(compressed.AsType(mlx.DTypeFloat32), 0, 2, 1), 1)
			compressedScores = mlx.Transpose(mlx.Matmul(allQueries, allKeys), 0, 2, 1, 3)
			visible := make([]bool, length*compressedCount)
			for token := range length {
				for key := range compressedCount {
					visible[token*compressedCount+key] = (key+1)*ratio <= queryStart+token+1
				}
			}
			mask = mlx.Reshape(mlx.FromValues(visible, length, compressedCount), 1, int32(length), 1, int32(compressedCount))
		}
		compressedScores = mlx.MulScalar(compressedScores, scale)
		allScores = append(allScores, mlx.Where(mask, compressedScores, mlx.FromValue(float32(-math.MaxFloat32))))
	}
	sinkLogits := mlx.Reshape(sink.AsType(mlx.DTypeFloat32), 1, 1, int32(heads), 1)
	sinkLogits = mlx.BroadcastTo(sinkLogits, int32(batch), int32(length), int32(heads), 1)
	allScores = append(allScores, sinkLogits)
	probabilities := mlx.SoftmaxAxis(mlx.Concatenate(allScores, -1), -1, true)
	rawProbabilities := probabilities.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, window))
	values := mlx.ExpandDims(selected, 2)
	out := mlx.Squeeze(mlx.Matmul(mlx.ExpandDims(rawProbabilities, -2), values), -2)
	if compressedCount > 0 {
		compressedProbabilities := probabilities.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(window, window+compressedCount))
		var compressedOutput *mlx.Array
		if chosen != nil {
			compressedValues := mlx.ExpandDims(chosen, 2)
			compressedOutput = mlx.Squeeze(mlx.Matmul(mlx.ExpandDims(compressedProbabilities, -2), compressedValues), -2)
		} else {
			compressedProbabilities = mlx.Transpose(compressedProbabilities, 0, 2, 1, 3)
			compressedValues := mlx.ExpandDims(compressed.AsType(mlx.DTypeFloat32), 1)
			compressedOutput = mlx.Transpose(mlx.Matmul(compressedProbabilities, compressedValues), 0, 2, 1, 3)
		}
		out = mlx.Add(out, compressedOutput)
	}
	return out.AsType(q.DType())
}

type compressor struct {
	KeyValue nn.LinearLayer
	Gate     nn.LinearLayer
	Norm     *nn.RMSNorm
	Ratio    int
}

// encode writes a compressed key at the raw position that completed its
// group. This keeps the source layer's cache at one offset per input token,
// including across prefix snapshots and single-token decode steps.
func (c *compressor) encode(x, previous *mlx.Array, start int, frequencies []float32) (compressed, unrotated, values, scores *mlx.Array) {
	batch, length, width := x.Dim(0), x.Dim(1), c.KeyValue.OutputDim()
	input := x
	if c.Ratio > 1 {
		input = x.AsType(mlx.DTypeFloat32)
	}
	values = c.KeyValue.Forward(input).AsType(mlx.DTypeFloat32)
	if c.Ratio == 1 {
		scores = mlx.Zeros(mlx.DTypeFloat32, batch, length, int(width))
		positions := make([]int32, length)
		for i := range positions {
			positions[i] = int32(start + i)
		}
		latent := c.Norm.Forward(values.AsType(x.DType()), 0)
		rotated := mlx.Squeeze(mlx.RotateTail(mlx.ExpandDims(latent, 2), mlx.FromValues(positions, batch, length), frequencies, false), 2)
		return rotated.AsType(mlx.DTypeFloat32), latent, values, scores
	}

	scores = c.Gate.Forward(x.AsType(mlx.DTypeFloat32)).AsType(mlx.DTypeFloat32)
	pairedValues, pairedScores := values, scores
	groupStart := start
	if start%c.Ratio != 0 {
		if previous == nil || previous.Dim(2) < start {
			panic(fmt.Sprintf("DeepSeek compressor has no history at token %d", start))
		}
		priorValue := mlx.Squeeze(previous.Slice(mlx.Slice(), mlx.Slice(2, 3), mlx.Slice(start-1, start), mlx.Slice()), 1)
		priorScore := mlx.Squeeze(previous.Slice(mlx.Slice(), mlx.Slice(3, 4), mlx.Slice(start-1, start), mlx.Slice()), 1)
		pairedValues = mlx.Concatenate([]*mlx.Array{priorValue, values}, 1)
		pairedScores = mlx.Concatenate([]*mlx.Array{priorScore, scores}, 1)
		groupStart--
	}
	groups := pairedValues.Dim(1) / c.Ratio
	if groups == 0 {
		zero := mlx.Zeros(mlx.DTypeFloat32, batch, length, int(width))
		return zero, zero.AsType(x.DType()), values, scores
	}
	complete := groups * c.Ratio
	pairedValues = pairedValues.Slice(mlx.Slice(), mlx.Slice(0, complete), mlx.Slice())
	pairedScores = pairedScores.Slice(mlx.Slice(), mlx.Slice(0, complete), mlx.Slice())
	pairedValues = mlx.Reshape(pairedValues, int32(batch), int32(groups), int32(c.Ratio), width)
	pairedScores = mlx.Reshape(pairedScores, int32(batch), int32(groups), int32(c.Ratio), width)
	pooled := mlx.Sum(mlx.Mul(pairedValues, mlx.SoftmaxAxis(pairedScores, 2, true)), 2, false)
	pooled = c.Norm.Forward(pooled.AsType(x.DType()), 0)
	positions := make([]int32, groups)
	for i := range positions {
		positions[i] = int32(groupStart + i*c.Ratio)
	}
	rotated := mlx.Squeeze(mlx.RotateTail(mlx.ExpandDims(pooled, 2), mlx.FromValues(positions, batch, groups), frequencies, false), 2)

	indices := make([]int32, length)
	visible := make([]float32, length)
	for i := range indices {
		position := start + i
		if (position+1)%c.Ratio == 0 {
			indices[i] = int32((position + 1 - c.Ratio - groupStart) / c.Ratio)
			visible[i] = 1
		}
	}
	selected := mlx.Take(rotated, mlx.FromValues(indices, length), 1)
	selectedUnrotated := mlx.Take(pooled, mlx.FromValues(indices, length), 1)
	mask := mlx.FromValues(visible, 1, length, 1)
	return mlx.Mul(selected.AsType(mlx.DTypeFloat32), mask),
		mlx.Mul(selectedUnrotated, mask.AsType(selectedUnrotated.DType())), values, scores
}

func sourceKeys(history *mlx.Array, channel, ratio int) *mlx.Array {
	count := history.Dim(2) / ratio
	if count == 0 {
		return nil
	}
	positions := make([]int32, count)
	for i := range positions {
		positions[i] = int32((i+1)*ratio - 1)
	}
	selected := history.Slice(mlx.Slice(), mlx.Slice(channel, channel+1), mlx.Slice(), mlx.Slice())
	return mlx.Squeeze(mlx.Take(selected, mlx.FromValues(positions, count), 2), 1)
}

// groupedOutput applies the published block-diagonal output projection. One
// quantized bank belongs to each group of attention heads; no cross-group
// weights exist in the checkpoint.
type groupedOutput struct {
	Weight     *mlx.Array
	Scales     *mlx.Array
	Groups     int
	InputWidth int
	Rank       int
}

func newGroupedOutput(weight, scales *mlx.Array, groups, inputWidth, rank int) (*groupedOutput, error) {
	if weight == nil || scales == nil || groups <= 0 || inputWidth%32 != 0 || rank <= 0 ||
		weight.NumDims() != 2 || scales.NumDims() != 2 ||
		weight.DType() != mlx.DTypeUint32 || scales.DType() != mlx.DTypeUint8 ||
		weight.Dim(0) != groups*rank || weight.Dim(1) != inputWidth/4 ||
		scales.Dim(0) != groups*rank || scales.Dim(1) != inputWidth/32 {
		return nil, fmt.Errorf("invalid DeepSeek grouped output projection")
	}
	return &groupedOutput{
		Weight: mlx.Reshape(weight, int32(groups), int32(rank), int32(inputWidth/4)),
		Scales: mlx.Reshape(scales, int32(groups), int32(rank), int32(inputWidth/32)),
		Groups: groups, InputWidth: inputWidth, Rank: rank,
	}, nil
}

func (p *groupedOutput) forward(heads *mlx.Array) *mlx.Array {
	batch, length := heads.Dim(0), heads.Dim(1)
	grouped := mlx.Reshape(heads, int32(batch*length*p.Groups), 1, 1, int32(p.InputWidth))
	indices := make([]uint32, batch*length*p.Groups)
	for i := range indices {
		indices[i] = uint32(i % p.Groups)
	}
	projected := mlx.GatherQMM(grouped, p.Weight, p.Scales, nil, nil,
		mlx.FromValues(indices, len(indices), 1), true, 32, 8, "mxfp8", nil, false)
	return mlx.Reshape(projected, int32(batch), int32(length), int32(p.Groups*p.Rank))
}

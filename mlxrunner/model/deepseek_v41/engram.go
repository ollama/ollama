package deepseek_v41

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type engramLayer struct {
	Table   *hostEngramTable
	Project nn.LinearLayer
	Query   *mlx.Array
	Key     *mlx.Array
	Copies  int32
	Width   int32
	Epsilon float32
}

func (e *engramLayer) forward(streams *mlx.Array, rowIDs []int64, tokenMask *mlx.Array) (*mlx.Array, error) {
	if e.Table == nil || e.Project == nil || e.Query == nil || e.Key == nil {
		return nil, fmt.Errorf("engram layer is incomplete")
	}
	batch, length := streams.Dim(0), streams.Dim(1)
	const columns = 24
	if len(rowIDs) != batch*length*columns {
		return nil, fmt.Errorf("engram has %d row IDs, want %d", len(rowIDs), batch*length*columns)
	}
	rows, err := e.Table.lookup(rowIDs, batch, length, columns)
	if err != nil {
		return nil, err
	}
	projected := e.Project.Forward(mlx.Reshape(rows.AsType(streams.DType()), int32(batch), int32(length), columns*256))
	key := projected.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, int(e.Copies*e.Width)))
	value := projected.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(int(e.Copies*e.Width), mlx.End))
	key = mlx.Reshape(key.AsType(mlx.DTypeFloat32), int32(batch), int32(length), e.Copies, e.Width)
	hidden := streams.AsType(mlx.DTypeFloat32)
	queryScale := mlx.Mul(e.Query.AsType(mlx.DTypeFloat32), e.Key.AsType(mlx.DTypeFloat32))
	normX := mlx.RSqrt(mlx.AddScalar(mlx.Mean(mlx.Mul(hidden, hidden), -1, false), e.Epsilon))
	normK := mlx.RSqrt(mlx.AddScalar(mlx.Mean(mlx.Mul(key, key), -1, false), e.Epsilon))
	dot := mlx.Sum(mlx.Mul(mlx.Mul(hidden, queryScale), key), -1, false)
	dot = mlx.MulScalar(mlx.Mul(dot, mlx.Mul(normX, normK)), 1/float32(math.Sqrt(float64(e.Width))))
	clamped := mlx.Maximum(dot.Abs(), mlx.FromValue(float32(1e-6)))
	gate := mlx.Sigmoid(mlx.Mul(dot.Sign(), clamped.Sqrt()))
	if tokenMask != nil {
		gate = mlx.Where(mlx.ExpandDims(tokenMask, -1), gate, mlx.Zeros(mlx.DTypeFloat32, gate.Dims()...))
	}
	result := mlx.Add(hidden, mlx.Mul(mlx.ExpandDims(gate, -1), mlx.ExpandDims(value.AsType(mlx.DTypeFloat32), -2)))
	return result.AsType(streams.DType()), nil
}

// hashCurrentTokens keeps the raw token history in a regular cache slot, so
// rollback and speculative snapshots use the runner's cache lifecycle. Only
// the previous three IDs cross to the host on each decode step.
func (m *Model) hashCurrentTokens(b *batch.Batch, state cache.Layer) map[int][]int64 {
	batchSize, length := b.InputIDs.Dim(0), b.InputIDs.Dim(1)
	ids := mlx.Contiguous(b.InputIDs.AsType(mlx.DTypeInt32), false).Ints()
	var valid []bool
	if len(b.Media) != 0 {
		positions := imagePositions(b)
		valid = make([]bool, len(positions))
		for i := range valid {
			valid[i] = !positions[i]
		}
	}
	result := make(map[int][]int64, len(m.Config.EngramLayerIDs))
	for _, id := range m.Config.EngramLayerIDs {
		result[id] = make([]int64, 0, len(ids)*(m.Config.EngramMaxNGramSize-1)*m.Config.EngramNumHeads)
	}
	var views []*mlx.Array
	if state != nil {
		view := state.(cache.Attention).View(b)
		views = make([]*mlx.Array, batchSize)
		if rowViews := view.Rows(); rowViews != nil {
			for row, history := range rowViews {
				views[row] = history.K()
			}
		} else {
			views[0] = view.K()
		}
	}
	for row := range batchSize {
		hasher, err := newEngramHasher(m.EngramLayout, m.TokenMap, m.HashMultipliers, m.Config.EngramPadTokenID)
		if err != nil {
			panic(err)
		}
		if state != nil && b.SeqOffsets[row] > 0 {
			previous := views[row]
			if previous == nil || previous.Dim(2) != int(b.SeqOffsets[row]) {
				panic(fmt.Sprintf("engram token cache has %v tokens, want %d", previous, b.SeqOffsets[row]))
			}
			start := max(previous.Dim(2)-len(hasher.history), 0)
			tail := previous.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, previous.Dim(2)), mlx.Slice())
			history := mlx.Contiguous(tail.AsType(mlx.DTypeInt32), false).Ints()
			for i := range history {
				id := history[len(history)-1-i]
				if id < 0 {
					hasher.history[i] = -1
				} else if int(id) < len(m.TokenMap) {
					hasher.history[i] = int64(m.TokenMap[id])
				} else {
					panic(fmt.Sprintf("engram history token %d is out of range", id))
				}
			}
		}
		var rowValid []bool
		if valid != nil {
			rowValid = valid[row*length : (row+1)*length]
		}
		rows, err := hasher.hashChunk(ids[row*length:(row+1)*length], rowValid)
		if err != nil {
			panic(err)
		}
		for _, byLayer := range rows {
			for layer, columnRows := range byLayer {
				id := m.Config.EngramLayerIDs[layer]
				result[id] = append(result[id], columnRows...)
			}
		}
	}
	if state != nil {
		current := b.InputIDs.AsType(mlx.DTypeInt32)
		if valid != nil {
			current = mlx.Where(mlx.FromValues(valid, batchSize, length), current, mlx.FromValue(-1))
		}
		current = mlx.Reshape(current, int32(batchSize), 1, int32(length), 1)
		state.(cache.Attention).Update(b, current, current)
	}
	return result
}

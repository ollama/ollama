package qwen3_5

import (
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// IsolatedLayout lets a state attend all questions while each question
// attends only the state and itself. The state includes its separator token.
// Recurrent layers restart later questions from that same state.
type IsolatedLayout struct {
	ContextLen int32
	Chunks     []int32
}

func (l *IsolatedLayout) tokenPositions(length int32) []int32 {
	positions := make([]int32, length)
	for i := range l.ContextLen {
		positions[i] = i
	}
	start := l.ContextLen
	for _, size := range l.Chunks {
		for i := range size {
			positions[start+i] = l.ContextLen + i
		}
		start += size
	}
	return positions
}

func (l *IsolatedLayout) attentionMask(length int32) nn.AttentionMask {
	groups := make([]int32, length)
	for i := range l.ContextLen {
		groups[i] = -1
	}
	start := l.ContextLen
	for group, size := range l.Chunks {
		for i := range size {
			groups[start+i] = int32(group)
		}
		start += size
	}
	ids := mlx.FromValues(groups, int(length))
	query := ids.Reshape(int(length), 1)
	key := ids.Reshape(1, int(length))
	state := mlx.FromValues([]int32{-1}, 1)
	zero := mlx.FromValues([]float32{0}, 1)
	blocked := mlx.FromValues([]float32{float32(math.Inf(-1))}, 1)
	allowed := mlx.Where(query.Equal(key), zero, blocked)
	allowed = mlx.Where(key.Equal(state), zero, allowed)
	allowed = mlx.Where(query.Equal(state), zero, allowed)
	return nn.ArrayMask(allowed.Reshape(1, 1, int(length), int(length)))
}

// forwardIsolated reproduces the recurrent state at each later question's
// start. The first question already starts immediately after the state.
func (g *GatedDeltaNet) forwardIsolated(x *mlx.Array, b *batch.Batch, cfg *Config, layout *IsolatedLayout) *mlx.Array {
	length := int32(x.Dim(1))
	out := g.Forward(x, b, nil, 1, length, cfg)
	if len(layout.Chunks) < 2 {
		return out
	}
	parts := []*mlx.Array{out.Slice(mlx.Slice(), mlx.Slice(0, int(layout.ContextLen+layout.Chunks[0])), mlx.Slice())}
	start := layout.ContextLen + layout.Chunks[0]
	state := x.Slice(mlx.Slice(), mlx.Slice(0, int(layout.ContextLen)), mlx.Slice())
	for _, size := range layout.Chunks[1:] {
		question := x.Slice(mlx.Slice(), mlx.Slice(int(start), int(start+size)), mlx.Slice())
		restarted := mlx.Concatenate([]*mlx.Array{state, question}, 1)
		n := layout.ContextLen + size
		row := &batch.Batch{SeqOffsets: []int32{0}, SeqQueryLens: []int32{n}}
		output := g.Forward(restarted, row, nil, 1, n, cfg)
		parts = append(parts, output.Slice(mlx.Slice(), mlx.Slice(int(layout.ContextLen), int(n)), mlx.Slice()))
		start += size
	}
	return mlx.Concatenate(parts, 1)
}

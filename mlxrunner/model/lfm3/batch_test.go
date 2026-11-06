package lfm3

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestShortConvBatchedDecode(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		cfg := &Config{HiddenSize: 2, ConvLCache: 3}
		// Binary fractions keep this small fixture exact across convolution
		// implementations, so the comparison tests cache ownership.
		conv := &ShortConv{
			InProj: nn.NewLinear(mlx.FromValues([]float32{
				1, 0, 0, 1, 0.5, 0, 0, 0.5, 1, 0, 0, 1,
			}, 6, 2), nil),
			OutProj: nn.NewLinear(mlx.FromValues([]float32{1, 0, 0, 1}, 2, 2), nil),
			Conv1D:  nn.NewConv1d(mlx.FromValues([]float32{0.25, 0.5, 0.75, -0.25, 0.5, 0.75}, 2, 3, 1), nil, 1, 0, 1, 2),
		}
		var together, alone [2]*cache.RecurrentCache
		for i := range together {
			together[i] = cache.NewRecurrentCache(2, 2, 1, 1, 1)
			alone[i] = cache.NewRecurrentCache(2, 2, 1, 1, 1)
			defer together[i].Free()
			defer alone[i].Free()
			// Seed unequal histories before requests join a shared decode.
			length := 3 + i
			values := make([]float32, 2*length)
			for j := range values {
				values[j] = float32(i+1) + float32(j)/8
			}
			b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, length), SeqOffsets: []int32{0}, SeqQueryLens: []int32{int32(length)}}
			x := mlx.FromValues(values, 1, length, 2)
			mlx.Eval(conv.Forward(x, b, together[i], 1, int32(length), cfg), conv.Forward(x, b, alone[i], 1, int32(length), cfg))
		}
		// Row order changes, then a peer leaves and rejoins. Each cache must
		// follow its request rather than its position in the current batch.
		for step, order := range [][]int{{0, 1}, {1, 0}, {1}, {0, 1}} {
			n := []int{1, 3, 2, 4}[step]
			rows := make([][]cache.Cache, len(order))
			values := make([]float32, 2*n*len(order))
			b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, len(order), n)}
			var want []float32
			for row, id := range order {
				rows[row] = []cache.Cache{together[id]}
				b.SeqOffsets = append(b.SeqOffsets, int32(together[id].Offset()))
				b.SeqQueryLens = append(b.SeqQueryLens, int32(n))
				for j := range n {
					values[2*(row*n+j)], values[2*(row*n+j)+1] = float32(id+1)/2, float32(step+j+1)/4
				}
				single := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, n), SeqOffsets: []int32{int32(alone[id].Offset())}, SeqQueryLens: []int32{int32(n)}}
				x := mlx.FromValues(values[2*row*n:2*(row+1)*n], 1, n, 2)
				want = append(want, conv.Forward(x, single, alone[id], 1, int32(n), cfg).Floats()...)
			}
			layers, err := cache.Batch(rows)
			if err != nil {
				t.Fatal(err)
			}
			got := conv.Forward(mlx.FromValues(values, len(order), n, 2), b, layers[0], int32(len(order)), int32(n), cfg).Floats()
			if len(got) == 0 || len(got) != len(want) {
				t.Fatalf("step %d order %v value count = %d, want %d", step, order, len(got), len(want))
			}
			for i := range want {
				if delta := math.Abs(float64(got[i] - want[i])); math.IsNaN(delta) || delta > 1e-5 {
					t.Fatalf("step %d order %v output %d = %g, want %g", step, order, i, got[i], want[i])
				}
			}
			for _, id := range order {
				if together[id].Offset() != alone[id].Offset() {
					t.Fatalf("step %d request %d cache offset = %d, want %d", step, id, together[id].Offset(), alone[id].Offset())
				}
			}
		}
	})
}

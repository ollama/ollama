package hy4_flash

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestIndexPoolAcrossChunksAndRewind(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const width, length = 8, 19
		weights := mlx.Sin(mlx.Arange(0, width*width, 1, mlx.DTypeFloat32)).Reshape(width, width)
		unit := mlx.AddScalar(mlx.Zeros(mlx.DTypeBFloat16, width), 1)
		cfg := &textConfig{IndexRatio: 4, IndexTopK: 2, IndexDim: width}
		shape := attentionShape{rotary: 4, theta: 10000, sections: [3]int{1, 1, 1}}
		pool := &indexer{
			TokenKey:  nn.NewLinear(weights.AsType(mlx.DTypeBFloat16), nil),
			TokenNorm: nn.NewRMSNorm(unit, 1e-5),
			PoolValue: nn.NewLinear(weights, nil),
			PoolGate:  nn.NewLinear(weights.Transpose(1, 0), nil),
			GroupNorm: nn.NewRMSNorm(unit, 1e-5),
			SlotBias:  mlx.Zeros(mlx.DTypeFloat32, cfg.IndexRatio, width),
		}
		ids := mlx.Arange(0, length, 1, mlx.DTypeInt32).Reshape(1, length)
		x := mlx.Cos(mlx.Arange(0, length*width, 1, mlx.DTypeFloat32)).AsType(mlx.DTypeBFloat16).Reshape(1, length, width)
		layout := &positionLayout{positions: make([][3]int32, length)}
		for j := range length {
			p := int32(j)
			layout.positions[j] = [3]int32{p, p + 1, p + 3}
		}
		run := func(start, end int, slot cache.Attention) indexHistory {
			b := &batch.Batch{
				InputIDs:     ids.Slice(mlx.Slice(), mlx.Slice(start, end)),
				SeqOffsets:   []int32{int32(start)},
				SeqQueryLens: []int32{int32(end - start)},
				Layout:       []any{layout},
			}
			input := x.Slice(mlx.Slice(), mlx.Slice(start, end), mlx.Slice())
			return pool.histories(input, b, slot, shape, cfg)[0]
		}
		full := run(0, length, nil).groups.AsType(mlx.DTypeFloat32).Floats()
		slot := cache.NewKVCache()
		defer slot.Free()
		slot.PrepareSnapshots([]int{5})
		run(0, 7, slot)
		snap := slot.TakeSnapshots()[0]
		defer snap.Close()
		check := func(got indexHistory) {
			t.Helper()
			if got.tokens != length || !slices.Equal(got.groups.AsType(mlx.DTypeFloat32).Floats(), full) {
				t.Fatal("cached groups differ from full replay")
			}
		}
		run(7, 11, slot)
		check(run(11, length, slot))
		slot = cache.NewKVCache()
		defer slot.Free()
		if !slot.Restore(snap, 5) {
			t.Fatal("restore across a partial group failed")
		}
		for pos := 5; pos < length-1; pos++ {
			run(pos, pos+1, slot)
		}
		check(run(length-1, length, slot))

		left, right := cache.NewKVCache(), cache.NewKVCache()
		defer left.Free()
		defer right.Free()
		run(0, 3, left)
		combined, err := cache.Batch([][]cache.Cache{{left}, {right}})
		if err != nil {
			t.Fatal(err)
		}
		const chunk = 9
		b := &batch.Batch{
			InputIDs:     mlx.Zeros(mlx.DTypeInt32, 2, chunk),
			SeqOffsets:   []int32{3, 0},
			SeqQueryLens: []int32{chunk, chunk},
			Layout:       []any{layout, layout},
		}
		inputs := mlx.Concatenate([]*mlx.Array{
			x.Slice(mlx.Slice(), mlx.Slice(3, 3+chunk), mlx.Slice()),
			x.Slice(mlx.Slice(), mlx.Slice(0, chunk), mlx.Slice()),
		}, 0)
		rows := pool.histories(inputs, b, combined[0].(cache.Attention), shape, cfg)
		for row, count := range []int{3 + chunk, chunk} {
			if rows[row].tokens != count || rows[row].groups.Dim(2) != count/cfg.IndexRatio {
				t.Fatalf("row %d returned the wrong token or group count", row)
			}
		}
	})
}

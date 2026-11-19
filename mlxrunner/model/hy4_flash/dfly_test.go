package hy4_flash

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
)

func TestDFlyConditioningCarriesEachRowsSelection(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		draft := &dflyDraft{
			Layers:     []dflyLayer{{sharedSlot: 1}, {sharedSlot: 3}},
			indexWidth: 4,
		}
		b := &batch.Batch{SeqOffsets: []int32{1, 4}, SeqQueryLens: []int32{2, 2}}
		hidden := mlx.FromValues([]float32{0, 1, 2, 3}, 2, 2, 1).AsType(mlx.DTypeBFloat16)
		b.Memo.Put(draftSelectionKey{slot: 1, row: 1}, mlx.FromValues([]int32{0, 2, 4, 5, 1, 3, 4, 5}, 1, 2, 4))
		b.Memo.Put(draftSelectionKey{slot: 3, row: 1}, mlx.FromValues([]int32{1, 2, 4, 5, 0, 3, 4, 5}, 1, 2, 4))

		features := draft.conditioning(b, hidden)
		if got := features.Dims(); !slices.Equal(got, []int{2, 2, 9}) {
			t.Fatalf("feature shape = %v, want [2 2 9]", got)
		}
		want := []float32{
			0, -1, -1, -1, -1, -1, -1, -1, -1,
			1, -1, -1, -1, -1, -1, -1, -1, -1,
			2, 0, 2, 4, 5, 1, 2, 4, 5,
			3, 1, 3, 4, 5, 0, 3, 4, 5,
		}
		if got := features.Floats(); !slices.Equal(got, want) {
			t.Fatalf("conditioned features = %v, want %v", got, want)
		}
	})
}

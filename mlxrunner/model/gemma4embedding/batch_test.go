package gemma4embedding

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
)

func TestScatterMediaUsesRequestRow(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		m := &Model{Cfg: &Config{}}
		m.Cfg.HiddenSize = 2
		h := mlx.Zeros(mlx.DTypeFloat32, 2, 3, 2)
		b := &batch.Batch{SeqQueryLens: []int32{3, 2}, Media: []batch.MediaItem{
			{Seq: 1, Pos: 1, Opaque: preparedAudio{numTokens: 1}, Features: mlx.FromValues([]float32{7, 8}, 1, 2)},
			{Seq: 0, Pos: 0, Opaque: preparedAudio{numTokens: 1}, Features: mlx.FromValues([]float32{3, 4}, 1, 2)},
		}}
		got, err := m.scatterMedia(h, b)
		if err != nil {
			t.Fatal(err)
		}
		want := []float32{3, 4, 0, 0, 0, 0, 0, 0, 7, 8, 0, 0}
		if values := got.Floats(); !slices.Equal(values, want) {
			t.Fatalf("media rows %v, want %v", values, want)
		}
		b.Media[0].Pos = 2
		if _, err := m.scatterMedia(h, b); err == nil {
			t.Fatal("accepted media past row's real query length")
		}
	})
}

// Multiple layers must keep padded queries from contaminating real keys later.
func TestPaddedForwardMatchesIndependent(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		cfg := tinyTextConfig()
		m, err := loadModelTensors(cfg, tinyTextTensors(cfg))
		if err != nil {
			t.Fatal(err)
		}
		rows := [][]int32{{1, 5, 2}, {1, 9, 7, 6, 3, 8, 2}, {1, 4, 6, 2}}
		const length = 7
		ids := make([]int32, len(rows)*length)
		lens := make([]int32, len(rows))
		for i, row := range rows {
			copy(ids[i*length:], row)
			lens[i] = int32(len(row))
		}
		got, _ := m.Forward(&batch.Batch{InputIDs: mlx.FromValues(ids, len(rows), length), SeqOffsets: make([]int32, len(rows)), SeqQueryLens: lens}, nil)
		for i, row := range rows {
			want, _ := m.Forward(&batch.Batch{InputIDs: mlx.FromValues(row, 1, len(row)), SeqOffsets: []int32{0}, SeqQueryLens: []int32{int32(len(row))}}, nil)
			actual := got.Slice(mlx.Slice(i, i+1), mlx.Slice(0, len(row)), mlx.Slice()).Floats()
			expected := want.Floats()
			if len(actual) == 0 || len(actual) != len(expected) {
				t.Fatalf("row %d value count = %d, want %d", i, len(actual), len(expected))
			}
			for j, v := range expected {
				if delta := math.Abs(float64(actual[j] - v)); math.IsNaN(delta) || delta > 1e-4 {
					t.Fatalf("row %d value %d = %g, independent %g", i, j, actual[j], v)
				}
			}
		}
	})
}

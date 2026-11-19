package hy4_flash

import (
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
)

func TestMappedEngramRows(t *testing.T) {
	const name = "engram.weight"
	path := filepath.Join(t.TempDir(), "table.safetensors")
	data := make([]byte, 8*2*2)
	for i := range len(data) / 2 {
		bits := math.Float32bits(float32(i + 1))
		binary.LittleEndian.PutUint16(data[2*i:], uint16(bits>>16))
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	_, err = io.Copy(f, safetensors.BuildPackedSafetensorsReader([]*safetensors.TensorData{
		safetensors.NewTensorDataFromBytes(name, "BF16", []int32{8, 2}, data),
	}))
	closeErr := f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}

	cfg := config{Text: textConfig{EngramOrder: 3, EngramHeads: 1, EngramWidth: 2, EngramPadID: 0}, ImageTokenID: 5, VideoTokenID: 6}
	e, err := openEngram(path, name, [][]int{{3}, {5}}, []int64{1, 3, 5}, &cfg.Text)
	if err != nil {
		t.Fatal(err)
	}
	defer e.table.Close()
	e.tokenMap = []int32{1, 0, 2, 3, 4, 5, 6}

	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			chunk := func(ids []int32, off int32) *batch.Batch {
				return &batch.Batch{InputIDs: mlx.FromValues(ids, 1, len(ids)), SeqOffsets: []int32{off}, SeqQueryLens: []int32{int32(len(ids))}}
			}
			slot := cache.NewKVCache()
			defer slot.Free()
			prefill := chunk([]int32{2, 5}, 0)
			engramTokens(prefill, slot, cfg.Text.EngramOrder)
			next := chunk([]int32{3, 0, 4}, 2)
			rows := engramTokens(next, slot, cfg.Text.EngramOrder)
			if len(rows) != 1 || !slices.Equal(rows[0], []int32{2, 5, 3, 0, 4}) {
				t.Fatalf("unexpected cached history %v", rows)
			}
			embeddings, mask, err := e.gather(rows, next, &cfg)
			if err != nil {
				t.Fatal(err)
			}
			// The previous image token participates in hashes for later text.
			// It contributes no Engram output at its own position.
			want := []float32{1, 2, 9, 10, 0, 0, 0, 0, 3, 4, 13, 14}
			if got := embeddings.Floats(); !slices.Equal(got, want) {
				t.Fatalf("gathered %v, want %v", got, want)
			}
			if got := mask.AsType(mlx.DTypeInt32).Ints(); !slices.Equal(got, []int32{1, 0, 1}) {
				t.Fatalf("Engram mask = %v", got)
			}
		})
	})
}

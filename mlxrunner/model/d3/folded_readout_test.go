package d3

import (
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func bf16Bytes(t *testing.T, values []float32) []byte {
	t.Helper()
	out := make([]byte, 0, len(values)*2)
	for _, v := range values {
		bits := math.Float32bits(v)
		out = binary.LittleEndian.AppendUint16(out, uint16((bits+0x7fff+(bits>>16)&1)>>16))
	}
	return out
}

func TestFoldedReadoutRejectsCompetingWeight(t *testing.T) {
	m := &Model{foldedReadout: &foldedReadout{}}
	err := m.LoadWeights(map[string]*mlx.Array{"weight": {}})
	if err == nil || !strings.Contains(err.Error(), "cannot also load weight") {
		t.Fatalf("LoadWeights error = %v, want competing readout rejection", err)
	}
}

func TestFoldedReadoutLoadsDesignatedRows(t *testing.T) {
	const (
		vocab  = 8
		hidden = 4
	)
	rows := make([]float32, 0, vocab*hidden)
	for r := range vocab {
		for range hidden {
			rows = append(rows, float32(r)+0.5)
		}
	}
	path := filepath.Join(t.TempDir(), "lm_head.safetensors")
	blob, err := io.ReadAll(safetensors.BuildPackedSafetensorsReader([]*safetensors.TensorData{
		safetensors.NewTensorDataFromBytes(lmHeadReadoutTensor, "BF16", []int32{vocab, hidden}, bf16Bytes(t, rows)),
	}))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, blob, 0o600); err != nil {
		t.Fatal(err)
	}

	folded := &foldedReadout{path: path, rows: []int32{5, 0, 7}, width: hidden}
	mlxtest.Run(t, func(t *mlxtest.T) {
		out, err := folded.load()
		if err != nil {
			t.Fatal(err)
		}
		if got := out.Dims(); len(got) != 2 || got[0] != 3 || got[1] != hidden {
			t.Fatalf("readout dims = %v, want [3 %d]", got, hidden)
		}
		values := out.Floats()
		for i, row := range folded.rows {
			for col := range hidden {
				if want := float32(row) + 0.5; values[i*hidden+col] != want {
					t.Fatalf("readout[%d][%d] = %v, want row %d (%v)", i, col, values[i*hidden+col], row, want)
				}
			}
		}
	})
}

package deepseek_v41

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/mlxrunner/model"
)

func TestHostEngramReadsOnlySelectedRows(t *testing.T) {
	const name = "layers.1.engram.embed.weight"
	const words, scaleColumns = 64, 8
	weightBytes := make([]byte, 3*words*4)
	for i := range 3 * words {
		binary.LittleEndian.PutUint32(weightBytes[i*4:], uint32(i+1))
	}
	scaleBytes := make([]byte, 3*scaleColumns)
	for i := range scaleBytes {
		scaleBytes[i] = byte(i + 1)
	}
	blob := safetensors.BuildPackedSafetensorsReaderWithMetadata([]*safetensors.TensorData{
		safetensors.NewTensorDataFromBytes(name, "U32", []int32{3, words}, weightBytes),
		safetensors.NewTensorDataFromBytes(name+".scale", "U8", []int32{3, scaleColumns}, scaleBytes),
	}, map[string]string{"quant_type": "MXFP8", "group_size": "32"})
	path := filepath.Join(t.TempDir(), "table.safetensors")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.ReadFrom(blob); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	table, err := openHostEngramTable(path, name, &model.TensorQuantInfo{QuantType: "MXFP8", GroupSize: 32})
	if err != nil {
		t.Fatal(err)
	}
	defer table.close()
	weights, scales, err := table.readRows([]int64{2, 0, 2})
	if err != nil {
		t.Fatal(err)
	}
	wantScales := append(append(append([]byte{}, scaleBytes[2*scaleColumns:3*scaleColumns]...), scaleBytes[:scaleColumns]...), scaleBytes[2*scaleColumns:3*scaleColumns]...)
	if !slices.Equal(scales, wantScales) || weights[0] != uint32(2*words+1) ||
		weights[words] != 1 || weights[2*words] != uint32(2*words+1) {
		t.Fatalf("selected rows = %v, %v", weights, scales)
	}
	if _, _, err := table.readRows([]int64{3}); err == nil {
		t.Fatal("out-of-range row accepted")
	}
	for _, quant := range []*model.TensorQuantInfo{
		{QuantType: "MXFP4", GroupSize: 32},
		{QuantType: "NVFP4", GroupSize: 16},
		{QuantType: "MXFP8", GroupSize: 16},
	} {
		if _, err := openHostEngramTable(path, name, quant); err == nil {
			t.Fatalf("unsupported engram quantization %+v accepted", quant)
		}
	}
}

func TestModelCloseReleasesEngramMappings(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "engram-*")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write([]byte{1}); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	mapping, err := model.OpenHostMapping(f.Name())
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{EngramTables: map[int]*hostEngramTable{1: {mapping: mapping}}}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := mapping.Bytes(0, 1); err == nil {
		t.Fatal("host mapping remains readable after model close")
	}
	if err := m.Close(); err != nil {
		t.Fatalf("second model close: %v", err)
	}
}

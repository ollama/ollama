package client

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/x/create"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/safetensors"
)

func TestDecodeSourceFP8TensorAcceptsWeightScale(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX unavailable: %v", err)
	}

	weight := mlx.FromValues([]uint8{0, 1, 2, 3}, 2, 2)
	scale := mlx.FromValues([]float32{1}, 1, 1).AsType(mlx.DTypeBFloat16)
	got, err := decodeSourceFP8Tensor(weight, scale)
	if err != nil {
		t.Fatal(err)
	}
	mlx.Eval(got)
	if dims := got.Dims(); len(dims) != 2 || dims[0] != 2 || dims[1] != 2 {
		t.Fatalf("decoded dims = %v, want [2 2]", dims)
	}
}

func TestQuantizePackedGroupFromGoroutine(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX unavailable: %v", err)
	}

	groupName := "model.layers.0.moe.experts"
	blobData, err := io.ReadAll(safetensors.BuildPackedSafetensorsReader([]*safetensors.TensorData{
		safetensors.NewTensorDataFromBytes(
			groupName+".0.gate_proj.weight",
			"F32",
			[]int32{64, 64},
			float32Bytes(64*64, 1),
		),
		safetensors.NewTensorDataFromBytes(
			groupName+".1.gate_proj.weight",
			"F32",
			[]int32{64, 64},
			float32Bytes(64*64, 4097),
		),
	}))
	if err != nil {
		t.Fatal(err)
	}

	resultCh := make(chan []byte, 1)
	errCh := make(chan error, 1)
	go func() {
		out, err := QuantizePackedGroup(groupName, []create.PackedTensorInput{
			{
				Name:     groupName + ".0.gate_proj.weight",
				Dtype:    "F32",
				Shape:    []int32{64, 64},
				Quantize: "int4",
				Reader:   bytes.NewReader(blobData),
			},
			{
				Name:     groupName + ".1.gate_proj.weight",
				Dtype:    "F32",
				Shape:    []int32{64, 64},
				Quantize: "int4",
				Reader:   bytes.NewReader(blobData),
			},
		})
		if err != nil {
			errCh <- err
			return
		}
		resultCh <- out
	}()

	var out []byte
	select {
	case err := <-errCh:
		t.Fatal(err)
	case out = <-resultCh:
	}

	path := filepath.Join(t.TempDir(), "quantized.safetensors")
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatal(err)
	}

	metas, err := safetensors.ReadBlobMetas(path)
	if err != nil {
		t.Fatal(err)
	}

	want := map[string]bool{
		"model.layers.0.moe.switch_mlp.gate_proj.weight":       true,
		"model.layers.0.moe.switch_mlp.gate_proj.weight.scale": true,
	}
	for _, meta := range metas {
		delete(want, meta.Name)
	}
	if len(want) != 0 {
		t.Fatalf("missing quantized packed tensors: %v", want)
	}
}

func float32Bytes(n int, start float32) []byte {
	data := make([]byte, n*4)
	for i := range n {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(start+float32(i)))
	}
	return data
}

package qwen4_exp

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"slices"
	"testing"

	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestHostPLELookupMatchesResidentEmbedding(t *testing.T) {
	const (
		prefix       = "model.language_model.layers.1.ple"
		shardCount   = 2
		rowsPerShard = 3
		rowWidth     = 16
	)
	weights := []uint32{
		0x01234567, 0x89abcdef,
		0x11111111, 0x22222222,
		0x33333333, 0x44444444,
		0x55555555, 0x66666666,
		0x77777777, 0x88888888,
		0x99999999, 0xaaaaaaaa,
	}
	scales := []uint8{0x38, 0x40, 0x48, 0x38, 0x40, 0x48}

	t.Setenv("OLLAMA_MODELS", t.TempDir())
	layers := make([]manifest.Layer, 0, shardCount)
	for shard := range shardCount {
		name := fmt.Sprintf("%s.ple_embedding.ngram_embedding.shard_%d.weight", prefix, shard)
		start := shard * rowsPerShard
		end := start + rowsPerShard
		weightData := safetensors.NewTensorDataFromBytes(name, "U32", []int32{rowsPerShard, rowWidth / 8}, uint32Bytes(weights[start*2:end*2]))
		scaleData := safetensors.NewTensorDataFromBytes(name+".scale", "U8", []int32{rowsPerShard, rowWidth / 16}, scales[start:end])
		blob, err := io.ReadAll(safetensors.BuildPackedSafetensorsReaderWithMetadata([]*safetensors.TensorData{weightData, scaleData}, map[string]string{
			"group_size": "16",
			"quant_type": "nvfp4",
		}))
		if err != nil {
			t.Fatal(err)
		}
		layer, err := manifest.NewLayer(bytes.NewReader(blob), manifest.MediaTypeImageTensor)
		if err != nil {
			t.Fatal(err)
		}
		layer.Name = name
		layers = append(layers, layer)
	}
	layerByName := make(map[string]manifest.Layer, len(layers))
	for _, layer := range layers {
		layerByName[layer.Name] = layer
	}
	table, err := openHostPLETable(layerByName, prefix, shardCount, rowWidth)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(table.close)

	mlxtest.Run(t, func(t *mlxtest.T) {
		ids := mlx.FromValues([]int64{5, 0, 3, 3, 2, 4}, 2, 3)
		got := table.lookup(ids).AsType(mlx.DTypeFloat32)
		want := (&nn.QuantizedEmbedding{
			Weight:    mlx.FromValues(weights, shardCount*rowsPerShard, rowWidth/8),
			Scales:    mlx.FromValues(scales, shardCount*rowsPerShard, rowWidth/16),
			GroupSize: 16,
			Bits:      4,
			Mode:      "nvfp4",
		}).Forward(ids).AsType(mlx.DTypeFloat32)
		mlx.Eval(got, want)
		if !slices.Equal(got.Floats(), want.Floats()) {
			t.Fatalf("host lookup = %v, want %v", got.Floats(), want.Floats())
		}
	})
}

func uint32Bytes(values []uint32) []byte {
	data := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(data[i*4:], value)
	}
	return data
}

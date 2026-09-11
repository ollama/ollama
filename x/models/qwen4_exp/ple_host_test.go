package qwen4_exp

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/safetensors"
)

func TestHostPLELookupMatchesResidentEmbedding(t *testing.T) {
	mlxtest.Setup(t)

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

	dir := t.TempDir()
	layers := make([]manifest.ManifestLayer, 0, shardCount)
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
		digest := fmt.Sprintf("sha256:%d", shard)
		if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("sha256-%d", shard)), blob, 0o600); err != nil {
			t.Fatal(err)
		}
		layers = append(layers, manifest.ManifestLayer{
			MediaType: "application/vnd.ollama.image.tensor",
			Digest:    digest,
			Name:      name,
		})
	}
	modelManifest := &manifest.ModelManifest{
		Manifest: &manifest.Manifest{Layers: layers},
		BlobDir:  dir,
	}
	layerByName := make(map[string]manifest.ManifestLayer, len(layers))
	for _, layer := range layers {
		layerByName[layer.Name] = layer
	}
	table, err := openHostPLETable(modelManifest, layerByName, prefix, shardCount, rowWidth)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(table.close)

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
}

func uint32Bytes(values []uint32) []byte {
	data := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(data[i*4:], value)
	}
	return data
}

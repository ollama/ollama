package qwen4_exp

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"slices"
	"sort"
	"strings"

	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
)

var errHostPLEUnsupported = errors.New("host-backed PLE requires standard NVFP4")

type hostPLEShard struct {
	mapping *hostMapping

	weightStart   int64
	scaleStart    int64
	rows          int
	weightColumns int
	scaleColumns  int
	globalScale   float32 // MLX representation; zero means no scale was stored.
}

type hostPLETable struct {
	shards         []*hostPLEShard
	rows           int
	rowWidth       int
	hasGlobalScale bool
}

func (m *Model) SkippedTensorNames() []string {
	var names []string
	for prefix, table := range m.hostPLE {
		for i := range table.shards {
			name := fmt.Sprintf("%s.ple_embedding.ngram_embedding.shard_%d.weight", prefix, i)
			names = append(names, name, name+".scale", name+".global_scale")
		}
	}
	return names
}

func openHostPLETables(root *model.Root, cfg *Config) (map[string]*hostPLETable, error) {
	headCount := (cfg.NGramSize - 1) * cfg.HeadsPerNGram
	if headCount <= 0 || cfg.PLEEmbedDim%headCount != 0 {
		return nil, nil
	}
	rowWidth := int(cfg.PLEEmbedDim / headCount)

	layers := make(map[string]manifest.Layer)
	for _, layer := range root.Manifest.TensorLayers() {
		layers[layer.Name] = layer
	}

	tables := make(map[string]*hostPLETable, len(cfg.PLELayerIDs))
	rowsPerShard := 0
	for _, id := range cfg.PLELayerIDs {
		prefix := fmt.Sprintf("model.language_model.layers.%d.ple", id-1)
		table, err := openHostPLETable(layers, root.AllTensorQuant(), prefix, int(cfg.SplitNGramParts), rowWidth)
		if errors.Is(err, errHostPLEUnsupported) {
			closeHostPLETables(tables)
			return nil, nil
		}
		if err != nil {
			closeHostPLETables(tables)
			return nil, fmt.Errorf("%s: %w", prefix, err)
		}
		tables[prefix] = table
		rowsPerShard = table.rows
	}
	slog.Info("enabled host-backed Qwen PLE", "layers", len(tables), "shards_per_layer", cfg.SplitNGramParts, "rows_per_shard", rowsPerShard)
	return tables, nil
}

func openHostPLETable(layers map[string]manifest.Layer, quant map[string]*model.TensorQuantInfo, prefix string, shardCount, rowWidth int) (*hostPLETable, error) {
	table := &hostPLETable{
		shards:   make([]*hostPLEShard, shardCount),
		rowWidth: rowWidth,
	}
	for i := range shardCount {
		name := fmt.Sprintf("%s.ple_embedding.ngram_embedding.shard_%d.weight", prefix, i)
		layer, ok := layers[name]
		if !ok {
			table.close()
			return nil, fmt.Errorf("missing tensor layer %q", name)
		}
		path, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			table.close()
			return nil, fmt.Errorf("resolve %s: %w", name, err)
		}
		shard, err := openHostPLEShard(path, name, rowWidth, quant[name])
		if err != nil {
			table.close()
			return nil, fmt.Errorf("open %s: %w", name, err)
		}
		if i == 0 {
			table.rows = shard.rows
		} else if shard.rows != table.rows {
			shard.mapping.close()
			table.close()
			return nil, fmt.Errorf("shard %d has %d rows, want %d", i, shard.rows, table.rows)
		}
		table.shards[i] = shard
		table.hasGlobalScale = table.hasGlobalScale || shard.globalScale != 0
	}
	return table, nil
}

func openHostPLEShard(path, name string, rowWidth int, quant *model.TensorQuantInfo) (*hostPLEShard, error) {
	// NVFP4 always uses groups of 16; prequantized imports may omit the field.
	if quant == nil || !strings.EqualFold(quant.QuantType, "nvfp4") || (quant.GroupSize != 0 && quant.GroupSize != 16) {
		return nil, errHostPLEUnsupported
	}
	extractor, err := safetensors.OpenForExtraction(path)
	if err != nil {
		return nil, err
	}
	defer extractor.Close()
	weight, err := extractor.GetTensor(name)
	if err != nil {
		return nil, err
	}
	scale, err := extractor.GetTensor(name + ".scale")
	if err != nil {
		return nil, err
	}
	if weight.Dtype != "U32" || len(weight.Shape) != 2 || weight.Shape[0] <= 0 || int64(weight.Shape[1])*8 != int64(rowWidth) {
		return nil, fmt.Errorf("weight has dtype=%s shape=%v, want U32 [rows,%d]", weight.Dtype, weight.Shape, rowWidth/8)
	}
	if scale.Dtype != "U8" || len(scale.Shape) != 2 || scale.Shape[0] != weight.Shape[0] || int64(scale.Shape[1])*16 != int64(rowWidth) {
		return nil, fmt.Errorf("scale has dtype=%s shape=%v, want U8 [%d,%d]", scale.Dtype, scale.Shape, weight.Shape[0], rowWidth/16)
	}
	globalScale, err := readHostGlobalScale(extractor, name)
	if err != nil {
		return nil, err
	}
	mapping, err := openHostMapping(path)
	if err != nil {
		return nil, err
	}

	return &hostPLEShard{
		mapping:       mapping,
		weightStart:   weight.Offset(),
		scaleStart:    scale.Offset(),
		rows:          int(weight.Shape[0]),
		weightColumns: int(weight.Shape[1]),
		scaleColumns:  int(scale.Shape[1]),
		globalScale:   globalScale,
	}, nil
}

func readHostGlobalScale(extractor *safetensors.TensorExtractor, name string) (float32, error) {
	if !slices.Contains(extractor.ListTensors(), name+".global_scale") {
		return 0, nil
	}
	entry, err := extractor.GetTensor(name + ".global_scale")
	if err != nil {
		return 0, err
	}
	scalar := true
	for _, dim := range entry.Shape {
		scalar = scalar && dim == 1
	}
	if entry.Dtype != "F32" || !scalar || entry.Size != 4 {
		return 0, fmt.Errorf("global scale has dtype=%s shape=%v, want scalar F32", entry.Dtype, entry.Shape)
	}
	var data [4]byte
	if _, err := io.ReadFull(entry.Reader(), data[:]); err != nil {
		return 0, err
	}
	value := math.Float32frombits(binary.LittleEndian.Uint32(data[:])) * mlx.Nvfp4MaxProduct
	if !(value > 0) || math.IsInf(float64(value), 0) {
		return 0, fmt.Errorf("global scale is not finite and positive")
	}
	return value, nil
}

func closeHostPLETables(tables map[string]*hostPLETable) {
	for _, table := range tables {
		table.close()
	}
}

func (t *hostPLETable) close() {
	for _, shard := range t.shards {
		if shard != nil {
			shard.mapping.close()
		}
	}
}

func (t *hostPLETable) lookup(globalIDs *mlx.Array) *mlx.Array {
	shape := globalIDs.Dims()
	ids := globalIDs.AsType(mlx.DTypeInt32).Ints()
	order := make([]int, len(ids))
	maxID := len(t.shards) * t.rows
	for i, id := range ids {
		if id < 0 || int(id) >= maxID {
			panic(fmt.Sprintf("Qwen PLE row ID %d outside [0,%d)", id, maxID))
		}
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool {
		return ids[order[i]] < ids[order[j]]
	})

	weightColumns := t.shards[0].weightColumns
	scaleColumns := t.shards[0].scaleColumns
	weights := make([]uint32, len(ids)*weightColumns)
	scales := make([]uint8, len(ids)*scaleColumns)
	previousID := int32(-1)
	previousOutput := -1
	for _, output := range order {
		id := ids[output]
		weightOutput := weights[output*weightColumns : (output+1)*weightColumns]
		scaleOutput := scales[output*scaleColumns : (output+1)*scaleColumns]
		if id == previousID {
			copy(weightOutput, weights[previousOutput*weightColumns:(previousOutput+1)*weightColumns])
			copy(scaleOutput, scales[previousOutput*scaleColumns:(previousOutput+1)*scaleColumns])
			continue
		}

		shard := t.shards[int(id)/t.rows]
		row := int64(int(id) % t.rows)
		weightBytes, err := shard.mapping.bytes(
			shard.weightStart+row*int64(weightColumns*4),
			shard.weightStart+(row+1)*int64(weightColumns*4),
		)
		if err != nil {
			panic(fmt.Sprintf("read Qwen PLE weight row %d: %v", id, err))
		}
		for i := range weightColumns {
			weightOutput[i] = binary.LittleEndian.Uint32(weightBytes[i*4:])
		}
		scaleBytes, err := shard.mapping.bytes(
			shard.scaleStart+row*int64(scaleColumns),
			shard.scaleStart+(row+1)*int64(scaleColumns),
		)
		if err != nil {
			panic(fmt.Sprintf("read Qwen PLE scale row %d: %v", id, err))
		}
		copy(scaleOutput, scaleBytes)
		previousID = id
		previousOutput = output
	}

	weightShape := append(append([]int(nil), shape...), weightColumns)
	scaleShape := append(append([]int(nil), shape...), scaleColumns)
	weight := mlx.FromValues(weights, weightShape...)
	scale := mlx.FromValues(scales, scaleShape...)
	var globalScale *mlx.Array
	if t.hasGlobalScale {
		// IDs can interleave shards, each with its own global scale. Apply the
		// selected shard's scale per row, including repeated IDs.
		values := make([]float32, len(ids))
		for i, id := range ids {
			values[i] = t.shards[int(id)/t.rows].globalScale
			if values[i] == 0 {
				values[i] = mlx.Nvfp4MaxProduct
			}
		}
		globalScale = mlx.FromValues(values, shape...)
	}
	return mlx.Dequantize(weight, scale, nil, 16, 4, "nvfp4", globalScale)
}

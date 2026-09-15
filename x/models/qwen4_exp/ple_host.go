package qwen4_exp

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strconv"
	"strings"

	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

var errHostPLEUnsupported = errors.New("host-backed PLE requires standard NVFP4")

type safetensorEntry struct {
	DType       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

type hostPLEShard struct {
	mapping *hostMapping

	weightStart   int64
	scaleStart    int64
	rows          int
	weightColumns int
	scaleColumns  int
}

type hostPLETable struct {
	shards   []*hostPLEShard
	rows     int
	rowWidth int
}

func openHostPLETables(m *manifest.ModelManifest, cfg *Config) (map[string]*hostPLETable, error) {
	headCount := (cfg.NGramSize - 1) * cfg.HeadsPerNGram
	if headCount <= 0 || cfg.PLEEmbedDim%headCount != 0 {
		return nil, nil
	}
	rowWidth := int(cfg.PLEEmbedDim / headCount)

	layers := make(map[string]manifest.ManifestLayer)
	for _, layer := range m.GetTensorLayers("") {
		layers[layer.Name] = layer
	}

	tables := make(map[string]*hostPLETable, len(cfg.PLELayerIDs))
	rowsPerShard := 0
	for _, id := range cfg.PLELayerIDs {
		prefix := fmt.Sprintf("model.language_model.layers.%d.ple", id-1)
		table, err := openHostPLETable(m, layers, prefix, int(cfg.SplitNGramParts), rowWidth)
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

func openHostPLETable(m *manifest.ModelManifest, layers map[string]manifest.ManifestLayer, prefix string, shardCount, rowWidth int) (*hostPLETable, error) {
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
		shard, err := openHostPLEShard(m.BlobPath(layer.Digest), name, rowWidth)
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
	}
	return table, nil
}

func openHostPLEShard(path, name string, rowWidth int) (*hostPLEShard, error) {
	mapping, err := openHostMapping(path)
	if err != nil {
		return nil, err
	}
	fail := func(err error) (*hostPLEShard, error) {
		mapping.close()
		return nil, err
	}

	prefix, err := mapping.bytes(0, 8)
	if err != nil {
		return fail(err)
	}
	headerSize := int64(binary.LittleEndian.Uint64(prefix))
	headerData, err := mapping.bytes(8, 8+headerSize)
	if err != nil {
		return fail(err)
	}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerData, &header); err != nil {
		return fail(fmt.Errorf("parse safetensors header: %w", err))
	}
	var metadata map[string]string
	if err := json.Unmarshal(header["__metadata__"], &metadata); err != nil {
		return fail(errHostPLEUnsupported)
	}
	if !strings.EqualFold(metadata["quant_type"], "nvfp4") || metadata["group_size"] != strconv.Itoa(16) {
		return fail(errHostPLEUnsupported)
	}

	var weight, scale safetensorEntry
	if err := json.Unmarshal(header[name], &weight); err != nil {
		return fail(fmt.Errorf("parse weight header: %w", err))
	}
	if err := json.Unmarshal(header[name+".scale"], &scale); err != nil {
		return fail(errHostPLEUnsupported)
	}
	if weight.DType != "U32" || len(weight.Shape) != 2 || weight.Shape[0] <= 0 || weight.Shape[1]*8 != int64(rowWidth) {
		return fail(fmt.Errorf("weight has dtype=%s shape=%v, want U32 [rows,%d]", weight.DType, weight.Shape, rowWidth/8))
	}
	if scale.DType != "U8" || len(scale.Shape) != 2 || scale.Shape[0] != weight.Shape[0] || scale.Shape[1]*16 != int64(rowWidth) {
		return fail(fmt.Errorf("scale has dtype=%s shape=%v, want U8 [%d,%d]", scale.DType, scale.Shape, weight.Shape[0], rowWidth/16))
	}

	dataStart := 8 + headerSize
	weightStart := dataStart + weight.DataOffsets[0]
	weightEnd := dataStart + weight.DataOffsets[1]
	scaleStart := dataStart + scale.DataOffsets[0]
	scaleEnd := dataStart + scale.DataOffsets[1]
	if _, err := mapping.bytes(weightStart, weightEnd); err != nil {
		return fail(err)
	}
	if _, err := mapping.bytes(scaleStart, scaleEnd); err != nil {
		return fail(err)
	}
	if weightEnd-weightStart != weight.Shape[0]*weight.Shape[1]*4 || scaleEnd-scaleStart != scale.Shape[0]*scale.Shape[1] {
		return fail(fmt.Errorf("tensor byte sizes do not match their shapes"))
	}

	return &hostPLEShard{
		mapping:       mapping,
		weightStart:   weightStart,
		scaleStart:    scaleStart,
		rows:          int(weight.Shape[0]),
		weightColumns: int(weight.Shape[1]),
		scaleColumns:  int(scale.Shape[1]),
	}, nil
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
	return mlx.Dequantize(weight, scale, nil, 16, 4, "nvfp4", nil)
}

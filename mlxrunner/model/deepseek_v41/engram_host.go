package deepseek_v41

import (
	"encoding/binary"
	"fmt"

	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
)

// hostEngramTable keeps a published engram tensor on disk. Only the hashed
// rows needed for a batch are copied into MLX. The underlying blob must live
// on local storage; the package, rather than the vault source, owns its bytes.
type hostEngramTable struct {
	mapping     *model.HostMapping
	rows        int64
	weightStart int64
	scaleStart  int64
	words       int
	scales      int
}

func openHostEngramTable(path, name string, quant *model.TensorQuantInfo) (_ *hostEngramTable, err error) {
	if quant == nil || quant.QuantType != "MXFP8" || quant.GroupSize != 32 {
		return nil, fmt.Errorf("host engram %s requires the published MXFP8 layout", name)
	}
	ext, err := safetensors.OpenForExtraction(path)
	if err != nil {
		return nil, err
	}
	defer ext.Close()
	weight, err := ext.GetTensor(name)
	if err != nil {
		return nil, err
	}
	scale, err := ext.GetTensor(name + ".scale")
	if err != nil {
		return nil, err
	}
	if weight.Dtype != "U32" || scale.Dtype != "U8" || len(weight.Shape) != 2 || len(scale.Shape) != 2 ||
		weight.Shape[0] < 1 || weight.Shape[0] != scale.Shape[0] || weight.Shape[1] != 64 || scale.Shape[1] != 8 ||
		weight.Size != int64(weight.Shape[0])*int64(weight.Shape[1])*4 ||
		scale.Size != int64(scale.Shape[0])*int64(scale.Shape[1]) {
		return nil, fmt.Errorf("host engram %s has invalid weight/scale layout: %s %v and %s %v", name, weight.Dtype, weight.Shape, scale.Dtype, scale.Shape)
	}
	mapping, err := model.OpenHostMapping(path)
	if err != nil {
		return nil, err
	}
	return &hostEngramTable{
		mapping: mapping, rows: int64(weight.Shape[0]), weightStart: weight.Offset(), scaleStart: scale.Offset(),
		words: int(weight.Shape[1]), scales: int(scale.Shape[1]),
	}, nil
}

func (t *hostEngramTable) close() error {
	if t == nil || t.mapping == nil {
		return nil
	}
	err := t.mapping.Close()
	t.mapping = nil
	return err
}

// readRows keeps disk reads independent of MLX evaluation so I/O failures
// surface before a kernel is dispatched.
func (t *hostEngramTable) readRows(ids []int64) ([]uint32, []uint8, error) {
	weights := make([]uint32, len(ids)*t.words)
	scales := make([]uint8, len(ids)*t.scales)
	for output, id := range ids {
		if id < 0 || id >= t.rows {
			return nil, nil, fmt.Errorf("engram row %d outside [0,%d)", id, t.rows)
		}
		weightStart := t.weightStart + id*int64(t.words*4)
		row, err := t.mapping.Bytes(weightStart, weightStart+int64(t.words*4))
		if err != nil {
			return nil, nil, fmt.Errorf("read engram weight row %d: %w", id, err)
		}
		for col := range t.words {
			weights[output*t.words+col] = binary.LittleEndian.Uint32(row[col*4:])
		}
		start := output * t.scales
		scaleStart := t.scaleStart + id*int64(t.scales)
		scaleRow, err := t.mapping.Bytes(scaleStart, scaleStart+int64(t.scales))
		if err != nil {
			return nil, nil, fmt.Errorf("read engram scale row %d: %w", id, err)
		}
		copy(scales[start:start+t.scales], scaleRow)
	}
	return weights, scales, nil
}

func (t *hostEngramTable) lookup(ids []int64, shape ...int) (*mlx.Array, error) {
	count := 1
	for _, dim := range shape {
		if dim < 0 {
			return nil, fmt.Errorf("invalid engram lookup shape %v", shape)
		}
		count *= dim
	}
	if count != len(ids) {
		return nil, fmt.Errorf("engram lookup shape %v has %d entries, want %d", shape, count, len(ids))
	}
	weights, scales, err := t.readRows(ids)
	if err != nil {
		return nil, err
	}
	wShape := append(append([]int(nil), shape...), t.words)
	sShape := append(append([]int(nil), shape...), t.scales)
	w := mlx.FromValues(weights, wShape...)
	s := mlx.FromValues(scales, sShape...)
	return mlx.Dequantize(w, s, nil, 32, 8, "mxfp8", nil), nil
}

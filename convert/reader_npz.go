package convert

import (
	"encoding/binary"
	"fmt"
	"io"
	"log/slog"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"
	"github.com/sbinet/npyio/npz"
)

type adapterTensor struct {
	path  string
	dtype string
	*tensorBase
}

func DetectNPZ(fn string) (bool, error) {
	f, err := npz.Open(fn)
	if err != nil {
		return false, err
	}
	defer f.Close()

	if len(f.Keys()) > 0 && strings.HasSuffix(f.Keys()[0], ".npy") {
		return true, nil
	}

	return false, nil
}

func parseNPZ(fn string) ([]Tensor, error) {
	var ts []Tensor

	f, err := npz.Open(fn)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	for _, name := range f.Keys() {
		slog.Info(fmt.Sprintf("reading layer '%s'", name))
		h := f.Header(name)

		shape := make([]uint64, 2)
		for cnt, v := range h.Descr.Shape {
			// llamacpp expects the loraB layer to be reversed
			if strings.Contains(name, "lora_b") {
				shape[len(shape)-cnt-1] = uint64(v)
			} else {
				shape[cnt] = uint64(v)
			}
		}

		dtypeMap := map[string]string{
			"<f2": "F16",
			"<f4": "F32",
		}
		dtype, ok := dtypeMap[h.Descr.Type]
		if !ok {
			return nil, fmt.Errorf("Unknown type '%s' for '%s'", h.Descr.Type, name)
		}

		ts = append(ts, adapterTensor{
			path:  fn,
			dtype: dtype,
			tensorBase: &tensorBase{
				name:  name,
				shape: shape,
			},
		})
	}
	return ts, nil
}

func (t adapterTensor) Kind() uint32 {
	switch t.dtype {
	case "F32":
		return 0
	case "F16":
		return 1
	}
	return 0
}

func (t adapterTensor) WriteTo(w io.Writer) (int64, error) {
	f, err := npz.Open(t.path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	switch t.dtype {
	case "F32":
		var f32s []float32
		err = f.Read(t.tensorBase.name, &f32s)
		if err != nil {
			return 0, err
		}

		// ggla expects the loraB to be transposed
		if strings.Contains(t.tensorBase.name, "lora_b") {
			f32s, err = transpose(f32s, t.tensorBase.shape)
			if err != nil {
				return 0, err
			}
		}

		return 0, binary.Write(w, binary.LittleEndian, f32s)
	}

	return 0, fmt.Errorf("unknown data type: %s", t.dtype)
}

func transpose(f32s []float32, shape []uint64) ([]float32, error) {
	if len(shape) != 2 {
		return nil, fmt.Errorf("only 2 dimensions supported for transpose")
	}

	// the shape is already backward
	n := tensor.New(tensor.WithShape(int(shape[1]), int(shape[0])), tensor.WithBacking(f32s))
	if err := n.T(1, 0); err != nil {
		return nil, err
	}
	if err := n.Transpose(); err != nil {
		return nil, err
	}
	ts, err := native.SelectF32(n, 1)
	if err != nil {
		return nil, err
	}
	f32s = make([]float32, 0)
	for _, t := range ts {
		f32s = append(f32s, t...)
	}
	return f32s, nil
}

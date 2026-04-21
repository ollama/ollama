package compatmigrate

import (
	"fmt"
	"io"
	"slices"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/fs/gguf"
)

func splitSourceTensorDim(t *sourceTensor, dim int, names ...string) ([]*ggml.Tensor, error) {
	if dim < 0 || dim >= len(t.shape) {
		return nil, fmt.Errorf("split dimension %d out of range for tensor %s shape %v", dim, t.name, t.shape)
	}
	if len(names) == 0 {
		return nil, fmt.Errorf("split tensor %s: no target names", t.name)
	}
	if t.shape[dim]%uint64(len(names)) != 0 {
		return nil, fmt.Errorf("split tensor %s: dimension %d size %d is not divisible by %d", t.name, dim, t.shape[dim], len(names))
	}

	out := make([]*ggml.Tensor, 0, len(names))
	offset := 0
	for _, name := range names {
		part := int(t.shape[dim] / uint64(len(names)))
		start := offset
		end := offset + part
		offset = end

		shape := slices.Clone(t.shape)
		shape[dim] = uint64(part)

		writer := t.Clone()
		writer.shape = slices.Clone(shape)
		writer.SetRepacker(func(_ string, data []float32, sourceShape []uint64) ([]float32, error) {
			dims := make([]int, len(sourceShape))
			for i := range sourceShape {
				dims[i] = int(sourceShape[i])
			}

			slice := slices.Repeat([]tensor.Slice{nil}, len(sourceShape))
			slice[dim] = tensor.S(start, end)

			var tt tensor.Tensor = tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))
			tt, err := tt.Slice(slice...)
			if err != nil {
				return nil, err
			}

			tt = tensor.Materialize(tt)
			if err := tt.Reshape(tt.Shape().TotalSize()); err != nil {
				return nil, err
			}

			return native.VectorF32(tt.(*tensor.Dense))
		})
		outType := compatOutputTensorType(name, t.name, t.info.Type)
		if outType != t.info.Type {
			writer.SetOutputType(outType)
		}

		out = append(out, &ggml.Tensor{
			Name:     name,
			Kind:     uint32(outType),
			Shape:    shape,
			WriterTo: writer,
		})
	}

	return out, nil
}

func concatSourceTensorsDim(dim int, name string, tensors ...*sourceTensor) (*ggml.Tensor, error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("concat tensor %s: no source tensors", name)
	}
	if dim < 0 || dim >= len(tensors[0].shape) {
		return nil, fmt.Errorf("concat tensor %s: dimension %d out of range for shape %v", name, dim, tensors[0].shape)
	}

	kind := compatOutputTensorType(name, tensors[0].name, tensors[0].info.Type)
	shape := slices.Clone(tensors[0].shape)
	shape[dim] = 0
	for _, t := range tensors {
		if compatOutputTensorType(name, t.name, t.info.Type) != kind {
			return nil, fmt.Errorf("concat tensor %s: mixed tensor types %v and %v", name, kind, t.info.Type)
		}
		if len(t.shape) != len(shape) {
			return nil, fmt.Errorf("concat tensor %s: rank mismatch %v vs %v", name, shape, t.shape)
		}
		for i := range shape {
			if i == dim {
				continue
			}
			if t.shape[i] != shape[i] {
				return nil, fmt.Errorf("concat tensor %s: incompatible shape %v vs %v", name, shape, t.shape)
			}
		}
		shape[dim] += t.shape[dim]
	}

	return &ggml.Tensor{
		Name:     name,
		Kind:     uint32(kind),
		Shape:    shape,
		WriterTo: sourceTensorConcat{tensors: tensors, dim: dim, kind: kind},
	}, nil
}

func concatSourceTensorsSlowDim(dim int, name string, tensors ...*sourceTensor) (*ggml.Tensor, error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("concat tensor %s: no source tensors", name)
	}
	if dim < 0 || dim >= len(tensors[0].shape) {
		return nil, fmt.Errorf("concat tensor %s: dimension %d out of range for shape %v", name, dim, tensors[0].shape)
	}
	if dim != len(tensors[0].shape)-1 {
		return nil, fmt.Errorf("concat tensor %s: dimension %d is not the slowest dimension for shape %v", name, dim, tensors[0].shape)
	}

	kind := compatOutputTensorType(name, tensors[0].name, tensors[0].info.Type)
	shape := slices.Clone(tensors[0].shape)
	shape[dim] = 0
	for _, t := range tensors {
		if compatOutputTensorType(name, t.name, t.info.Type) != kind {
			return nil, fmt.Errorf("concat tensor %s: mixed tensor types %v and %v", name, kind, t.info.Type)
		}
		if len(t.shape) != len(shape) {
			return nil, fmt.Errorf("concat tensor %s: rank mismatch %v vs %v", name, shape, t.shape)
		}
		for i := range shape {
			if i == dim {
				continue
			}
			if t.shape[i] != shape[i] {
				return nil, fmt.Errorf("concat tensor %s: incompatible shape %v vs %v", name, shape, t.shape)
			}
		}
		shape[dim] += t.shape[dim]
	}

	return &ggml.Tensor{
		Name:     name,
		Kind:     uint32(kind),
		Shape:    shape,
		WriterTo: sourceTensorSlowDimConcat{tensors: tensors},
	}, nil
}

func concatSourceTensorsDimKind(dim int, name string, kind gguf.TensorType, tensors ...*sourceTensor) (*ggml.Tensor, error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("concat tensor %s: no source tensors", name)
	}
	if dim < 0 || dim >= len(tensors[0].shape) {
		return nil, fmt.Errorf("concat tensor %s: dimension %d out of range for shape %v", name, dim, tensors[0].shape)
	}

	shape := slices.Clone(tensors[0].shape)
	shape[dim] = 0
	for _, t := range tensors {
		if len(t.shape) != len(shape) {
			return nil, fmt.Errorf("concat tensor %s: rank mismatch %v vs %v", name, shape, t.shape)
		}
		for i := range shape {
			if i == dim {
				continue
			}
			if t.shape[i] != shape[i] {
				return nil, fmt.Errorf("concat tensor %s: incompatible shape %v vs %v", name, shape, t.shape)
			}
		}
		shape[dim] += t.shape[dim]
	}

	return &ggml.Tensor{
		Name:     name,
		Kind:     uint32(kind),
		Shape:    shape,
		WriterTo: sourceTensorConcat{tensors: tensors, dim: dim, kind: kind},
	}, nil
}

func stackSourceTensors(name string, tensors ...*sourceTensor) (*ggml.Tensor, error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("stack tensor %s: no source tensors", name)
	}

	kind := compatOutputTensorType(name, tensors[0].name, tensors[0].info.Type)
	rawCopy := kind == tensors[0].info.Type
	baseShape := slices.Clone(tensors[0].shape)
	for _, t := range tensors {
		outType := compatOutputTensorType(name, t.name, t.info.Type)
		if outType != kind {
			return nil, fmt.Errorf("stack tensor %s: mixed tensor types %v and %v", name, kind, t.info.Type)
		}
		rawCopy = rawCopy && outType == t.info.Type
		if !slices.Equal(t.shape, baseShape) {
			return nil, fmt.Errorf("stack tensor %s: incompatible shape %v vs %v", name, baseShape, t.shape)
		}
	}

	shape := slices.Clone(baseShape)
	if len(shape) >= 3 && shape[2] == 1 {
		shape[2] = uint64(len(tensors))
	} else {
		shape = append(shape, uint64(len(tensors)))
	}

	var writer io.WriterTo = sourceTensorStack{tensors: tensors, kind: kind}
	if rawCopy {
		writer = sourceTensorSlowDimConcat{tensors: tensors}
	}

	return &ggml.Tensor{
		Name:     name,
		Kind:     uint32(kind),
		Shape:    shape,
		WriterTo: writer,
	}, nil
}

type sourceTensorConcat struct {
	tensors []*sourceTensor
	dim     int
	kind    gguf.TensorType
}

func (c sourceTensorConcat) GGUFWriteMemoryEstimate() uint64 {
	var sourceBytes, inputFloatBytes, outputElements uint64
	for _, source := range c.tensors {
		sourceBytes = saturatingAdd(sourceBytes, int64ToUint64(source.info.NumBytes()))
		elements := tensorElementCount(source.shape)
		inputFloatBytes = saturatingAdd(inputFloatBytes, saturatingMul(elements, 4))
		outputElements = saturatingAdd(outputElements, elements)
	}

	return saturatingAdd(sourceBytes, inputFloatBytes, saturatingMul(outputElements, 4), ggufTensorBytes(c.kind, outputElements))
}

func (c sourceTensorConcat) WriteTo(w io.Writer) (int64, error) {
	parts := make([]tensor.Tensor, 0, len(c.tensors))
	for _, source := range c.tensors {
		data, err := readSourceTensorFloatData(source)
		if err != nil {
			return 0, err
		}

		dims := make([]int, len(source.shape))
		for i := range source.shape {
			dims[i] = int(source.shape[i])
		}

		parts = append(parts, tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data)))
	}

	merged, err := tensor.Concat(c.dim, parts[0], parts[1:]...)
	if err != nil {
		return 0, err
	}
	merged = tensor.Materialize(merged)
	if err := merged.Reshape(merged.Shape().TotalSize()); err != nil {
		return 0, err
	}

	data, err := native.VectorF32(merged.(*tensor.Dense))
	if err != nil {
		return 0, err
	}

	n, err := encodeFloatTensor(w, c.kind, data)
	return int64(n), err
}

type sourceTensorSlowDimConcat struct {
	tensors []*sourceTensor
}

func (c sourceTensorSlowDimConcat) GGUFWriteMemoryEstimate() uint64 {
	total := uint64(compatCopyBufferSize)
	for _, source := range c.tensors {
		total = saturatingAdd(total, int64ToUint64(source.info.NumBytes()))
	}
	return total
}

func (c sourceTensorSlowDimConcat) WriteTo(w io.Writer) (int64, error) {
	var total int64
	for _, source := range c.tensors {
		n, err := source.WriteTo(w)
		total += n
		if err != nil {
			return total, err
		}
	}
	return total, nil
}

type sourceTensorStack struct {
	tensors []*sourceTensor
	kind    gguf.TensorType
}

func (s sourceTensorStack) GGUFWriteMemoryEstimate() uint64 {
	var sourceBytes, inputFloatBytes, outputElements uint64
	for _, source := range s.tensors {
		sourceBytes = saturatingAdd(sourceBytes, int64ToUint64(source.info.NumBytes()))
		elements := tensorElementCount(source.shape)
		inputFloatBytes = saturatingAdd(inputFloatBytes, saturatingMul(elements, 4))
		outputElements = saturatingAdd(outputElements, elements)
	}

	return saturatingAdd(sourceBytes, inputFloatBytes, ggufTensorBytes(s.kind, outputElements))
}

func (s sourceTensorStack) WriteTo(w io.Writer) (int64, error) {
	var total int64
	for _, source := range s.tensors {
		data, err := readSourceTensorFloatData(source)
		if err != nil {
			return total, err
		}
		n, err := encodeFloatTensor(w, s.kind, data)
		total += int64(n)
		if err != nil {
			return total, err
		}
	}
	return total, nil
}

func ggufTensorBytes(kind gguf.TensorType, elements uint64) uint64 {
	return uint64(float64(elements) * kind.NumBytes())
}

func readSourceTensorFloatData(t *sourceTensor) ([]float32, error) {
	info := t.info
	var r io.Reader
	if t.readerAt != nil {
		r = io.NewSectionReader(t.readerAt, t.dataOffset+int64(t.info.Offset), t.info.NumBytes())
	} else {
		f, err := gguf.Open(t.path)
		if err != nil {
			return nil, err
		}
		defer f.Close()

		info, r, err = f.TensorReader(t.info.Name)
		if err != nil {
			return nil, err
		}
	}

	dataBytes, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}

	return decodeTensorData(info.Type, dataBytes, info.Shape)
}

func copyTensorPrefix(name string, t *sourceTensor, shape []uint64) *ggml.Tensor {
	writer := t.Clone()
	writer.shape = slices.Clone(shape)
	writer.byteLimit = gguf.TensorInfo{Name: name, Shape: shape, Type: t.info.Type}.NumBytes()
	return &ggml.Tensor{
		Name:     name,
		Kind:     uint32(t.info.Type),
		Shape:    slices.Clone(shape),
		WriterTo: writer,
	}
}

func decodeTensorData(kind gguf.TensorType, b []byte, shape []uint64) ([]float32, error) {
	switch kind {
	case gguf.TensorTypeQ8_0, gguf.TensorTypeQ5_0, gguf.TensorTypeQ4_K, gguf.TensorTypeQ6_K:
		if len(shape) == 0 {
			return nil, fmt.Errorf("decoded tensor shape is empty for %v", kind)
		}

		cols := int(shape[0])
		rows := 1
		if len(shape) > 1 {
			rows = int(shape[1])
		}
		if rows <= 0 || cols <= 0 {
			return nil, fmt.Errorf("invalid tensor shape %v for %v", shape, kind)
		}

		rowBytes := gguf.TensorInfo{Name: "row", Shape: []uint64{uint64(cols)}, Type: kind}.NumBytes()
		if rowBytes <= 0 {
			return nil, fmt.Errorf("unsupported row width %d for %v", cols, kind)
		}
		if want := int(rowBytes) * rows; len(b) != want {
			return nil, fmt.Errorf("tensor has %d bytes, expected %d for shape %v type %v", len(b), want, shape, kind)
		}

		out := make([]float32, cols*rows)
		for row := range rows {
			start := row * int(rowBytes)
			end := start + int(rowBytes)
			decoded, err := decodeTensorRow(kind, b[start:end], cols)
			if err != nil {
				return nil, err
			}
			copy(out[row*cols:], decoded)
		}
		return out, nil
	default:
		return decodeFloatTensor(kind, b)
	}
}

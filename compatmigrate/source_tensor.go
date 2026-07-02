package compatmigrate

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"reflect"
	"slices"
	"strings"
	"sync"
	"unsafe"

	"github.com/d4l3k/go-bfloat16"
	"github.com/x448/float16"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/fs/gguf"
)

func readAllSourceTensors(src *SourceModel) ([]*sourceTensor, error) {
	return readSourceTensors(src.GGUFPath, src.GGUF, src.GGUFData, src.GGUFDataOffset)
}

func readAllProjectorTensors(src *SourceModel) ([]*sourceTensor, error) {
	if src.ProjectorGGUF == nil {
		return nil, fmt.Errorf("source manifest %s has no projector layer", src.Source.DisplayShortest())
	}
	return readSourceTensors(src.ProjectorPath, src.ProjectorGGUF, src.ProjectorData, src.ProjectorDataOffset)
}

func readSourceTensors(path string, g *gguf.File, readerAt io.ReaderAt, dataOffset int64) ([]*sourceTensor, error) {
	var tensors []*sourceTensor
	for _, info := range g.TensorInfos() {
		if !info.Valid() {
			continue
		}
		tensors = append(tensors, &sourceTensor{
			path:       path,
			readerAt:   readerAt,
			dataOffset: dataOffset,
			info:       info,
			name:       info.Name,
			shape:      slices.Clone(info.Shape),
		})
	}
	return tensors, nil
}

func copyTensor(name string, t *sourceTensor) *ggml.Tensor {
	writer := t.Clone()
	outType := compatOutputTensorType(name, t.name, t.info.Type)
	if outType != t.info.Type {
		writer.SetOutputType(outType)
	}
	return &ggml.Tensor{
		Name:     name,
		Kind:     uint32(outType),
		Shape:    slices.Clone(t.shape),
		WriterTo: writer,
	}
}

type repacker func(name string, data []float32, shape []uint64) ([]float32, error)

type sourceTensor struct {
	path       string
	readerAt   io.ReaderAt
	dataOffset int64
	info       gguf.TensorInfo
	name       string
	shape      []uint64
	repacker   repacker
	byteLimit  int64
	outputType gguf.TensorType
	hasOutput  bool
}

func (t *sourceTensor) Clone() *sourceTensor {
	return &sourceTensor{
		path:       t.path,
		readerAt:   t.readerAt,
		dataOffset: t.dataOffset,
		info:       t.info,
		name:       t.name,
		shape:      slices.Clone(t.shape),
		repacker:   t.repacker,
		byteLimit:  t.byteLimit,
		outputType: t.outputType,
		hasOutput:  t.hasOutput,
	}
}

func (t *sourceTensor) SetRepacker(fn repacker) {
	t.repacker = fn
}

func (t *sourceTensor) SetOutputType(kind gguf.TensorType) {
	t.outputType = kind
	t.hasOutput = true
}

func (t *sourceTensor) encodedType() gguf.TensorType {
	if t.hasOutput {
		return t.outputType
	}
	return t.info.Type
}

func (t *sourceTensor) GGUFWriteMemoryEstimate() uint64 {
	if t.repacker == nil && !t.hasOutput {
		return uint64(compatCopyBufferSize)
	}

	sourceBytes := int64ToUint64(t.info.NumBytes())
	floatBytes := saturatingMul(tensorElementCount(t.info.Shape), 4)
	if t.repacker != nil {
		// Repackers often return a new slice while the decoded source slice is
		// still live, so account for both float buffers.
		floatBytes = saturatingMul(floatBytes, 2)
	}
	outputBytes := int64ToUint64(gguf.TensorInfo{
		Name:  t.name,
		Shape: t.shape,
		Type:  t.encodedType(),
	}.NumBytes())

	return saturatingAdd(sourceBytes, floatBytes, outputBytes)
}

func (t *sourceTensor) WriteTo(w io.Writer) (int64, error) {
	info := t.info
	var r io.Reader
	if t.readerAt != nil {
		r = io.NewSectionReader(t.readerAt, t.dataOffset+int64(t.info.Offset), t.info.NumBytes())
	} else {
		f, err := gguf.Open(t.path)
		if err != nil {
			return 0, err
		}
		defer f.Close()

		info, r, err = f.TensorReader(t.info.Name)
		if err != nil {
			return 0, err
		}
	}

	outType := t.encodedType()
	if t.repacker == nil && !t.hasOutput {
		buf := compatCopyBufferPool.Get().(*[]byte)
		defer compatCopyBufferPool.Put(buf)
		if t.byteLimit > 0 {
			return copyNBuffer(w, r, t.byteLimit, *buf)
		}
		return io.CopyBuffer(w, r, *buf)
	}

	dataBytes, err := io.ReadAll(r)
	if err != nil {
		return 0, err
	}

	data, err := decodeTensorData(info.Type, dataBytes, info.Shape)
	if err != nil {
		return 0, err
	}

	if t.repacker != nil {
		data, err = t.repacker(t.name, data, info.Shape)
		if err != nil {
			return 0, err
		}
	}

	n, err := encodeFloatTensor(w, outType, data)
	return int64(n), err
}

const compatCopyBufferSize = 1 << 20

var compatCopyBufferPool = newCompatCopyBufferPool()

func newCompatCopyBufferPool() sync.Pool {
	return sync.Pool{
		New: func() any {
			buf := make([]byte, compatCopyBufferSize)
			return &buf
		},
	}
}

func copyNBuffer(dst io.Writer, src io.Reader, n int64, buf []byte) (int64, error) {
	written, err := io.CopyBuffer(dst, io.LimitReader(src, n), buf)
	if written != n && err == nil {
		err = io.ErrUnexpectedEOF
	}
	return written, err
}

func tensorElementCount(shape []uint64) uint64 {
	var count uint64 = 1
	for _, dim := range shape {
		count = saturatingMul(count, dim)
	}
	return count
}

func int64ToUint64(n int64) uint64 {
	if n <= 0 {
		return 0
	}
	return uint64(n)
}

func saturatingAdd(values ...uint64) uint64 {
	var out uint64
	for _, value := range values {
		if out > ^uint64(0)-value {
			return ^uint64(0)
		}
		out += value
	}
	return out
}

func saturatingMul(a, b uint64) uint64 {
	if b != 0 && a > ^uint64(0)/b {
		return ^uint64(0)
	}
	return a * b
}

func compatOutputTensorType(targetName, sourceName string, sourceType gguf.TensorType) gguf.TensorType {
	if compatTensorMustBeF32(targetName) || compatTensorMustBeF32(sourceName) {
		return gguf.TensorTypeF32
	}
	return sourceType
}

func compatTensorMustBeF32(name string) bool {
	return strings.HasSuffix(name, ".ffn_gate_inp.weight") ||
		strings.HasSuffix(name, ".bias") ||
		strings.HasSuffix(name, ".shortconv.conv.weight") ||
		strings.HasSuffix(name, ".ssm_conv1d.weight") ||
		strings.HasPrefix(name, "a.conv1d.") ||
		strings.Contains(name, ".conv_dw.") ||
		name == "token_types.weight" ||
		name == "v.class_embd" ||
		name == "v.cls_embd" ||
		name == "v.positional_embedding_vlm" ||
		strings.HasPrefix(name, "v.patch_embd.weight") ||
		strings.HasPrefix(name, "v.patch_embedding.weight") ||
		name == "v.patch_conv.weight" ||
		name == "v.position_embd" ||
		name == "v.position_embd.weight" ||
		name == "v.position_embedding.weight" ||
		name == "v.tile_position_embd.weight" ||
		name == "v.pre_tile_position_embd.weight" ||
		name == "v.post_tile_position_embd.weight" ||
		name == "s.position_embd" ||
		strings.HasSuffix(name, "rel_pos_h") ||
		strings.HasSuffix(name, "rel_pos_w")
}

func decodeFloatTensor(kind gguf.TensorType, b []byte) ([]float32, error) {
	switch kind {
	case gguf.TensorTypeF32:
		out := make([]float32, len(b)/4)
		for i := range out {
			bits := binary.LittleEndian.Uint32(b[i*4:])
			out[i] = math.Float32frombits(bits)
		}
		return out, nil
	case gguf.TensorTypeF16:
		out := make([]float32, len(b)/2)
		for i := range out {
			bits := binary.LittleEndian.Uint16(b[i*2:])
			out[i] = float16.Frombits(bits).Float32()
		}
		return out, nil
	case gguf.TensorTypeBF16:
		out := make([]float32, len(b)/2)
		for i := range out {
			out[i] = bfloat16.ToFloat32(bfloat16.FromBytes(b[i*2 : i*2+2]))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("repacking tensor type %v is unsupported", kind)
	}
}

func encodeFloatTensor(w io.Writer, kind gguf.TensorType, data []float32) (int, error) {
	switch kind {
	case gguf.TensorTypeF32:
		buf := make([]byte, 4*len(data))
		for i, v := range data {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
		return w.Write(buf)
	case gguf.TensorTypeF16:
		buf := make([]byte, 2*len(data))
		for i, v := range data {
			binary.LittleEndian.PutUint16(buf[i*2:], float16.Fromfloat32(v).Bits())
		}
		return w.Write(buf)
	case gguf.TensorTypeBF16:
		return w.Write(bfloat16.EncodeFloat32(data))
	default:
		return 0, fmt.Errorf("encoding tensor type %v is unsupported", kind)
	}
}

func rawGGUFValue(v gguf.Value) any {
	rv := reflect.ValueOf(&v).Elem()
	field := rv.FieldByName("value")
	return reflect.NewAt(field.Type(), unsafe.Pointer(field.UnsafeAddr())).Elem().Interface()
}

func normalizeGGUFValue(v any) any {
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		if rv.Type().Bits() <= 32 {
			return int32(rv.Int())
		}
		return rv.Int()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		if rv.Type().Bits() <= 32 {
			return uint32(rv.Uint())
		}
		return rv.Uint()
	case reflect.Float32:
		return float32(rv.Float())
	case reflect.Float64:
		return float32(rv.Float())
	case reflect.Bool:
		return rv.Bool()
	case reflect.String:
		return rv.String()
	case reflect.Slice:
		switch rv.Type().Elem().Kind() {
		case reflect.String:
			out := make([]string, rv.Len())
			for i := range out {
				out[i] = rv.Index(i).String()
			}
			return out
		case reflect.Bool:
			out := make([]bool, rv.Len())
			for i := range out {
				out[i] = rv.Index(i).Bool()
			}
			return out
		case reflect.Int8:
			out := make([]int8, rv.Len())
			for i := range out {
				out[i] = int8(rv.Index(i).Int())
			}
			return out
		case reflect.Uint8:
			out := make([]uint8, rv.Len())
			for i := range out {
				out[i] = uint8(rv.Index(i).Uint())
			}
			return out
		case reflect.Float32, reflect.Float64:
			out := make([]float32, rv.Len())
			for i := range out {
				out[i] = float32(rv.Index(i).Convert(reflect.TypeOf(float64(0))).Float())
			}
			return out
		case reflect.Int, reflect.Int16, reflect.Int32, reflect.Int64:
			out := make([]int32, rv.Len())
			for i := range out {
				out[i] = int32(rv.Index(i).Int())
			}
			return out
		case reflect.Uint, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			out := make([]uint32, rv.Len())
			for i := range out {
				out[i] = uint32(rv.Index(i).Uint())
			}
			return out
		}
	}

	return v
}

func requiredBytesFromSource(src *SourceModel) uint64 {
	var sourceBytes uint64
	for _, layer := range src.Manifest.Layers {
		if layer.MediaType == "application/vnd.ollama.image.model" {
			sourceBytes += uint64(layer.Size)
		}
	}
	if sourceBytes == 0 {
		return compatMigrationHeadroom
	}
	return sourceBytes + sourceBytes/4 + compatMigrationHeadroom
}

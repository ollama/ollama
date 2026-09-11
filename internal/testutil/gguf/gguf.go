package gguftest

import (
	"cmp"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/gguf"
)

const (
	typeUint8 uint32 = iota
	typeInt8
	typeUint16
	typeInt16
	typeUint32
	typeInt32
	typeFloat32
	typeBool
	typeString
	typeArray
	typeUint64
	typeInt64
	typeFloat64
)

type KV map[string]any

type Tensor struct {
	Name   string
	Type   gguf.TensorType
	Shape  []uint64
	Offset uint64
	io.WriterTo
}

func (t *Tensor) Size() uint64 {
	size, _ := tensorSize(t)
	return size
}

func Write(f *os.File, kv KV, tensors []*Tensor) error {
	architecture, _ := kv["general.architecture"].(string)
	if architecture == "" {
		return fmt.Errorf("architecture not set")
	}

	if _, err := f.WriteString("GGUF"); err != nil {
		return err
	}
	for _, value := range []any{uint32(3), uint64(len(tensors)), uint64(len(kv))} {
		if err := binary.Write(f, binary.LittleEndian, value); err != nil {
			return err
		}
	}

	keys := make([]string, 0, len(kv))
	for key := range kv {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	for _, key := range keys {
		if err := writeKeyValue(f, architecture, key, kv[key]); err != nil {
			return err
		}
	}

	tensors = slices.Clone(tensors)
	slices.SortStableFunc(tensors, func(a, b *Tensor) int {
		return cmp.Or(cmp.Compare(tensorBlock(a.Name), tensorBlock(b.Name)), cmp.Compare(a.Name, b.Name))
	})

	alignment := uint64Value(kv["general.alignment"], 32)
	if alignment == 0 || alignment > math.MaxInt64 {
		return fmt.Errorf("invalid alignment %d", alignment)
	}

	var dataSize uint64
	for _, tensor := range tensors {
		tensor.Offset = dataSize
		if err := writeTensorInfo(f, tensor); err != nil {
			return err
		}
		size, err := tensorSize(tensor)
		if err != nil {
			return err
		}
		if size > math.MaxUint64-dataSize {
			return fmt.Errorf("tensor data size overflows")
		}
		dataSize += size
		padding := padding(dataSize, alignment)
		if padding > math.MaxUint64-dataSize {
			return fmt.Errorf("tensor padding overflows")
		}
		dataSize += padding
	}

	offset, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if len(tensors) == 0 {
		return nil
	}
	if err := writeZeros(f, padding(uint64(offset), alignment)); err != nil {
		return err
	}

	for i, tensor := range tensors {
		size, err := tensorSize(tensor)
		if err != nil {
			return err
		}
		if tensor.WriterTo == nil {
			if err := writeZeros(f, size); err != nil {
				return err
			}
		} else {
			n, err := tensor.WriteTo(f)
			if err != nil {
				return err
			}
			if n < 0 || uint64(n) != size {
				return fmt.Errorf("tensor %q wrote %d bytes, want %d", tensor.Name, n, size)
			}
		}
		if i+1 < len(tensors) {
			if err := writeZeros(f, padding(size, alignment)); err != nil {
				return err
			}
		}
	}

	return nil
}

func writeKeyValue(w io.Writer, architecture, key string, value any) error {
	if !strings.HasPrefix(key, architecture+".") &&
		!strings.HasPrefix(key, "general.") &&
		!strings.HasPrefix(key, "adapter.") &&
		!strings.HasPrefix(key, "split.") &&
		!strings.HasPrefix(key, "tokenizer.") {
		key = architecture + "." + key
	}
	if err := writeStringData(w, key); err != nil {
		return err
	}

	switch value := value.(type) {
	case uint8:
		return writeScalar(w, typeUint8, value)
	case int8:
		return writeScalar(w, typeInt8, value)
	case uint16:
		return writeScalar(w, typeUint16, value)
	case int16:
		return writeScalar(w, typeInt16, value)
	case uint32:
		return writeScalar(w, typeUint32, value)
	case gguf.FileType:
		return writeScalar(w, typeUint32, uint32(value))
	case int32:
		return writeScalar(w, typeInt32, value)
	case uint64:
		return writeScalar(w, typeUint64, value)
	case int64:
		return writeScalar(w, typeInt64, value)
	case float32:
		return writeScalar(w, typeFloat32, value)
	case float64:
		return writeScalar(w, typeFloat64, value)
	case bool:
		return writeScalar(w, typeBool, value)
	case string:
		if err := binary.Write(w, binary.LittleEndian, typeString); err != nil {
			return err
		}
		return writeStringData(w, value)
	case []uint8:
		return writeArray(w, typeUint8, value)
	case []int8:
		return writeArray(w, typeInt8, value)
	case []uint16:
		return writeArray(w, typeUint16, value)
	case []int16:
		return writeArray(w, typeInt16, value)
	case []uint32:
		return writeArray(w, typeUint32, value)
	case []int32:
		return writeArray(w, typeInt32, value)
	case []uint64:
		return writeArray(w, typeUint64, value)
	case []int64:
		return writeArray(w, typeInt64, value)
	case []float32:
		return writeArray(w, typeFloat32, value)
	case []float64:
		return writeArray(w, typeFloat64, value)
	case []bool:
		return writeArray(w, typeBool, value)
	case []string:
		if err := writeArrayHeader(w, typeString, len(value)); err != nil {
			return err
		}
		for _, item := range value {
			if err := writeStringData(w, item); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("unsupported metadata type %T for %q", value, key)
	}
}

func writeScalar[T any](w io.Writer, kind uint32, value T) error {
	if err := binary.Write(w, binary.LittleEndian, kind); err != nil {
		return err
	}
	return binary.Write(w, binary.LittleEndian, value)
}

func writeArray[S ~[]E, E any](w io.Writer, kind uint32, values S) error {
	if err := writeArrayHeader(w, kind, len(values)); err != nil {
		return err
	}
	return binary.Write(w, binary.LittleEndian, values)
}

func writeArrayHeader(w io.Writer, kind uint32, size int) error {
	for _, value := range []any{typeArray, kind, uint64(size)} {
		if err := binary.Write(w, binary.LittleEndian, value); err != nil {
			return err
		}
	}
	return nil
}

func writeStringData(w io.Writer, value string) error {
	if err := binary.Write(w, binary.LittleEndian, uint64(len(value))); err != nil {
		return err
	}
	_, err := io.WriteString(w, value)
	return err
}

func writeTensorInfo(w io.Writer, tensor *Tensor) error {
	if tensor == nil || tensor.Name == "" {
		return fmt.Errorf("invalid tensor")
	}
	if len(tensor.Shape) > gguf.MaxTensorDims {
		return fmt.Errorf("tensor %q has %d dimensions", tensor.Name, len(tensor.Shape))
	}
	if err := writeStringData(w, tensor.Name); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(len(tensor.Shape))); err != nil {
		return err
	}
	for _, dimension := range tensor.Shape {
		if err := binary.Write(w, binary.LittleEndian, dimension); err != nil {
			return err
		}
	}
	for _, value := range []any{uint32(tensor.Type), tensor.Offset} {
		if err := binary.Write(w, binary.LittleEndian, value); err != nil {
			return err
		}
	}
	return nil
}

func tensorSize(tensor *Tensor) (uint64, error) {
	info := gguf.TensorInfo{Name: tensor.Name, Shape: tensor.Shape, Type: tensor.Type}
	size := info.NumBytes()
	if size < 0 {
		return 0, fmt.Errorf("tensor %q size overflows", tensor.Name)
	}
	return uint64(size), nil
}

func tensorBlock(name string) int {
	var block int
	if _, err := fmt.Sscanf(name, "blk.%d.", &block); err != nil {
		return math.MaxInt
	}
	return block
}

func uint64Value(value any, defaultValue uint64) uint64 {
	switch value := value.(type) {
	case uint8:
		return uint64(value)
	case uint16:
		return uint64(value)
	case uint32:
		return uint64(value)
	case uint64:
		return value
	default:
		return defaultValue
	}
}

func padding(offset, alignment uint64) uint64 {
	return (alignment - offset%alignment) % alignment
}

func writeZeros(w io.Writer, size uint64) error {
	var zeros [4096]byte
	for size > 0 {
		n := min(size, uint64(len(zeros)))
		written, err := w.Write(zeros[:n])
		if err != nil {
			return err
		}
		if written == 0 {
			return io.ErrShortWrite
		}
		size -= uint64(written)
	}
	return nil
}

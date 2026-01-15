package ggml

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"runtime"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs"
	"golang.org/x/sync/errgroup"
)

type containerGGUF struct {
	ByteOrder binary.ByteOrder

	Version uint32

	V1 struct {
		NumTensor uint32
		NumKV     uint32
	}

	V2 struct {
		NumTensor uint64
		NumKV     uint64
	}

	V3 struct {
		NumTensor uint64
		NumKV     uint64
	}

	maxArraySize int
}

func (c *containerGGUF) Name() string {
	return "gguf"
}

func (c *containerGGUF) Decode(rs io.ReadSeeker) (model, error) {
	if err := binary.Read(rs, c.ByteOrder, &c.Version); err != nil {
		return nil, err
	}

	var err error
	switch c.Version {
	case 1:
		err = binary.Read(rs, c.ByteOrder, &c.V1)
	case 2:
		err = binary.Read(rs, c.ByteOrder, &c.V2)
	default:
		err = binary.Read(rs, c.ByteOrder, &c.V3)
	}
	if err != nil {
		return nil, err
	}

	model := newGGUF(c)
	if err := model.Decode(rs); err != nil {
		return nil, err
	}

	return model, nil
}

const (
	ggufTypeUint8 uint32 = iota
	ggufTypeInt8
	ggufTypeUint16
	ggufTypeInt16
	ggufTypeUint32
	ggufTypeInt32
	ggufTypeFloat32
	ggufTypeBool
	ggufTypeString
	ggufTypeArray
	ggufTypeUint64
	ggufTypeInt64
	ggufTypeFloat64
)

type gguf struct {
	*containerGGUF

	kv      KV
	tensors []*Tensor

	parameters   uint64
	tensorOffset uint64

	scratch [16 << 10]byte
}

func newGGUF(container *containerGGUF) *gguf {
	return &gguf{
		containerGGUF: container,
		kv:            make(KV),
	}
}

func (llm *gguf) KV() KV {
	return llm.kv
}

func (llm *gguf) Tensors() Tensors {
	return Tensors{
		items:  llm.tensors,
		Offset: llm.tensorOffset,
	}
}

func (llm *gguf) numTensor() uint64 {
	switch llm.Version {
	case 1:
		return uint64(llm.V1.NumTensor)
	case 2:
		return llm.V2.NumTensor
	default:
		return llm.V3.NumTensor
	}
}

func (llm *gguf) numKV() uint64 {
	switch llm.Version {
	case 1:
		return uint64(llm.V1.NumKV)
	case 2:
		return llm.V2.NumKV
	default:
		return llm.V3.NumKV
	}
}

func (llm *gguf) Decode(rs io.ReadSeeker) error {
	// decode key-values
	for i := 0; uint64(i) < llm.numKV(); i++ {
		k, err := readGGUFString(llm, rs)
		if err != nil {
			return err
		}

		t, err := readGGUF[uint32](llm, rs)
		if err != nil {
			return err
		}

		var v any
		switch t {
		case ggufTypeUint8:
			v, err = readGGUF[uint8](llm, rs)
		case ggufTypeInt8:
			v, err = readGGUF[int8](llm, rs)
		case ggufTypeUint16:
			v, err = readGGUF[uint16](llm, rs)
		case ggufTypeInt16:
			v, err = readGGUF[int16](llm, rs)
		case ggufTypeUint32:
			v, err = readGGUF[uint32](llm, rs)
		case ggufTypeInt32:
			v, err = readGGUF[int32](llm, rs)
		case ggufTypeUint64:
			v, err = readGGUF[uint64](llm, rs)
		case ggufTypeInt64:
			v, err = readGGUF[int64](llm, rs)
		case ggufTypeFloat32:
			v, err = readGGUF[float32](llm, rs)
		case ggufTypeFloat64:
			v, err = readGGUF[float64](llm, rs)
		case ggufTypeBool:
			v, err = readGGUF[bool](llm, rs)
		case ggufTypeString:
			v, err = readGGUFString(llm, rs)
		case ggufTypeArray:
			v, err = readGGUFArray(llm, rs)
		default:
			return fmt.Errorf("invalid type: %d", t)
		}

		if err != nil {
			return err
		}

		llm.kv[k] = v
	}

	// decode tensors
	for range llm.numTensor() {
		name, err := readGGUFString(llm, rs)
		if err != nil {
			return fmt.Errorf("failed to read tensor name: %w", err)
		}

		// dims is the number of dimensions in the tensor
		dims, err := readGGUF[uint32](llm, rs)
		if err != nil {
			return fmt.Errorf("failed to read tensor dimensions: %w", err)
		}

		shape := make([]uint64, dims)
		for i := 0; uint32(i) < dims; i++ {
			shape[i], err = readGGUF[uint64](llm, rs)
			if err != nil {
				return fmt.Errorf("failed to read tensor shape: %w", err)
			}
		}

		kind, err := readGGUF[uint32](llm, rs)
		if err != nil {
			return fmt.Errorf("failed to read tensor kind: %w", err)
		}

		offset, err := readGGUF[uint64](llm, rs)
		if err != nil {
			return fmt.Errorf("failed to read tensor offset: %w", err)
		}

		tensor := Tensor{
			Name:   name,
			Kind:   kind,
			Offset: offset,
			Shape:  shape[:],
		}

		llm.tensors = append(llm.tensors, &tensor)
		llm.parameters += tensor.Elements()
	}

	// patch KV with parameter count
	llm.kv["general.parameter_count"] = llm.parameters

	alignment := llm.kv.Uint("general.alignment", 32)

	offset, err := rs.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}

	padding := ggufPadding(offset, int64(alignment))
	llm.tensorOffset = uint64(offset + padding)

	for _, tensor := range llm.tensors {
		offset, err := rs.Seek(0, io.SeekCurrent)
		if err != nil {
			return fmt.Errorf("failed to get current offset: %w", err)
		}

		padding := ggufPadding(offset, int64(alignment))
		if _, err := rs.Seek(padding, io.SeekCurrent); err != nil {
			return fmt.Errorf("failed to seek to init padding: %w", err)
		}

		if _, err := rs.Seek(int64(tensor.Size()), io.SeekCurrent); err != nil {
			return fmt.Errorf("failed to seek to tensor: %w", err)
		}
	}

	return nil
}

func readGGUF[T any](llm *gguf, r io.Reader) (T, error) {
	var t T
	err := binary.Read(r, llm.ByteOrder, &t)
	return t, err
}

func writeGGUF[V any](w io.Writer, t uint32, v V) error {
	if err := binary.Write(w, binary.LittleEndian, t); err != nil {
		return err
	}

	return binary.Write(w, binary.LittleEndian, v)
}

func readGGUFV1String(llm *gguf, r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, llm.ByteOrder, &length); err != nil {
		return "", err
	}

	var b bytes.Buffer
	if _, err := io.CopyN(&b, r, int64(length)); err != nil {
		return "", err
	}

	// gguf v1 strings are null-terminated
	b.Truncate(b.Len() - 1)

	return b.String(), nil
}

func readGGUFV1StringsData(llm *gguf, r io.Reader, a *array[string]) (any, error) {
	for i := range a.size {
		if a.values != nil {
			e, err := readGGUFV1String(llm, r)
			if err != nil {
				return nil, err
			}

			a.values[i] = e
		} else {
			_ = discardGGUFString(llm, r)
		}
	}

	return a, nil
}

func discardGGUFString(llm *gguf, r io.Reader) error {
	buf := llm.scratch[:8]
	_, err := io.ReadFull(r, buf)
	if err != nil {
		return err
	}

	size := int(llm.ByteOrder.Uint64(buf))
	for size > 0 {
		n, err := r.Read(llm.scratch[:min(size, cap(llm.scratch))])
		if err != nil {
			return err
		}
		size -= n
	}
	return nil
}

func readGGUFString(llm *gguf, r io.Reader) (string, error) {
	if llm.Version == 1 {
		return readGGUFV1String(llm, r)
	}

	buf := llm.scratch[:8]
	_, err := io.ReadFull(r, buf)
	if err != nil {
		return "", err
	}

	length := int(llm.ByteOrder.Uint64(buf))
	if length > len(llm.scratch) {
		buf = make([]byte, length)
	} else {
		buf = llm.scratch[:length]
	}
	clear(buf)

	_, err = io.ReadFull(r, buf)
	if err != nil {
		return "", err
	}
	return string(buf), nil
}

func writeGGUFString(w io.Writer, s string) error {
	if err := binary.Write(w, binary.LittleEndian, ggufTypeString); err != nil {
		return err
	}

	if err := binary.Write(w, binary.LittleEndian, uint64(len(s))); err != nil {
		return err
	}

	_, err := io.Copy(w, strings.NewReader(s))
	return err
}

func readGGUFStringsData(llm *gguf, r io.Reader, a *array[string]) (any, error) {
	for i := range a.size {
		if a.values != nil {
			e, err := readGGUFString(llm, r)
			if err != nil {
				return nil, err
			}

			a.values[i] = e
		} else {
			discardGGUFString(llm, r)
		}
	}

	return a, nil
}

type array[T any] struct {
	// size is the actual size of the array
	size int

	// values is the array of values. this is nil if the array is larger than configured maxSize
	values []T
}

func (a *array[T]) MarshalJSON() ([]byte, error) {
	return json.Marshal(a.values)
}

func newArray[T any](size, maxSize int) *array[T] {
	a := array[T]{size: size}
	if maxSize < 0 || size <= maxSize {
		a.values = make([]T, size)
	}
	return &a
}

func readGGUFArray(llm *gguf, r io.Reader) (any, error) {
	t, err := readGGUF[uint32](llm, r)
	if err != nil {
		return nil, err
	}

	n, err := readGGUF[uint64](llm, r)
	if err != nil {
		return nil, err
	}

	switch t {
	case ggufTypeUint8:
		a := newArray[uint8](int(n), llm.maxArraySize)
		return readGGUFArrayData(llm, r, a)
	case ggufTypeInt8:
		a := newArray[int8](int(n), llm.maxArraySize)
		return readGGUFArrayData(llm, r, a)
	case ggufTypeUint16:
		a := newArray[uint16](int(n), llm.maxArraySize)
		return readGGUFArrayData(llm, r, a)
	case ggufTypeInt16:
		a := newArray[int16](int(n), llm.maxArraySize)
		return readGGUFArrayData(llm, r, a)
	case ggufTypeUint32:
		a := newArray[uint32](int(n), llm.maxArraySize)
		return readGGUFArrayData(llm, r, a)
	case ggufTypeInt32:
		a := newArray[int32](int(n), llm.maxArraySize)
		return readGGUFArrayData(llm, r, a)
	case ggufTypeUint64:
		a := newArray[uint64](int(n), llm.maxArraySize)
		return readGGUFArrayData(llm, r, a)
	case ggufTypeInt64:
		a := newArray[int64](int(n), llm.maxArraySize)
		return readGGUFArrayData(llm, r, a)
	case ggufTypeFloat32:
		a := newArray[float32](int(n), llm.maxArraySize)
		return readGGUFArrayData(llm, r, a)
	case ggufTypeFloat64:
		a := newArray[float64](int(n), llm.maxArraySize)
		return readGGUFArrayData(llm, r, a)
	case ggufTypeBool:
		a := newArray[bool](int(n), llm.maxArraySize)
		return readGGUFArrayData(llm, r, a)
	case ggufTypeString:
		a := newArray[string](int(n), llm.maxArraySize)
		if llm.Version == 1 {
			return readGGUFV1StringsData(llm, r, a)
		}

		return readGGUFStringsData(llm, r, a)
	default:
		return nil, fmt.Errorf("invalid array type: %d", t)
	}
}

func readGGUFArrayData[T any](llm *gguf, r io.Reader, a *array[T]) (any, error) {
	for i := range a.size {
		e, err := readGGUF[T](llm, r)
		if err != nil {
			return nil, err
		}

		if a.values != nil {
			a.values[i] = e
		}
	}

	return a, nil
}

// writeGGUFArray writes a slice s of type E to the write with a gguf type of t
func writeGGUFArray[S ~[]E, E any](w io.Writer, t uint32, s S) error {
	if err := binary.Write(w, binary.LittleEndian, ggufTypeArray); err != nil {
		return err
	}

	if err := binary.Write(w, binary.LittleEndian, t); err != nil {
		return err
	}

	if err := binary.Write(w, binary.LittleEndian, uint64(len(s))); err != nil {
		return err
	}

	if t == ggufTypeString {
		for _, e := range any(s).([]string) {
			if err := binary.Write(w, binary.LittleEndian, uint64(len(e))); err != nil {
				return err
			}

			if err := binary.Write(w, binary.LittleEndian, []byte(e)); err != nil {
				return err
			}
		}
		return nil
	}

	return binary.Write(w, binary.LittleEndian, s)
}

func WriteGGUF(f *os.File, kv fs.Config, ts []*Tensor) error {
	arch := kv.String("general.architecture")
	if arch == "" {
		return fmt.Errorf("architecture not set")
	}

	if err := binary.Write(f, binary.LittleEndian, []byte("GGUF")); err != nil {
		return err
	}

	if err := binary.Write(f, binary.LittleEndian, uint32(3)); err != nil {
		return err
	}

	if err := binary.Write(f, binary.LittleEndian, uint64(len(ts))); err != nil {
		return err
	}

	if err := binary.Write(f, binary.LittleEndian, uint64(kv.Len())); err != nil {
		return err
	}

	for _, key := range slices.Sorted(kv.Keys()) {
		if err := ggufWriteKV(f, arch, key, kv.Value(key)); err != nil {
			return err
		}
	}

	slices.SortStableFunc(
		ts,
		func(a, b *Tensor) int {
			return cmp.Or(
				cmp.Compare(a.block(), b.block()),
				cmp.Compare(a.Name, b.Name),
			)
		},
	)

	alignment := kv.Uint("general.alignment", 32)

	var s uint64
	for i := range ts {
		ts[i].Offset = s
		if err := ggufWriteTensorInfo(f, ts[i]); err != nil {
			return err
		}
		s += ts[i].Size()
		s += uint64(ggufPadding(int64(s), int64(alignment)))
	}

	offset, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	offset += ggufPadding(offset, int64(alignment))

	var g errgroup.Group
	g.SetLimit(runtime.GOMAXPROCS(0))
	// TODO consider reducing if tensors size * gomaxprocs is larger than free memory
	for _, t := range ts {
		w := io.NewOffsetWriter(f, offset+int64(t.Offset))
		g.Go(func() error {
			_, err := t.WriteTo(w)
			return err
		})
	}

	return g.Wait()
}

func ggufWriteKV(ws io.WriteSeeker, arch, k string, v any) error {
	if !strings.HasPrefix(k, arch+".") &&
		!strings.HasPrefix(k, "general.") &&
		!strings.HasPrefix(k, "adapter.") &&
		!strings.HasPrefix(k, "tokenizer.") {
		k = arch + "." + k
	}

	slog.Debug(k, "type", fmt.Sprintf("%T", v))
	if err := binary.Write(ws, binary.LittleEndian, uint64(len(k))); err != nil {
		return err
	}

	if err := binary.Write(ws, binary.LittleEndian, []byte(k)); err != nil {
		return err
	}

	var err error
	switch v := v.(type) {
	case int32:
		err = writeGGUF(ws, ggufTypeInt32, v)
	case int64:
		err = writeGGUF(ws, ggufTypeInt64, v)
	case uint32, FileType:
		err = writeGGUF(ws, ggufTypeUint32, v)
	case uint64:
		err = writeGGUF(ws, ggufTypeUint64, v)
	case float32:
		err = writeGGUF(ws, ggufTypeFloat32, v)
	case bool:
		err = writeGGUF(ws, ggufTypeBool, v)
	case string:
		err = writeGGUFString(ws, v)
	case []int32:
		err = writeGGUFArray(ws, ggufTypeInt32, v)
	case *array[int32]:
		err = writeGGUFArray(ws, ggufTypeInt32, v.values)
	case []int64:
		err = writeGGUFArray(ws, ggufTypeInt64, v)
	case *array[int64]:
		err = writeGGUFArray(ws, ggufTypeInt64, v.values)
	case []uint32:
		err = writeGGUFArray(ws, ggufTypeUint32, v)
	case *array[uint32]:
		err = writeGGUFArray(ws, ggufTypeUint32, v.values)
	case []float32:
		err = writeGGUFArray(ws, ggufTypeFloat32, v)
	case *array[float32]:
		err = writeGGUFArray(ws, ggufTypeFloat32, v.values)
	case []string:
		err = writeGGUFArray(ws, ggufTypeString, v)
	case *array[string]:
		err = writeGGUFArray(ws, ggufTypeString, v.values)
	case []bool:
		err = writeGGUFArray(ws, ggufTypeBool, v)
	case *array[bool]:
		err = writeGGUFArray(ws, ggufTypeBool, v.values)
	default:
		return fmt.Errorf("improper type for '%s'", k)
	}

	return err
}

func ggufWriteTensorInfo(ws io.WriteSeeker, t *Tensor) error {
	slog.Debug(t.Name, "kind", t.Kind, "shape", t.Shape, "offset", t.Offset)
	if err := binary.Write(ws, binary.LittleEndian, uint64(len(t.Name))); err != nil {
		return err
	}

	if err := binary.Write(ws, binary.LittleEndian, []byte(t.Name)); err != nil {
		return err
	}

	if err := binary.Write(ws, binary.LittleEndian, uint32(len(t.Shape))); err != nil {
		return err
	}

	for _, n := range t.Shape {
		if err := binary.Write(ws, binary.LittleEndian, n); err != nil {
			return err
		}
	}

	if err := binary.Write(ws, binary.LittleEndian, t.Kind); err != nil {
		return err
	}

	return binary.Write(ws, binary.LittleEndian, t.Offset)
}

func ggufPadding(offset, align int64) int64 {
	return (align - offset%align) % align
}

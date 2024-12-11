package llm

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"slices"
	"strings"

	"golang.org/x/exp/maps"
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

func (c *containerGGUF) canCollectArray(size int) bool {
	return c.maxArraySize < 0 || size <= c.maxArraySize
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

func (llm *gguf) Tensors() *Tensors {
	return &Tensors{
		Items:  llm.tensors,
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
		llm.parameters += tensor.parameters()
	}

	// patch KV with parameter count
	llm.kv["general.parameter_count"] = llm.parameters

	alignment, ok := llm.kv["general.alignment"].(uint32)
	if !ok {
		alignment = 32
	}

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

type array struct {
	size   int
	values []any
}

func (a *array) MarshalJSON() ([]byte, error) {
	return json.Marshal(a.values)
}

func readGGUFV1Array(llm *gguf, r io.Reader) (*array, error) {
	t, err := readGGUF[uint32](llm, r)
	if err != nil {
		return nil, err
	}

	n, err := readGGUF[uint32](llm, r)
	if err != nil {
		return nil, err
	}

	a := &array{size: int(n)}
	if llm.canCollectArray(int(n)) {
		a.values = make([]any, 0, int(n))
	}

	for i := range n {
		var e any
		switch t {
		case ggufTypeUint8:
			e, err = readGGUF[uint8](llm, r)
		case ggufTypeInt8:
			e, err = readGGUF[int8](llm, r)
		case ggufTypeUint16:
			e, err = readGGUF[uint16](llm, r)
		case ggufTypeInt16:
			e, err = readGGUF[int16](llm, r)
		case ggufTypeUint32:
			e, err = readGGUF[uint32](llm, r)
		case ggufTypeInt32:
			e, err = readGGUF[int32](llm, r)
		case ggufTypeUint64:
			e, err = readGGUF[uint64](llm, r)
		case ggufTypeInt64:
			e, err = readGGUF[int64](llm, r)
		case ggufTypeFloat32:
			e, err = readGGUF[float32](llm, r)
		case ggufTypeFloat64:
			e, err = readGGUF[float64](llm, r)
		case ggufTypeBool:
			e, err = readGGUF[bool](llm, r)
		case ggufTypeString:
			e, err = readGGUFV1String(llm, r)
		default:
			return nil, fmt.Errorf("invalid array type: %d", t)
		}
		if err != nil {
			return nil, err
		}

		if a.values != nil {
			a.values[i] = e
		}
	}

	return a, nil
}

func readGGUFArray(llm *gguf, r io.Reader) (*array, error) {
	if llm.Version == 1 {
		return readGGUFV1Array(llm, r)
	}

	t, err := readGGUF[uint32](llm, r)
	if err != nil {
		return nil, err
	}

	n, err := readGGUF[uint64](llm, r)
	if err != nil {
		return nil, err
	}

	a := &array{size: int(n)}
	if llm.canCollectArray(int(n)) {
		a.values = make([]any, int(n))
	}

	for i := range n {
		var e any
		switch t {
		case ggufTypeUint8:
			e, err = readGGUF[uint8](llm, r)
		case ggufTypeInt8:
			e, err = readGGUF[int8](llm, r)
		case ggufTypeUint16:
			e, err = readGGUF[uint16](llm, r)
		case ggufTypeInt16:
			e, err = readGGUF[int16](llm, r)
		case ggufTypeUint32:
			e, err = readGGUF[uint32](llm, r)
		case ggufTypeInt32:
			e, err = readGGUF[int32](llm, r)
		case ggufTypeUint64:
			e, err = readGGUF[uint64](llm, r)
		case ggufTypeInt64:
			e, err = readGGUF[int64](llm, r)
		case ggufTypeFloat32:
			e, err = readGGUF[float32](llm, r)
		case ggufTypeFloat64:
			e, err = readGGUF[float64](llm, r)
		case ggufTypeBool:
			e, err = readGGUF[bool](llm, r)
		case ggufTypeString:
			if a.values != nil {
				e, err = readGGUFString(llm, r)
			} else {
				err = discardGGUFString(llm, r)
			}
		default:
			return nil, fmt.Errorf("invalid array type: %d", t)
		}
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

	return binary.Write(w, binary.LittleEndian, s)
}

func WriteGGUF(ws io.WriteSeeker, kv KV, ts []Tensor) error {
	if err := binary.Write(ws, binary.LittleEndian, []byte("GGUF")); err != nil {
		return err
	}

	if err := binary.Write(ws, binary.LittleEndian, uint32(3)); err != nil {
		return err
	}

	if err := binary.Write(ws, binary.LittleEndian, uint64(len(ts))); err != nil {
		return err
	}

	if err := binary.Write(ws, binary.LittleEndian, uint64(len(kv))); err != nil {
		return err
	}

	keys := maps.Keys(kv)
	slices.Sort(keys)

	for _, key := range keys {
		if err := ggufWriteKV(ws, key, kv[key]); err != nil {
			return err
		}
	}

	slices.SortStableFunc(ts, func(a, b Tensor) int {
		if i, j := a.block(), b.block(); i < 0 && j > 0 {
			return 1
		} else if i > 0 && j < 0 {
			return -1
		} else {
			return cmp.Compare(i, j)
		}
	})

	var s uint64
	for _, t := range ts {
		t.Offset = s
		if err := ggufWriteTensorInfo(ws, t); err != nil {
			return err
		}
		s += t.Size()
	}

	var alignment int64 = 32
	for _, t := range ts {
		if err := ggufWriteTensor(ws, t, alignment); err != nil {
			return err
		}
	}

	offset, err := ws.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if err := binary.Write(ws, binary.LittleEndian, bytes.Repeat([]byte{0}, int(ggufPadding(offset, alignment)))); err != nil {
		return err
	}

	return nil
}

func ggufWriteKV(ws io.WriteSeeker, k string, v any) error {
	slog.Debug(k, "type", fmt.Sprintf("%T", v))
	if err := binary.Write(ws, binary.LittleEndian, uint64(len(k))); err != nil {
		return err
	}

	if err := binary.Write(ws, binary.LittleEndian, []byte(k)); err != nil {
		return err
	}

	var err error
	switch v := v.(type) {
	case uint32:
		err = writeGGUF(ws, ggufTypeUint32, v)
	case float32:
		err = writeGGUF(ws, ggufTypeFloat32, v)
	case bool:
		err = writeGGUF(ws, ggufTypeBool, v)
	case string:
		err = writeGGUFString(ws, v)
	case []int32:
		err = writeGGUFArray(ws, ggufTypeInt32, v)
	case []uint32:
		err = writeGGUFArray(ws, ggufTypeUint32, v)
	case []float32:
		err = writeGGUFArray(ws, ggufTypeFloat32, v)
	case []string:
		if err := binary.Write(ws, binary.LittleEndian, ggufTypeArray); err != nil {
			return err
		}

		if err := binary.Write(ws, binary.LittleEndian, ggufTypeString); err != nil {
			return err
		}

		if err := binary.Write(ws, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}

		for _, e := range v {
			if err := binary.Write(ws, binary.LittleEndian, uint64(len(e))); err != nil {
				return err
			}

			if err := binary.Write(ws, binary.LittleEndian, []byte(e)); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("improper type for '%s'", k)
	}

	return err
}

func ggufWriteTensorInfo(ws io.WriteSeeker, t Tensor) error {
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

	for i := range len(t.Shape) {
		if err := binary.Write(ws, binary.LittleEndian, t.Shape[len(t.Shape)-i-1]); err != nil {
			return err
		}
	}

	if err := binary.Write(ws, binary.LittleEndian, t.Kind); err != nil {
		return err
	}

	return binary.Write(ws, binary.LittleEndian, t.Offset)
}

func ggufWriteTensor(ws io.WriteSeeker, t Tensor, alignment int64) error {
	offset, err := ws.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}

	if err := binary.Write(ws, binary.LittleEndian, bytes.Repeat([]byte{0}, int(ggufPadding(offset, alignment)))); err != nil {
		return err
	}

	_, err = t.WriteTo(ws)
	return err
}

func ggufPadding(offset, align int64) int64 {
	return (align - offset%align) % align
}

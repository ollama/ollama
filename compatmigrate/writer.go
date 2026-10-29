package compatmigrate

// Minimal serial GGUF writer for one-shot compatibility migrations; kept
// unexported so GGUF writing stays local to this package.

import (
	"cmp"
	"encoding/binary"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/gguf"
)

// outKV is the metadata of the GGUF file a migration is writing.
type outKV map[string]any

func (kv outKV) Len() int {
	return len(kv)
}

// Keys returns the metadata keys in sorted order, matching what the reader
// expects of well-formed GGUF files.
func (kv outKV) Keys() []string {
	return slices.Sorted(maps.Keys(kv))
}

func (kv outKV) Value(key string) any {
	return kv[key]
}

func (kv outKV) String(key string) string {
	v, _ := kv[key].(string)
	return v
}

func (kv outKV) Uint(key string, fallback uint64) uint64 {
	switch v := kv[key].(type) {
	case uint32:
		return uint64(v)
	case uint64:
		return v
	}
	return fallback
}

// outTensor is one tensor of the GGUF file a migration is writing. WriterTo
// provides the tensor payload.
type outTensor struct {
	Name   string
	Kind   uint32
	Offset uint64

	// Shape is the number of elements in each dimension
	Shape []uint64

	io.WriterTo
}

// block returns the tensor's layer index, so tensors sort with their layer.
// Tensors outside any layer sort after them.
func (t *outTensor) block() (n int) {
	if _, err := fmt.Sscanf(t.Name, "blk.%d.", &n); err != nil {
		return math.MaxInt
	}

	return
}

// size returns the tensor's size in bytes, padding the final block out to a
// whole block.
func (t *outTensor) size() uint64 {
	var elements uint64 = 1
	for _, n := range t.Shape {
		elements *= n
	}
	return uint64(float64(elements) * gguf.TensorType(t.Kind).NumBytes())
}

// writeGGUF writes kv and tensors to f as a GGUF v3 file.
func writeGGUF(f *os.File, kv outKV, ts []*outTensor) error {
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

	for _, key := range kv.Keys() {
		if err := writeKV(f, arch, key, kv.Value(key)); err != nil {
			return err
		}
	}

	slices.SortStableFunc(
		ts,
		func(a, b *outTensor) int {
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
		if err := writeTensorInfo(f, ts[i]); err != nil {
			return err
		}
		s += ts[i].size()
		s += uint64(padding(int64(s), int64(alignment)))
	}

	offset, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	offset += padding(offset, int64(alignment))

	for _, t := range ts {
		if _, err := t.WriteTo(io.NewOffsetWriter(f, offset+int64(t.Offset))); err != nil {
			return err
		}
	}

	return nil
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

func writeKV(w io.Writer, arch, k string, v any) error {
	if !strings.HasPrefix(k, arch+".") &&
		!strings.HasPrefix(k, "general.") &&
		!strings.HasPrefix(k, "adapter.") &&
		!strings.HasPrefix(k, "tokenizer.") {
		k = arch + "." + k
	}

	if err := binary.Write(w, binary.LittleEndian, uint64(len(k))); err != nil {
		return err
	}

	if err := binary.Write(w, binary.LittleEndian, []byte(k)); err != nil {
		return err
	}

	switch v := v.(type) {
	case int32:
		return writeValue(w, ggufTypeInt32, v)
	case int64:
		return writeValue(w, ggufTypeInt64, v)
	case uint32:
		return writeValue(w, ggufTypeUint32, v)
	case uint64:
		return writeValue(w, ggufTypeUint64, v)
	case float32:
		return writeValue(w, ggufTypeFloat32, v)
	case float64:
		return writeValue(w, ggufTypeFloat64, v)
	case bool:
		return writeValue(w, ggufTypeBool, v)
	case string:
		return writeString(w, v)
	case []int8:
		return writeArray(w, ggufTypeInt8, v)
	case []uint8:
		return writeArray(w, ggufTypeUint8, v)
	case []int32:
		return writeArray(w, ggufTypeInt32, v)
	case []int64:
		return writeArray(w, ggufTypeInt64, v)
	case []uint32:
		return writeArray(w, ggufTypeUint32, v)
	case []uint64:
		return writeArray(w, ggufTypeUint64, v)
	case []float32:
		return writeArray(w, ggufTypeFloat32, v)
	case []string:
		return writeArray(w, ggufTypeString, v)
	case []bool:
		return writeArray(w, ggufTypeBool, v)
	default:
		return fmt.Errorf("improper type for '%s'", k)
	}
}

func writeTensorInfo(w io.Writer, t *outTensor) error {
	if err := binary.Write(w, binary.LittleEndian, uint64(len(t.Name))); err != nil {
		return err
	}

	if err := binary.Write(w, binary.LittleEndian, []byte(t.Name)); err != nil {
		return err
	}

	if err := binary.Write(w, binary.LittleEndian, uint32(len(t.Shape))); err != nil {
		return err
	}

	for _, n := range t.Shape {
		if err := binary.Write(w, binary.LittleEndian, n); err != nil {
			return err
		}
	}

	if err := binary.Write(w, binary.LittleEndian, t.Kind); err != nil {
		return err
	}

	return binary.Write(w, binary.LittleEndian, t.Offset)
}

func writeValue[V any](w io.Writer, t uint32, v V) error {
	if err := binary.Write(w, binary.LittleEndian, t); err != nil {
		return err
	}

	return binary.Write(w, binary.LittleEndian, v)
}

func writeString(w io.Writer, s string) error {
	if err := binary.Write(w, binary.LittleEndian, ggufTypeString); err != nil {
		return err
	}

	if err := binary.Write(w, binary.LittleEndian, uint64(len(s))); err != nil {
		return err
	}

	_, err := io.Copy(w, strings.NewReader(s))
	return err
}

func writeArray[S ~[]E, E any](w io.Writer, t uint32, s S) error {
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

func padding(offset, align int64) int64 {
	return (align - offset%align) % align
}

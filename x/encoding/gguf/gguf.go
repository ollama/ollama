package gguf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/ollama/ollama/x/types/structs"
)

// TODO(bmizerany): determine  a more reasonable value for MaxDimensions.

// MaxDimensions is the maximum number of dimensions a tensor can have.
const MaxDimensions uint32 = 1e6

// Errors
var (
	// ErrBadMagic is returned when the magic bytes at the start of the
	// file. This is useful for detecting if the file is not a gguf
	// file.
	ErrBadMagic = errors.New("gguf: bad magic")

	ErrUnsupportedVersion = errors.New("gguf: unsupported version")
	ErrMangled            = errors.New("gguf: mangled data")
)

type Type uint32

const (
	TypeF32   Type = 0
	TypeF16   Type = 1
	TypeQ4_0  Type = 2
	TypeQ4_1  Type = 3
	TypeQ5_0  Type = 6
	TypeQ5_1  Type = 7
	TypeQ8_0  Type = 8
	TypeQ8_1  Type = 9
	TypeQ2_K  Type = 10
	TypeQ3_K  Type = 11
	TypeQ4_K  Type = 12
	TypeQ5_K  Type = 13
	TypeQ6_K  Type = 14
	TypeQ8_K  Type = 15
	TypeI8    Type = 16
	TypeI16   Type = 17
	TypeI32   Type = 18
	TypeCount Type = 19
)

var typeNames = map[Type]string{
	TypeF32:   "F32",
	TypeF16:   "F16",
	TypeQ4_0:  "Q4_0",
	TypeQ4_1:  "Q4_1",
	TypeQ5_0:  "Q5_0",
	TypeQ5_1:  "Q5_1",
	TypeQ8_0:  "Q8_0",
	TypeQ8_1:  "Q8_1",
	TypeQ2_K:  "Q2_K",
	TypeQ3_K:  "Q3_K",
	TypeQ4_K:  "Q4_K",
	TypeQ5_K:  "Q5_K",
	TypeQ6_K:  "Q6_K",
	TypeQ8_K:  "Q8_K",
	TypeI8:    "I8",
	TypeI16:   "I16",
	TypeI32:   "I32",
	TypeCount: "COUNT",
}

func (t Type) String() string {
	if name := typeNames[t]; name != "" {
		return name
	}
	return fmt.Sprintf("(!unknown_type %d!)", t)
}

// ValueType is the type of a metadata value.
type ValueType uint32

func (t ValueType) String() string {
	if name := metaTypeNames[t]; name != "" {
		return name
	}
	return fmt.Sprintf("(!unknown_value_type %d!)", t)
}

const (
	ValueTypeUint8   ValueType = 0
	ValueTypeInt8    ValueType = 1
	ValueTypeUint16  ValueType = 2
	ValueTypeInt16   ValueType = 3
	ValueTypeUint32  ValueType = 4
	ValueTypeInt32   ValueType = 5
	ValueTypeFloat32 ValueType = 6
	ValueTypeBool    ValueType = 7
	ValueTypeString  ValueType = 8
	ValueTypeArray   ValueType = 9
	ValueTypeUint64  ValueType = 10
	ValueTypeInt64   ValueType = 11
	ValueTypeFloat64 ValueType = 12
)

var metaTypeNames = map[ValueType]string{
	ValueTypeUint8:   "uint8",
	ValueTypeInt8:    "int8",
	ValueTypeUint16:  "uint16",
	ValueTypeInt16:   "int16",
	ValueTypeUint32:  "uint32",
	ValueTypeInt32:   "int32",
	ValueTypeFloat32: "float32",
	ValueTypeBool:    "bool",
	ValueTypeString:  "string",
	ValueTypeArray:   "array",
	ValueTypeUint64:  "uint64",
	ValueTypeInt64:   "int64",
	ValueTypeFloat64: "float64",
}

type TensorInfo struct {
	Name       string
	Dimensions []uint64
	Type       Type
	Offset     uint64
	Size       uint64
}

type MetaValue struct {
	Type  ValueType
	Value []byte
}

func (v MetaValue) String() string {
	var b strings.Builder
	b.WriteString(v.Type.String())
	b.WriteString("(")
	switch v.Type {
	case ValueTypeArray:
		b.WriteString("[...]")
	case ValueTypeString:
		b.WriteString(strconv.Quote(string(v.Value)))
	case ValueTypeBool:
		if len(v.Value) == 0 {
			b.WriteString("(!invalid bool)")
		}
		switch v.Value[0] {
		case 0:
			b.WriteString("false")
		case 1:
			b.WriteString("true")
		default:
			b.WriteString("!invalid bool")
		}
	case ValueTypeUint8, ValueTypeInt8, ValueTypeUint16, ValueTypeInt16, ValueTypeUint32, ValueTypeInt32, ValueTypeUint64, ValueTypeInt64, ValueTypeFloat32, ValueTypeFloat64:
		var buf [8]byte
		if len(v.Value) < 8 {
			copy(buf[:], v.Value)
		}
		fmt.Fprintf(&b, "%v", binary.LittleEndian.Uint64(buf[:]))
	default:
		fmt.Fprintf(&b, "%v", v.Value)
	}
	b.WriteString(")")
	return b.String()
}

type MetaEntry struct {
	Key    string
	Type   ValueType
	Values []MetaValue
}

func (e MetaEntry) String() string {
	if len(e.Values) == 0 {
		return ""
	}
	return string(e.Values[0].Value)
}

func (e MetaEntry) Uint32() uint32 {
	if len(e.Values) == 0 {
		return 0
	}
	return binary.LittleEndian.Uint32(e.Values[0].Value)
}

func (e MetaEntry) FileType() Type {
	if len(e.Values) == 0 {
		return TypeCount
	}
	return Type(e.Uint32())
}

func (e MetaEntry) GoString() string {
	var b strings.Builder
	b.WriteString(e.Key)
	b.WriteString(": ")
	b.WriteString(e.Type.String())
	b.WriteString("(")
	for i, v := range e.Values {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(v.String())
	}
	b.WriteString(")")
	return b.String()
}

type Info struct {
	_ structs.Incomparable //  prevent comparison of Info values so we can change the implementation later

	Version  int
	FileType Type
}

func Stat(path string) (Info, error) {
	f, err := os.Open(path)
	if err != nil {
		return Info{}, err
	}
	defer f.Close()
	return StatReader(f)
}

// StatReader reads the header information from r and returns an Info
// struct with the version and file type.
//
// It returns an error if any.
//
// As a special case, it returns ErrBadMagic if the file does not start with
// the magic bytes. This can be used to detect if the file is not a GGUF
// file.
func StatReader(r io.ReadSeeker) (Info, error) {
	if _, err := r.Seek(0, 0); err != nil {
		return Info{}, err
	}
	f, err := ReadFile(r)
	if err != nil {
		return Info{}, err
	}
	info := Info{Version: f.Version()}
	for m, err := range f.Metadata {
		if err != nil {
			return Info{}, err
		}
		if m.Key == "general.file_type" {
			if m.Type != ValueTypeUint32 {
				return Info{}, fmt.Errorf("unexpected type for metadata key %q: %v, want %v", m.Key, m.Type, ValueTypeUint32)
			}
			info.FileType = m.FileType()
		}
	}
	return info, nil
}

type File struct {
	version       uint32
	numMetaValues uint64
	numTensors    uint64

	gr *ggufReader
}

// ReadFile reads header information from r and returns a File, ready for
// iteration over Metadata and Tensors.
func ReadFile(r io.Reader) (*File, error) {
	f, err := readFile(r)
	if err != nil {
		return nil, err
	}
	return f, nil
}

func (f *File) Version() int {
	return int(f.version)
}

// Metadata iterates over the metadata in the file. It must be exhausted
// before calling Tensors.
//
// It is not resumable.
func (f *File) Metadata(yield func(MetaEntry, error) bool) {
	var n int
	for range f.numMetaValues {
		meta, err := f.gr.readMetaEntry()
		if err != nil {
			err = fmt.Errorf("error reading metadata entry %d: %w", n, err)
			yield(MetaEntry{}, err)
			return
		}
		if !yield(meta, nil) {
			return
		}
		n++
	}
}

// Tensors iterates over the tensors in the file. It must only be called
// after exhausting the metadata iterator.
//
// It is not resumable.
func (f *File) Tensors(yield func(TensorInfo, error) bool) {
	var last TensorInfo
	for range f.numTensors {
		info, err := f.gr.readTensorInfo()

		// If the last tensor had a valid offset, yield it.
		//
		// NOTE: No tensor should have an offset of 0 because the
		// offset is the start of the tensor data which is always
		// afer the magic bytes, version, numMetaValues, and
		// numTensors, which MUST all be non-zero bytes as per the
		// GGUF spec.
		if last.Offset > 0 {
			if !yield(last, err) {
				return
			}
		}
		if err != nil {
			yield(TensorInfo{}, err)
			return
		}

		// Tensor data does not include size, so we need to
		// calculate it based on the offset of the previous tensor
		// offset to the current.
		offset0 := last.Offset
		last = info
		last.Size = info.Offset - offset0
	}
	if last.Offset > 0 {
		yield(last, nil)
	}
}

var magicBytes = []byte{0x47, 0x47, 0x55, 0x46}

func readFile(r io.Reader) (*File, error) {
	gr := &ggufReader{r: &reader{r: r}}
	magic, err := gr.next(4)
	if err != nil {
		return nil, errors.Join(err, ErrBadMagic)
	}
	if !bytes.Equal(magic, magicBytes) {
		return nil, ErrBadMagic
	}
	version, err := gr.readUint32()
	if err != nil {
		return nil, err
	}
	if version != 3 {
		return nil, fmt.Errorf("%w: %d", ErrUnsupportedVersion, version)
	}
	numTensors, err := gr.readUint64()
	if err != nil {
		return nil, err
	}
	numMetaValues, err := gr.readUint64()
	if err != nil {
		return nil, err
	}
	info := &File{
		version: version,

		numMetaValues: numMetaValues,
		numTensors:    numTensors,
		gr:            gr,
	}
	return info, nil
}

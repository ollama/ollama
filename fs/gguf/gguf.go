package gguf

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"iter"
	"os"
	"slices"
	"strings"
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

var ErrUnsupported = errors.New("unsupported")

// maxTensorDims is a defensive cap on the number of dimensions any tensor
// may declare. The GGUF format in practice uses 1-4 dimensions; 16 is far
// beyond any legitimate value and protects shape-array allocation from a
// malformed header that would otherwise allocate gigabytes of memory.
const maxTensorDims = 16

type File struct {
	Magic   [4]byte
	Version uint32

	keyValues *lazy[KeyValue]
	tensors   *lazy[TensorInfo]
	offset    int64

	file   *os.File
	reader *bufferedReader
	bts    []byte

	// fileSize is the size in bytes of the underlying GGUF file. Length
	// fields read out of the file (string lengths, array counts, tensor
	// shape dims) are bounded by this value so an attacker cannot supply
	// e.g. 0xFFFFFFFFFFFFFFFF and trigger an out-of-range or unbounded
	// allocation. A value of 0 means the size is unknown.
	fileSize int64
}

// validateLength returns an error if n cannot be a valid count of bytes or
// elements for this file: it must be non-negative when cast to int64
// (guards uint64→int sign flip on 64-bit hosts) and must not exceed the
// declared file size when known.
func (f *File) validateLength(n uint64) error {
	if int64(n) < 0 {
		return fmt.Errorf("invalid gguf length: %d", n)
	}
	if f.fileSize > 0 && int64(n) > f.fileSize {
		return fmt.Errorf("gguf length %d exceeds file size %d", n, f.fileSize)
	}
	return nil
}

// elementDiskSize returns the minimum number of bytes a single element of the
// given GGUF scalar array type occupies in the file. For fixed-width types
// this is the encoded width, which also equals the in-memory width, so
// bounding the element count by it bounds the slice allocation as well. For
// strings it is the 8-byte uint64 length prefix every string carries; the
// string payload is bounded separately by validateLength when it is read.
func elementDiskSize(t uint32) int {
	switch t {
	case typeUint8, typeInt8, typeBool:
		return 1
	case typeUint16, typeInt16:
		return 2
	case typeUint32, typeInt32, typeFloat32:
		return 4
	case typeUint64, typeInt64, typeFloat64:
		return 8
	case typeString:
		return 8
	default:
		return 1
	}
}

// validateArrayLength returns an error if an array of n elements, each
// occupying at least elemSize bytes on disk, cannot fit in this file. It
// guards the uint64→int sign flip and, more importantly, rejects element
// counts whose backing allocation (n * elemSize bytes) would exceed the file.
// readArrayData/readArrayString here allocate make([]T, n) unconditionally
// with no maxArraySize cap, so validateLength alone (which only bounds the
// raw count by the byte size of the file) would still allow an allocation of
// up to 8-16x fileSize for uint64/float64/string arrays. Bounding by the
// per-element disk size closes that memory-amplification path. The comparison
// uses division so n * elemSize can never overflow, and because a well-formed
// file must physically contain elemSize bytes per element, no valid file is
// rejected.
func (f *File) validateArrayLength(n uint64, elemSize int) error {
	if int64(n) < 0 {
		return fmt.Errorf("invalid gguf array length: %d", n)
	}
	if elemSize < 1 {
		elemSize = 1
	}
	if f.fileSize > 0 && n > uint64(f.fileSize)/uint64(elemSize) {
		return fmt.Errorf("gguf array of %d elements (%d bytes each) exceeds file size %d", n, elemSize, f.fileSize)
	}
	return nil
}

func Open(path string) (f *File, err error) {
	f = &File{bts: make([]byte, 4096)}
	f.file, err = os.Open(path)
	if err != nil {
		return nil, err
	}

	if fi, statErr := f.file.Stat(); statErr == nil {
		f.fileSize = fi.Size()
	}

	f.reader = newBufferedReader(f.file, 32<<10)

	if err := binary.Read(f.reader, binary.LittleEndian, &f.Magic); err != nil {
		return nil, err
	}

	if bytes.Equal(f.Magic[:], []byte("gguf")) {
		return nil, fmt.Errorf("%w file type %v", ErrUnsupported, f.Magic)
	}

	if err := binary.Read(f.reader, binary.LittleEndian, &f.Version); err != nil {
		return nil, err
	}

	if f.Version < 2 {
		return nil, fmt.Errorf("%w version %v", ErrUnsupported, f.Version)
	}

	f.tensors, err = newLazy(f, f.readTensor)
	if err != nil {
		return nil, err
	}

	f.tensors.successFunc = func() error {
		offset := f.reader.offset

		alignment := cmp.Or(f.KeyValue("general.alignment").Int(), 32)
		f.offset = offset + (alignment-offset%alignment)%alignment
		return nil
	}

	f.keyValues, err = newLazy(f, f.readKeyValue)
	if err != nil {
		return nil, err
	}

	return f, nil
}

func (f *File) readTensor() (TensorInfo, error) {
	name, err := readString(f)
	if err != nil {
		return TensorInfo{}, err
	}

	dims, err := read[uint32](f)
	if err != nil {
		return TensorInfo{}, err
	}

	if dims > maxTensorDims {
		return TensorInfo{}, fmt.Errorf("invalid tensor dimensions: %d", dims)
	}

	shape := make([]uint64, dims)
	for i := range dims {
		shape[i], err = read[uint64](f)
		if err != nil {
			return TensorInfo{}, err
		}
		if err := f.validateLength(shape[i]); err != nil {
			return TensorInfo{}, fmt.Errorf("invalid tensor shape: %w", err)
		}
	}

	type_, err := read[uint32](f)
	if err != nil {
		return TensorInfo{}, err
	}

	offset, err := read[uint64](f)
	if err != nil {
		return TensorInfo{}, err
	}

	return TensorInfo{
		Name:   name,
		Offset: offset,
		Shape:  shape,
		Type:   TensorType(type_),
	}, nil
}

func (f *File) readKeyValue() (KeyValue, error) {
	key, err := readString(f)
	if err != nil {
		return KeyValue{}, err
	}

	t, err := read[uint32](f)
	if err != nil {
		return KeyValue{}, err
	}

	value, err := func() (any, error) {
		switch t {
		case typeUint8:
			return read[uint8](f)
		case typeInt8:
			return read[int8](f)
		case typeUint16:
			return read[uint16](f)
		case typeInt16:
			return read[int16](f)
		case typeUint32:
			return read[uint32](f)
		case typeInt32:
			return read[int32](f)
		case typeUint64:
			return read[uint64](f)
		case typeInt64:
			return read[int64](f)
		case typeFloat32:
			return read[float32](f)
		case typeFloat64:
			return read[float64](f)
		case typeBool:
			return read[bool](f)
		case typeString:
			return readString(f)
		case typeArray:
			return readArray(f)
		default:
			return nil, fmt.Errorf("%w type %d", ErrUnsupported, t)
		}
	}()
	if err != nil {
		return KeyValue{}, err
	}

	return KeyValue{
		Key:   key,
		Value: Value{value},
	}, nil
}

func read[T any](f *File) (t T, err error) {
	err = binary.Read(f.reader, binary.LittleEndian, &t)
	return t, err
}

func readString(f *File) (string, error) {
	n, err := read[uint64](f)
	if err != nil {
		return "", err
	}

	if err := f.validateLength(n); err != nil {
		return "", fmt.Errorf("invalid string length: %w", err)
	}

	if int(n) > len(f.bts) {
		f.bts = make([]byte, n)
	}

	bts := f.bts[:n]
	if _, err := io.ReadFull(f.reader, bts); err != nil {
		return "", err
	}
	defer clear(bts)

	return string(bts), nil
}

func readArray(f *File) (any, error) {
	t, err := read[uint32](f)
	if err != nil {
		return nil, err
	}

	n, err := read[uint64](f)
	if err != nil {
		return nil, err
	}

	if err := f.validateArrayLength(n, elementDiskSize(t)); err != nil {
		return nil, fmt.Errorf("invalid array length: %w", err)
	}

	switch t {
	case typeUint8:
		return readArrayData[uint8](f, n)
	case typeInt8:
		return readArrayData[int8](f, n)
	case typeUint16:
		return readArrayData[uint16](f, n)
	case typeInt16:
		return readArrayData[int16](f, n)
	case typeUint32:
		return readArrayData[uint32](f, n)
	case typeInt32:
		return readArrayData[int32](f, n)
	case typeUint64:
		return readArrayData[uint64](f, n)
	case typeInt64:
		return readArrayData[int64](f, n)
	case typeFloat32:
		return readArrayData[float32](f, n)
	case typeFloat64:
		return readArrayData[float64](f, n)
	case typeBool:
		return readArrayData[bool](f, n)
	case typeString:
		return readArrayString(f, n)
	default:
		return nil, fmt.Errorf("%w type %d", ErrUnsupported, t)
	}
}

func readArrayData[T any](f *File, n uint64) (s []T, err error) {
	s = make([]T, n)
	for i := range n {
		e, err := read[T](f)
		if err != nil {
			return nil, err
		}

		s[i] = e
	}

	return s, nil
}

func readArrayString(f *File, n uint64) (s []string, err error) {
	s = make([]string, n)
	for i := range n {
		e, err := readString(f)
		if err != nil {
			return nil, err
		}

		s[i] = e
	}

	return s, nil
}

func (f *File) Close() error {
	f.keyValues.stop()
	f.tensors.stop()
	return f.file.Close()
}

func (f *File) KeyValue(key string) KeyValue {
	if !strings.HasPrefix(key, "general.") && !strings.HasPrefix(key, "tokenizer.") {
		key = f.KeyValue("general.architecture").String() + "." + key
	}

	if index := slices.IndexFunc(f.keyValues.values, func(kv KeyValue) bool {
		return kv.Key == key
	}); index >= 0 {
		return f.keyValues.values[index]
	}

	for keyValue, ok := f.keyValues.next(); ok; keyValue, ok = f.keyValues.next() {
		if keyValue.Key == key {
			return keyValue
		}
	}

	return KeyValue{}
}

func (f *File) NumKeyValues() int {
	return int(f.keyValues.count)
}

func (f *File) KeyValues() iter.Seq2[int, KeyValue] {
	return f.keyValues.All()
}

func (f *File) TensorInfo(name string) TensorInfo {
	if index := slices.IndexFunc(f.tensors.values, func(t TensorInfo) bool {
		return t.Name == name
	}); index >= 0 {
		return f.tensors.values[index]
	}

	// fast-forward through key values if we haven't already
	_ = f.keyValues.rest()
	for tensor, ok := f.tensors.next(); ok; tensor, ok = f.tensors.next() {
		if tensor.Name == name {
			return tensor
		}
	}

	return TensorInfo{}
}

func (f *File) NumTensors() int {
	return int(f.tensors.count)
}

func (f *File) TensorInfos() iter.Seq2[int, TensorInfo] {
	// fast forward through key values if we haven't already
	f.keyValues.rest()
	return f.tensors.All()
}

func (f *File) TensorReader(name string) (TensorInfo, io.Reader, error) {
	t := f.TensorInfo(name)
	if t.NumBytes() == 0 {
		return TensorInfo{}, nil, fmt.Errorf("tensor %s not found", name)
	}

	// fast forward through tensor info if we haven't already
	_ = f.tensors.rest()
	return t, io.NewSectionReader(f.file, f.offset+int64(t.Offset), t.NumBytes()), nil
}

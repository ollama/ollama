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
	MaxStringLength = 16 << 20
	MaxArraySize    = 64 << 20

	MaxTensorDims = 4
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

type File struct {
	Magic   [4]byte
	Version uint32

	keyValues *lazy[KeyValue]
	tensors   *lazy[TensorInfo]
	offset    int64

	file   *os.File
	reader *bufferedReader
	bts    []byte
}

func Open(path string) (f *File, err error) {
	f = &File{bts: make([]byte, 4096)}
	f.file, err = os.Open(path)
	if err != nil {
		return nil, err
	}

	f.reader = newBufferedReader(f.file, 32<<10)

	if err := binary.Read(f.reader, binary.LittleEndian, &f.Magic); err != nil {
		return nil, err
	}

	if !bytes.Equal(f.Magic[:], []byte("GGUF")) {
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
		if alignment <= 0 {
			return fmt.Errorf("%w alignment %d", ErrUnsupported, alignment)
		}
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
	if dims > MaxTensorDims {
		return TensorInfo{}, fmt.Errorf("%w tensor dimensions %d exceeds maximum %d", ErrUnsupported, dims, MaxTensorDims)
	}

	shape := make([]uint64, dims)
	for i := range dims {
		shape[i], err = read[uint64](f)
		if err != nil {
			return TensorInfo{}, err
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

	ti := TensorInfo{
		Name:   name,
		Offset: offset,
		Shape:  shape,
		Type:   TensorType(type_),
	}
	if _, ok := ti.numBytes(); !ok {
		return TensorInfo{}, fmt.Errorf("%w tensor %q size overflows", ErrUnsupported, ti.Name)
	}
	return ti, nil
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

	length, err := checkedLength(n, "string", MaxStringLength)
	if err != nil {
		return "", err
	}

	if length > len(f.bts) {
		f.bts = make([]byte, length)
	}

	bts := f.bts[:length]
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
	size, err := checkedLength(n, "array size", MaxArraySize)
	if err != nil {
		return nil, err
	}

	s = make([]T, size)
	for i := range size {
		e, err := read[T](f)
		if err != nil {
			return nil, err
		}

		s[i] = e
	}

	return s, nil
}

func readArrayString(f *File, n uint64) (s []string, err error) {
	size, err := checkedLength(n, "array size", MaxArraySize)
	if err != nil {
		return nil, err
	}

	s = make([]string, size)
	for i := range size {
		e, err := readString(f)
		if err != nil {
			return nil, err
		}

		s[i] = e
	}

	return s, nil
}

func checkedLength(n uint64, kind string, max uint64) (int, error) {
	if n > uint64(maxInt()) {
		return 0, fmt.Errorf("%s %d exceeds maximum %d", kind, n, maxInt())
	}
	if n > max {
		return 0, fmt.Errorf("%s %d exceeds maximum %d", kind, n, max)
	}
	return int(n), nil
}

func maxInt() int {
	return int(^uint(0) >> 1)
}

func maxInt64() uint64 {
	return 1<<63 - 1
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
	if err := f.Err(); err != nil {
		return TensorInfo{}, nil, err
	}
	if t.Name == "" {
		return TensorInfo{}, nil, fmt.Errorf("tensor %s not found", name)
	}
	numBytes, ok := t.numBytes()
	if !ok {
		return TensorInfo{}, nil, fmt.Errorf("%w tensor %q size overflows", ErrUnsupported, t.Name)
	}
	if numBytes == 0 {
		return TensorInfo{}, nil, fmt.Errorf("tensor %s not found", name)
	}

	// fast forward through tensor info if we haven't already
	f.tensors.rest()
	if err := f.Err(); err != nil {
		return TensorInfo{}, nil, err
	}

	if t.Offset > maxInt64() {
		return TensorInfo{}, nil, fmt.Errorf("%w tensor %q offset %d exceeds maximum %d", ErrUnsupported, t.Name, t.Offset, maxInt64())
	}
	offset := f.offset + int64(t.Offset)
	if offset < f.offset {
		return TensorInfo{}, nil, fmt.Errorf("%w tensor %q offset overflows", ErrUnsupported, t.Name)
	}

	fileInfo, err := f.file.Stat()
	if err != nil {
		return TensorInfo{}, nil, err
	}
	if numBytes > fileInfo.Size()-offset {
		return TensorInfo{}, nil, fmt.Errorf("%w tensor %q offset+size exceeds file size", ErrUnsupported, t.Name)
	}

	return t, io.NewSectionReader(f.file, offset, numBytes), nil
}

func (f *File) Err() error {
	// Key/value and tensor metadata are read lazily, so parse errors can surface
	// after Open succeeds.
	if f.keyValues != nil {
		if err := f.keyValues.Err(); err != nil {
			return err
		}
	}
	if f.tensors != nil {
		if err := f.tensors.Err(); err != nil {
			return err
		}
	}
	return nil
}

package gguf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"iter"
)

type ggufReader struct {
	r *reader
	n int
}

func (r *ggufReader) readMetaEntry() (MetaEntry, error) {
	key, err := r.readString()
	if err != nil {
		return MetaEntry{}, err
	}
	typ, err := r.readValueType()
	if err != nil {
		return MetaEntry{}, err
	}
	var values []MetaValue
	for v, err := range r.readMetaValues(typ) {
		if err != nil {
			err = fmt.Errorf("(key=%q type=%s): %w", key, typ, err)
			return MetaEntry{}, err
		}
		values = append(values, v)
	}
	return MetaEntry{
		Key:    string(key),
		Type:   typ,
		Values: values,
	}, nil
}

func (r *ggufReader) readMetaValue(typ ValueType) (MetaValue, error) {
	var value []byte
	var err error
	switch typ {
	case ValueTypeUint8, ValueTypeInt8:
		value, err = r.next(1)
	case ValueTypeUint16, ValueTypeInt16:
		value, err = r.next(2)
	case ValueTypeUint32, ValueTypeInt32, ValueTypeFloat32:
		value, err = r.next(4)
	case ValueTypeUint64, ValueTypeInt64, ValueTypeFloat64:
		value, err = r.next(8)
	case ValueTypeBool:
		value, err = r.next(1)
	case ValueTypeString:
		value, err = r.readString()
	case ValueTypeArray:
		err = fmt.Errorf("nested arrays are not supported")
	default:
		err = fmt.Errorf("unsupported metadata type: %d", typ)
	}
	if err != nil {
		return MetaValue{}, err
	}
	return MetaValue{
		Type:  typ,
		Value: bytes.Clone(value),
	}, nil
}

func (r *ggufReader) readMetaValues(typ ValueType) iter.Seq2[MetaValue, error] {
	return func(yield func(MetaValue, error) bool) {
		if typ == ValueTypeArray {
			atyp, err := r.readValueType()
			if err != nil {
				err = fmt.Errorf("invalid type: %w", err)
				yield(MetaValue{}, err)
				return
			}
			n, err := r.readUint64()
			if err != nil {
				err = fmt.Errorf("invalid length: %w", err)
				yield(MetaValue{}, err)
				return
			}
			for i := range n {
				v, err := r.readMetaValue(atyp)
				if err != nil {
					err = fmt.Errorf("invalid entry (type=%s) %d: %w", atyp, i, err)
					yield(MetaValue{}, err)
					return
				}
				if !yield(v, nil) {
					return
				}
			}
		} else {
			v, err := r.readMetaValue(typ)
			if err != nil {
				err = fmt.Errorf("error reading metadata value: %w", err)
				yield(MetaValue{}, err)
				return
			}
			yield(v, nil)
		}
	}
}

func (r *ggufReader) readValueType() (ValueType, error) {
	typ, err := r.readUint32()
	return ValueType(typ), err
}

func (r *ggufReader) readTensorInfo() (TensorInfo, error) {
	name, err := r.readString()
	if err != nil {
		return TensorInfo{}, err
	}

	numDimensions, err := r.readUint32()
	if err != nil {
		return TensorInfo{}, err
	}
	if numDimensions > MaxDimensions {
		return TensorInfo{}, fmt.Errorf("%w: dimensions length (%d) exceeds %d", ErrMangled, numDimensions, MaxDimensions)
	}

	dims := make([]uint64, numDimensions)
	for i := range dims {
		d, err := r.readUint64()
		if err != nil {
			return TensorInfo{}, err
		}
		dims[i] = d
	}
	typ, err := r.readUint32()
	if err != nil {
		return TensorInfo{}, err
	}
	offset, err := r.readUint64()
	if err != nil {
		return TensorInfo{}, err
	}

	// TODO(bmizerany): check offset is multiple of ALIGNMENT
	return TensorInfo{
		Name:       string(name),
		Dimensions: dims,
		Type:       Type(typ),
		Offset:     offset,
	}, nil
}

func (r *ggufReader) next(n int) ([]byte, error) {
	if n < 0 {
		return nil, errors.Join(fmt.Errorf("invalid read length: %d", n), ErrMangled)
	}
	w := r.r.window()
	for len(w) < n {
		if r.r.extend() == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		w = r.r.window()
	}
	r.r.release(n)
	r.n += n
	return w[:n], nil
}

func (r *ggufReader) readString() ([]byte, error) {
	n, err := r.readUint64()
	if err != nil {
		return nil, err
	}
	// TODO(bmizerany): limit max string length
	return r.next(int(n))
}

func (r *ggufReader) readUint32() (uint32, error) {
	b, err := r.next(4)
	if err != nil {
		return 0, err
	}
	n := binary.LittleEndian.Uint32(b)
	return n, nil
}

func (r *ggufReader) readUint64() (uint64, error) {
	b, err := r.next(8)
	if err != nil {
		return 0, err
	}
	n := binary.LittleEndian.Uint64(b)
	return n, nil
}

package compatmigrate

import (
	"encoding/binary"
	"fmt"
	"io"
	"slices"

	"github.com/x448/float16"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/fs/gguf"
)

const (
	qk8_0       = 32
	qk5_0       = 32
	qkK         = 256
	q8_0BlkSize = 34
	q5_0BlkSize = 22
	q4KBlkSize  = 144
	q6KBlkSize  = 210
)

type sourceTensorRow struct {
	tensor *sourceTensor
	row    uint64
	shape  []uint64
}

func copyTensorRowF32(name string, t *sourceTensor, row uint64) (*ggml.Tensor, error) {
	if len(t.shape) < 2 {
		return nil, fmt.Errorf("tensor %s shape %v cannot be row-sliced", t.name, t.shape)
	}
	if row >= t.shape[1] {
		return nil, fmt.Errorf("row %d out of range for tensor %s shape %v", row, t.name, t.shape)
	}

	return &ggml.Tensor{
		Name:     name,
		Kind:     uint32(gguf.TensorTypeF32),
		Shape:    []uint64{t.shape[0]},
		WriterTo: &sourceTensorRow{tensor: t, row: row, shape: []uint64{t.shape[0]}},
	}, nil
}

func (r *sourceTensorRow) WriteTo(w io.Writer) (int64, error) {
	t := r.tensor
	rowBytes := gguf.TensorInfo{Name: t.name, Shape: r.shape, Type: t.info.Type}.NumBytes()
	if rowBytes <= 0 {
		return 0, fmt.Errorf("unsupported row tensor type %v for %s", t.info.Type, t.name)
	}

	readerAt := t.readerAt
	dataOffset := t.dataOffset
	var g *gguf.File
	var err error
	if readerAt == nil {
		g, err = gguf.Open(t.path)
		if err != nil {
			return 0, err
		}
		defer g.Close()
		readerAt = g.ReaderAt()
		dataOffset = g.TensorDataOffset()
	}

	b := make([]byte, rowBytes)
	offset := dataOffset + int64(t.info.Offset) + int64(r.row)*rowBytes
	if _, err := readerAt.ReadAt(b, offset); err != nil {
		return 0, err
	}

	data, err := decodeTensorRow(t.info.Type, b, int(r.shape[0]))
	if err != nil {
		return 0, err
	}

	n, err := encodeFloatTensor(w, gguf.TensorTypeF32, data)
	return int64(n), err
}

func (r *sourceTensorRow) GGUFWriteMemoryEstimate() uint64 {
	rowBytes := int64ToUint64(gguf.TensorInfo{Name: r.tensor.name, Shape: r.shape, Type: r.tensor.info.Type}.NumBytes())
	floatBytes := saturatingMul(tensorElementCount(r.shape), 4)
	return saturatingAdd(rowBytes, floatBytes, floatBytes)
}

func decodeTensorRow(kind gguf.TensorType, b []byte, n int) ([]float32, error) {
	switch kind {
	case gguf.TensorTypeF32, gguf.TensorTypeF16, gguf.TensorTypeBF16:
		out, err := decodeFloatTensor(kind, b)
		if err != nil {
			return nil, err
		}
		if len(out) != n {
			return nil, fmt.Errorf("decoded %d values for %v row, expected %d", len(out), kind, n)
		}
		return out, nil
	case gguf.TensorTypeQ8_0:
		return decodeQ8_0Row(b, n)
	case gguf.TensorTypeQ5_0:
		return decodeQ5_0Row(b, n)
	case gguf.TensorTypeQ4_K:
		return decodeQ4KRow(b, n)
	case gguf.TensorTypeQ6_K:
		return decodeQ6KRow(b, n)
	default:
		return nil, fmt.Errorf("row extraction from tensor type %v is unsupported", kind)
	}
}

func decodeQ8_0Row(b []byte, n int) ([]float32, error) {
	if n%qk8_0 != 0 {
		return nil, fmt.Errorf("q8_0 row length %d is not divisible by %d", n, qk8_0)
	}
	if want := n / qk8_0 * q8_0BlkSize; len(b) != want {
		return nil, fmt.Errorf("q8_0 row has %d bytes, expected %d", len(b), want)
	}

	out := make([]float32, n)
	for block := range n / qk8_0 {
		base := block * q8_0BlkSize
		d := decodeF16(b[base:])
		qs := b[base+2 : base+q8_0BlkSize]
		for i, q := range qs {
			out[block*qk8_0+i] = d * float32(int8(q))
		}
	}
	return out, nil
}

func decodeQ5_0Row(b []byte, n int) ([]float32, error) {
	if n%qk5_0 != 0 {
		return nil, fmt.Errorf("q5_0 row length %d is not divisible by %d", n, qk5_0)
	}
	if want := n / qk5_0 * q5_0BlkSize; len(b) != want {
		return nil, fmt.Errorf("q5_0 row has %d bytes, expected %d", len(b), want)
	}

	out := make([]float32, n)
	for block := range n / qk5_0 {
		base := block * q5_0BlkSize
		d := decodeF16(b[base:])
		qh := binary.LittleEndian.Uint32(b[base+2:])
		qs := b[base+6 : base+q5_0BlkSize]

		outOffset := block * qk5_0
		for j, q := range qs {
			loHighBit := uint8(((qh >> j) << 4) & 0x10)
			hiHighBit := uint8((qh >> (j + 12)) & 0x10)
			out[outOffset+j] = d * float32(int((q&0x0f)|loHighBit)-16)
			out[outOffset+j+qk5_0/2] = d * float32(int((q>>4)|hiHighBit)-16)
		}
	}
	return out, nil
}

func decodeQ4KRow(b []byte, n int) ([]float32, error) {
	if n%qkK != 0 {
		return nil, fmt.Errorf("q4_k row length %d is not divisible by %d", n, qkK)
	}
	if want := n / qkK * q4KBlkSize; len(b) != want {
		return nil, fmt.Errorf("q4_k row has %d bytes, expected %d", len(b), want)
	}

	out := make([]float32, n)
	for block := range n / qkK {
		base := block * q4KBlkSize
		d := decodeF16(b[base:])
		min := decodeF16(b[base+2:])
		scales := b[base+4 : base+16]
		qs := b[base+16 : base+q4KBlkSize]

		outOffset := block * qkK
		scaleIndex := 0
		qOffset := 0
		for range qkK / 64 {
			sc, m := scaleMinK4(scaleIndex, scales)
			d1 := d * float32(sc)
			m1 := min * float32(m)
			sc, m = scaleMinK4(scaleIndex+1, scales)
			d2 := d * float32(sc)
			m2 := min * float32(m)

			for l := range 32 {
				out[outOffset+l] = d1*float32(qs[qOffset+l]&0x0f) - m1
			}
			for l := range 32 {
				out[outOffset+32+l] = d2*float32(qs[qOffset+l]>>4) - m2
			}

			outOffset += 64
			qOffset += 32
			scaleIndex += 2
		}
	}
	return out, nil
}

func scaleMinK4(j int, q []byte) (uint8, uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0x0f) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4)
}

func decodeQ6KRow(b []byte, n int) ([]float32, error) {
	if n%qkK != 0 {
		return nil, fmt.Errorf("q6_k row length %d is not divisible by %d", n, qkK)
	}
	if want := n / qkK * q6KBlkSize; len(b) != want {
		return nil, fmt.Errorf("q6_k row has %d bytes, expected %d", len(b), want)
	}

	out := make([]float32, n)
	for block := range n / qkK {
		base := block * q6KBlkSize
		ql := b[base : base+128]
		qh := b[base+128 : base+192]
		scales := b[base+192 : base+208]
		d := decodeF16(b[base+208:])

		outOffset := block * qkK
		qlOffset := 0
		qhOffset := 0
		scaleOffset := 0
		for range qkK / 128 {
			for l := range 32 {
				is := l / 16
				q1 := int((ql[qlOffset+l+0]&0x0f)|(((qh[qhOffset+l]>>0)&3)<<4)) - 32
				q2 := int((ql[qlOffset+l+32]&0x0f)|(((qh[qhOffset+l]>>2)&3)<<4)) - 32
				q3 := int((ql[qlOffset+l+0]>>4)|(((qh[qhOffset+l]>>4)&3)<<4)) - 32
				q4 := int((ql[qlOffset+l+32]>>4)|(((qh[qhOffset+l]>>6)&3)<<4)) - 32

				out[outOffset+l+0] = d * float32(int8(scales[scaleOffset+is+0])) * float32(q1)
				out[outOffset+l+32] = d * float32(int8(scales[scaleOffset+is+2])) * float32(q2)
				out[outOffset+l+64] = d * float32(int8(scales[scaleOffset+is+4])) * float32(q3)
				out[outOffset+l+96] = d * float32(int8(scales[scaleOffset+is+6])) * float32(q4)
			}
			outOffset += 128
			qlOffset += 64
			qhOffset += 32
			scaleOffset += 8
		}
	}
	return out, nil
}

func decodeF16(b []byte) float32 {
	return float16.Frombits(binary.LittleEndian.Uint16(b)).Float32()
}

func f32Tensor(name string, shape []uint64, data []float32) *ggml.Tensor {
	return &ggml.Tensor{
		Name:     name,
		Kind:     uint32(gguf.TensorTypeF32),
		Shape:    slices.Clone(shape),
		WriterTo: tensorFloat32Writer(data),
	}
}

type tensorFloat32Writer []float32

func (t tensorFloat32Writer) WriteTo(w io.Writer) (int64, error) {
	n, err := encodeFloatTensor(w, gguf.TensorTypeF32, []float32(t))
	return int64(n), err
}

func (t tensorFloat32Writer) GGUFWriteMemoryEstimate() uint64 {
	return saturatingMul(uint64(len(t)), 4)
}

package mlx

// #include "generated.h"
import "C"

import (
	"encoding/binary"
	"log/slog"
	"reflect"
	"strings"
	"time"
	"unsafe"

	"github.com/ollama/ollama/logutil"
)

type tensorDesc struct {
	name    string
	inputs  []*Tensor
	numRefs int
}

func (d tensorDesc) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("name", d.name),
		slog.Int("inputs", len(d.inputs)),
		slog.Int("num_refs", d.numRefs),
	)
}

type Tensor struct {
	ctx  C.mlx_array
	desc tensorDesc
}

// constructor utilities

func New(name string, inputs ...*Tensor) *Tensor {
	t := &Tensor{
		desc: tensorDesc{
			name:   name,
			inputs: inputs,
		},
	}

	for _, input := range inputs {
		input.desc.numRefs++
	}
	logutil.Trace("New", "t", t)
	return t
}

type scalarTypes interface {
	~bool | ~int | ~float32 | ~float64 | ~complex64
}

func FromValue[T scalarTypes](t T) *Tensor {
	tt := New("")
	switch v := any(t).(type) {
	case bool:
		tt.ctx = C.mlx_array_new_bool(C.bool(v))
	case int:
		tt.ctx = C.mlx_array_new_int(C.int(v))
	case float32:
		tt.ctx = C.mlx_array_new_float32(C.float(v))
	case float64:
		tt.ctx = C.mlx_array_new_float64(C.double(v))
	case complex64:
		tt.ctx = C.mlx_array_new_complex(C.float(real(v)), C.float(imag(v)))
	default:
		panic("unsupported type")
	}
	return tt
}

type arrayTypes interface {
	~bool | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~int8 | ~int16 | ~int32 | ~int64 |
		~float32 | ~float64 |
		~complex64
}

func FromValues[S ~[]E, E arrayTypes](s S, shape ...int) *Tensor {
	if len(shape) == 0 {
		panic("shape must be provided for non-scalar tensors")
	}

	cShape := make([]C.int, len(shape))
	for i := range shape {
		cShape[i] = C.int(shape[i])
	}

	var dtype DType
	switch reflect.TypeOf(s).Elem().Kind() {
	case reflect.Bool:
		dtype = DTypeBool
	case reflect.Uint8:
		dtype = DTypeUint8
	case reflect.Uint16:
		dtype = DTypeUint16
	case reflect.Uint32:
		dtype = DTypeUint32
	case reflect.Uint64:
		dtype = DTypeUint64
	case reflect.Int8:
		dtype = DTypeInt8
	case reflect.Int16:
		dtype = DTypeInt16
	case reflect.Int32:
		dtype = DTypeInt32
	case reflect.Int64:
		dtype = DTypeInt64
	case reflect.Float32:
		dtype = DTypeFloat32
	case reflect.Float64:
		dtype = DTypeFloat64
	case reflect.Complex64:
		dtype = DTypeComplex64
	default:
		panic("unsupported type")
	}

	bts := make([]byte, binary.Size(s))
	if _, err := binary.Encode(bts, binary.LittleEndian, s); err != nil {
		panic(err)
	}

	tt := New("")
	tt.ctx = C.mlx_array_new_data(unsafe.Pointer(&bts[0]), unsafe.SliceData(cShape), C.int(len(cShape)), C.mlx_dtype(dtype))
	return tt
}

func (t *Tensor) Set(other *Tensor) {
	other.desc.numRefs++
	t.desc.inputs = []*Tensor{other}
	C.mlx_array_set(&t.ctx, other.ctx)
}

func (t *Tensor) Clone() *Tensor {
	tt := New(t.desc.name, t.desc.inputs...)
	C.mlx_array_set(&tt.ctx, t.ctx)
	return tt
}

// misc. utilities

func (t *Tensor) Valid() bool {
	return t.ctx.ctx != nil
}

func (t *Tensor) String() string {
	str := C.mlx_string_new()
	defer C.mlx_string_free(str)
	C.mlx_array_tostring(&str, t.ctx)
	return strings.TrimSpace(C.GoString(C.mlx_string_data(str)))
}

func (t *Tensor) LogValue() slog.Value {
	attrs := []slog.Attr{slog.Any("", t.desc)}
	if t.Valid() {
		attrs = append(attrs,
			slog.Any("dtype", t.DType()),
			slog.Any("shape", t.Dims()),
			slog.Int("num_bytes", t.NumBytes()),
		)
	}
	return slog.GroupValue(attrs...)
}

// shape utilities

func (t Tensor) Size() int {
	return int(C.mlx_array_size(t.ctx))
}

func (t Tensor) NumBytes() int {
	return int(C.mlx_array_nbytes(t.ctx))
}

func (t Tensor) NumDims() int {
	return int(C.mlx_array_ndim(t.ctx))
}

func (t Tensor) Dims() []int {
	dims := make([]int, t.NumDims())
	for i := range dims {
		dims[i] = t.Dim(i)
	}

	return dims
}

func (t Tensor) Dim(dim int) int {
	return int(C.mlx_array_dim(t.ctx, C.int(dim)))
}

func (t Tensor) DType() DType {
	return DType(C.mlx_array_dtype(t.ctx))
}

// data utilities

func (t Tensor) Int() int {
	var item C.int64_t
	C.mlx_array_item_int64(&item, t.ctx)
	return int(item)
}

func (t Tensor) Float() float64 {
	var item C.double
	C.mlx_array_item_float64(&item, t.ctx)
	return float64(item)
}

func (t Tensor) Ints() []int {
	ints := make([]int, t.Size())
	for i, f := range unsafe.Slice(C.mlx_array_data_int32(t.ctx), len(ints)) {
		ints[i] = int(f)
	}
	return ints
}

func (t Tensor) Floats() []float32 {
	floats := make([]float32, t.Size())
	for i, f := range unsafe.Slice(C.mlx_array_data_float32(t.ctx), len(floats)) {
		floats[i] = float32(f)
	}
	return floats
}

func Free(s ...*Tensor) (n int) {
	now := time.Now()
	defer func() {
		if n > 0 {
			logutil.Trace("Freed tensors", "num_bytes", PrettyBytes(n), "took", time.Since(now))
		}
	}()

	free := make([]*Tensor, 0, 8192)
	fn := func(t *Tensor) {
		if t.Valid() {
			free = append(free, t.desc.inputs...)
			t.desc.numRefs--
			if t.desc.numRefs <= 0 {
				logutil.Trace("Free", "t", t)
				n += t.NumBytes()
				C.mlx_array_free(t.ctx)
				t.ctx.ctx = nil
			}
		}
	}

	for _, t := range s {
		fn(t)
	}

	for len(free) > 0 {
		tail := free[len(free)-1]
		free = free[:len(free)-1]
		fn(tail)
	}

	return n
}

//go:build mlx

package mlx

// #include "generated.h"
import "C"

import (
	"encoding/binary"
	"fmt"
	"log/slog"
	"reflect"
	"sort"
	"strings"
	"unsafe"

	"github.com/ollama/ollama/logutil"
)

type Array struct {
	ctx    C.mlx_array
	name   string
	pinned bool
}

var arrays []*Array

// constructor utilities

func New(name string) *Array {
	t := &Array{name: name}
	arrays = append(arrays, t)
	return t
}

type scalarTypes interface {
	~bool | ~int | ~float32 | ~float64 | ~complex64
}

func FromValue[T scalarTypes](t T) *Array {
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

func FromValues[S ~[]E, E arrayTypes](s S, shape ...int) *Array {
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

func (t *Array) Set(other *Array) {
	C.mlx_array_set(&t.ctx, other.ctx)
}

func (t *Array) Clone() *Array {
	tt := New(t.name)
	C.mlx_array_set(&tt.ctx, t.ctx)
	return tt
}

// lifecycle utilities

// Pin marks arrays as in-use so they are retained during Sweep.
func Pin(s ...*Array) {
	for _, t := range s {
		if t != nil {
			t.pinned = true
		}
	}
}

// Unpin marks arrays as no longer in-use, allowing Sweep to free them.
func Unpin(s ...*Array) {
	for _, t := range s {
		if t != nil {
			t.pinned = false
		}
	}
}

// Sweep releases all unpinned arrays, primarily intermediate tensors. MLX will truly
// free them when there are no other references, including dependencies in the graph.
func Sweep() {
	n := 0
	for _, t := range arrays {
		if t.pinned && t.Valid() {
			arrays[n] = t
			n++
		} else if t.Valid() {
			C.mlx_array_free(t.ctx)
			t.ctx.ctx = nil
		}
	}
	arrays = arrays[:n]
}

// misc. utilities

func (t *Array) Valid() bool {
	return t.ctx.ctx != nil
}

func (t *Array) String() string {
	str := C.mlx_string_new()
	defer C.mlx_string_free(str)
	C.mlx_array_tostring(&str, t.ctx)
	return strings.TrimSpace(C.GoString(C.mlx_string_data(str)))
}

func (t *Array) LogValue() slog.Value {
	attrs := []slog.Attr{
		slog.String("name", t.name),
		slog.Bool("pinned", t.pinned),
	}
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

func (t Array) Size() int {
	return int(C.mlx_array_size(t.ctx))
}

func (t Array) NumBytes() int {
	return int(C.mlx_array_nbytes(t.ctx))
}

func (t Array) NumDims() int {
	return int(C.mlx_array_ndim(t.ctx))
}

func (t Array) Dims() []int {
	dims := make([]int, t.NumDims())
	for i := range dims {
		dims[i] = t.Dim(i)
	}

	return dims
}

func (t Array) Dim(dim int) int {
	return int(C.mlx_array_dim(t.ctx, C.int(dim)))
}

func (t Array) DType() DType {
	return DType(C.mlx_array_dtype(t.ctx))
}

// data utilities

func (t Array) Int() int {
	var item C.int64_t
	C.mlx_array_item_int64(&item, t.ctx)
	return int(item)
}

func (t Array) Float() float64 {
	var item C.double
	C.mlx_array_item_float64(&item, t.ctx)
	return float64(item)
}

func (t Array) Ints() []int {
	ints := make([]int, t.Size())
	for i, f := range unsafe.Slice(C.mlx_array_data_int32(t.ctx), len(ints)) {
		ints[i] = int(f)
	}
	return ints
}

func (t Array) Floats() []float32 {
	floats := make([]float32, t.Size())
	for i, f := range unsafe.Slice(C.mlx_array_data_float32(t.ctx), len(floats)) {
		floats[i] = float32(f)
	}
	return floats
}

func (t Array) Save(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	C.mlx_save(cName, t.ctx)
	return nil
}

// LogArrays logs all live arrays, sorted by size
func LogArrays() {
	sort.Slice(arrays, func(i, j int) bool {
		return arrays[i].NumBytes() > arrays[j].NumBytes()
	})

	for _, t := range arrays {
		nb := t.NumBytes()
		logutil.Trace(fmt.Sprintf("tensor %-60s %5s %5s %v", t.name, t.DType(), PrettyBytes(nb), t.Dims()))
	}
	logutil.Trace(fmt.Sprintf("tensors total: %d, size: %s", len(arrays), PrettyBytes(ActiveMemory())))
}

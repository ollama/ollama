package ml

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"strings"
)

type Config interface {
	Architecture() string
	String(string, ...string) string
	Uint(string, ...uint32) uint32
	Float(string, ...float32) float32

	Strings(string, ...[]string) []string
	Uints(string, ...[]uint32) []uint32
}

type Backend interface {
	Config() Config
	Get(name string) Tensor
	NewContext() Context
}

var backends = make(map[string]func(*os.File) (Backend, error))

func RegisterBackend(name string, f func(*os.File) (Backend, error)) {
	if _, ok := backends[name]; ok {
		panic("backend: backend already registered")
	}

	backends[name] = f
}

func NewBackend(f *os.File) (Backend, error) {
	if backend, ok := backends["ggml"]; ok {
		return backend(f)
	}

	return nil, fmt.Errorf("unsupported backend")
}

type Context interface {
	Zeros(dtype DType, shape ...int) Tensor
	FromFloatSlice(s []float32, shape ...int) (Tensor, error)
	FromIntSlice(s []int32, shape ...int) (Tensor, error)

	Forward(Tensor)
	Compute(Tensor) Tensor
	Close() error
}

type Tensor interface {
	Dim(n int) int64
	Stride(n int) int64

	Shape() []int64
	DType() DType

	Bytes() []byte
	Floats() []float32

	Add(ctx Context, t2 Tensor) Tensor
	Mul(ctx Context, t2 Tensor) Tensor
	Mulmat(ctx Context, t2 Tensor) Tensor

	Softmax(ctx Context) Tensor
	LayerNorm(ctx Context, weight, bias Tensor, eps float32) Tensor
	RMSNorm(ctx Context, weight Tensor, eps float32) Tensor
	Scale(ctx Context, s float64) Tensor

	Conv2D(ctx Context, weight Tensor, s0, s1, p0, p1, d0, d1 int) Tensor
	RoPE(ctx Context, positionIDs, ropeFactors Tensor, dim uint32, base, scale float32) Tensor

	Tanh(ctx Context) Tensor
	GELU(ctx Context) Tensor
	SILU(ctx Context) Tensor

	Reshape(ctx Context, shape ...int64) Tensor
	View(ctx Context, offset int, shape ...int) Tensor
	Permute(ctx Context, shape ...int) Tensor
	Contiguous(ctx Context) Tensor

	Pad(ctx Context, shape ...int64) Tensor
	Unpad(ctx Context, shape ...int64) Tensor

	Stack(ctx Context, dim int, s ...Tensor) Tensor
	Concat(ctx Context, t2 Tensor, dim int) Tensor
	Rows(ctx Context, t2 Tensor) Tensor
	Copy(ctx Context, t2 Tensor) Tensor
}

type number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64 |
		~complex64 | ~complex128
}

func mul[T number](s ...T) T {
	p := T(1)
	for _, v := range s {
		p *= v
	}

	return p
}

type DumpOptions struct {
	// Items is the number of elements to print at the beginning and end of each dimension.
	Items int64

	// Precision is the number of decimal places to print. Applies to float32 and float64.
	Precision int
}

func Dump(t Tensor, opts ...DumpOptions) string {
	if len(opts) < 1 {
		opts = append(opts, DumpOptions{
			Items:     3,
			Precision: 4,
		})
	}

	switch t.DType() {
	case DTypeF32:
		return dump[[]float32](t, opts[0])
	case DTypeI32:
		return dump[[]int32](t, opts[0])
	default:
		return "<unsupported>"
	}
}

func dump[S ~[]E, E number](t Tensor, opts DumpOptions) string {
	bts := t.Bytes()
	if bts == nil {
		return "<nil>"
	}

	s := make(S, mul(t.Shape()...))
	if err := binary.Read(bytes.NewBuffer(t.Bytes()), binary.LittleEndian, &s); err != nil {
		panic(err)
	}

	shape := t.Shape()

	var sb strings.Builder
	var f func([]int64, int64)
	f = func(dims []int64, stride int64) {
		prefix := strings.Repeat(" ", len(shape)-len(dims)+1)
		fmt.Fprint(&sb, "[")
		defer func() { fmt.Fprint(&sb, "]") }()
		for i := int64(0); i < dims[0]; i++ {
			if i >= opts.Items && i < dims[0]-opts.Items {
				fmt.Fprint(&sb, "..., ")
				// skip to next printable element
				skip := dims[0] - 2*opts.Items
				if len(dims) > 1 {
					stride += mul(append(dims[1:], skip)...)
					fmt.Fprint(&sb, strings.Repeat("\n", len(dims)-1), prefix)
				}
				i += skip - 1
			} else if len(dims) > 1 {
				f(dims[1:], stride)
				stride += mul(dims[1:]...)
				if i < dims[0]-1 {
					fmt.Fprint(&sb, ",", strings.Repeat("\n", len(dims)-1), prefix)
				}
			} else {
				fmt.Fprint(&sb, s[stride+i])
				if i < dims[0]-1 {
					fmt.Fprint(&sb, ", ")
				}
			}
		}
	}
	f(shape, 0)

	return sb.String()
}

type DType int

const (
	DTypeF32 DType = iota
	DTypeI32
	DTypeOther
)

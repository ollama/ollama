package ml

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"log/slog"
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
	be := os.Getenv("OLLAMA_BACKEND")
	if be == "" {
		be = "ggml"
		slog.Info("Defaulting to " + be + ". Set OLLAMA_BACKEND to override")
	}
	slog.Info("Loading new engine", "backend", be)
	if backend, ok := backends[be]; ok {
		return backend(f)
	}

	return nil, fmt.Errorf("unsupported backend")
}

type Context interface {
	Zeros(dtype DType, shape ...int) Tensor

	// TODO - the (Tensor, error) return pattern makes this impossible to
	// one-line in cases where we need to pass a scalar into a function that
	// requires a Tensor leading to overly verbose impls.  Consider a Must* API.
	FromFloatSlice(s []float32, shape ...int) (Tensor, error)
	FromIntSlice(s []int32, shape ...int) (Tensor, error)

	Forward(Tensor)
	Compute(Tensor) Tensor
	MaxTensors() int
	Close()

	// TODO remove this before merging - temporary debugging aid
	Abort(Tensor) // Evaluate the graph up to this point, retrieve the data from the tensor and dump it to a json file for comparision
}

// Usage:
//
//	if sdpa, ok := ctx.(ml.FastScaledDotProductAttention); ok {
//	  hiddenState = sdpa.FastScaledDotProductAttention(...)
//	} else {
//	  // manual sdpa
//	}
type FastScaledDotProductAttention interface {
	FastScaledDotProductAttention(queries, keys, values Tensor, scale float32, mask Tensor) Tensor
}

// Usage:
//
//	if su, ok := ctx.(ml.SliceUpdate); ok {
//	  su.SliceUpdate(...)
//	} else {
//	  // view + copy operations
//	}
type SliceUpdate interface {
	SliceUpdate(target, source Tensor, start, stop, strides []int)
}

type Tensor interface {
	Dim(n int) int
	Stride(n int) int

	Shape() []int
	DType() DType

	Bytes() []byte
	Floats() []float32

	Add(ctx Context, t2 Tensor) Tensor
	Mul(ctx Context, t2 Tensor) Tensor
	Mulmat(ctx Context, t2 Tensor) Tensor

	Softmax(ctx Context) Tensor // TODO axis parameter?
	LayerNorm(ctx Context, weight, bias Tensor, eps float32) Tensor
	RMSNorm(ctx Context, weight Tensor, eps float32) Tensor
	Scale(ctx Context, s float64) Tensor

	Conv2D(ctx Context, weight Tensor, s0, s1, p0, p1, d0, d1 int) Tensor

	RoPE(ctx Context, positionIDs, ropeFactors, freqs Tensor, dim uint32, base, scale float32) Tensor

	Tanh(ctx Context) Tensor
	GELU(ctx Context) Tensor
	SILU(ctx Context) Tensor

	Reshape(ctx Context, shape ...int) Tensor
	View(ctx Context, offset int, shape, stride []int) Tensor
	Permute(ctx Context, shape ...int) Tensor
	Contiguous(ctx Context) Tensor

	Pad(ctx Context, shape ...int) Tensor
	Unpad(ctx Context, shape ...int) Tensor

	Stack(ctx Context, dim int, s ...Tensor) Tensor
	Concat(ctx Context, t2 Tensor, dim int) Tensor
	Rows(ctx Context, t2 Tensor) Tensor
	Copy(ctx Context, t2 Tensor) Tensor
	Repeat(ctx Context, repeats, axis int) Tensor
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
	Items int

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
	var f func([]int, int)
	f = func(dims []int, stride int) {
		prefix := strings.Repeat(" ", len(shape)-len(dims)+1)
		fmt.Fprint(&sb, "[")
		defer func() { fmt.Fprint(&sb, "]") }()
		for i := 0; i < dims[0]; i++ {
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

func (dt DType) String() string {
	switch dt {
	case DTypeF32:
		return "float32"
	case DTypeI32:
		return "int32"
	default:
		return "unknon"
	}
}

func (dt DType) Sizeof() int64 {
	// TODO call underlying API?
	switch dt {
	case DTypeF32:
		return 4
	case DTypeI32:
		return 4
	default:
		panic("unrecognized type")
	}
}

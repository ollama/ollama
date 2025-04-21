package ml

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs"
)

type Backend interface {
	Config() fs.Config
	Get(name string) Tensor
	NewContext() Context
	NewContextSize(size int) Context
}

// BackendCacheConfig should be implemented by backends that need special output
// from the cache to meet specific requirements. It is frequently implemented in
// conjunction with ScaledDotProductAttention.
type BackendCacheConfig interface {
	CacheConfig() CacheConfig
}

// CacheConfig controls optimizations (mostly backend-specific) that may transform
// the output the cache to work better with specific kernels.
type CacheConfig struct {
	// CachePadding specifies the multiple for the number of tokens of cache history
	// that will be returned from cache Get for k, v and mask. The capacity of the
	// cache itself will also be increased to a multiple of this size if needed.
	CachePadding int

	// PermutedV performs Permute(ctx, 1, 2, 0, 3) on v tensors stored via Put
	// and return the permuted version via Get. This uses the cache copy operation
	// to avoid a Contiguous call on the permuted tensor.
	PermutedV bool

	// MaskDType specifies the data type for generating the mask. If unset it will
	// default to DTypeF32.
	MaskDType DType

	// MaskBatchPadding specifies the multiple for the batch size dimension in the mask.
	// Any position that does not correspond to an actual token will be filled with -Inf.
	MaskBatchPadding int
}

// BackendParams controls how the backend loads and executes models
type BackendParams struct {
	// Progress is a callback function that allows reporting percentage completion
	// of model loading
	Progress func(float32)

	// NumThreads sets the number of threads to use if running on the CPU
	NumThreads int

	// MainGPU is the index of the primary GPU to use
	MainGPU int

	// NumGPULayers is the number of layers to offload to GPUs
	NumGPULayers int

	// TensorSplit is the fraction of the model to offload to each GPU
	TensorSplit []float32

	// FlashAttention indicates that we should use a fused flash attention kernel
	FlashAttention bool
}

var backends = make(map[string]func(context.Context, *os.File, BackendParams) (Backend, error))

func RegisterBackend(name string, f func(context.Context, *os.File, BackendParams) (Backend, error)) {
	if _, ok := backends[name]; ok {
		panic("backend: backend already registered")
	}

	backends[name] = f
}

func NewBackend(ctx context.Context, f *os.File, params BackendParams) (Backend, error) {
	if backend, ok := backends["ggml"]; ok {
		return backend(ctx, f, params)
	}

	return nil, fmt.Errorf("unsupported backend")
}

type Context interface {
	Empty(dtype DType, shape ...int) Tensor
	Zeros(dtype DType, shape ...int) Tensor
	FromFloatSlice(s []float32, shape ...int) (Tensor, error)
	FromIntSlice(s []int32, shape ...int) (Tensor, error)

	// Arange creates a 1D tensor with values within an interval (start, stop] increased by step.
	Arange(start, stop, step float32, dtype DType) Tensor

	Forward(...Tensor) Context
	Compute(...Tensor)

	// Reserve is analogous to Compute but rather than executing a
	// graph, simply preallocates memory. Typically called with a
	// worst case graph to ensure all resources are available for
	// for future inference.
	Reserve() error

	MaxGraphNodes() int
	Close()

	// Input returns a context appropriate for creating tensors that are
	// inputs to the model (which includes things like output locations)
	Input() Context

	// Layer returns a context appropriate for creating intermediate tensors
	Layer(int) Context
}

type Tensor interface {
	Dim(n int) int
	Stride(n int) int

	Shape() []int
	DType() DType

	Bytes() []byte
	Floats() []float32

	Neg(ctx Context) Tensor
	Add(ctx Context, t2 Tensor) Tensor
	Mul(ctx Context, t2 Tensor) Tensor
	Mulmat(ctx Context, t2 Tensor) Tensor
	MulmatFullPrec(ctx Context, t2 Tensor) Tensor

	Softmax(ctx Context) Tensor
	LayerNorm(ctx Context, weight, bias Tensor, eps float32) Tensor
	RMSNorm(ctx Context, weight Tensor, eps float32) Tensor
	Scale(ctx Context, s float64) Tensor

	AvgPool2D(ctx Context, k, s int, p float32) Tensor
	Conv2D(ctx Context, weight Tensor, s0, s1, p0, p1, d0, d1 int) Tensor

	RoPE(ctx Context, positionIDs, ropeFactors Tensor, dim, ropeType uint32, base, scale float32) Tensor
	IM2Col(ctx Context, weight Tensor, s0, s1, p0, p1, d0, d1 int) Tensor

	Sin(ctx Context) Tensor
	Cos(ctx Context) Tensor
	Tanh(ctx Context) Tensor
	GELU(ctx Context) Tensor
	SILU(ctx Context) Tensor

	Reshape(ctx Context, shape ...int) Tensor
	View(ctx Context, offset int, shape ...int) Tensor
	Permute(ctx Context, shape ...int) Tensor
	Contiguous(ctx Context) Tensor
	Set(ctx Context, t2 Tensor, offset int, strides ...int) Tensor

	Pad(ctx Context, shape ...int) Tensor
	Unpad(ctx Context, shape ...int) Tensor

	Stack(ctx Context, dim int, s ...Tensor) Tensor

	// Repeat repeats the tensor n times along dimension dim
	Repeat(ctx Context, dim, n int) Tensor
	Concat(ctx Context, t2 Tensor, dim int) Tensor
	Rows(ctx Context, t2 Tensor) Tensor
	Copy(ctx Context, t2 Tensor) Tensor
	Duplicate(ctx Context) Tensor
}

// ScaledDotProductAttention implements a fused attention
// operation equivalent to following code on a tensor named
// query:
//
// query = query.Permute(ctx, 0, 2, 1, 3)
// key = key.Permute(ctx, 0, 2, 1, 3)
// value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)
//
// kq := key.MulmatFullPrec(ctx, query)
//
// kq = kq.Scale(ctx, scale)
//
//	if mask != nil {
//		kq = kq.Add(ctx, mask)
//	}
//
// kq = kq.Softmax(ctx)
//
// kqv := value.Mulmat(ctx, kq)
// return kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
type ScaledDotProductAttention interface {
	ScaledDotProductAttention(ctx Context, key, value, mask Tensor, scale float64) Tensor
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

func Dump(ctx Context, t Tensor, opts ...DumpOptions) string {
	if len(opts) < 1 {
		opts = append(opts, DumpOptions{
			Items:     3,
			Precision: 4,
		})
	}

	switch t.DType() {
	case DTypeF32:
		return dump[[]float32](ctx, t, opts[0].Items, func(f float32) string {
			return strconv.FormatFloat(float64(f), 'f', opts[0].Precision, 32)
		})
	case DTypeF16, DTypeQ80, DTypeQ40:
		f32 := ctx.Input().Empty(DTypeF32, t.Shape()...)
		f32 = t.Copy(ctx, f32)
		return dump[[]float32](ctx, f32, opts[0].Items, func(f float32) string {
			return strconv.FormatFloat(float64(f), 'f', opts[0].Precision, 32)
		})
	case DTypeI32:
		return dump[[]int32](ctx, t, opts[0].Items, func(i int32) string {
			return strconv.FormatInt(int64(i), 10)
		})
	default:
		return "<unsupported>"
	}
}

func dump[S ~[]E, E number](ctx Context, t Tensor, items int, fn func(E) string) string {
	if t.Bytes() == nil {
		ctx.Forward(t).Compute(t)
	}

	s := make(S, mul(t.Shape()...))
	if err := binary.Read(bytes.NewBuffer(t.Bytes()), binary.LittleEndian, &s); err != nil {
		panic(err)
	}

	shape := t.Shape()
	slices.Reverse(shape)

	var sb strings.Builder
	var f func([]int, int)
	f = func(dims []int, stride int) {
		prefix := strings.Repeat(" ", len(shape)-len(dims)+1)
		sb.WriteString("[")
		defer func() { sb.WriteString("]") }()
		for i := 0; i < dims[0]; i++ {
			if i >= items && i < dims[0]-items {
				sb.WriteString("..., ")
				// skip to next printable element
				skip := dims[0] - 2*items
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
				text := fn(s[stride+i])
				if len(text) > 0 && text[0] != '-' {
					sb.WriteString(" ")
				}

				sb.WriteString(text)
				if i < dims[0]-1 {
					sb.WriteString(", ")
				}
			}
		}
	}
	f(shape, 0)

	return sb.String()
}

type DType int

const (
	DTypeOther DType = iota
	DTypeF32
	DTypeF16
	DTypeQ80
	DTypeQ40
	DTypeI32
)

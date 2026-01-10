package ml

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs"
)

type Backend interface {
	// Close frees all memory associated with this backend
	Close()

	Load(ctx context.Context, progress func(float32)) error

	// BackendMemory returns the memory allocations that were made for this model
	BackendMemory() BackendMemory

	Config() fs.Config
	Get(name string) Tensor
	NewContext() Context
	NewContextSize(size int) Context

	// Enumerate the devices available for inference via this backend
	BackendDevices() []DeviceInfo
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
}

// BackendParams controls how the backend loads and executes models
type BackendParams struct {
	// AllocMemory causes the backend to allocate memory for the model. If
	// false, this is only being used for discovering the required amount of
	// memory and cannot load the model for running.
	AllocMemory bool

	// NumThreads sets the number of threads to use if running on the CPU
	NumThreads int

	// GPULayers is the set of layers to offload to GPUs
	GPULayers GPULayersList

	// FlashAttention indicates that we should use a fused flash attention kernel
	FlashAttention FlashAttentionType
}

var backends = make(map[string]func(string, BackendParams) (Backend, error))

func RegisterBackend(name string, f func(string, BackendParams) (Backend, error)) {
	if _, ok := backends[name]; ok {
		panic("backend: backend already registered")
	}

	backends[name] = f
}

func NewBackend(modelPath string, params BackendParams) (Backend, error) {
	if backend, ok := backends["ggml"]; ok {
		return backend(modelPath, params)
	}

	return nil, fmt.Errorf("unsupported backend")
}

type Context interface {
	Empty(dtype DType, shape ...int) Tensor
	Zeros(dtype DType, shape ...int) Tensor
	FromBytes(dtype DType, s []byte, shape ...int) Tensor
	FromFloats(s []float32, shape ...int) Tensor
	FromInts(s []int32, shape ...int) Tensor

	// Arange creates a 1D tensor with values within an interval (start, stop] increased by step.
	Arange(start, stop, step float32, dtype DType) Tensor

	Forward(...Tensor) Context

	// SetBatchSize provides a hint on the batch size to optimize processing
	// Uses heuristics if not set
	SetBatchSize(int)

	Compute(...Tensor)
	ComputeWithNotify(func(), ...Tensor) // notify callback once compute has begun

	// Reserve is analogous to Compute but rather than executing a
	// graph, simply preallocates memory. Typically called with a
	// worst case graph to ensure all resources are available for
	// for future inference.
	Reserve()

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
	Cast(ctx Context, dtype DType) Tensor

	Bytes() []byte
	Floats() []float32

	FromBytes([]byte)
	FromFloats([]float32)
	FromInts([]int32)

	Add(ctx Context, t2 Tensor) Tensor
	Sub(ctx Context, t2 Tensor) Tensor
	Mul(ctx Context, t2 Tensor) Tensor
	Div(ctx Context, t2 Tensor) Tensor

	Mulmat(ctx Context, t2 Tensor) Tensor
	MulmatFullPrec(ctx Context, t2 Tensor) Tensor
	MulmatID(ctx Context, t2, ids Tensor) Tensor
	AddID(ctx Context, t2, ids Tensor) Tensor

	Softmax(ctx Context) Tensor
	L2Norm(ctx Context, eps float32) Tensor
	LayerNorm(ctx Context, weight, bias Tensor, eps float32) Tensor
	RMSNorm(ctx Context, weight Tensor, eps float32) Tensor
	Scale(ctx Context, s float64) Tensor
	SumRows(ctx Context) Tensor

	AvgPool2D(ctx Context, k, s int, p float32) Tensor
	Conv2D(ctx Context, weight Tensor, s0, s1, p0, p1, d0, d1 int) Tensor
	Conv3D(ctx Context, weight Tensor, c, s0, s1, s2, p0, p1, p2, d0, d1, d2 int) Tensor

	IM2Col(ctx Context, weight Tensor, s0, s1, p0, p1, d0, d1 int) Tensor

	Sin(ctx Context) Tensor
	Cos(ctx Context) Tensor
	Tanh(ctx Context) Tensor
	GELU(ctx Context, up ...Tensor) Tensor
	QuickGELU(ctx Context, up ...Tensor) Tensor
	SILU(ctx Context, up ...Tensor) Tensor
	RELU(ctx Context, up ...Tensor) Tensor
	Sigmoid(ctx Context) Tensor

	// AlphaLimitSILU is a variant of SILU that clamps the input to the range [-limit, limit]
	SILUAlphaLimit(ctx Context, up Tensor, alpha, limit float32) Tensor

	Reshape(ctx Context, shape ...int) Tensor
	View(ctx Context, offset int, shape ...int) Tensor
	Permute(ctx Context, shape ...int) Tensor
	Contiguous(ctx Context, shape ...int) Tensor

	Pad(ctx Context, shape ...int) Tensor

	Stack(ctx Context, dim int, s ...Tensor) Tensor

	// Repeat repeats the tensor n times along dimension dim
	Repeat(ctx Context, dim, n int) Tensor
	Concat(ctx Context, t2 Tensor, dim int) Tensor
	Rows(ctx Context, t2 Tensor) Tensor
	SetRows(ctx Context, src Tensor, idxs Tensor) Tensor
	Copy(ctx Context, t2 Tensor) Tensor
	Duplicate(ctx Context) Tensor

	Slice(ctx Context, dim, low, high, step int) Tensor
	Chunk(ctx Context, dim int, size int) []Tensor
	ChunkSections(ctx Context, dim int, sections ...int) []Tensor

	TopK(ctx Context, k int) Tensor
	Argsort(ctx Context) Tensor
	Mean(ctx Context) Tensor
	Variance(ctx Context) Tensor
	Stddev(ctx Context) Tensor
	Sqr(ctx Context) Tensor
	Sqrt(ctx Context) Tensor

	Interpolate(ctx Context, dims [4]int, samplingMode SamplingMode) Tensor
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
//
// cacheConfigApplied indicates whether the optimizations requested through CacheConfig have been performed
type ScaledDotProductAttention interface {
	ScaledDotProductAttention(ctx Context, key, value, mask, sinks Tensor, vmla Tensor, scale float64, cacheConfigApplied bool) Tensor
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

type DumpOptions func(*dumpOptions)

// DumpWithPrecision sets the number of decimal places to print. Applies to float32 and float64.
func DumpWithPrecision(n int) DumpOptions {
	return func(opts *dumpOptions) {
		opts.Precision = n
	}
}

// DumpWithThreshold sets the threshold for printing the entire tensor. If the number of elements
// is less than or equal to this value, the entire tensor will be printed. Otherwise, only the
// beginning and end of each dimension will be printed.
func DumpWithThreshold(n int) DumpOptions {
	return func(opts *dumpOptions) {
		opts.Threshold = n
	}
}

// DumpWithEdgeItems sets the number of elements to print at the beginning and end of each dimension.
func DumpWithEdgeItems(n int) DumpOptions {
	return func(opts *dumpOptions) {
		opts.EdgeItems = n
	}
}

type dumpOptions struct {
	Precision, Threshold, EdgeItems int
}

func Dump(ctx Context, t Tensor, optsFuncs ...DumpOptions) string {
	opts := dumpOptions{Precision: 4, Threshold: 1000, EdgeItems: 3}
	for _, optsFunc := range optsFuncs {
		optsFunc(&opts)
	}

	if mul(t.Shape()...) <= opts.Threshold {
		opts.EdgeItems = math.MaxInt
	}

	switch t.DType() {
	case DTypeF32:
		return dump[[]float32](ctx, t, opts.EdgeItems, func(f float32) string {
			return strconv.FormatFloat(float64(f), 'f', opts.Precision, 32)
		})
	case DTypeF16, DTypeQ80, DTypeQ40:
		f32 := ctx.Input().Empty(DTypeF32, t.Shape()...)
		f32 = t.Copy(ctx, f32)
		return dump[[]float32](ctx, f32, opts.EdgeItems, func(f float32) string {
			return strconv.FormatFloat(float64(f), 'f', opts.Precision, 32)
		})
	case DTypeI32:
		return dump[[]int32](ctx, t, opts.EdgeItems, func(i int32) string {
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
	DTypeMXFP4
)

type SamplingMode int

const (
	SamplingModeNearest SamplingMode = iota
	SamplingModeBilinear
)

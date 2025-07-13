package ml

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"log/slog"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs"
)

type Backend interface {
	Load(ctx context.Context, progress func(float32)) error

	// BackendMemory returns the memory allocations that were made for this model
	BackendMemory() BackendMemory

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

// ErrNoMem is returned when panicing due to insufficient memory. It includes
// the attempted memory allocation.
type ErrNoMem struct {
	BackendMemory
}

func (e ErrNoMem) Error() string {
	return fmt.Sprintf("insufficient memory - required allocations: %+v", e.BackendMemory)
}

type AllocationStatus int

const (
	// Unallocated memory - have not yet attempted to allocate
	Unallocated AllocationStatus = iota

	// Failed memory - tried to allocate the memory and did not succeed
	Failed

	// Allocated memory = tried and succeeded to allocate memory
	Allocated
)

// Memory is the size of an allocation and whether it was successful.
type Memory struct {
	Size   uint64
	Status AllocationStatus
}

func (m Memory) String() string {
	s := fmt.Sprint(m.Size)

	switch m.Status {
	case Unallocated:
		s += "U"
	case Failed:
		s += "F"
	case Allocated:
		s += "A"
	}

	return s
}

// DeviceMemory provides a breakdown of the memory needed
// per device, such as a CPU or GPU.
type DeviceMemory struct {
	// Name is the name of the device as labeled by the backend. It
	// may not be persistent across instances of the runner.
	Name string

	// ID is an identifier for the device for matching with system
	// management libraries.
	ID string

	// Weights is the per-layer memory needed for the model weights.
	Weights []Memory

	// Cache is the per-layer memory needed for the KV cache.
	Cache []Memory

	// Graph is the size of the compute graph. It is not per-layer.
	Graph Memory
}

func memoryPresent(mem []Memory) bool {
	return slices.ContainsFunc(mem, func(m Memory) bool { return m.Size != 0 })
}

func (m DeviceMemory) LogValue() slog.Value {
	var attrs []slog.Attr
	if memoryPresent(m.Weights) {
		attrs = append(attrs, slog.Any("Weights", m.Weights))
	}

	if memoryPresent(m.Cache) {
		attrs = append(attrs, slog.Any("Cache", m.Cache))
	}

	if m.Graph.Size != 0 {
		attrs = append(attrs, slog.Any("Graph", m.Graph))
	}

	if len(attrs) > 0 && m.ID != "" {
		attrs = append([]slog.Attr{slog.String("ID", m.ID)}, attrs...)
	}

	return slog.GroupValue(attrs...)
}

// BackendMemory provides the amount of memory required to load the model
// per device based on the BackendParams. In some cases, not all required
// allocations will be known at this point. However, the size of the most recent
// allocation is guaranteed to be provided so that if it failed, the caller can
// accommodate that to make forward progress.
type BackendMemory struct {
	// InputsWeights are always located on the CPU and cannot be moved
	InputWeights Memory

	// CPU model components are located in system memory. This does not
	// include unified memory allocated through the GPU.
	CPU DeviceMemory

	// GPU model components are located on one or more GPUs.
	GPUs []DeviceMemory
}

func (m BackendMemory) LogValue() slog.Value {
	var attrs []slog.Attr
	if m.InputWeights.Size != 0 {
		attrs = append(attrs, slog.Any("InputWeights", m.InputWeights))
	}

	attrs = append(attrs, slog.Any(m.CPU.Name, m.CPU))
	for _, g := range m.GPUs {
		attrs = append(attrs, slog.Any(g.Name, g))
	}

	return slog.GroupValue(attrs...)
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
	FromFloatSlice(s []float32, shape ...int) Tensor
	FromIntSlice(s []int32, shape ...int) Tensor

	// Arange creates a 1D tensor with values within an interval (start, stop] increased by step.
	Arange(start, stop, step float32, dtype DType) Tensor

	Forward(...Tensor) Context
	Compute(...Tensor)

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

	Bytes() []byte
	Floats() []float32

	Neg(ctx Context) Tensor
	Add(ctx Context, t2 Tensor) Tensor
	Sub(ctx Context, t2 Tensor) Tensor
	Mul(ctx Context, t2 Tensor) Tensor
	Div(ctx Context, t2 Tensor) Tensor

	Mulmat(ctx Context, t2 Tensor) Tensor
	MulmatFullPrec(ctx Context, t2 Tensor) Tensor
	MulmatID(ctx Context, t2, ids Tensor) Tensor

	Softmax(ctx Context) Tensor
	LayerNorm(ctx Context, weight, bias Tensor, eps float32) Tensor
	RMSNorm(ctx Context, weight Tensor, eps float32) Tensor
	Scale(ctx Context, s float64) Tensor
	SumRows(ctx Context) Tensor

	AvgPool2D(ctx Context, k, s int, p float32) Tensor
	Conv2D(ctx Context, weight Tensor, s0, s1, p0, p1, d0, d1 int) Tensor

	IM2Col(ctx Context, weight Tensor, s0, s1, p0, p1, d0, d1 int) Tensor

	Sin(ctx Context) Tensor
	Cos(ctx Context) Tensor
	Tanh(ctx Context) Tensor
	GELU(ctx Context) Tensor
	SILU(ctx Context) Tensor
	RELU(ctx Context) Tensor
	Sigmoid(ctx Context) Tensor

	Reshape(ctx Context, shape ...int) Tensor
	View(ctx Context, offset int, shape ...int) Tensor
	Permute(ctx Context, shape ...int) Tensor
	Contiguous(ctx Context) Tensor
	Set(ctx Context, t2 Tensor, offset int, strides ...int) Tensor

	Pad(ctx Context, shape ...int) Tensor

	Stack(ctx Context, dim int, s ...Tensor) Tensor

	// Repeat repeats the tensor n times along dimension dim
	Repeat(ctx Context, dim, n int) Tensor
	Concat(ctx Context, t2 Tensor, dim int) Tensor
	Rows(ctx Context, t2 Tensor) Tensor
	Copy(ctx Context, t2 Tensor) Tensor
	Duplicate(ctx Context) Tensor

	TopK(ctx Context, k int) Tensor
	Argsort(ctx Context) Tensor
	Mean(ctx Context) Tensor
	Variance(ctx Context) Tensor
	Stddev(ctx Context) Tensor
	Sqr(ctx Context) Tensor
	Sqrt(ctx Context) Tensor
	Clamp(ctx Context, min, max float32) Tensor
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
)

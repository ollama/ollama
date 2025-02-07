package mlx

/*
#cgo CPPFLAGS: -I${SRCDIR}/../../../build/_deps/mlx-c-src
#cgo LDFLAGS: -L${SRCDIR}/../../../build/lib/ollama/ -lmlxc -lmlx
#cgo LDFLAGS: -framework Accelerate
#cgo LDFLAGS: -Wl,-rpath,${SRCDIR}/../../../build/lib/ollama/
#include <stdlib.h>
#include "mlx/c/array.h"
#include "mlx/c/fast.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"
#include "mlx/c/transforms.h"
#include "mlx/c/error.h"
static inline size_t stride(const mlx_array a, int i) {return mlx_array_strides(a)[i];}

extern void goStackTrace();
static void error_handler(const char *msg, void* data) {
	fprintf(stderr, "MLX error: %s\n", msg);
	goStackTrace();
	exit(-1);
}
static void set_error_handler() {mlx_set_error_handler(&error_handler, NULL, NULL);}
*/
import "C"

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"unsafe"

	fs "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	"golang.org/x/sync/errgroup"
)

func init() {
	ml.RegisterBackend("mlx", New)
	C.set_error_handler()
}

//export goStackTrace
func goStackTrace() {
	debug.PrintStack()
}

func New(r *os.File) (ml.Backend, error) {
	meta, n, err := fs.Decode(r, -1)
	if err != nil {
		return nil, err
	}

	// TODO all this loading logic will be replaced by the new model loading abstraction, including any necessary transformations

	tensors := make(map[string]*Array, len(meta.Tensors().Items()))
	sr := io.NewSectionReader(r, int64(meta.Tensors().Offset), n-int64(meta.Tensors().Offset))

	slog.Info("initializing MLX GPU backend")
	stream := C.mlx_default_gpu_stream_new()

	var g errgroup.Group
	var mu sync.Mutex
	for _, t := range meta.Tensors().Items() {
		g.Go(func() error {
			var b bytes.Buffer
			n, err := io.Copy(&b, io.NewSectionReader(sr, int64(t.Offset), int64(t.Size())))
			if err != nil {
				return err
			}

			if n != int64(t.Size()) {
				return fmt.Errorf("expected %d bytes, got %d", t.Size(), n)
			}

			cbytes := C.CBytes(b.Bytes())
			defer C.free(cbytes)

			// Inverted
			shape := make([]C.int, len(t.Shape))
			i := len(t.Shape) - 1
			for _, dim := range t.Shape {
				shape[i] = C.int(dim)
				i--
			}

			// TODO Quantization types
			// ref: https://github.com/ml-explore/mlx/blob/main/mlx/io/gguf_quants.cpp
			var dtype C.mlx_dtype
			switch t.Kind {
			case 0:
				dtype = C.MLX_FLOAT32
			case 1:
				dtype = C.MLX_FLOAT16
			default:
				return fmt.Errorf("unsupported dtype %d", t.Kind)
			}

			mu.Lock()
			defer mu.Unlock()

			var a C.mlx_array
			r := C.mlx_array_new_data(
				cbytes,
				(*C.int)(&shape[0]),
				C.int(len(shape)),
				dtype,
			)

			// Q/K are are mutated and we need to reverse that mutation
			// TODO - this is only for llama based models and shouldn't be applied universally
			// but only applies to some backends at the moment...  maybe?
			if strings.HasSuffix(t.Name, "attn_q.weight") || strings.HasSuffix(t.Name, "attn_q.bias") || strings.HasSuffix(t.Name, "attn_k.weight") || strings.HasSuffix(t.Name, "attn_k.bias") {

				// TODO - is this code memory access safe, or does the delayed processing cause potential memory access after Go frees the stack?

				// TODO performance: Since these operations are ~static yet cause a lot of additional nodes in the graph
				// Ideally these should be applied "on the fly" at load time, so the tensor has the data ready to go.
				defer C.mlx_array_free(r)

				var n_head uint64
				if strings.Contains(t.Name, "attn_q") {
					n_head = meta.KV().HeadCount() // Q
				} else {
					n_head = meta.KV().HeadCountKV() // K
				}
				tmpShape := []C.int{C.int(n_head), C.int(math.Floor(math.Floor(float64(shape[0]) / float64(n_head) / float64(2)))), 2, shape[1]}
				var shaped C.mlx_array
				C.mlx_reshape(&shaped, r, (*C.int)(&tmpShape[0]), C.size_t(len(tmpShape)), stream)
				defer C.mlx_array_free(shaped)
				var swapped C.mlx_array
				C.mlx_swapaxes(
					&swapped,
					shaped,
					1,
					2,
					stream,
				)
				defer C.mlx_array_free(swapped)

				var reshaped C.mlx_array
				C.mlx_reshape(
					&reshaped,
					swapped,
					(*C.int)(&shape[0]),
					C.size_t(len(shape)),
					stream,
				)
				defer C.mlx_array_free(reshaped)
				C.mlx_transpose_all(
					&a,
					reshaped,
					stream,
				)
			} else if strings.Contains(t.Name, "token_embd.weight") {
				// TODO bug in model code?  Why is this one special compared to all the rest?
				a = r
			} else {
				// TODO performance: this should be done to the data as it's loaded, not add additional operations in the graph
				C.mlx_transpose_all(
					&a,
					r,
					stream,
				)
				defer C.mlx_array_free(r)
			}

			tmp := newArray(a)
			tmp.name = t.Name
			tensors[t.Name] = tmp
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}

	// TODO - This seems to have zero performance benefit
	// slog.Info("evaling loading ops")
	// v := C.mlx_vector_array_new()
	// for _, a := range tensors {
	// 	C.mlx_vector_array_append_value(v, a.a)
	// }
	// C.mlx_eval(v)
	// slog.Info("finished load")

	return &Backend{
		meta:    meta,
		tensors: tensors,
	}, nil
}

type Backend struct {
	meta    *fs.GGML
	tensors map[string]*Array
}

// Config implements ml.Backend.
func (b *Backend) Config() ml.Config {
	return b.meta.KV()
}

// Get implements ml.Backend.
func (b *Backend) Get(name string) ml.Tensor {
	if a, ok := b.tensors[name]; ok {
		return a
	}

	return nil
}

func (b *Backend) NewContext() ml.Context {
	return &Context{
		stream: C.mlx_default_gpu_stream_new(),
	}
}

type Context struct {
	stream C.mlx_stream
}

// Close implements ml.Context.
func (c *Context) Close() {
	// C.mlx_synchronize(c.stream) // ???
	C.mlx_stream_free(c.stream)
}

// Compute implements ml.Context.
func (c *Context) Compute(t ml.Tensor) ml.Tensor {
	c.Forward(t)
	return t
}

// Forward implements ml.Context.
func (c *Context) Forward(t ml.Tensor) {
	C.mlx_array_eval(t.(*Array).a)
}

// FromFloatSlice implements ml.Context.
func (c *Context) FromFloatSlice(s []float32, shape ...int) (ml.Tensor, error) {
	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		cshape[i] = C.int(dim)
	}
	return newArray(
		C.mlx_array_new_data(
			unsafe.Pointer(&s[0]),
			(*C.int)(&cshape[0]),
			C.int(len(cshape)),
			C.MLX_FLOAT32,
		),
	), nil
}

// FromIntSlice implements ml.Context.
func (c *Context) FromIntSlice(s []int32, shape ...int) (ml.Tensor, error) {
	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		cshape[i] = C.int(dim)
	}
	return newArray(
		C.mlx_array_new_data(
			unsafe.Pointer(&s[0]),
			(*C.int)(&cshape[0]),
			C.int(len(cshape)),
			C.MLX_INT32,
		),
	), nil
}

// Zeros implements ml.Context.
func (c *Context) Zeros(dtype ml.DType, shape ...int) ml.Tensor {
	if len(shape) < 1 || len(shape) > 4 {
		panic("unsupported number of dimensions")
	}
	for _, dim := range shape {
		if dim < 1 {
			panic("invalid shape")
		}
	}
	var dt C.mlx_dtype
	switch dtype {
	case 0:
		dt = C.MLX_FLOAT32
	case 1:
		dt = C.MLX_FLOAT16
	default:
		panic("unsupported dtype")
	}
	sh := make([]C.int, len(shape))
	for i, s := range shape {
		sh[i] = (C.int)(s)
	}

	var r C.mlx_array
	C.mlx_zeros(
		&r,
		&sh[0],
		(C.size_t)(len(sh)),
		dt,
		c.stream,
	)
	return newArray(r)
}

func (c *Context) MaxTensors() int {
	// TODO actually wire up correctly
	return 9999
}

type Array struct {
	name string
	a    C.mlx_array
}

// Wire up finalizer so we can free the underlying mlx array when the Go Array is GC'd
func newArray(a C.mlx_array) *Array {
	// TODO measure impact and if this slows things down, make it conditional on some debugging flag at load time
	var name string
	_, f, l, ok := runtime.Caller(2)
	if ok {
		name = fmt.Sprintf("%s:%d", f, l)
	}

	t := &Array{
		name: name,
		a:    a,
	}
	runtime.SetFinalizer(t, func(a *Array) {
		C.mlx_array_free(a.a)
	})
	return t
}

func (a *Array) LogValue() slog.Value {
	// TODO this forces eval on every log message - find a pattern to make this configurable to aid in debugging
	// str := C.mlx_string_new()
	// C.mlx_array_tostring(&str, a.a)
	// s := C.mlx_string_data(str)
	// defer C.mlx_string_free(str)
	dims := int(C.mlx_array_ndim(a.a))
	strides := make([]int, dims)
	for i := range strides {
		strides[i] = int(C.stride(a.a, (C.int)(i)))
	}

	return slog.GroupValue(
		slog.String("name", a.name),
		slog.Any("shape", a.Shape()),
		slog.Any("strides", strides),
		// slog.String("values", C.GoString(s)),
	)
}

// Add implements ml.Tensor.
func (a *Array) Add(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	var r C.mlx_array
	C.mlx_add(
		&r,
		a.a,
		a2.(*Array).a,
		ctx.(*Context).stream,
	)
	return newArray(r)
}

// Bytes implements ml.Tensor.
func (a *Array) Bytes() []byte {
	l := (int)(C.mlx_array_nbytes(a.a))
	data := C.mlx_array_data_uint8(a.a)
	if data == nil {
		panic("nil data, wasn't eval'd")
	}
	return unsafe.Slice((*byte)(data), l)
}

// Concat implements ml.Tensor.
func (a *Array) Concat(ctx ml.Context, a2 ml.Tensor, dim int) ml.Tensor {
	panic("unimplemented")
}

// Contiguous implements ml.Tensor.
func (a *Array) Contiguous(ctx ml.Context) ml.Tensor {
	var r C.mlx_array
	C.mlx_contiguous(
		&r,
		a.a,
		true, // TODO ???
		ctx.(*Context).stream,
	)
	return newArray(r)
}

// Conv2D implements ml.Tensor.
func (a *Array) Conv2D(ctx ml.Context, weight ml.Tensor, s0 int, s1 int, p0 int, p1 int, d0 int, d1 int) ml.Tensor {
	panic("unimplemented")
}

// Copy implements ml.Tensor.
func (a *Array) Copy(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	C.mlx_copy(
		&a2.(*Array).a,
		a.a,
		ctx.(*Context).stream,
	)
	// TODO - view?
	return newArray(a2.(*Array).a)
}

// DType implements ml.Tensor.
func (a *Array) DType() ml.DType {
	switch C.mlx_array_dtype(a.a) {
	// case	C.MLX_BOOL:
	// case	C.MLX_UINT8:
	// case	C.MLX_UINT16:
	// case	C.MLX_UINT32:
	// case	C.MLX_UINT64:
	// case	C.MLX_INT8:
	// case	C.MLX_INT16:
	case C.MLX_INT32:
		return ml.DTypeI32
	// case	C.MLX_INT64:
	// case	C.MLX_FLOAT16:
	case C.MLX_FLOAT32:
		return ml.DTypeF32
	default:
		panic("unsupported dtype")
	}
}

// Dim implements ml.Tensor.
func (a *Array) Dim(n int) int {
	return int(C.mlx_array_dim(a.a, C.int(n)))
}

// Floats implements ml.Tensor.
func (a *Array) Floats() []float32 {
	f32sLen := (int)(C.mlx_array_size(a.a))
	data := C.mlx_array_data_float32(a.a)
	if data == nil {
		panic("nil data, wasn't eval'd")
	}
	f32s := unsafe.Slice((*float32)(data), f32sLen)
	return f32s
}

// GELU implements ml.Tensor.
func (a *Array) GELU(ctx ml.Context) ml.Tensor {
	panic("unimplemented")
}

// Mul implements ml.Tensor.
func (a *Array) Mul(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	var r C.mlx_array
	C.mlx_multiply(
		&r,
		a.a,
		a2.(*Array).a,
		ctx.(*Context).stream,
	)
	return newArray(r)
}

// Mulmat implements ml.Tensor.
func (a *Array) Mulmat(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	var r C.mlx_array
	s := a.Shape()
	strides := make([]int, len(s))
	for i := range s {
		strides[i] = a.Stride(i)
	}
	sb := a2.Shape()
	stridesb := make([]int, len(sb))
	for i := range sb {
		stridesb[i] = a2.Stride(i)
	}
	C.mlx_matmul(&r,
		a2.(*Array).a,
		a.a,
		ctx.(*Context).stream)
	return newArray(r)
}

// LayerNorm implements ml.Tensor.
func (a *Array) LayerNorm(ctx ml.Context, w, b ml.Tensor, eps float32) ml.Tensor {
	var r C.mlx_array
	C.mlx_fast_layer_norm(
		&r,
		a.a,
		w.(*Array).a,
		b.(*Array).a,
		C.float(eps),
		ctx.(*Context).stream,
	)
	return newArray(r)
}

// Pad implements ml.Tensor.
func (a *Array) Pad(ctx ml.Context, shape ...int) ml.Tensor {
	panic("unimplemented")
}

// Permute implements ml.Tensor.
func (a *Array) Permute(ctx ml.Context, shape ...int) ml.Tensor {
	ndim := min(C.mlx_array_ndim(a.a), C.size_t(len(shape)))
	var r C.mlx_array
	sh := make([]C.int, ndim)
	for i := range ndim {
		sh[i] = (C.int)(shape[i])
		if int(sh[i]) >= int(ndim) {
			slog.Error("Permute error", "tensor", a, "shape", shape)
			panic("invalid pemute call")
		}
	}
	C.mlx_transpose(
		&r,
		a.a,
		&sh[0],
		ndim,
		ctx.(*Context).stream,
	)
	return newArray(r)
}

// RMSNorm implements ml.Tensor.
func (a *Array) RMSNorm(ctx ml.Context, w ml.Tensor, eps float32) ml.Tensor {
	var r C.mlx_array
	C.mlx_fast_rms_norm(
		&r,
		a.a,
		w.(*Array).a,
		C.float(eps),
		ctx.(*Context).stream,
	)
	return newArray(r)
}

// Reshape implements ml.Tensor.
func (a *Array) Reshape(ctx ml.Context, shape ...int) ml.Tensor {
	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		cshape[i] = C.int(dim)
	}
	var r C.mlx_array
	C.mlx_reshape(&r, a.a, (*C.int)(&cshape[0]), C.size_t(len(cshape)), ctx.(*Context).stream)
	return newArray(r)
}

/* MLX breadcrumb for Fast RoPE
a (array) – Input array.
dims (int) – The feature dimensions to be rotated. If the input feature is larger than dims then the rest is left unchanged.
traditional (bool) – If set to True choose the traditional implementation which rotates consecutive dimensions.
base (float, optional) – The base used to compute angular frequency for each dimension in the positional encodings. Exactly one of base and freqs must be None.
scale (float) – The scale used to scale the positions.
offset (int or array) – The position offset to start at.
freqs (array, optional) – Optional frequencies to use with RoPE. If set, the base parameter must be None. Default: None.
*/

// Rope implements ml.Tensor.
func (a *Array) RoPE(
	ctx ml.Context,
	positionIDs ml.Tensor, // Unused in MLX
	ropeFactors ml.Tensor, // Unused in MLX
	freqs ml.Tensor,
	dim uint32,
	base float32,
	scale float32,
) ml.Tensor {
	a = a.Reshape(ctx, append([]int{1}, a.Shape()...)...).Permute(ctx, 0, 2, 1, 3).(*Array)
	// TODO figure out how to get offset wired up
	offset := 0
	var r C.mlx_array
	var b C.mlx_optional_float
	var _freqs C.mlx_array
	if base == 0 {
		base = 10000
	}
	if freqs == nil || len(freqs.Shape()) == 0 {
		b.value = C.float(base)
		b.has_value = true
	} else {
		_freqs = freqs.(*Array).a
	}

	C.mlx_fast_rope(
		&r,
		a.a,
		C.int(dim),
		false, // traditional=false
		b,
		C.float(scale),
		C.int(offset),
		_freqs,
		ctx.(*Context).stream,
	)

	res := newArray(r).Permute(ctx, 0, 2, 1, 3)
	return res.Reshape(ctx, res.Shape()[1:]...)
}

// Rows implements ml.Tensor.
func (a *Array) Rows(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	var r C.mlx_array

	// HACK!
	// If the indicies is greater than 2 dimensions, assume axis 1
	var axis C.int
	if C.mlx_array_ndim(a2.(*Array).a) > 1 {
		axis = 1
	} else {
		axis = 0
	}
	C.mlx_take(&r, a.a, a2.(*Array).a, axis, ctx.(*Context).stream)
	return newArray(r)
}

// SILU implements ml.Tensor.
func (a *Array) SILU(ctx ml.Context) ml.Tensor {
	var sig C.mlx_array
	C.mlx_sigmoid(
		&sig,
		a.a,
		ctx.(*Context).stream,
	)
	var r C.mlx_array
	C.mlx_multiply(
		&r,
		a.a,
		sig,
		ctx.(*Context).stream,
	)
	return newArray(r)
}

// Scale implements ml.Tensor.
func (a *Array) Scale(ctx ml.Context, s float64) ml.Tensor {
	scale := C.mlx_array_new_float(C.float(s))
	var r C.mlx_array
	C.mlx_multiply(
		&r,
		a.a,
		scale,
		ctx.(*Context).stream,
	)
	return newArray(r)
}

// Shape implements ml.Tensor.
func (a *Array) Shape() []int {
	shape := make([]int, C.mlx_array_ndim(a.a))
	for i := range shape {
		shape[i] = int(C.mlx_array_dim(a.a, C.int(i)))
	}

	return shape
}

// Softmax implements ml.Tensor.
func (a *Array) Softmax(ctx ml.Context) ml.Tensor {
	var r C.mlx_array
	axes := []C.int{-1}
	C.mlx_softmax(
		&r,
		a.a,
		&axes[0],
		C.size_t(len(axes)),
		false, //TODO - precise?
		ctx.(*Context).stream,
	)
	return newArray(r)
}

// Stack implements ml.Tensor.
func (a *Array) Stack(ctx ml.Context, dim int, s ...ml.Tensor) ml.Tensor {
	panic("unimplemented")
}

// Stride implements ml.Tensor.
func (a *Array) Stride(n int) int {
	return (int)(C.stride(a.a, (C.int)(n)))
}

// Tanh implements ml.Tensor.
func (a *Array) Tanh(ctx ml.Context) ml.Tensor {
	panic("unimplemented")
}

// Unpad implements ml.Tensor.
func (a *Array) Unpad(ctx ml.Context, shape ...int) ml.Tensor {
	panic("unimplemented")
}

// View implements ml.Tensor.
func (a *Array) View(ctx ml.Context, offset int, shape []int, stride []int) ml.Tensor {
	if len(stride)+1 != len(shape) {
		panic(fmt.Sprintf("malformed view request: shape=%v stride=%v", shape, stride))
	}

	var r C.mlx_array
	var sh []C.int
	var st []C.size_t
	var stp *C.size_t
	switch len(shape) {
	case 1:
		sh = []C.int{
			C.int(shape[0]),
		}
	case 2:
		sh = []C.int{
			C.int(shape[0]),
			C.int(shape[1]),
		}
		// st = []C.size_t{
		// 	C.size_t(stride[0]),
		// }
	case 3:
		sh = []C.int{
			C.int(shape[0]),
			C.int(shape[1]),
			C.int(shape[2]),
		}
		// st = []C.size_t{
		// 	C.size_t(stride[0]),
		// 	C.size_t(stride[1]),
		// }
	case 4:
		sh = []C.int{
			C.int(shape[0]),
			C.int(shape[1]),
			C.int(shape[2]),
			C.int(shape[3]),
		}
		// st = []C.size_t{
		// 	C.size_t(stride[0]),
		// 	C.size_t(stride[1]),
		// 	C.size_t(stride[2]),
		// }
	default:
		panic("unsupported number of dimensions")
	}
	if len(st) > 0 {
		stp = (*C.size_t)(unsafe.Pointer(&st[0]))
	}
	C.mlx_as_strided(
		&r,
		a.a,
		(*C.int)(unsafe.Pointer(&sh[0])),
		C.size_t(len(sh)),
		stp,
		C.size_t(len(st)),
		C.size_t(offset),
		ctx.(*Context).stream,
	)

	return newArray(r)
}

func (a *Array) Repeat(ctx ml.Context, repeats, axis int) ml.Tensor {
	var r C.mlx_array
	C.mlx_repeat(
		&r,
		a.a,
		(C.int)(repeats),
		(C.int)(axis),
		ctx.(*Context).stream)
	return newArray(r)
}

func (ctx *Context) FastScaledDotProductAttention(queries, keys, values ml.Tensor, scale float32, mask ml.Tensor) ml.Tensor {
	var r C.mlx_array
	var m C.mlx_array
	if mask != nil {
		m = mask.(*Array).a
	}

	C.mlx_fast_scaled_dot_product_attention(
		&r,
		queries.(*Array).a,
		keys.(*Array).a,
		values.(*Array).a,
		C.float(scale),
		m,
		C.mlx_optional_int{},
		ctx.stream,
	)
	return newArray(r)
}

func (ctx *Context) SliceUpdate(target, source ml.Tensor, start, stop, strides []int) {
	cStart := make([]C.int, len(start))
	for i := range start {
		cStart[i] = C.int(start[i])
	}
	cStop := make([]C.int, len(stop))
	for i := range stop {
		cStop[i] = C.int(stop[i])
	}
	cStrides := make([]C.int, len(strides))
	for i := range strides {
		cStrides[i] = C.int(strides[i])
	}
	C.mlx_slice_update(
		&target.(*Array).a,
		target.(*Array).a,
		source.(*Array).a,
		(*C.int)(unsafe.Pointer(&cStart[0])),
		C.size_t(len(cStart)),
		(*C.int)(unsafe.Pointer(&cStop[0])),
		C.size_t(len(cStop)),
		(*C.int)(unsafe.Pointer(&cStrides[0])),
		C.size_t(len(cStrides)),
		ctx.stream,
	)
}

// TODO remove this before merging - temporary debugging aid
func (c *Context) Abort(t ml.Tensor) {
	// str := C.mlx_string_new()
	// C.mlx_array_tostring(&str, t.(*Array).a)
	// s := C.mlx_string_data(str)
	// defer C.mlx_string_free(str)
	debug.PrintStack()
	// fmt.Printf("shape%v\n", t.Shape())
	// fmt.Println(C.GoString(s))

	c.Compute(t)
	f32 := t.Floats()

	filename := os.Getenv("OLLAMA_BACKEND") + ".json"
	slog.Info("Writing tensors to", "filename", filename)
	f, err := os.Create(filename)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	encoder := json.NewEncoder(f)
	err = encoder.Encode(f32)
	if err != nil {
		panic(err)
	}

	os.Exit(1)
}

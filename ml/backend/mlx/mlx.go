package mlx

// #cgo CPPFLAGS: -I${SRCDIR}/../../../build/_deps/mlx-c-src
// #cgo LDFLAGS: -L${SRCDIR}/../../../build/lib -lmlxc -lmlx
// #cgo LDFLAGS: -framework Accelerate
// #cgo LDFLAGS: -Wl,-rpath,${SRCDIR}/../../../build/lib
// #include <stdlib.h>
// #include "mlx/c/array.h"
// #include "mlx/c/fast.h"
// #include "mlx/c/ops.h"
// #include "mlx/c/stream.h"
import "C"

import (
	"bytes"
	"fmt"
	"io"
	"log/slog"
	"os"
	"sync"
	"unsafe"

	fs "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	"golang.org/x/sync/errgroup"
)

func init() {
	ml.RegisterBackend("mlx", New)
}

func New(r *os.File) (ml.Backend, error) {
	meta, n, err := fs.Decode(r, -1)
	if err != nil {
		return nil, err
	}

	tensors := make(map[string]*Array, len(meta.Tensors().Items()))
	sr := io.NewSectionReader(r, int64(meta.Tensors().Offset), n-int64(meta.Tensors().Offset))

	stream := C.mlx_default_cpu_stream_new()

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

			shape := make([]C.int, len(t.Shape))
			for i, dim := range t.Shape {
				shape[i] = C.int(dim)
			}

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
			C.mlx_transpose_all(
				&a,
				C.mlx_array_new_data(
					cbytes,
					(*C.int)(&shape[0]),
					C.int(len(shape)),
					dtype,
				),
				stream,
			)

			tensors[t.Name] = &Array{
				name: t.Name,
				a:    a,
			}
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}

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
		stream: C.mlx_default_cpu_stream_new(),
	}
}

type Context struct {
	stream C.mlx_stream
}

// Close implements ml.Context.
func (c *Context) Close() error {
	panic("unimplemented")
}

// Compute implements ml.Context.
func (c *Context) Compute(ml.Tensor) ml.Tensor {
	panic("unimplemented")
}

// Forward implements ml.Context.
func (c *Context) Forward(ml.Tensor) {
	panic("unimplemented")
}

// FromFloatSlice implements ml.Context.
func (c *Context) FromFloatSlice(s []float32, shape ...int) (ml.Tensor, error) {
	panic("unimplemented")
}

// FromIntSlice implements ml.Context.
func (c *Context) FromIntSlice(s []int32, shape ...int) (ml.Tensor, error) {
	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		cshape[i] = C.int(dim)
	}

	return &Array{
		a: C.mlx_array_new_data(
			unsafe.Pointer(&s[0]),
			(*C.int)(&cshape[0]),
			C.int(len(cshape)),
			C.MLX_INT32,
		),
	}, nil
}

// Zeros implements ml.Context.
func (c *Context) Zeros(dtype ml.DType, shape ...int) ml.Tensor {
	panic("unimplemented")
}

type Array struct {
	name string
	a    C.mlx_array
}

func (a *Array) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("name", a.name),
		slog.Any("shape", a.Shape()),
	)
}

// Add implements ml.Tensor.
func (a *Array) Add(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	panic("unimplemented")
}

// Bytes implements ml.Tensor.
func (a *Array) Bytes() []byte {
	panic("unimplemented")
}

// Concat implements ml.Tensor.
func (a *Array) Concat(ctx ml.Context, a2 ml.Tensor, dim int) ml.Tensor {
	panic("unimplemented")
}

// Contiguous implements ml.Tensor.
func (a *Array) Contiguous(ctx ml.Context) ml.Tensor {
	panic("unimplemented")
}

// Conv2D implements ml.Tensor.
func (a *Array) Conv2D(ctx ml.Context, weight ml.Tensor, s0 int, s1 int, p0 int, p1 int, d0 int, d1 int) ml.Tensor {
	panic("unimplemented")
}

// Copy implements ml.Tensor.
func (a *Array) Copy(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	panic("unimplemented")
}

// DType implements ml.Tensor.
func (a *Array) DType() ml.DType {
	panic("unimplemented")
}

// Dim implements ml.Tensor.
func (a *Array) Dim(n int) int64 {
	return int64(C.mlx_array_dim(a.a, C.int(n)))
}

// Floats implements ml.Tensor.
func (a *Array) Floats() []float32 {
	panic("unimplemented")
}

// GELU implements ml.Tensor.
func (a *Array) GELU(ctx ml.Context) ml.Tensor {
	panic("unimplemented")
}

// Mul implements ml.Tensor.
func (a *Array) Mul(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	panic("unimplemented")
}

// Mulmat implements ml.Tensor.
func (a *Array) Mulmat(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	slog.Info("mulmat", "a", a, "a2", a2)
	var r C.mlx_array
	C.mlx_matmul(&r, a2.(*Array).a, a.Permute(1, 0, 2, 3), ctx.(*Context).stream)
	return &Array{a: r}
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
	return &Array{a: r}
}

// Pad implements ml.Tensor.
func (a *Array) Pad(ctx ml.Context, shape ...int64) ml.Tensor {
	panic("unimplemented")
}

// Permute implements ml.Tensor.
func (a *Array) Permute(ctx ml.Context, shape ...int) ml.Tensor {
	panic("unimplemented")
}

// RMSNorm implements ml.Tensor.
func (a *Array) RMSNorm(ctx ml.Context, w, b ml.Tensor, eps float32) ml.Tensor {
	var r C.mlx_array
	C.mlx_fast_rms_norm(
		&r,
		a.a,
		w.(*Array).a,
		C.float(eps),
		ctx.(*Context).stream,
	)
	return &Array{a: r}
}

// Reshape implements ml.Tensor.
func (a *Array) Reshape(ctx ml.Context, shape ...int64) ml.Tensor {
	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		cshape[i] = C.int(dim)
	}

	var r C.mlx_array
	C.mlx_reshape(&r, a.a, (*C.int)(&cshape[0]), C.size_t(len(cshape)), ctx.(*Context).stream)
	return &Array{a: r}
}

// Rope implements ml.Tensor.
func (a *Array) Rope(ctx ml.Context, positionIDs ml.Tensor, ropeFactors ml.Tensor, dim uint32, base float32, scale float32) ml.Tensor {
	panic("unimplemented")
}

// Rows implements ml.Tensor.
func (a *Array) Rows(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	var r C.mlx_array
	slog.Info("rows", "a", a, "a2", a2)
	C.mlx_take(&r, a.a, a2.(*Array).a, 0, ctx.(*Context).stream)
	return &Array{a: r}
}

// SILU implements ml.Tensor.
func (a *Array) SILU(ctx ml.Context) ml.Tensor {
	panic("unimplemented")
}

// Scale implements ml.Tensor.
func (a *Array) Scale(ctx ml.Context, s float64) ml.Tensor {
	panic("unimplemented")
}

// Shape implements ml.Tensor.
func (a *Array) Shape() []int64 {
	shape := make([]int64, C.mlx_array_ndim(a.a))
	for i := range shape {
		shape[i] = int64(C.mlx_array_dim(a.a, C.int(i)))
	}

	return shape
}

// Softmax implements ml.Tensor.
func (a *Array) Softmax(ctx ml.Context) ml.Tensor {
	panic("unimplemented")
}

// Stack implements ml.Tensor.
func (a *Array) Stack(ctx ml.Context, dim int, s ...ml.Tensor) ml.Tensor {
	panic("unimplemented")
}

// Stride implements ml.Tensor.
func (a *Array) Stride(n int) int64 {
	panic("unimplemented")
}

// Tanh implements ml.Tensor.
func (a *Array) Tanh(ctx ml.Context) ml.Tensor {
	panic("unimplemented")
}

// Unpad implements ml.Tensor.
func (a *Array) Unpad(ctx ml.Context, shape ...int64) ml.Tensor {
	panic("unimplemented")
}

// View implements ml.Tensor.
func (a *Array) View(ctx ml.Context, offset int, shape ...int) ml.Tensor {
	panic("unimplemented")
}

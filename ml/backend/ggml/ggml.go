package ggml

// #cgo CPPFLAGS: -I${SRCDIR}/ggml/include
// #include <stdlib.h>
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-cpu.h"
// #include "ggml-backend.h"
import "C"

import (
	"bytes"
	"fmt"
	"io"
	"log/slog"
	"os"
	"unsafe"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"

	_ "github.com/ollama/ollama/ml/backend/ggml/ggml"
	_ "github.com/ollama/ollama/ml/backend/ggml/ggml/ggml-cpu"
)

func newCPUBackend() *C.struct_ggml_backend {
	return C.ggml_backend_cpu_init()
}

type Backend struct {
	c  *C.struct_ggml_context
	b  *C.struct_ggml_backend
	bb *C.struct_ggml_backend_buffer

	ggml.KV
	ggml.Tensors
}

func New(r *os.File) (ml.Backend, error) {
	f, _, err := ggml.Decode(r, -1)
	if err != nil {
		return nil, err
	}

	slog.Info(
		"",
		"architecture", f.KV().Architecture(),
		"file_type", f.KV().FileType(),
		"name", f.KV().String("general.name"),
		"description", f.KV().String("general.description"),
		"num_tensors", len(f.Tensors().Items),
		"num_key_values", len(f.KV()),
	)

	c := C.ggml_init(C.struct_ggml_init_params{
		mem_size:   C.size_t(len(f.Tensors().Items)) * C.ggml_tensor_overhead(),
		mem_buffer: nil,
		no_alloc:   true,
	})

	for _, t := range f.Tensors().Items {
		func() {
			cname := C.CString(t.Name)
			defer C.free(unsafe.Pointer(cname))

			tt := C.ggml_new_tensor(c, t.Kind, C.int(len(t.Shape)), (*C.int64_t)(unsafe.Pointer(&t.Shape[0])))
			C.ggml_set_name(tt, cname)
		}()
	}

	b := newBackend()
	bb := C.ggml_backend_alloc_ctx_tensors(c, b)

	var g errgroup.Group
	for _, t := range f.Tensors().Items {
		g.Go(func() error {
			var b bytes.Buffer
			n, err := io.Copy(&b, io.NewSectionReader(r, int64(f.Tensors().Offset+t.Offset), int64(t.Size())))
			if err != nil {
				return err
			}

			if n != int64(t.Size()) {
				return fmt.Errorf("expected %d bytes, got %d", t.Size(), n)
			}

			cname := C.CString(t.Name)
			defer C.free(unsafe.Pointer(cname))

			cbytes := C.CBytes(b.Bytes())
			defer C.free(cbytes)

			C.ggml_backend_tensor_set(C.ggml_get_tensor(c, cname), cbytes, 0, C.size_t(n))
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}

	return &Backend{c, b, bb, f.KV(), f.Tensors()}, nil
}

func init() {
	ml.RegisterBackend("ggml", New)
}

func (b *Backend) Config() ml.Config {
	return b.KV
}

func (b *Backend) Get(name string) ml.Tensor {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	if t := C.ggml_get_tensor(b.c, cname); t != nil {
		return &Tensor{t}
	}

	return nil
}

func (b *Backend) NewContext() ml.Context {
	n := max(8192, len(b.Tensors.Items)*5)
	bts := make([]byte, C.size_t(n)*C.ggml_tensor_overhead()+C.ggml_graph_overhead_custom(C.size_t(n), false))
	c := C.ggml_init(C.struct_ggml_init_params{
		mem_buffer: unsafe.Pointer(&bts[0]),
		mem_size:   C.size_t(len(bts)),
		no_alloc:   true,
	})
	return &Context{
		b: b.b,
		c: c,
		g: C.ggml_new_graph_custom(c, C.size_t(n), false),
	}
}

type Context struct {
	b *C.struct_ggml_backend
	c *C.struct_ggml_context
	g *C.struct_ggml_cgraph
}

func (c *Context) Forward(t ml.Tensor) {
	C.ggml_build_forward_expand(c.g, t.(*Tensor).t)
}

func (c *Context) Compute(t ml.Tensor) ml.Tensor {
	c.Forward(t)

	a := C.ggml_gallocr_new(C.ggml_backend_get_default_buffer_type(c.b))
	C.ggml_gallocr_alloc_graph(a, c.g)
	slog.Debug("compute graph memory", "require", format.HumanBytes2(uint64(C.ggml_gallocr_get_buffer_size(a, 0))))

	C.ggml_backend_graph_compute(c.b, c.g)
	return &Tensor{
		C.ggml_graph_node(c.g, C.ggml_graph_n_nodes(c.g)-1),
	}
}

func (c Context) Zeros(dtype ml.DType, shape ...int) ml.Tensor {
	if len(shape) < 1 || len(shape) > 4 {
		panic("unsupported number of dimensions")
	}

	for _, dim := range shape {
		if dim < 1 {
			panic("invalid shape")
		}
	}

	var t *C.struct_ggml_tensor
	switch dtype {
	case ml.DTypeF32:
		t = C.ggml_new_tensor(c.c, C.GGML_TYPE_F32, C.int(len(shape)), (*C.int64_t)(unsafe.Pointer(&shape[0])))
	case ml.DTypeI32:
		t = C.ggml_new_tensor(c.c, C.GGML_TYPE_I32, C.int(len(shape)), (*C.int64_t)(unsafe.Pointer(&shape[0])))
	default:
		panic("unsupported dtype")
	}

	b := C.ggml_backend_alloc_buffer(c.b, C.ggml_nbytes(t))
	C.ggml_backend_tensor_alloc(b, t, C.ggml_backend_buffer_get_base(b))
	C.ggml_set_f32(t, 0.)
	return &Tensor{t}
}

func fromSlice[S ~[]E, E float32 | int32](ctx Context, s S, shape []int, dtype uint32) (ml.Tensor, error) {
	n := len(s)
	for _, v := range shape {
		n /= v
	}

	if n != 1 {
		return nil, fmt.Errorf("invalid shape %v for %d elements", shape, len(s))
	}

	t := C.ggml_new_tensor(ctx.c, dtype, C.int(len(shape)), (*C.int64_t)(unsafe.Pointer(&shape[0])))
	b := C.ggml_backend_alloc_buffer(ctx.b, C.ggml_nbytes(t))
	C.ggml_backend_tensor_alloc(b, t, C.ggml_backend_buffer_get_base(b))
	C.ggml_backend_tensor_set(t, unsafe.Pointer(&s[0]), 0, C.ggml_nbytes(t))
	return &Tensor{t}, nil
}

func (c Context) FromFloatSlice(s []float32, shape ...int) (ml.Tensor, error) {
	return fromSlice(c, s, shape, C.GGML_TYPE_F32)
}

func (c Context) FromIntSlice(s []int32, shape ...int) (ml.Tensor, error) {
	return fromSlice(c, s, shape, C.GGML_TYPE_I32)
}

func (c *Context) Close() error {
	C.ggml_free(c.c)
	return nil
}

type Tensor struct {
	t *C.struct_ggml_tensor
}

func (t *Tensor) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("name", C.GoString(C.ggml_get_name(t.t))),
		slog.String("type", C.GoString(C.ggml_type_name(t.t._type))),
		slog.Any("shape", t.Shape()),
	)
}

func (t *Tensor) Dim(n int) int64 {
	return int64(t.t.ne[n])
}

func (t *Tensor) Stride(n int) int64 {
	return int64(t.t.nb[n])
}

func (t *Tensor) Shape() []int64 {
	shape := make([]int64, C.ggml_n_dims(t.t))
	for i := range shape {
		shape[i] = t.Dim(i)
	}

	return shape
}

func (t *Tensor) Bytes() []byte {
	if bts := C.ggml_get_data(t.t); bts != nil {
		return C.GoBytes(bts, C.int(C.ggml_nbytes(t.t)))
	}

	return nil
}

func (t *Tensor) Floats() []float32 {
	if s := C.ggml_get_data_f32(t.t); s != nil {
		f32s := make([]float32, C.ggml_nelements(t.t))
		for i, v := range unsafe.Slice(s, C.ggml_nelements(t.t)) {
			f32s[i] = float32(v)
		}

		return f32s
	}

	return nil
}

func (t *Tensor) DType() ml.DType {
	switch t.t._type {
	case C.GGML_TYPE_F32:
		return ml.DTypeF32
	case C.GGML_TYPE_I32:
		return ml.DTypeI32
	default:
		return ml.DTypeOther
	}
}

func (t *Tensor) Add(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		C.ggml_add(ctx.(*Context).c, t.t, t2.(*Tensor).t),
	}
}

func (t *Tensor) Stack(ctx ml.Context, dim int, s ...ml.Tensor) ml.Tensor {
	if len(s) > 0 {
		return t.Concat(ctx, s[0].Stack(ctx, dim, s[1:]...), dim)
	}

	return t
}

func (t *Tensor) Concat(ctx ml.Context, t2 ml.Tensor, dim int) ml.Tensor {
	return &Tensor{
		C.ggml_concat(ctx.(*Context).c, t.t, t2.(*Tensor).t, C.int(dim)),
	}
}

func (t *Tensor) Contiguous(ctx ml.Context) ml.Tensor {
	return &Tensor{
		C.ggml_cont(ctx.(*Context).c, t.t),
	}
}

func (t *Tensor) Mul(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		C.ggml_mul(ctx.(*Context).c, t.t, t2.(*Tensor).t),
	}
}

func (t *Tensor) Mulmat(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		C.ggml_mul_mat(ctx.(*Context).c, t.t, t2.(*Tensor).t),
	}
}

func (t *Tensor) Norm(ctx ml.Context, eps float32) ml.Tensor {
	return &Tensor{
		C.ggml_norm(ctx.(*Context).c, t.t, (C.float)(eps)),
	}
}

func (t *Tensor) RMSNorm(ctx ml.Context, eps float32) ml.Tensor {
	return &Tensor{
		C.ggml_rms_norm(ctx.(*Context).c, t.t, C.float(eps)),
	}
}

func (t *Tensor) Pad(ctx ml.Context, shape ...int64) ml.Tensor {
	if len(shape) != 4 {
		panic("expected 4 dimensions")
	}

	return &Tensor{
		C.ggml_pad(ctx.(*Context).c, t.t, C.int(shape[0]), C.int(shape[1]), C.int(shape[2]), C.int(shape[3])),
	}
}

func (t *Tensor) Permute(ctx ml.Context, shape ...int) ml.Tensor {
	if len(shape) != 4 {
		panic("expected 4 dimensions")
	}

	return &Tensor{
		C.ggml_permute(ctx.(*Context).c, t.t, C.int(shape[0]), C.int(shape[1]), C.int(shape[2]), C.int(shape[3])),
	}
}

func (t *Tensor) Rows(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		C.ggml_get_rows(ctx.(*Context).c, t.t, t2.(*Tensor).t),
	}
}

func (t *Tensor) Copy(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		C.ggml_cpy(ctx.(*Context).c, t.t, t2.(*Tensor).t),
	}
}

func (t *Tensor) Reshape(ctx ml.Context, shape ...int64) ml.Tensor {
	switch len(shape) {
	case 1:
		return &Tensor{
			C.ggml_reshape_1d(ctx.(*Context).c, t.t, C.int64_t(shape[0])),
		}
	case 2:
		return &Tensor{
			C.ggml_reshape_2d(ctx.(*Context).c, t.t, C.int64_t(shape[0]), C.int64_t(shape[1])),
		}
	case 3:
		return &Tensor{
			C.ggml_reshape_3d(ctx.(*Context).c, t.t, C.int64_t(shape[0]), C.int64_t(shape[1]), C.int64_t(shape[2])),
		}
	case 4:
		return &Tensor{
			C.ggml_reshape_4d(ctx.(*Context).c, t.t, C.int64_t(shape[0]), C.int64_t(shape[1]), C.int64_t(shape[2]), C.int64_t(shape[3])),
		}
	default:
		panic("unsupported number of dimensions")
	}
}

func (t *Tensor) Scale(ctx ml.Context, s float64) ml.Tensor {
	return &Tensor{
		C.ggml_scale(ctx.(*Context).c, t.t, (C.float)(s)),
	}
}

func (t *Tensor) Softmax(ctx ml.Context) ml.Tensor {
	return &Tensor{
		C.ggml_soft_max(ctx.(*Context).c, t.t),
	}
}

func (t *Tensor) Tanh(ctx ml.Context) ml.Tensor {
	return &Tensor{
		C.ggml_tanh_inplace(ctx.(*Context).c, t.t),
	}
}

func (t *Tensor) Unpad(ctx ml.Context, shape ...int64) ml.Tensor {
	if len(shape) != 4 {
		panic("expected 4 dimensions")
	}

	return &Tensor{
		C.ggml_unpad(ctx.(*Context).c, t.t, C.int(shape[0]), C.int(shape[1]), C.int(shape[2]), C.int(shape[3])),
	}
}

func (t *Tensor) View(ctx ml.Context, offset int, shape ...int) ml.Tensor {
	switch len(shape) {
	case 1:
		return &Tensor{
			C.ggml_view_1d(ctx.(*Context).c, t.t, C.int64_t(shape[0]), C.size_t(offset)),
		}
	case 3:
		return &Tensor{
			C.ggml_view_2d(ctx.(*Context).c, t.t,
				C.int64_t(shape[0]), C.int64_t(shape[2]),
				C.size_t(shape[1]),
				C.size_t(offset)),
		}
	case 5:
		return &Tensor{
			C.ggml_view_3d(ctx.(*Context).c, t.t,
				C.int64_t(shape[0]), C.int64_t(shape[2]), C.int64_t(shape[4]),
				C.size_t(shape[1]), C.size_t(shape[3]),
				C.size_t(offset)),
		}
	case 7:
		return &Tensor{
			C.ggml_view_4d(ctx.(*Context).c, t.t,
				C.int64_t(shape[0]), C.int64_t(shape[2]), C.int64_t(shape[4]), C.int64_t(shape[6]),
				C.size_t(shape[1]), C.size_t(shape[3]), C.size_t(shape[5]),
				C.size_t(offset)),
		}
	default:
		panic("unsupported number of dimensions")
	}
}

const (
	ropeTypeNorm C.int = iota
)

func (t *Tensor) Rope(ctx ml.Context, positionIDs, ropeFactors ml.Tensor, ropeDim uint32, ropeBase, ropeScale float32) ml.Tensor {
	return &Tensor{
		C.ggml_rope_ext(
			ctx.(*Context).c, t.t, positionIDs.(*Tensor).t, ropeFactors.(*Tensor).t,
			C.int(ropeDim),
			131072,       // YaRN n_ctx_train
			ropeTypeNorm, // ROPE_TYPE_NORM
			C.float(ropeBase),
			C.float(ropeScale),
			0.,  // YaRN ext_factor
			1.,  // YaRN attn_factor
			32., // YaRN beta_fast
			1.,  // YaRN beta_slow
		),
	}
}

func (t *Tensor) GELU(ctx ml.Context) ml.Tensor {
	return &Tensor{
		C.ggml_gelu_inplace(ctx.(*Context).c, t.t),
	}
}

func (t *Tensor) SILU(ctx ml.Context) ml.Tensor {
	return &Tensor{
		C.ggml_silu_inplace(ctx.(*Context).c, t.t),
	}
}

func (t *Tensor) Conv2D(ctx ml.Context, t2 ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	return &Tensor{
		C.ggml_conv_2d(ctx.(*Context).c, t.t, t2.(*Tensor).t, C.int(s0), C.int(s1), C.int(p0), C.int(p1), C.int(d0), C.int(d1)),
	}
}

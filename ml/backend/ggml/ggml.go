package ggml

/*
#cgo CPPFLAGS: -I${SRCDIR}/ggml/include
#include <stdlib.h>
#include <stdint.h>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
static struct ggml_backend_feature * getBackendFeatures(void *fp, ggml_backend_reg_t reg) {return ((ggml_backend_get_features_t)(fp))(reg);}
static struct ggml_backend_feature * getNextBackendFeatures(struct ggml_backend_feature * feature) { return &feature[1];}

typedef enum {COMP_UNKNOWN,COMP_GCC,COMP_CLANG} COMPILER;
COMPILER inline get_compiler() {
#if defined(__clang__)
	return COMP_CLANG;
#elif defined(__GNUC__)
	return COMP_GCC;
#else
	return UNKNOWN_COMPILER;
#endif
}

*/
import "C"

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"sync"
	"unsafe"

	"github.com/ollama/ollama/format"
	fs "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	"golang.org/x/sync/errgroup"

	ggml "github.com/ollama/ollama/ml/backend/ggml/ggml/src"
)

type device struct {
	d *C.struct_ggml_backend_device
}

func (d device) LogValue() slog.Value {
	var free, total uint64
	C.ggml_backend_dev_memory(d.d, (*C.size_t)(&free), (*C.size_t)(&total))

	kind := "unknown"
	switch C.ggml_backend_dev_type(d.d) {
	case C.GGML_BACKEND_DEVICE_TYPE_CPU:
		kind = "cpu"
	case C.GGML_BACKEND_DEVICE_TYPE_GPU:
		kind = "gpu"
	case C.GGML_BACKEND_DEVICE_TYPE_ACCEL:
		kind = "accel"
	}

	return slog.GroupValue(
		slog.String("name", C.GoString(C.ggml_backend_dev_name(d.d))),
		slog.String("description", C.GoString(C.ggml_backend_dev_description(d.d))),
		slog.String("kind", kind),
		slog.String("free", format.HumanBytes2(free)),
		slog.String("total", format.HumanBytes2(total)),
	)
}

var devices = sync.OnceValue(func() []device {
	ggml.OnceLoad()

	s := make([]device, C.ggml_backend_dev_count())
	for i := range s {
		s[i] = device{C.ggml_backend_dev_get(C.size_t(i))}
	}

	return s
})

type Backend struct {
	meta       *fs.GGML
	cpus, gpus []Context
	tensors    map[string]*Context

	sched *C.struct_ggml_backend_sched
}

func New(r *os.File, params ml.BackendParams) (ml.Backend, error) {
	meta, n, err := fs.Decode(r, -1)
	if err != nil {
		return nil, err
	}

	slog.Info(
		"",
		"architecture", meta.KV().Architecture(),
		"file_type", meta.KV().FileType(),
		"name", meta.KV().String("general.name"),
		"description", meta.KV().String("general.description"),
		"num_tensors", len(meta.Tensors().Items()),
		"num_key_values", len(meta.KV()),
	)

	var cpus, gpus []Context
	for _, d := range devices() {
		switch C.ggml_backend_dev_type(d.d) {
		case C.GGML_BACKEND_DEVICE_TYPE_CPU,
			C.GGML_BACKEND_DEVICE_TYPE_ACCEL:
			slog.Info("cpu", "device", d)
			cpus = append(cpus, Context{
				ctx: C.ggml_init(C.struct_ggml_init_params{
					mem_size: C.size_t(int(C.ggml_tensor_overhead()) * (len(meta.Tensors().Items()) + 1 + int(meta.KV().BlockCount())*2)),
					no_alloc: true,
				}),
				backend: C.ggml_backend_dev_init(d.d, nil),
			})
		case C.GGML_BACKEND_DEVICE_TYPE_GPU:
			slog.Info("gpu", "device", d)
			gpus = append(gpus, Context{
				ctx: C.ggml_init(C.struct_ggml_init_params{
					mem_size: C.size_t(int(C.ggml_tensor_overhead()) * (len(meta.Tensors().Items()) + 1 + int(meta.KV().BlockCount())*2)),
					no_alloc: true,
				}),
				backend: C.ggml_backend_dev_init(d.d, nil),
			})
		}
	}

	ctxFunc := func(s []Context) (*Context, error) {
		for _, e := range s {
			return &e, nil
		}

		return nil, fmt.Errorf("no devices available")
	}

	tensors := make(map[*fs.Tensor]*Context, len(meta.Tensors().Items()))
	for _, t := range meta.Tensors().Items() {
		c, err := ctxFunc(append(gpus, cpus...))
		if err != nil {
			return nil, err
		}

		func() {
			tt := C.ggml_new_tensor(c.ctx, t.Kind, C.int(len(t.Shape)), (*C.int64_t)(unsafe.Pointer(&t.Shape[0])))

			cname := C.CString(t.Name)
			defer C.free(unsafe.Pointer(cname))
			C.ggml_set_name(tt, cname)

			tensors[t] = c
		}()
	}

	for _, b := range append(gpus, cpus...) {
		C.ggml_backend_alloc_ctx_tensors(b.ctx, b.backend)
	}

	sr := io.NewSectionReader(r, int64(meta.Tensors().Offset), n-int64(meta.Tensors().Offset))

	var g errgroup.Group
	for t, c := range tensors {
		g.Go(func() error {
			bts := make([]byte, t.Size())
			n, err := io.ReadFull(io.NewSectionReader(sr, int64(t.Offset), int64(t.Size())), bts)
			if err != nil {
				return err
			}

			if n != int(t.Size()) {
				return fmt.Errorf("expected %d bytes, got %d", t.Size(), n)
			}

			cname := C.CString(t.Name)
			defer C.free(unsafe.Pointer(cname))

			C.ggml_backend_tensor_set(C.ggml_get_tensor(c.ctx, cname), unsafe.Pointer(&bts[0]), 0, C.size_t(n))
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}

	backends := make([]*C.struct_ggml_backend, len(gpus)+len(cpus))
	bufts := make([]*C.struct_ggml_backend_buffer_type, len(gpus)+len(cpus))
	for i, c := range append(gpus, cpus...) {
		backends[i] = c.backend
		bufts[i] = C.ggml_backend_get_default_buffer_type(c.backend)
	}

	return &Backend{
		meta: meta,
		cpus: cpus,
		gpus: gpus,
		sched: C.ggml_backend_sched_new(
			(*C.ggml_backend_t)(unsafe.Pointer(&backends[0])),
			(*C.ggml_backend_buffer_type_t)(unsafe.Pointer(&bufts[0])),
			C.int(len(backends)),
			C.size_t(max(8192, len(meta.Tensors().Items())*5)),
			true,
		),
	}, nil
}

func init() {
	ml.RegisterBackend("ggml", New)
}

func (b *Backend) Config() ml.Config {
	return b.meta.KV()
}

func (b *Backend) Get(name string) ml.Tensor {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	for _, c := range append(b.gpus, b.cpus...) {
		if t := C.ggml_get_tensor(c.ctx, cname); t != nil {
			return &Tensor{t: t}
		}
	}

	return nil
}

func (b *Backend) NewContext() ml.Context {
	nodes := max(8192, len(b.meta.Tensors().Items())*5)
	c := C.ggml_init(C.struct_ggml_init_params{
		mem_buffer: nil,
		mem_size:   C.size_t(nodes)*C.ggml_tensor_overhead() + C.ggml_graph_overhead_custom(C.size_t(nodes), false),
		no_alloc:   true,
	})

	backends := make([]*C.struct_ggml_backend, len(b.gpus)+len(b.cpus))
	for i, c := range append(b.gpus, b.cpus...) {
		backends[i] = c.backend
	}

	return &Context{
		b:       b,
		ctx:     c,
		backend: backends[0],
		nodes:   nodes,
	}
}

type Context struct {
	b       *Backend
	ctx     *C.struct_ggml_context
	backend *C.struct_ggml_backend

	graph *C.struct_ggml_cgraph
	nodes int
}

func (c *Context) Forward(tensors ...ml.Tensor) ml.Context {
	if c.graph == nil {
		c.graph = C.ggml_new_graph_custom(c.ctx, C.size_t(c.nodes), false)
	}

	for _, tensor := range tensors {
		C.ggml_build_forward_expand(c.graph, tensor.(*Tensor).t)
	}

	return c
}

func (c *Context) Compute(tensors ...ml.Tensor) {
	C.ggml_backend_sched_graph_compute_async(c.b.sched, c.graph)
	C.ggml_backend_sched_reset(c.b.sched)

	needSync := true
	sync := func() {
		if needSync {
			C.ggml_backend_sched_synchronize(c.b.sched)
			needSync = false
		}
	}

	for _, t := range tensors {
		if C.ggml_nbytes(t.(*Tensor).t) > 0 {
			t.(*Tensor).sync = sync
		}
	}
}

func (c *Context) MaxTensors() int {
	return c.nodes
}

func shapeToGGML(shape []int) *C.int64_t {
	sh := make([]C.int64_t, len(shape))
	for i, s := range shape {
		sh[i] = (C.int64_t)(s)
	}

	return &sh[0]
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
		t = C.ggml_new_tensor(c.ctx, C.GGML_TYPE_F32, C.int(len(shape)), shapeToGGML(shape))
	case ml.DTypeF16:
		t = C.ggml_new_tensor(c.ctx, C.GGML_TYPE_F16, C.int(len(shape)), shapeToGGML(shape))
	case ml.DTypeI32:
		t = C.ggml_new_tensor(c.ctx, C.GGML_TYPE_I32, C.int(len(shape)), shapeToGGML(shape))
	default:
		panic("unsupported dtype")
	}

	b := C.ggml_backend_alloc_buffer(c.backend, C.ggml_nbytes(t))
	C.ggml_backend_tensor_alloc(b, t, C.ggml_backend_buffer_get_base(b))
	C.ggml_set_zero(t)
	return &Tensor{t: t}
}

func fromSlice[S ~[]E, E float32 | int32](ctx Context, s S, shape []int, dtype uint32) (ml.Tensor, error) {
	n := len(s)

	if n == 0 {
		var shape C.int64_t = 0
		t := C.ggml_new_tensor(ctx.ctx, dtype, 1, &shape)
		return &Tensor{t: t}, nil
	}

	for _, v := range shape {
		n /= v
	}

	if n != 1 {
		return nil, fmt.Errorf("invalid shape %v for %d elements", shape, len(s))
	}

	t := C.ggml_new_tensor(ctx.ctx, dtype, C.int(len(shape)), shapeToGGML(shape))
	b := C.ggml_backend_alloc_buffer(ctx.backend, C.ggml_nbytes(t))
	C.ggml_backend_tensor_alloc(b, t, C.ggml_backend_buffer_get_base(b))
	C.ggml_backend_tensor_set(t, unsafe.Pointer(&s[0]), 0, C.ggml_nbytes(t))
	return &Tensor{t: t}, nil
}

func (c Context) FromFloatSlice(s []float32, shape ...int) (ml.Tensor, error) {
	return fromSlice(c, s, shape, C.GGML_TYPE_F32)
}

func (c Context) FromIntSlice(s []int32, shape ...int) (ml.Tensor, error) {
	return fromSlice(c, s, shape, C.GGML_TYPE_I32)
}

func (c *Context) Close() {
	if c != nil {
		C.ggml_free(c.ctx)
	}
}

type Tensor struct {
	t    *C.struct_ggml_tensor
	sync func()
}

func (t *Tensor) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("name", C.GoString(C.ggml_get_name(t.t))),
		slog.String("type", C.GoString(C.ggml_type_name(t.t._type))),
		slog.Any("shape", t.Shape()),
	)
}

func (t *Tensor) Dim(n int) int {
	return int(t.t.ne[n])
}

func (t *Tensor) Stride(n int) int {
	return int(t.t.nb[n])
}

func (t *Tensor) Shape() []int {
	shape := make([]int, C.ggml_n_dims(t.t))
	for i := range shape {
		shape[i] = t.Dim(i)
	}

	return shape
}

func (t *Tensor) Bytes() (data []byte) {
	if t.sync != nil {
		data = make([]byte, C.ggml_nbytes(t.t))

		t.sync()
		C.ggml_backend_tensor_get(t.t, unsafe.Pointer(&data[0]), 0, C.ggml_nbytes(t.t))
	}

	return
}

func (t *Tensor) Floats() (data []float32) {
	if t.sync != nil {
		data = make([]float32, C.ggml_nelements(t.t))

		t.sync()
		C.ggml_backend_tensor_get(t.t, unsafe.Pointer(&data[0]), 0, C.ggml_nbytes(t.t))
	}

	return
}

func (t *Tensor) DType() ml.DType {
	switch t.t._type {
	case C.GGML_TYPE_F32:
		return ml.DTypeF32
	case C.GGML_TYPE_F16:
		return ml.DTypeF16
	case C.GGML_TYPE_I32:
		return ml.DTypeI32
	default:
		return ml.DTypeOther
	}
}

func (t *Tensor) Add(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		t: C.ggml_add(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
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
		t: C.ggml_concat(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, C.int(dim)),
	}
}

func (t *Tensor) Contiguous(ctx ml.Context) ml.Tensor {
	return &Tensor{
		t: C.ggml_cont(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) Mul(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		t: C.ggml_mul(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

func (t *Tensor) Mulmat(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		t: C.ggml_mul_mat(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

func (t *Tensor) MulmatFullPrec(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	mul := C.ggml_mul_mat(ctx.(*Context).ctx, t.t, t2.(*Tensor).t)
	C.ggml_mul_mat_set_prec(mul, C.GGML_PREC_F32)

	return &Tensor{
		t: mul,
	}
}

func (t *Tensor) LayerNorm(ctx ml.Context, w, b ml.Tensor, eps float32) ml.Tensor {
	tt := (&Tensor{t: C.ggml_norm(ctx.(*Context).ctx, t.t, C.float(eps))}).Mul(ctx, w)
	if b != nil {
		tt = tt.Add(ctx, b)
	}

	return tt
}

func (t *Tensor) RMSNorm(ctx ml.Context, w ml.Tensor, eps float32) ml.Tensor {
	return (&Tensor{t: C.ggml_rms_norm(ctx.(*Context).ctx, t.t, C.float(eps))}).Mul(ctx, w)
}

func (t *Tensor) Pad(ctx ml.Context, shape ...int) ml.Tensor {
	if len(shape) != 4 {
		panic("expected 4 dimensions")
	}

	return &Tensor{
		t: C.ggml_pad(ctx.(*Context).ctx, t.t, C.int(shape[0]), C.int(shape[1]), C.int(shape[2]), C.int(shape[3])),
	}
}

func (t *Tensor) Permute(ctx ml.Context, shape ...int) ml.Tensor {
	if len(shape) != 4 {
		panic("expected 4 dimensions")
	}

	return &Tensor{
		t: C.ggml_permute(ctx.(*Context).ctx, t.t, C.int(shape[0]), C.int(shape[1]), C.int(shape[2]), C.int(shape[3])),
	}
}

func (t *Tensor) Rows(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		t: C.ggml_get_rows(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

func (t *Tensor) Copy(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		t: C.ggml_cpy(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

func (t *Tensor) Reshape(ctx ml.Context, shape ...int) ml.Tensor {
	switch len(shape) {
	case 1:
		return &Tensor{
			t: C.ggml_reshape_1d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0])),
		}
	case 2:
		return &Tensor{
			t: C.ggml_reshape_2d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), C.int64_t(shape[1])),
		}
	case 3:
		return &Tensor{
			t: C.ggml_reshape_3d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), C.int64_t(shape[1]), C.int64_t(shape[2])),
		}
	case 4:
		return &Tensor{
			t: C.ggml_reshape_4d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), C.int64_t(shape[1]), C.int64_t(shape[2]), C.int64_t(shape[3])),
		}
	default:
		panic("unsupported number of dimensions")
	}
}

func (t *Tensor) Scale(ctx ml.Context, s float64) ml.Tensor {
	return &Tensor{
		t: C.ggml_scale(ctx.(*Context).ctx, t.t, (C.float)(s)),
	}
}

func (t *Tensor) Softmax(ctx ml.Context) ml.Tensor {
	return &Tensor{
		t: C.ggml_soft_max(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) Tanh(ctx ml.Context) ml.Tensor {
	return &Tensor{
		t: C.ggml_tanh_inplace(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) Unpad(ctx ml.Context, shape ...int) ml.Tensor {
	if len(shape) != 4 {
		panic("expected 4 dimensions")
	}

	return &Tensor{
		t: C.ggml_unpad(ctx.(*Context).ctx, t.t, C.int(shape[0]), C.int(shape[1]), C.int(shape[2]), C.int(shape[3])),
	}
}

func (t *Tensor) View(ctx ml.Context, offset int, shape ...int) ml.Tensor {
	switch len(shape) {
	case 1:
		return &Tensor{
			t: C.ggml_view_1d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), C.size_t(offset)),
		}
	case 3:
		return &Tensor{
			t: C.ggml_view_2d(ctx.(*Context).ctx, t.t,
				C.int64_t(shape[0]), C.int64_t(shape[2]),
				C.size_t(shape[1]),
				C.size_t(offset)),
		}
	case 5:
		return &Tensor{
			t: C.ggml_view_3d(ctx.(*Context).ctx, t.t,
				C.int64_t(shape[0]), C.int64_t(shape[2]), C.int64_t(shape[4]),
				C.size_t(shape[1]), C.size_t(shape[3]),
				C.size_t(offset)),
		}
	case 7:
		return &Tensor{
			t: C.ggml_view_4d(ctx.(*Context).ctx, t.t,
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

func (t *Tensor) RoPE(ctx ml.Context, positionIDs, ropeFactors ml.Tensor, ropeDim uint32, ropeBase, ropeScale float32) ml.Tensor {
	if ropeFactors == nil {
		ropeFactors = &Tensor{}
	}

	dequant := t.t
	if C.ggml_is_quantized(t.t._type) {
		dequant = C.ggml_cast(ctx.(*Context).ctx, t.t, C.GGML_TYPE_F32)
	}

	return &Tensor{
		t: C.ggml_rope_ext(
			ctx.(*Context).ctx, dequant, positionIDs.(*Tensor).t, ropeFactors.(*Tensor).t,
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
		t: C.ggml_gelu_inplace(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) SILU(ctx ml.Context) ml.Tensor {
	return &Tensor{
		t: C.ggml_silu_inplace(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) Conv2D(ctx ml.Context, t2 ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	return &Tensor{
		t: C.ggml_conv_2d(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, C.int(s0), C.int(s1), C.int(p0), C.int(p1), C.int(d0), C.int(d1)),
	}
}

func (t *Tensor) ScaledDotProductAttention(ctx ml.Context, key, value, mask ml.Tensor, scale float64) ml.Tensor {
	var kqMask *C.struct_ggml_tensor
	if mask != nil {
		kqMask = mask.(*Tensor).t
	}

	kq := key.MulmatFullPrec(ctx, t)
	kq = &Tensor{
		t: C.ggml_soft_max_ext(ctx.(*Context).ctx, kq.(*Tensor).t, kqMask, C.float(scale), 0),
	}

	kqv := value.Mulmat(ctx, kq)
	return kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
}

func (b *Backend) SystemInfo() string {
	var compiler string
	switch C.get_compiler() {
	case C.COMP_UNKNOWN:
		compiler = "cgo(unknown_compiler)"
	case C.COMP_GCC:
		compiler = "cgo(gcc)"
	case C.COMP_CLANG:
		compiler = "cgo(clang)"
	}

	var s string
	for i := range C.ggml_backend_reg_count() {
		reg := C.ggml_backend_reg_get(i)
		fName := C.CString("ggml_backend_get_features")
		defer C.free(unsafe.Pointer(fName))
		get_features_fn := C.ggml_backend_reg_get_proc_address(reg, fName)
		if get_features_fn != nil {
			s += C.GoString(C.ggml_backend_reg_name(reg))
			s += " : "
			for features := C.getBackendFeatures(get_features_fn, reg); features.name != nil; features = C.getNextBackendFeatures(features) {
				s += C.GoString(features.name)
				s += " = "
				s += C.GoString(features.value)
				s += " | "
			}
		}
	}
	return s + compiler
}

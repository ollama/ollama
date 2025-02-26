package ggml

// #cgo CPPFLAGS: -I${SRCDIR}/ggml/include
// #include <stdlib.h>
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-cpu.h"
// #include "ggml-backend.h"
import "C"

import (
	"errors"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"maps"
	"os"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unsafe"

	"github.com/ollama/ollama/format"
	fs "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	ggml "github.com/ollama/ollama/ml/backend/ggml/ggml/src"
	"golang.org/x/sync/errgroup"
)

func devices() iter.Seq[*C.struct_ggml_backend_device] {
	return func(yield func(*C.struct_ggml_backend_device) bool) {
		ggml.OnceLoad()
		for i := range C.ggml_backend_dev_count() {
			if !yield(C.ggml_backend_dev_get(i)) {
				return
			}
		}
	}
}

type Backend struct {
	meta    *fs.GGML
	sched   *C.struct_ggml_backend_sched
	tensors map[string]*C.struct_ggml_tensor
	input   *C.struct_ggml_backend
	output  *C.struct_ggml_backend
	layers  map[int]*C.struct_ggml_backend

	flashAttention bool
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

	type dbt struct {
		d   *C.struct_ggml_backend_device
		bts []*C.struct_ggml_backend_buffer_type
	}

	var cpus, accels, gpus []*C.struct_ggml_backend_device
	for d := range devices() {
		switch C.ggml_backend_dev_type(d) {
		case C.GGML_BACKEND_DEVICE_TYPE_CPU:
			cpus = append(cpus, d)
		case C.GGML_BACKEND_DEVICE_TYPE_ACCEL:
			accels = append(accels, d)
		case C.GGML_BACKEND_DEVICE_TYPE_GPU:
			gpus = append(gpus, d)
		}
	}

	var cpuBufferTypes []*C.struct_ggml_backend_buffer_type
	for _, d := range append(accels, append(gpus, cpus...)...) {
		switch C.ggml_backend_dev_type(d) {
		case C.GGML_BACKEND_DEVICE_TYPE_CPU,
			C.GGML_BACKEND_DEVICE_TYPE_ACCEL:
			cpuBufferTypes = append(cpuBufferTypes, C.ggml_backend_dev_buffer_type(d))
		}
	}

	var sum uint64
	var cumsum []uint64

	var gpuBufferTypes []dbt
	for _, d := range gpus {
		var free, total C.size_t
		C.ggml_backend_dev_memory(d, &free, &total)
		sum += uint64(free)
		cumsum = append(cumsum, sum)

		bt := C.ggml_backend_dev_buffer_type(d)
		gpuBufferTypes = append(gpuBufferTypes, dbt{
			d:   d,
			bts: append([]*C.struct_ggml_backend_buffer_type{bt}, cpuBufferTypes...),
		})
	}

	splits := make([]float64, len(cumsum))
	for i := range splits {
		splits[i] = float64(cumsum[i]) / float64(sum)
	}

	input := dbt{C.ggml_backend_dev_by_type(C.GGML_BACKEND_DEVICE_TYPE_CPU), cpuBufferTypes}

	var blocks int
	for key, value := range meta.KV() {
		if strings.HasSuffix(key, ".block_count") {
			blocks += int(value.(uint32))
		}
	}

	indexFunc := func(i int) func(float64) bool {
		return func(f float64) bool {
			return float64(i)/float64(blocks+1) < f
		}
	}

	layers := make([]dbt, blocks)
	for i := range layers {
		layers[i] = gpuBufferTypes[slices.IndexFunc(splits, indexFunc(i))]
	}

	output := gpuBufferTypes[slices.IndexFunc(splits, indexFunc(blocks))]

	maxTensors := len(meta.Tensors().Items())
	maxTensors += 1
	maxTensors += blocks * 2

	type tensor struct {
		source *fs.Tensor
		target string
	}

	targets := make(map[string][]string)

	ctxs := make(map[*C.struct_ggml_backend_buffer_type]*C.struct_ggml_context)
	createTensor := func(t tensor, bts []*C.struct_ggml_backend_buffer_type) *C.struct_ggml_tensor {
		for _, bt := range bts {
			if _, ok := ctxs[bt]; !ok {
				ctxs[bt] = C.ggml_init(C.struct_ggml_init_params{
					mem_size: C.ggml_tensor_overhead() * C.size_t(maxTensors),
					no_alloc: true,
				})
			}

			targets[t.source.Name] = append(targets[t.source.Name], t.target)

			name := t.source.Name
			if t.target != "" {
				name = t.target
			}

			cname := C.CString(name)
			defer C.free(unsafe.Pointer(cname))
			if tt := C.ggml_get_tensor(ctxs[bt], cname); tt != nil {
				return tt
			}

			tt := C.ggml_new_tensor(ctxs[bt], t.source.Kind, C.int(len(t.source.Shape)), (*C.int64_t)(unsafe.Pointer(&t.source.Shape[0])))
			C.ggml_set_name(tt, cname)

			slog.Debug("created tensor", "name", name, "shape", t.source.Shape, "dtype", t.source.Kind, "buffer_type", C.GoString(C.ggml_backend_buft_name(bt)))
			//nolint:staticcheck // TODO: check if buffer type supports this tensor
			return tt
		}

		return nil
	}

	hasPart := func(s string, parts ...string) bool {
		split := strings.Split(s, ".")
		for _, part := range parts {
			if slices.Contains(split, part) {
				return true
			}
		}

		return false
	}

	for _, t := range meta.Tensors().Items() {
		switch {
		case hasPart(t.Name, "position_embd", "token_embd", "token_norm_embd", "token_types"):
			createTensor(tensor{source: t}, input.bts)
		case hasPart(t.Name, "cls", "output", "output_norm"):
			createTensor(tensor{source: t}, output.bts)
		default:
			if i := func() int {
				if fields := strings.FieldsFunc(t.Name, func(r rune) bool { return !unicode.IsNumber(r) }); len(fields) > 0 {
					if i, err := strconv.Atoi(fields[0]); err == nil {
						return i
					}
				}

				return -1
			}(); i >= 0 {
				createTensor(tensor{source: t}, layers[i].bts)
			} else {
				for i, layer := range layers {
					createTensor(tensor{
						source: t,
						target: "blk." + strconv.Itoa(i) + "." + t.Name,
					}, layer.bts)
				}
			}
		}
	}

	bbs := make(map[*C.struct_ggml_context][]*C.struct_ggml_backend_buffer, len(ctxs))

	for bt, c := range ctxs {
		if C.ggml_get_first_tensor(c) == nil {
			continue
		}

		b := C.ggml_backend_alloc_ctx_tensors_from_buft(c, bt)
		C.ggml_backend_buffer_set_usage(b, C.GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
		bbs[c] = append(bbs[c], b)
	}

	for bs := range maps.Values(bbs) {
		for _, b := range bs {
			slog.Info("model weights", "buffer", C.GoString(C.ggml_backend_buffer_name(b)), "size", format.HumanBytes2(uint64(C.ggml_backend_buffer_get_size(b))))
		}
	}

	tensors := make(map[string]*C.struct_ggml_tensor)
	for _, c := range ctxs {
		for t := C.ggml_get_first_tensor(c); t != nil; t = C.ggml_get_next_tensor(c, t) {
			tensors[C.GoString(C.ggml_get_name(t))] = t
		}
	}

	sr := io.NewSectionReader(r, int64(meta.Tensors().Offset), n-int64(meta.Tensors().Offset))
	var g errgroup.Group
	for _, t := range meta.Tensors().Items() {
		for _, target := range targets[t.Name] {
			g.Go(func() error {
				if target == "" {
					target = t.Name
				}

				tt, ok := tensors[target]
				if !ok {
					return fmt.Errorf("unassigned tensor: %s", t.Name)
				}

				bts := make([]byte, t.Size())
				n, err := io.ReadFull(io.NewSectionReader(sr, int64(t.Offset), int64(t.Size())), bts)
				if err != nil {
					return err
				}

				if n != len(bts) {
					return errors.New("short read")
				}

				cname := C.CString(t.Name)
				C.ggml_backend_tensor_set(tt, unsafe.Pointer(&bts[0]), 0, C.size_t(t.Size()))
				C.free(unsafe.Pointer(cname))

				return nil
			})
		}
	}

	if g.Wait() != nil {
		return nil, err
	}

	deviceBackends := make(map[*C.struct_ggml_backend_device]*C.struct_ggml_backend)
	var backends []*C.struct_ggml_backend
	var bufts []*C.struct_ggml_backend_buffer_type
	for _, d := range append(gpus, append(accels, cpus...)...) {
		b := C.ggml_backend_dev_init(d, nil)
		backends = append(backends, b)
		deviceBackends[d] = b

		bt := C.ggml_backend_get_default_buffer_type(b)
		if d := C.ggml_backend_get_device(b); C.ggml_backend_dev_type(d) == C.GGML_BACKEND_DEVICE_TYPE_CPU && len(gpus) > 0 {
			if hbt := C.ggml_backend_dev_host_buffer_type(d); hbt != nil {
				bt = hbt
			}
		}

		bufts = append(bufts, bt)

		slog.Info("compute graph", "backend", C.GoString(C.ggml_backend_name(b)), "buffer_type", C.GoString(C.ggml_backend_buft_name(bt)))
	}

	return &Backend{
		flashAttention: params.FlashAttention,
		meta:           meta,
		tensors:        tensors,
		sched: C.ggml_backend_sched_new(
			(*C.ggml_backend_t)(unsafe.Pointer(&backends[0])),
			(*C.ggml_backend_buffer_type_t)(unsafe.Pointer(&bufts[0])),
			C.int(len(backends)),
			C.size_t(max(8192, len(meta.Tensors().Items())*5)),
			true,
		),
		input:  deviceBackends[input.d],
		output: deviceBackends[output.d],
		layers: func() map[int]*C.struct_ggml_backend {
			m := make(map[int]*C.struct_ggml_backend)
			for i, layer := range layers {
				m[i] = deviceBackends[layer.d]
			}
			return m
		}(),
	}, nil
}

func init() {
	ml.RegisterBackend("ggml", New)
}

func (b *Backend) Config() ml.Config {
	return b.meta.KV()
}

func (b *Backend) Get(name string) ml.Tensor {
	if t, ok := b.tensors[name]; ok {
		return &Tensor{b: b, t: t}
	}

	return nil
}

func (b *Backend) NewContext() ml.Context {
	return b.NewContextSize(max(8192, len(b.meta.Tensors().Items())*5))
}

func (b *Backend) NewContextSize(n int) ml.Context {
	return &Context{
		b: b,
		ctx: C.ggml_init(C.struct_ggml_init_params{
			mem_size: C.size_t(n)*C.ggml_tensor_overhead() + C.ggml_graph_overhead_custom(C.size_t(n), false),
			no_alloc: true,
		}),
		backend:       C.ggml_backend_sched_get_backend(b.sched, 0),
		maxGraphNodes: n,
		input:         b.input,
		output:        b.output,
		layers:        b.layers,
	}
}

func (b *Backend) CacheConfig() ml.CacheConfig {
	if b.flashAttention {
		return ml.CacheConfig{CachePadding: 256, MaskDType: ml.DTypeF16, MaskBatchPadding: C.GGML_KQ_MASK_PAD}
	} else {
		return ml.CacheConfig{CachePadding: 32, PermutedV: true}
	}
}

type Context struct {
	b *Backend

	ctx   *C.struct_ggml_context
	graph *C.struct_ggml_cgraph

	// backend is the backend used for new tensors
	backend *C.struct_ggml_backend

	// input is the backend used for inputs
	input *C.struct_ggml_backend

	// output is the backend used for outputs
	output *C.struct_ggml_backend

	// output is the backend used for repeating layers
	layers map[int]*C.struct_ggml_backend

	maxGraphNodes int
}

func (c *Context) Input() ml.Context {
	if c.input != nil {
		return &Context{
			b:             c.b,
			ctx:           c.ctx,
			backend:       c.input,
			maxGraphNodes: c.maxGraphNodes,
		}
	}

	return c
}

func (c *Context) Output() ml.Context {
	if c.output != nil {
		return &Context{
			b:             c.b,
			ctx:           c.ctx,
			backend:       c.output,
			maxGraphNodes: c.maxGraphNodes,
		}
	}

	return c
}

func (c *Context) Layer(i int) ml.Context {
	if backend, ok := c.layers[i]; ok {
		return &Context{
			b:             c.b,
			ctx:           c.ctx,
			backend:       backend,
			maxGraphNodes: c.maxGraphNodes,
		}
	}

	return c
}

func (c *Context) Forward(tensors ...ml.Tensor) ml.Context {
	if c.graph == nil {
		c.graph = C.ggml_new_graph_custom(c.ctx, C.size_t(c.maxGraphNodes), false)
	}

	for _, tensor := range tensors {
		C.ggml_build_forward_expand(c.graph, tensor.(*Tensor).t)
	}

	return c
}

func (c *Context) Compute(tensors ...ml.Tensor) {
	C.ggml_backend_sched_reset(c.b.sched)
	C.ggml_backend_sched_alloc_graph(c.b.sched, c.graph)
	C.ggml_backend_sched_graph_compute_async(c.b.sched, c.graph)

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

func (c *Context) MaxGraphNodes() int {
	return c.maxGraphNodes
}

func shapeToGGML(shape []int) *C.int64_t {
	sh := make([]C.int64_t, len(shape))
	for i, s := range shape {
		sh[i] = C.int64_t(s)
	}

	return &sh[0]
}

func (c Context) newTensor(dtype ml.DType, shape []int) ml.Tensor {
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
	return &Tensor{b: c.b, t: t}
}

func (c Context) Empty(dtype ml.DType, shape ...int) ml.Tensor {
	return c.newTensor(dtype, shape)
}

func (c Context) Zeros(dtype ml.DType, shape ...int) ml.Tensor {
	t := c.newTensor(dtype, shape)
	C.ggml_set_zero(t.(*Tensor).t)
	return t
}

func checkShape[S ~[]E, E any](s S, shape ...int) error {
	n := len(s)
	for _, v := range shape {
		n /= v
	}

	if n != 1 {
		return fmt.Errorf("invalid shape: %v", shape)
	}

	return nil
}

func (c Context) FromFloatSlice(s []float32, shape ...int) (ml.Tensor, error) {
	if err := checkShape(s, shape...); err != nil {
		return nil, err
	}

	t := c.newTensor(ml.DTypeF32, shape)
	C.ggml_backend_tensor_set(t.(*Tensor).t, unsafe.Pointer(&s[0]), 0, C.ggml_nbytes(t.(*Tensor).t))
	return t, nil
}

func (c Context) FromIntSlice(s []int32, shape ...int) (ml.Tensor, error) {
	if err := checkShape(s, shape...); err != nil {
		return nil, err
	}

	t := c.newTensor(ml.DTypeI32, shape)
	C.ggml_backend_tensor_set(t.(*Tensor).t, unsafe.Pointer(&s[0]), 0, C.ggml_nbytes(t.(*Tensor).t))
	return t, nil
}

func (c Context) Close() {
	if c.ctx != nil {
		C.ggml_free(c.ctx)
	}
}

type Tensor struct {
	b    *Backend
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
		b: t.b,
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
		b: t.b,
		t: C.ggml_concat(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, C.int(dim)),
	}
}

func (t *Tensor) Contiguous(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_cont(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) Mul(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_mul(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

func (t *Tensor) Mulmat(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_mul_mat(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

func (t *Tensor) MulmatFullPrec(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	mul := C.ggml_mul_mat(ctx.(*Context).ctx, t.t, t2.(*Tensor).t)
	C.ggml_mul_mat_set_prec(mul, C.GGML_PREC_F32)

	return &Tensor{
		b: t.b,
		t: mul,
	}
}

func (t *Tensor) LayerNorm(ctx ml.Context, w, b ml.Tensor, eps float32) ml.Tensor {
	tt := (&Tensor{b: t.b, t: C.ggml_norm(ctx.(*Context).ctx, t.t, C.float(eps))}).Mul(ctx, w)
	if b != nil {
		tt = tt.Add(ctx, b)
	}

	return tt
}

func (t *Tensor) RMSNorm(ctx ml.Context, w ml.Tensor, eps float32) ml.Tensor {
	return (&Tensor{b: t.b, t: C.ggml_rms_norm(ctx.(*Context).ctx, t.t, C.float(eps))}).Mul(ctx, w)
}

func (t *Tensor) Pad(ctx ml.Context, shape ...int) ml.Tensor {
	if len(shape) != 4 {
		panic("expected 4 dimensions")
	}

	return &Tensor{
		b: t.b,
		t: C.ggml_pad(ctx.(*Context).ctx, t.t, C.int(shape[0]), C.int(shape[1]), C.int(shape[2]), C.int(shape[3])),
	}
}

func (t *Tensor) Permute(ctx ml.Context, shape ...int) ml.Tensor {
	if len(shape) != 4 {
		panic("expected 4 dimensions")
	}

	return &Tensor{
		b: t.b,
		t: C.ggml_permute(ctx.(*Context).ctx, t.t, C.int(shape[0]), C.int(shape[1]), C.int(shape[2]), C.int(shape[3])),
	}
}

func (t *Tensor) Rows(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_get_rows(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

func (t *Tensor) Copy(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_cpy(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

func (t *Tensor) Reshape(ctx ml.Context, shape ...int) ml.Tensor {
	switch len(shape) {
	case 1:
		return &Tensor{
			b: t.b,
			t: C.ggml_reshape_1d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0])),
		}
	case 2:
		return &Tensor{
			b: t.b,
			t: C.ggml_reshape_2d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), C.int64_t(shape[1])),
		}
	case 3:
		return &Tensor{
			b: t.b,
			t: C.ggml_reshape_3d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), C.int64_t(shape[1]), C.int64_t(shape[2])),
		}
	case 4:
		return &Tensor{
			b: t.b,
			t: C.ggml_reshape_4d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), C.int64_t(shape[1]), C.int64_t(shape[2]), C.int64_t(shape[3])),
		}
	default:
		panic("unsupported number of dimensions")
	}
}

func (t *Tensor) Scale(ctx ml.Context, s float64) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_scale(ctx.(*Context).ctx, t.t, (C.float)(s)),
	}
}

func (t *Tensor) Softmax(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_soft_max(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) Tanh(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_tanh_inplace(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) Unpad(ctx ml.Context, shape ...int) ml.Tensor {
	if len(shape) != 4 {
		panic("expected 4 dimensions")
	}

	return &Tensor{
		b: t.b,
		t: C.ggml_unpad(ctx.(*Context).ctx, t.t, C.int(shape[0]), C.int(shape[1]), C.int(shape[2]), C.int(shape[3])),
	}
}

func (t *Tensor) View(ctx ml.Context, offset int, shape ...int) ml.Tensor {
	switch len(shape) {
	case 1:
		return &Tensor{
			b: t.b,
			t: C.ggml_view_1d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), C.size_t(offset)),
		}
	case 3:
		return &Tensor{
			b: t.b,
			t: C.ggml_view_2d(ctx.(*Context).ctx, t.t,
				C.int64_t(shape[0]), C.int64_t(shape[2]),
				C.size_t(shape[1]),
				C.size_t(offset)),
		}
	case 5:
		return &Tensor{
			b: t.b,
			t: C.ggml_view_3d(ctx.(*Context).ctx, t.t,
				C.int64_t(shape[0]), C.int64_t(shape[2]), C.int64_t(shape[4]),
				C.size_t(shape[1]), C.size_t(shape[3]),
				C.size_t(offset)),
		}
	case 7:
		return &Tensor{
			b: t.b,
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
		ropeFactors = &Tensor{b: t.b}
	}

	dequant := t.t
	if C.ggml_is_quantized(t.t._type) {
		dequant = C.ggml_cast(ctx.(*Context).ctx, t.t, C.GGML_TYPE_F32)
	}

	return &Tensor{
		b: t.b,
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
		b: t.b,
		t: C.ggml_gelu_inplace(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) SILU(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_silu_inplace(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) Conv2D(ctx ml.Context, t2 ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_conv_2d(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, C.int(s0), C.int(s1), C.int(p0), C.int(p1), C.int(d0), C.int(d1)),
	}
}

func (t *Tensor) ScaledDotProductAttention(ctx ml.Context, key, value, mask ml.Tensor, scale float64) ml.Tensor {
	var kqMask *C.struct_ggml_tensor
	if mask != nil {
		kqMask = mask.(*Tensor).t
	}

	query := t.Permute(ctx, 0, 2, 1, 3)
	key = key.Permute(ctx, 0, 2, 1, 3)

	if t.b.flashAttention {
		value = value.Permute(ctx, 0, 2, 1, 3)

		kqv := C.ggml_flash_attn_ext(ctx.(*Context).ctx, query.(*Tensor).t, key.(*Tensor).t, value.(*Tensor).t, kqMask, C.float(scale), 0, 0)
		C.ggml_flash_attn_ext_set_prec(kqv, C.GGML_PREC_F32)
		return &Tensor{b: t.b, t: kqv}
	} else {
		kq := key.MulmatFullPrec(ctx, query)
		kq = &Tensor{
			b: t.b,
			t: C.ggml_soft_max_ext(ctx.(*Context).ctx, kq.(*Tensor).t, kqMask, C.float(scale), 0),
		}

		kqv := value.Mulmat(ctx, kq)
		return kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	}
}

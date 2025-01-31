package ggml

// #cgo CPPFLAGS: -I${SRCDIR}/ggml/include
// #include <stdlib.h>
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-cpu.h"
// #include "ggml-backend.h"
import "C"

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"unicode"
	"unsafe"

	"github.com/ollama/ollama/format"
	fs "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	"golang.org/x/sync/errgroup"

	ggml "github.com/ollama/ollama/ml/backend/ggml/ggml/src"
)

var rev = []C.int{3, 2, 1, 0}

func devices() []*C.struct_ggml_backend_device {
	ggml.OnceLoad()
	ds := make([]*C.struct_ggml_backend_device, C.ggml_backend_dev_count())
	for i := range ds {
		ds[i] = C.ggml_backend_dev_get(C.size_t(i))
	}

	return ds
}

type Backend struct {
	meta    *fs.GGML
	sched   *C.struct_ggml_backend_sched
	tensors map[string]*C.struct_ggml_tensor

	// input is the backend used for inputs
	input *C.struct_ggml_backend_buffer_type

	// layers is the backend used for repeating layers
	layers map[int]*C.struct_ggml_backend_buffer_type

	flashAttention bool

	// maxGraphNodes is the maximum allowed number of graph nodes in this scheduler
	maxGraphNodes int
}

func New(ctx context.Context, r *os.File, params ml.BackendParams) (ml.Backend, error) {
	meta, n, err := fs.Decode(r, -1)
	if err != nil {
		return nil, err
	}

	slog.Info(
		"initializing GGML backend",
		"architecture", meta.KV().Architecture(),
		"file_type", meta.KV().FileType(),
		"name", meta.KV().String("general.name"),
		"description", meta.KV().String("general.description"),
		"num_tensors", len(meta.Tensors().Items()),
		"num_key_values", len(meta.KV()),
	)

	type deviceBufferType struct {
		d   *C.struct_ggml_backend_device
		bts []*C.struct_ggml_backend_buffer_type
	}

	var cpus, accels, gpus []*C.struct_ggml_backend_device
	for _, d := range devices() {
		switch C.ggml_backend_dev_type(d) {
		case C.GGML_BACKEND_DEVICE_TYPE_CPU:
			if len(cpus) == 0 {
				// only the first cpu device should be used
				cpus = append(cpus, d)
			}
		case C.GGML_BACKEND_DEVICE_TYPE_ACCEL:
			accels = append(accels, d)
		case C.GGML_BACKEND_DEVICE_TYPE_GPU:
			gpus = append(gpus, d)
		}
	}

	// create list of buffer types for the cpu
	cpuDeviceBufferType := deviceBufferType{d: C.ggml_backend_dev_by_type(C.GGML_BACKEND_DEVICE_TYPE_CPU)}
	for _, d := range append(accels, append(gpus, cpus...)...) {
		switch C.ggml_backend_dev_type(d) {
		case C.GGML_BACKEND_DEVICE_TYPE_CPU,
			C.GGML_BACKEND_DEVICE_TYPE_ACCEL:
			cpuDeviceBufferType.bts = append(cpuDeviceBufferType.bts, C.ggml_backend_dev_buffer_type(d))
		}
	}

	// create list of buffer types for each gpu
	var gpuDeviceBufferTypes []deviceBufferType
	for _, d := range gpus {
		bt := C.ggml_backend_dev_buffer_type(d)
		gpuDeviceBufferTypes = append(gpuDeviceBufferTypes, deviceBufferType{
			d:   d,
			bts: append([]*C.struct_ggml_backend_buffer_type{bt}, cpuDeviceBufferType.bts...),
		})
	}

	useDefaultSplit := true
	for _, s := range params.TensorSplit {
		if s != 0 {
			useDefaultSplit = false
			break
		}
	}

	// calculate splits
	splits := make([]float32, len(gpus))
	if useDefaultSplit {
		// default: split on free memory
		for i := range splits {
			var free, total C.size_t
			C.ggml_backend_dev_memory(gpus[i], &free, &total)
			splits[i] = float32(free)
		}
	} else {
		splits = params.TensorSplit
	}

	var sum float32
	// cumulative sum of all splits
	for i := range splits {
		sum += splits[i]
		splits[i] = sum
	}

	// normalize splits
	for i := range splits {
		splits[i] /= sum
	}

	// inputs always use cpu
	input := cpuDeviceBufferType

	blocks := int(meta.KV().BlockCount())

	// define a range of gpu layers. anything outside of this range is assigned to the cpu
	gpuRangeStart := max(0, blocks-params.NumGPULayers)
	gpuRangeStop := min(gpuRangeStart+params.NumGPULayers, blocks+1)
	assignLayer := func(i int) deviceBufferType {
		if i < gpuRangeStart || i >= gpuRangeStop {
			return cpuDeviceBufferType
		}

		index := slices.IndexFunc(splits, func(f float32) bool { return float32(i-gpuRangeStart)/float32(gpuRangeStop-gpuRangeStart) < f })
		if index < 0 || index >= len(gpuDeviceBufferTypes) {
			return cpuDeviceBufferType
		}

		return gpuDeviceBufferTypes[index]
	}

	// repeating layers are assigned based on their index in reverse order, e.g. i / (block_count + 1)
	layers := make([]deviceBufferType, blocks)
	for i := range layers {
		layers[i] = assignLayer(i)
	}

	// outputs are assigned iff allowed by splits and configured number of gpu layers
	output := assignLayer(blocks)

	maxTensors := len(meta.Tensors().Items())
	maxTensors += 1
	// each layer has at most 2 extra tensors for rope operations
	maxTensors += blocks * 2

	type tensor struct {
		source *fs.Tensor
		target string
	}

	// some tensors are mapped to different names so keep a list
	targets := make(map[string][]string)

	// contexts are shared by tensors of the same buffer type
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

	contains := func(s string, parts ...string) bool {
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
		case contains(t.Name, "position_embd", "token_embd", "token_norm_embd", "token_types"):
			createTensor(tensor{source: t}, input.bts)
			if _, ok := meta.Tensors().GroupLayers()["output"]; !ok && t.Name == "token_embd.weight" {
				createTensor(tensor{source: t, target: "output.weight"}, output.bts)
			}
		case contains(t.Name, "cls", "output", "output_norm"):
			createTensor(tensor{source: t}, output.bts)
		case strings.HasPrefix(t.Name, "v.") || strings.HasPrefix(t.Name, "mm."):
			// TODO: assign vision tensors to the gpu if possible
			createTensor(tensor{source: t}, output.bts)
		case contains(t.Name, "rope_freqs", "rope_factors_long", "rope_factors_short"):
			// these tensors should be repeated per layer
			for i, layer := range layers {
				createTensor(tensor{
					source: t,
					target: "blk." + strconv.Itoa(i) + "." + t.Name,
				}, layer.bts)
			}
		default:
			layerIndex := -1
			if fields := strings.FieldsFunc(t.Name, func(r rune) bool { return !unicode.IsNumber(r) }); len(fields) > 0 {
				if i, err := strconv.Atoi(fields[0]); err == nil {
					layerIndex = i
				}
			}

			if layerIndex >= 0 {
				createTensor(tensor{source: t}, layers[layerIndex].bts)
			} else {
				// load all other tensors on the cpu
				createTensor(tensor{source: t}, input.bts)
			}
		}
	}

	// allocate buffers for each context
	bbs := make(map[*C.struct_ggml_context]*C.struct_ggml_backend_buffer, len(ctxs))
	for bt, c := range ctxs {
		if C.ggml_get_first_tensor(c) == nil {
			continue
		}

		b := C.ggml_backend_alloc_ctx_tensors_from_buft(c, bt)
		C.ggml_backend_buffer_set_usage(b, C.GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
		bbs[c] = b
	}

	for bs := range maps.Values(bbs) {
		slog.Info("model weights", "buffer", C.GoString(C.ggml_backend_buffer_name(bs)), "size", format.HumanBytes2(uint64(C.ggml_backend_buffer_get_size(bs))))
	}

	// map tensor names to tensors for easy lookup later
	tensors := make(map[string]*C.struct_ggml_tensor)
	for _, c := range ctxs {
		for t := C.ggml_get_first_tensor(c); t != nil; t = C.ggml_get_next_tensor(c, t) {
			tensors[C.GoString(C.ggml_get_name(t))] = t
		}
	}

	var doneBytes atomic.Uint64
	totalBytes := uint64(n) - meta.Tensors().Offset

	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(runtime.GOMAXPROCS(0))
	for _, t := range meta.Tensors().Items() {
		g.Go(func() error {
			tts := make([]*C.struct_ggml_tensor, max(1, len(targets[t.Name])))
			for i := range tts {
				target := targets[t.Name][i]
				if target == "" {
					target = t.Name
				}

				tt, ok := tensors[target]
				if !ok {
					return fmt.Errorf("unassigned tensor: %s", t.Name)
				}

				tts[i] = tt
			}

			sr := io.NewSectionReader(r, int64(meta.Tensors().Offset+t.Offset), int64(t.Size()))
			bts := make([]byte, 128*format.KibiByte)

			var s uint64
			for s < t.Size() {
				n, err := io.ReadFull(sr, bts[:min(len(bts), int(t.Size()-s))])
				if err != nil {
					return err
				}

				for _, tt := range tts {
					C.ggml_backend_tensor_set(tt, unsafe.Pointer(&bts[0]), C.size_t(s), C.size_t(n))
				}

				s += uint64(n)

				if params.Progress != nil {
					done := doneBytes.Add(uint64(n))
					params.Progress(float32(done) / float32(totalBytes))
				}
			}

			return nil
		})
	}

	// start a goroutine to cancel the errgroup if the parent context is done
	go func() {
		<-ctx.Done()
		g.Go(func() error {
			return ctx.Err()
		})
	}()

	if err := g.Wait(); err != nil {
		return nil, err
	}

	// map devices to backend buffer types so new tensors can be assigned to the correct device
	deviceBufferTypes := make(map[*C.struct_ggml_backend_device]*C.struct_ggml_backend_buffer_type)

	// create backends and buffer types used for the compute graph scheduler
	var schedBackends []*C.struct_ggml_backend
	var schedBufts []*C.struct_ggml_backend_buffer_type
	for _, d := range append(gpus, append(accels, cpus...)...) {
		b := C.ggml_backend_dev_init(d, nil)
		bt := C.ggml_backend_get_default_buffer_type(b)
		if d := C.ggml_backend_get_device(b); C.ggml_backend_dev_type(d) == C.GGML_BACKEND_DEVICE_TYPE_CPU && len(gpus) > 0 {
			// use the first gpu host buffer type for gpu if possible
			if hbt := C.ggml_backend_dev_host_buffer_type(gpus[0]); hbt != nil {
				bt = hbt
			}
		}

		deviceBufferTypes[d] = bt

		schedBackends = append(schedBackends, b)
		schedBufts = append(schedBufts, bt)

		slog.Info("compute graph", "backend", C.GoString(C.ggml_backend_name(b)), "buffer_type", C.GoString(C.ggml_backend_buft_name(bt)))

		if C.ggml_backend_is_cpu(b) {
			// set number of threads for cpu backend
			C.ggml_backend_cpu_set_n_threads(b, C.int(Threads(params.NumThreads)))
		}
	}

	maxGraphNodes := max(8192, len(meta.Tensors().Items())*5)
	return &Backend{
		flashAttention: params.FlashAttention,
		meta:           meta,
		tensors:        tensors,
		sched: C.ggml_backend_sched_new(
			(*C.ggml_backend_t)(unsafe.Pointer(&schedBackends[0])),
			(*C.ggml_backend_buffer_type_t)(unsafe.Pointer(&schedBufts[0])),
			C.int(len(schedBackends)),
			C.size_t(maxGraphNodes),
			C._Bool(len(gpus) > 1 && slices.Contains(gpus, output.d)),
		),
		input: deviceBufferTypes[input.d],
		layers: func() map[int]*C.struct_ggml_backend_buffer_type {
			m := make(map[int]*C.struct_ggml_backend_buffer_type)
			for i, layer := range layers {
				m[i] = deviceBufferTypes[layer.d]
			}
			return m
		}(),
		maxGraphNodes: maxGraphNodes,
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
		return &Tensor{b: b, t: t, nDims: int(C.ggml_n_dims(t))}
	}

	return nil
}

func (b *Backend) NewContext() ml.Context {
	return b.NewContextSize(b.maxGraphNodes)
}

func (b *Backend) NewContextSize(n int) ml.Context {
	if n > b.maxGraphNodes {
		panic(fmt.Errorf("requested number of graph nodes (%v) for new context exceeds maximum (%v)", n, b.maxGraphNodes))
	}

	return &Context{
		b:             b,
		maxGraphNodes: n,
		ctx: C.ggml_init(C.struct_ggml_init_params{
			mem_size: C.size_t(n)*C.ggml_tensor_overhead() + C.ggml_graph_overhead_custom(C.size_t(n), false),
			no_alloc: true,
		}),
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

	// buft is the buffer type used for new tensors
	buft *C.struct_ggml_backend_buffer_type

	// maxGraphNodes is the maximum allowed number of graph nodes in this context
	maxGraphNodes int
}

func (c Context) Input() ml.Context {
	if c.b.input != nil {
		return &Context{
			b:             c.b,
			ctx:           c.ctx,
			buft:          c.b.input,
			maxGraphNodes: c.maxGraphNodes,
		}
	}

	return &c
}

func (c Context) Layer(i int) ml.Context {
	if buft, ok := c.b.layers[i]; ok {
		return &Context{
			b:             c.b,
			ctx:           c.ctx,
			buft:          buft,
			maxGraphNodes: c.maxGraphNodes,
		}
	}

	return &c
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

func (c Context) Compute(tensors ...ml.Tensor) {
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

func (c Context) MaxGraphNodes() int {
	return c.maxGraphNodes
}

func shapeToGGML(shape []int) *C.int64_t {
	sh := make([]C.int64_t, len(shape))
	for i, s := range shape {
		sh[i] = C.int64_t(s)
	}

	return &sh[0]
}

func pad(length, pad C.size_t) C.size_t {
	return ((length + pad - 1) / pad) * pad
}

func (c Context) newTensor(dtype ml.DType, rshape []int) ml.Tensor {
	if c.buft == nil {
		panic("set Input, Output, or Layer before creating tensors")
	}

	var cdtype uint32
	switch dtype {
	case ml.DTypeF32:
		cdtype = C.GGML_TYPE_F32
	case ml.DTypeF16:
		cdtype = C.GGML_TYPE_F16
	case ml.DTypeQ80:
		cdtype = C.GGML_TYPE_Q8_0
	case ml.DTypeQ40:
		cdtype = C.GGML_TYPE_Q4_0
	case ml.DTypeI32:
		cdtype = C.GGML_TYPE_I32
	default:
		panic("unsupported dtype")
	}

	if len(rshape) < 1 || rshape[0] == 0 {
		var shape C.int64_t = 0
		return &Tensor{b: c.b, t: C.ggml_new_tensor(c.ctx, cdtype, 1, &shape), nDims: 1}
	} else if len(rshape) > 4 {
		panic("unsupported number of dimensions")
	}

	for _, dim := range rshape {
		if dim < 1 {
			panic("invalid shape")
		}
	}
	// Inverted
	shape := make([]int, len(rshape))
	i := len(rshape) - 1
	for _, dim := range rshape {
		shape[i] = dim
		i--
	}

	t := C.ggml_new_tensor(c.ctx, cdtype, C.int(len(shape)), shapeToGGML(shape))
	size := pad(C.ggml_backend_buft_get_alloc_size(c.buft, t), C.ggml_backend_buft_get_alignment(c.buft))
	b := C.ggml_backend_buft_alloc_buffer(c.buft, size)
	C.ggml_backend_tensor_alloc(b, t, C.ggml_backend_buffer_get_base(b))
	return &Tensor{b: c.b, t: t, nDims: len(shape)}
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

	if n == 0 {
		return nil
	}

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
	if len(s) > 0 {
		C.ggml_backend_tensor_set(t.(*Tensor).t, unsafe.Pointer(&s[0]), 0, C.ggml_nbytes(t.(*Tensor).t))
	}

	return t, nil
}

func (c Context) FromIntSlice(s []int32, shape ...int) (ml.Tensor, error) {
	if err := checkShape(s, shape...); err != nil {
		return nil, err
	}

	t := c.newTensor(ml.DTypeI32, shape)
	if len(s) > 0 {
		C.ggml_backend_tensor_set(t.(*Tensor).t, unsafe.Pointer(&s[0]), 0, C.ggml_nbytes(t.(*Tensor).t))
	}
	return t, nil
}

func (c Context) FromBytes(dtype ml.DType, s []uint8, shape ...int) ml.Tensor {
	// Unchecked to handle quantized types

	t := c.newTensor(dtype, shape)
	if len(s) > 0 {
		C.ggml_backend_tensor_set(t.(*Tensor).t, unsafe.Pointer(&s[0]), 0, C.ggml_nbytes(t.(*Tensor).t))
	}

	return t
}

func (c *Context) Close() {
	if c != nil {
		C.ggml_free(c.ctx)
	}
}

type Tensor struct {
	b *Backend
	t *C.struct_ggml_tensor

	// keep track of the number of dimensions
	// Since we reverse the shape, GGML considers a trailing "1" dimension as not present
	// and we can't actually trust the output of ggml_n_dims
	nDims int

	// keep track of permutations so we can accurately reverse stride information
	permuted []C.int

	sync func()
}

func (t *Tensor) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("name", C.GoString(C.ggml_get_name(t.t))),
		slog.String("type", C.GoString(C.ggml_type_name(t.t._type))),
		slog.Any("shape", t.Shape()),
		slog.Any("underlying shape", t.t.ne),
		slog.Any("underlying stride", t.t.nb),
	)
}

func (t *Tensor) Dim(n int) int {
	if t.nDims == 0 {
		// If this hits we likely forgot to copy the dimension to the returned tensor in some operation
		panic("zero dimension tensor")
	}
	r := rev[4-t.nDims:]
	return int(t.t.ne[r[n]])
}

func (t *Tensor) Stride(n int) int {
	var s int
	// GGML tracks strides in bytes, so we convert to elements for interop
	// nb[GGML_MAX_DIMS]; // stride in bytes:
	// nb[0] = ggml_type_size(type)
	// nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
	// nb[i] = nb[i-1] * ne[i-1]
	typeSize := C.ggml_type_size(t.t._type)
	blck_size := C.ggml_blck_size(t.t._type)

	if t.nDims == 0 {
		slog.Error("Stride", "tensor", t, "dim", n)
		panic("zero dimension tensor")
	}
	if n > t.nDims {
		return 0
	}
	r := rev[4-t.nDims:]
	if blck_size > 1 {
		// Quantized types require additional mapping to account for block size
		sh := make([]int, 4)
		if t.permuted != nil { // ggml_is_permuted can return false negative on some quantized shapes
			// If the tensor is permuted (and not contiguous) we need to adjust
			// the strides returned based on how it was permuted
			for i, d := range t.permuted {
				sh[i] = int(t.t.ne[int(d)])
			}
		} else {
			sh[0], sh[1], sh[2], sh[3] = int(t.t.ne[0]), int(t.t.ne[1]), int(t.t.ne[2]), int(t.t.ne[3])
		}
		synStride := make([]int, t.nDims)
		st := 1
		for i, d := range r {
			synStride[d] = st
			st *= sh[i]
		}
		if t.permuted != nil {
			for i, d := range r {
				if t.permuted[d] == r[n] {
					n = i
					break
				}
			}
		}
		s = synStride[n]
	} else {
		// This only works for non-quantized types with block size = 1
		s = int(t.t.nb[r[n]] / typeSize)
	}
	return s
}

func (t *Tensor) Shape() []int {
	shape := make([]int, t.nDims)
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
	case C.GGML_TYPE_Q8_0:
		return ml.DTypeQ80
	case C.GGML_TYPE_Q4_0:
		return ml.DTypeQ40
	case C.GGML_TYPE_I32:
		return ml.DTypeI32
	default:
		return ml.DTypeOther
	}
}

func (t *Tensor) Add(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b:     t.b,
		t:     C.ggml_add(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
		nDims: t.nDims,
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
		b:     t.b,
		t:     C.ggml_concat(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, C.int(dim)),
		nDims: max(t.nDims, t2.(*Tensor).nDims),
	}
}

func (t *Tensor) Contiguous(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b:     t.b,
		t:     C.ggml_cont(ctx.(*Context).ctx, t.t),
		nDims: t.nDims,
	}
}

func (t *Tensor) Mul(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b:     t.b,
		t:     C.ggml_mul(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
		nDims: t.nDims, // TODO should this be max(t.nDims, t2.nDims)?
	}
}

func (t *Tensor) Matmul(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return t.matmul(ctx, t2.(*Tensor), false)
}

func (a *Tensor) canBroadcast(b *Tensor) bool {
	if b.t.ne[2]%a.t.ne[2] != 0 {
		slog.Debug("unable to broadcast dimension 2", "a", a, "b", b)
		return false
	} else if b.t.ne[3]%a.t.ne[3] != 0 {
		slog.Debug("unable to broadcast dimension 3", "a", a, "b", b)
		return false
	}
	return true
}

func (b *Tensor) matmul(ctx ml.Context, a *Tensor, fullPrecision bool) ml.Tensor {
	// a: k columns, n rows => [ne03, ne02, n, k]
	// b: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
	// result is n columns, m rows => [ne03 * x, ne02 * y, m, n]

	var ap, bp, rp []C.int
	var nDims int
	switch [2]int{len(a.Shape()), len(b.Shape())} {
	case [2]int{1, 1}:
		nDims = 1
	case [2]int{2, 1}:
		ap = []C.int{1, 0, 2, 3}
		nDims = 1
	case [2]int{1, 2}:
		rp = []C.int{1, 0, 2, 3}
		nDims = 1
	case [2]int{3, 1}:
		a, b = b, a
		bp = []C.int{1, 0, 2, 3}
		rp = []C.int{2, 0, 1, 3}
		nDims = 2
	case [2]int{1, 3}:
		rp = []C.int{2, 0, 1, 3}
		nDims = 2
	case [2]int{4, 1}:
		a, b = b, a
		bp = []C.int{1, 0, 2, 3}
		rp = []C.int{3, 0, 1, 2}
		nDims = 3
	case [2]int{1, 4}:
		rp = []C.int{3, 0, 1, 2}
		nDims = 3
	case [2]int{2, 2}:
		ap = []C.int{1, 0, 2, 3}
		nDims = 2
	case [2]int{3, 2}:
		a, b = b, a
		bp = []C.int{1, 0, 2, 3}
		rp = []C.int{1, 0, 2, 3}
		nDims = 3
	case [2]int{2, 3}:
		ap = []C.int{1, 0, 2, 3}
		nDims = 3
	case [2]int{3, 3}:
		ap = []C.int{1, 0, 2, 3}
		nDims = 3
	case [2]int{4, 2}, [2]int{4, 3}:
		a, b = b, a
		bp = []C.int{1, 0, 2, 3}
		rp = []C.int{1, 0, 2, 3}
		nDims = 4
	case [2]int{2, 4}, [2]int{3, 4}:
		ap = []C.int{1, 0, 2, 3}
		nDims = 4
	case [2]int{4, 4}:
		if a.canBroadcast(b) {
			ap = []C.int{1, 0, 2, 3}
		} else {
			a, b = b, a
			bp = []C.int{1, 0, 2, 3}
			rp = []C.int{1, 0, 2, 3}
		}
		nDims = 4
	default:
		// Not reached
		panic("unhandled shape combination")
	}
	if len(ap) > 0 {
		a.t = C.ggml_permute(ctx.(*Context).ctx, a.t, ap[0], ap[1], ap[2], ap[3])
		// TODO - without this, CPU blows up, but this has a major performance impact
		a.t = C.ggml_cont(ctx.(*Context).ctx, a.t)
	}
	if len(bp) > 0 {
		b.t = C.ggml_permute(ctx.(*Context).ctx, b.t, bp[0], bp[1], bp[2], bp[3])
		// TODO - without this, CPU blows up, but this has a major performance impact
		b.t = C.ggml_cont(ctx.(*Context).ctx, b.t)
	}
	if a.t.ne[0] != b.t.ne[0] {
		slog.Error("last dimension does not match", "a", a, "b", b)
		panic("malformed tensors passed to Matmul")
	}
	if b.t.ne[2]%a.t.ne[2] != 0 {
		slog.Error("malformed tensor shapes in dim 2", "a", a, "b", b)
		panic(fmt.Sprintf("dim 2 - %d cannot broadcast on %d", b.t.ne[2], a.t.ne[2]))
	} else if b.t.ne[3]%a.t.ne[3] != 0 {
		slog.Error("malformed tensor shapes in dim 3", "a", a, "b", b)
		panic(fmt.Sprintf("dim 3 - %d cannot broadcast on %d", b.t.ne[3], a.t.ne[3]))
	}
	if a.t.nb[0] > a.t.nb[1] {
		slog.Error("tensor is not transposed", "a", a)
		panic("tensor is not transposed")
	}

	r := C.ggml_mul_mat(ctx.(*Context).ctx, a.t, b.t)
	if fullPrecision {
		C.ggml_mul_mat_set_prec(r, C.GGML_PREC_F32)
	}
	if len(rp) > 0 {
		r = C.ggml_permute(ctx.(*Context).ctx, r, rp[0], rp[1], rp[2], rp[3])
		r = C.ggml_cont(ctx.(*Context).ctx, r)
	}

	return &Tensor{
		b:     a.b,
		t:     r,
		nDims: nDims,
	}
}

func (t2 *Tensor) MatmulFullPrec(ctx ml.Context, t ml.Tensor) ml.Tensor {
	return t2.matmul(ctx, t.(*Tensor), true)
}

func (t *Tensor) LayerNorm(ctx ml.Context, w, b ml.Tensor, eps float32) ml.Tensor {
	tt := (&Tensor{b: t.b, t: C.ggml_norm(ctx.(*Context).ctx, t.t, C.float(eps)), nDims: t.nDims}).Mul(ctx, w)
	if b != nil {
		tt = tt.Add(ctx, b)
	}

	return tt
}

func (t *Tensor) RMSNorm(ctx ml.Context, w ml.Tensor, eps float32) ml.Tensor {
	return (&Tensor{b: t.b, t: C.ggml_rms_norm(ctx.(*Context).ctx, t.t, C.float(eps)), nDims: t.nDims}).Mul(ctx, w)
}

func (t *Tensor) Pad(ctx ml.Context, shape ...int) ml.Tensor {
	if len(shape) != 4 {
		panic("expected 4 dimensions")
	}
	var r *C.struct_ggml_tensor
	switch t.nDims {
	case 1:
		r = C.ggml_pad(ctx.(*Context).ctx, t.t, C.int(shape[0]), C.int(shape[1]), C.int(shape[2]), C.int(shape[3]))
	case 2:
		r = C.ggml_pad(ctx.(*Context).ctx, t.t, C.int(shape[1]), C.int(shape[0]), C.int(shape[2]), C.int(shape[3]))
	case 3:
		r = C.ggml_pad(ctx.(*Context).ctx, t.t, C.int(shape[2]), C.int(shape[1]), C.int(shape[0]), C.int(shape[3]))
	default:
		r = C.ggml_pad(ctx.(*Context).ctx, t.t, C.int(shape[3]), C.int(shape[2]), C.int(shape[1]), C.int(shape[0]))
	}
	return &Tensor{
		b:     t.b,
		t:     r,
		nDims: t.nDims,
	}
}

func (t *Tensor) Permute(ctx ml.Context, shape ...int) ml.Tensor {
	// The GGML pattern for ggml_permute is different than other tensor frameworks
	// GGML: [2, 0, 1, 3] means (t dim 0 goes to result dim 2, 1 goes to 0, 2 goes to 1, 3 goes to 3)
	// Others: [2, 0, 1, 3] == (result dim 0 comes from t dim 2, 1 comes from 0, 2 comes from 1, 3 comes from 3)
	//
	// The following mapping converts from the "comes from" input arguments to the "goes to" pattern GGML expects

	if len(shape) != 4 {
		panic("expected 4 dimensions")
	}
	rshape := []C.int{0, 1, 2, 3}
	switch t.nDims {
	case 2:
		rshape[0] = C.int(shape[0])
		rshape[1] = C.int(shape[1])
	case 3:
		rshape[0] = C.int(shape[0])
		rshape[1] = C.int(shape[1])
		rshape[2] = C.int(shape[2])
		switch shape[0]*100 + shape[1]*10 + shape[2] {
		case 21:
			rshape[0], rshape[1], rshape[2] = 1, 0, 2
		case 102:
			rshape[0], rshape[1], rshape[2] = 0, 2, 1
		}
	case 4:
		rshape[0] = C.int(shape[0])
		rshape[1] = C.int(shape[1])
		rshape[2] = C.int(shape[2])
		rshape[3] = C.int(shape[3])
		switch shape[0]*1000 + shape[1]*100 + shape[2]*10 + shape[3] {
		case 132:
			rshape[0], rshape[1], rshape[2], rshape[3] = 1, 0, 2, 3
		case 231:
			rshape[0], rshape[1], rshape[2], rshape[3] = 1, 2, 0, 3
		case 312:
			rshape[0], rshape[1], rshape[2], rshape[3] = 2, 0, 1, 3
		case 321:
			rshape[0], rshape[1], rshape[2], rshape[3] = 2, 1, 0, 3
		case 1023:
			rshape[0], rshape[1], rshape[2], rshape[3] = 0, 1, 3, 2
		case 1203:
			rshape[0], rshape[1], rshape[2], rshape[3] = 0, 2, 3, 1
		case 1302:
			rshape[0], rshape[1], rshape[2], rshape[3] = 2, 0, 3, 1
		case 1320:
			rshape[0], rshape[1], rshape[2], rshape[3] = 2, 1, 3, 0
		case 2013:
			rshape[0], rshape[1], rshape[2], rshape[3] = 0, 3, 1, 2
		case 2031:
			rshape[0], rshape[1], rshape[2], rshape[3] = 1, 3, 0, 2
		case 2103:
			rshape[0], rshape[1], rshape[2], rshape[3] = 0, 3, 2, 1
		case 2130:
			rshape[0], rshape[1], rshape[2], rshape[3] = 1, 3, 2, 0
		case 3021:
			rshape[0], rshape[1], rshape[2], rshape[3] = 3, 1, 0, 2
		case 3102:
			rshape[0], rshape[1], rshape[2], rshape[3] = 3, 0, 2, 1
		}
	}
	r := &Tensor{
		b:        t.b,
		t:        C.ggml_permute(ctx.(*Context).ctx, t.t, rshape[0], rshape[1], rshape[2], rshape[3]),
		nDims:    t.nDims,
		permuted: rshape,
	}
	return r
}

func (t *Tensor) Rows(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b:     t.b,
		t:     C.ggml_get_rows(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
		nDims: t.nDims,
	}
}

func (t *Tensor) Copy(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	r := &Tensor{
		b:     t.b,
		t:     C.ggml_cpy(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
		nDims: t.nDims,
	}
	return r
}

func (t *Tensor) Reshape(ctx ml.Context, shape ...int) ml.Tensor {
	// GGML does not handle -1 natively
	for i, sh := range shape {
		if sh == -1 {
			totalElems := 1
			for d := range t.nDims {
				totalElems *= int(t.t.ne[d])
			}
			otherElems := 1
			for _, osh := range shape {
				if osh != -1 {
					otherElems *= osh
				}
			}
			if otherElems > totalElems {
				slog.Error("Invalid request", "req", shape, "actual", t.Shape(), "totalElems", totalElems, "otherElems", otherElems)
				panic("impossible -1 shape request")
			}
			shape[i] = int(float64(totalElems) / float64(otherElems))
			break
		}
	}
	switch len(shape) {
	case 1:
		return &Tensor{
			b:     t.b,
			t:     C.ggml_reshape_1d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0])),
			nDims: len(shape),
		}
	case 2:
		return &Tensor{
			b:     t.b,
			t:     C.ggml_reshape_2d(ctx.(*Context).ctx, t.t, C.int64_t(shape[1]), C.int64_t(shape[0])),
			nDims: len(shape),
		}
	case 3:
		return &Tensor{
			b:     t.b,
			t:     C.ggml_reshape_3d(ctx.(*Context).ctx, t.t, C.int64_t(shape[2]), C.int64_t(shape[1]), C.int64_t(shape[0])),
			nDims: len(shape),
		}
	case 4:
		return &Tensor{
			b:     t.b,
			t:     C.ggml_reshape_4d(ctx.(*Context).ctx, t.t, C.int64_t(shape[3]), C.int64_t(shape[2]), C.int64_t(shape[1]), C.int64_t(shape[0])),
			nDims: len(shape),
		}
	default:
		panic("unsupported number of dimensions")
	}
}

func (t *Tensor) Scale(ctx ml.Context, s float64) ml.Tensor {
	return &Tensor{
		b:     t.b,
		t:     C.ggml_scale(ctx.(*Context).ctx, t.t, (C.float)(s)),
		nDims: t.nDims,
	}
}

func (t *Tensor) Softmax(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b:     t.b,
		t:     C.ggml_soft_max(ctx.(*Context).ctx, t.t),
		nDims: t.nDims,
	}
}

func (t *Tensor) Tanh(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b:     t.b,
		t:     C.ggml_tanh_inplace(ctx.(*Context).ctx, t.t),
		nDims: t.nDims,
	}
}

func (t *Tensor) Unpad(ctx ml.Context, shape ...int) ml.Tensor {
	if len(shape) != 4 {
		panic("expected 4 dimensions")
	}

	return &Tensor{
		b:     t.b,
		t:     C.ggml_unpad(ctx.(*Context).ctx, t.t, C.int(shape[3]), C.int(shape[2]), C.int(shape[1]), C.int(shape[0])),
		nDims: t.nDims, // TODO is this right?
	}
}

func (t *Tensor) AsStrided(ctx ml.Context, shape, stride []int, offset int) ml.Tensor {
	if len(stride) != len(shape) || len(stride) == 0 {
		panic(fmt.Sprintf("mismatch in length of shape=%v stride=%v", shape, stride))
	}
	if stride[len(stride)-1] != 1 {
		// GGML does not currently support jumping within the final row
		panic(fmt.Sprintf("final stride must be 1, got stride=%v", stride))
	}
	typeSize := C.ggml_type_size(t.t._type)
	blck_size := C.size_t(C.ggml_blck_size(t.t._type))
	strideBytes := make([]C.size_t, len(stride))
	var offsetBytes C.size_t
	permuted := []int{0, 1, 2, 3}
	if t.permuted != nil {
		permuted[0], permuted[1], permuted[2], permuted[3] = int(t.permuted[0]), int(t.permuted[1]), int(t.permuted[2]), int(t.permuted[3])
	}
	if blck_size > 1 {
		// Quantized type require additional mapping to account for block size
		//
		// Note: the returned tensor will report a synthetic stride which
		// matches the new shape, which is inconsistent with other row-order
		// frameworks.  Those expose the original tensors stride.  As such,
		// views of views for quantized types will not be possible today. If
		// this support is required for any future models, we'll need to record
		// the original stride information from the underlying quantized tensor
		// in the view tensor to be able to chain quantized views and properly
		// reverse the logic.
		//
		// GGML Does not support sub-row views on quantized types
		if shape[len(shape)-1] < int(t.t.ne[0]) {
			panic("AsStrided of quantized tensor must use whole rows")
		}

		// Intermediate single dimensions complicate the stride calculations
		// So we currently prevent them
		nonSingle := false
		for _, s := range shape {
			if s > 1 {
				nonSingle = true
			} else if nonSingle && s == 1 {
				panic("intermixed single dimensions not supported for quantized views")
			}
		}
		sh := make([]int, 4) // Original tensor shape, ggml order
		for i := range sh {
			sh[i] = int(t.t.ne[permuted[i]])
		}
		firstSynStride := C.size_t(sh[0]) // element based synthetic stride for the row
		d := len(stride) - 2
		for i := 0; i < len(stride) && d >= 0; i++ {
			strideBytes[i] = t.t.nb[permuted[i]] * C.size_t(stride[d]) / firstSynStride
			d--
		}
		offsetBytes = C.size_t(offset) * typeSize / blck_size
	} else {
		// non-quantized types
		offsetBytes = C.size_t(offset) * typeSize
		for i, d := range rev[5-len(stride):] {
			strideBytes[i] = C.size_t(stride[d]) * typeSize
		}
	}

	switch len(shape) {
	case 1:
		return &Tensor{
			b:     t.b,
			t:     C.ggml_view_1d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), offsetBytes),
			nDims: 1,
		}
	case 2:
		return &Tensor{
			b: t.b,
			t: C.ggml_view_2d(ctx.(*Context).ctx, t.t,
				C.int64_t(shape[1]), C.int64_t(shape[0]),
				strideBytes[0],
				offsetBytes),
			nDims: 2,
		}
	case 3:
		return &Tensor{
			b: t.b,
			t: C.ggml_view_3d(ctx.(*Context).ctx, t.t,
				C.int64_t(shape[2]), C.int64_t(shape[1]), C.int64_t(shape[0]),
				strideBytes[0], strideBytes[1],
				offsetBytes),
			nDims: 3,
		}
	case 4:
		return &Tensor{
			b: t.b,
			t: C.ggml_view_4d(ctx.(*Context).ctx, t.t,
				C.int64_t(shape[3]), C.int64_t(shape[2]), C.int64_t(shape[1]), C.int64_t(shape[0]),
				strideBytes[0], strideBytes[1], strideBytes[2],
				offsetBytes),
			nDims: 4,
		}
	default:
		panic("unsupported number of dimensions")
	}
}

const (
	ropeTypeNorm   C.int = 0
	ropeTypeNeox   C.int = 2
	ropeTypeMrope  C.int = 8
	ropeTypeVision C.int = 24
)

func (t *Tensor) RoPE(ctx ml.Context, positionIDs, ropeFactors ml.Tensor, ropeDim, ropeType uint32, ropeBase, ropeScale float32) ml.Tensor {
	if ropeFactors == nil {
		ropeFactors = &Tensor{b: t.b, nDims: 0}
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
			C.int(ropeType),
			131072, // YaRN n_ctx_train
			C.float(ropeBase),
			C.float(ropeScale),
			0.,  // YaRN ext_factor
			1.,  // YaRN attn_factor
			32., // YaRN beta_fast
			1.,  // YaRN beta_slow
		),
		nDims: t.nDims,
	}
}

func (t *Tensor) GELU(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b:     t.b,
		t:     C.ggml_gelu_inplace(ctx.(*Context).ctx, t.t),
		nDims: t.nDims,
	}
}

func (t *Tensor) SILU(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b:     t.b,
		t:     C.ggml_silu_inplace(ctx.(*Context).ctx, t.t),
		nDims: t.nDims,
	}
}

func (t *Tensor) Conv2D(ctx ml.Context, t2 ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	return &Tensor{
		b:     t.b,
		t:     C.ggml_conv_2d(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, C.int(s0), C.int(s1), C.int(p0), C.int(p1), C.int(d0), C.int(d1)),
		nDims: t.nDims,
	}
}

func (t *Tensor) AvgPool2D(ctx ml.Context, k, s int, p float32) ml.Tensor {
	return &Tensor{
		b:     t.b,
		t:     C.ggml_pool_2d(ctx.(*Context).ctx, t.t, C.GGML_OP_POOL_AVG, C.int(k), C.int(k), C.int(s), C.int(s), C.float(p), C.float(p)),
		nDims: t.nDims,
	}
}

func (t *Tensor) ScaledDotProductAttention(ctx ml.Context, key, value, mask ml.Tensor, scale float64) ml.Tensor {
	var kqMask *C.struct_ggml_tensor
	if mask != nil {
		kqMask = mask.(*Tensor).t
	}

	query := t.Permute(ctx, 1, 0, 2, 3)
	key = key.Permute(ctx, 1, 2, 0, 3)

	if t.b.flashAttention {
		value = value.Permute(ctx, 1, 0, 2, 3)

		// TODO - this hasn't been adjusted yet for row-order and probably doesn't work

		kqv := C.ggml_flash_attn_ext(ctx.(*Context).ctx, query.(*Tensor).t, key.(*Tensor).t, value.(*Tensor).t, kqMask, C.float(scale), 0, 0)
		C.ggml_flash_attn_ext_set_prec(kqv, C.GGML_PREC_F32)
		return &Tensor{b: t.b, t: kqv, nDims: t.nDims}
	} else {
		value = value.Permute(ctx, 0, 2, 1, 3)
		kq := query.MatmulFullPrec(ctx, key)
		kq = &Tensor{
			b:     t.b,
			t:     C.ggml_soft_max_ext(ctx.(*Context).ctx, kq.(*Tensor).t, kqMask, C.float(scale), 0),
			nDims: t.nDims,
		}

		kqv := kq.Matmul(ctx, value)
		return kqv.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	}
}

// TODO - DRY this out with New if possible
func newTestBackend(size int) *Backend {
	var cpus []*C.struct_ggml_backend_device
	for _, d := range devices() {
		switch C.ggml_backend_dev_type(d) {
		case C.GGML_BACKEND_DEVICE_TYPE_CPU:
			if len(cpus) == 0 {
				// only the first cpu device should be used
				cpus = append(cpus, d)
				break
			}
		}
	}
	var schedBackends []*C.struct_ggml_backend
	var schedBufts []*C.struct_ggml_backend_buffer_type
	b := C.ggml_backend_dev_init(cpus[0], nil)
	bt := C.ggml_backend_get_default_buffer_type(b)
	C.ggml_backend_cpu_set_n_threads(b, C.int(Threads(runtime.NumCPU())))
	// C.ggml_backend_cpu_set_n_threads(b, 1) // DEBUGGING
	schedBackends = append(schedBackends, b)
	schedBufts = append(schedBufts, bt)
	return &Backend{
		meta: nil,
		sched: C.ggml_backend_sched_new(
			(*C.ggml_backend_t)(unsafe.Pointer(&schedBackends[0])),
			(*C.ggml_backend_buffer_type_t)(unsafe.Pointer(&schedBufts[0])),
			C.int(len(schedBackends)),
			C.size_t(max(8192, size)),
			true,
		),
		input:         bt,
		maxGraphNodes: max(8192, size),
	}
}

func newTestContext(b *Backend, n int) *Context {
	n = max(8192, n)
	return &Context{
		b:             b,
		maxGraphNodes: n,
		ctx: C.ggml_init(C.struct_ggml_init_params{
			mem_size: C.size_t(n)*C.ggml_tensor_overhead() + C.ggml_graph_overhead_custom(C.size_t(n), false),
			no_alloc: true,
		}),
	}
}

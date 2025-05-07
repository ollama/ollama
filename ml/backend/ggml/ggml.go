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
	"errors"
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
	"github.com/ollama/ollama/fs"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	ggml "github.com/ollama/ollama/ml/backend/ggml/ggml/src"
	"golang.org/x/sync/errgroup"
)

func devices() []*C.struct_ggml_backend_device {
	ggml.OnceLoad()
	ds := make([]*C.struct_ggml_backend_device, C.ggml_backend_dev_count())
	for i := range ds {
		ds[i] = C.ggml_backend_dev_get(C.size_t(i))
	}

	return ds
}

type Backend struct {
	meta *fsggml.GGML

	sched         *C.struct_ggml_backend_sched
	schedBackends []*C.struct_ggml_backend
	schedBufts    []*C.struct_ggml_backend_buffer_type

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
	meta, n, err := fsggml.Decode(r, -1)
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
		source *fsggml.Tensor
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

			slog.Log(context.TODO(), logutil.LevelTrace, "created tensor", "name", name, "shape", t.source.Shape, "dtype", t.source.Kind, "buffer_type", C.GoString(C.ggml_backend_buft_name(bt)))
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
		if b == nil {
			return nil, fmt.Errorf("unable to allocate memory from device %v for model weights", C.GoString(C.ggml_backend_buft_name(bt)))
		}

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
		t := t
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

			// Create a new FD for each goroutine so that each FD is read sequentially, rather than
			// seeking around within an FD shared between all goroutines.
			file, err := os.Open(r.Name())
			if err != nil {
				slog.Warn("file open error", "file", r.Name(), "error", err)
				return err
			}
			defer file.Close()
			sr := io.NewSectionReader(file, int64(meta.Tensors().Offset+t.Offset), int64(t.Size()))
			bts := make([]byte, 128*format.KibiByte)

			var s uint64
			for s < t.Size() {
				// Stop if either the parent context has been canceled or if any of the other tensors returned an error
				if err := ctx.Err(); err != nil {
					return err
				}

				n, err := io.ReadFull(sr, bts[:min(len(bts), int(t.Size()-s))])
				if err != nil {
					slog.Warn("file read error", "file", r.Name(), "error", err)
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

		deviceBufferTypes[d] = bt

		schedBackends = append(schedBackends, b)
		schedBufts = append(schedBufts, bt)

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
			C._Bool(false),
		),
		schedBackends: schedBackends,
		schedBufts:    schedBufts,
		input:         deviceBufferTypes[input.d],
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

func (b *Backend) Config() fs.Config {
	return b.meta.KV()
}

func (b *Backend) Get(name string) ml.Tensor {
	if t, ok := b.tensors[name]; ok {
		return &Tensor{b: b, t: t}
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

	var allocatedBuffers []*C.struct_ggml_backend_buffer

	return &Context{
		b:             b,
		maxGraphNodes: n,
		ctx: C.ggml_init(C.struct_ggml_init_params{
			mem_size: C.size_t(n)*C.ggml_tensor_overhead() + C.ggml_graph_overhead_custom(C.size_t(n), false),
			no_alloc: true,
		}),
		allocatedBuffers: &allocatedBuffers,
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

	// allocatedBuffers are buffers for tensors that we have allocated in this context
	// so that we can free them when we close the context
	allocatedBuffers *[]*C.struct_ggml_backend_buffer

	// maxGraphNodes is the maximum allowed number of graph nodes in this context
	maxGraphNodes int
}

func (c *Context) Input() ml.Context {
	if c.b.input != nil {
		return &Context{
			b:                c.b,
			ctx:              c.ctx,
			buft:             c.b.input,
			allocatedBuffers: c.allocatedBuffers,
			maxGraphNodes:    c.maxGraphNodes,
		}
	}

	return c
}

func (c *Context) Layer(i int) ml.Context {
	if buft, ok := c.b.layers[i]; ok {
		return &Context{
			b:                c.b,
			ctx:              c.ctx,
			buft:             buft,
			allocatedBuffers: c.allocatedBuffers,
			maxGraphNodes:    c.maxGraphNodes,
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

func (c *Context) Reserve() error {
	if !C.ggml_backend_sched_reserve(c.b.sched, c.graph) {
		C.ggml_backend_sched_reset(c.b.sched)
		return errors.New("failed to reserve graph")
	}

	slog.Debug("compute graph", "nodes", C.ggml_graph_n_nodes(c.graph), "splits", C.ggml_backend_sched_get_n_splits(c.b.sched))
	for i := range c.b.schedBackends {
		size := C.ggml_backend_sched_get_buffer_size(c.b.sched, c.b.schedBackends[i])
		slog.Info("compute graph", "backend", C.GoString(C.ggml_backend_name(c.b.schedBackends[i])), "buffer_type", C.GoString(C.ggml_backend_buft_name(c.b.schedBufts[i])),
			"size", format.HumanBytes2(uint64(size)))
	}

	C.ggml_backend_sched_reset(c.b.sched)

	return nil
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

func pad(length, pad C.size_t) C.size_t {
	return ((length + pad - 1) / pad) * pad
}

func (c *Context) newTensor(dtype ml.DType, shape []int) (ml.Tensor, error) {
	if c.buft == nil {
		panic("set Input or Layer before creating tensors")
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

	if len(shape) < 1 || shape[0] == 0 {
		var shape C.int64_t = 0
		return &Tensor{b: c.b, t: C.ggml_new_tensor(c.ctx, cdtype, 1, &shape)}, nil
	} else if len(shape) > 4 {
		panic("unsupported number of dimensions")
	}

	for _, dim := range shape {
		if dim < 1 {
			panic("invalid shape")
		}
	}

	t := C.ggml_new_tensor(c.ctx, cdtype, C.int(len(shape)), shapeToGGML(shape))
	size := pad(C.ggml_backend_buft_get_alloc_size(c.buft, t), C.ggml_backend_buft_get_alignment(c.buft))
	b := C.ggml_backend_buft_alloc_buffer(c.buft, size)
	if b == nil {
		return nil, fmt.Errorf("unable to allocate %v from device %v for new tensor", format.HumanBytes2(uint64(size)), C.GoString(C.ggml_backend_buft_name(c.buft)))
	}
	*c.allocatedBuffers = append(*c.allocatedBuffers, b)

	C.ggml_backend_tensor_alloc(b, t, C.ggml_backend_buffer_get_base(b))
	return &Tensor{b: c.b, t: t}, nil
}

func (c *Context) Empty(dtype ml.DType, shape ...int) ml.Tensor {
	t, err := c.newTensor(dtype, shape)
	if err != nil {
		panic(err)
	}

	return t
}

func (c *Context) Zeros(dtype ml.DType, shape ...int) ml.Tensor {
	t, err := c.newTensor(dtype, shape)
	if err != nil {
		panic(err)
	}

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

func (c *Context) FromFloatSlice(s []float32, shape ...int) (ml.Tensor, error) {
	if err := checkShape(s, shape...); err != nil {
		return nil, err
	}

	t, err := c.newTensor(ml.DTypeF32, shape)
	if err != nil {
		return nil, err
	}

	if len(s) > 0 {
		C.ggml_backend_tensor_set(t.(*Tensor).t, unsafe.Pointer(&s[0]), 0, C.ggml_nbytes(t.(*Tensor).t))
	}

	return t, nil
}

func (c *Context) FromIntSlice(s []int32, shape ...int) (ml.Tensor, error) {
	if err := checkShape(s, shape...); err != nil {
		return nil, err
	}

	t, err := c.newTensor(ml.DTypeI32, shape)
	if err != nil {
		return nil, err
	}

	if len(s) > 0 {
		C.ggml_backend_tensor_set(t.(*Tensor).t, unsafe.Pointer(&s[0]), 0, C.ggml_nbytes(t.(*Tensor).t))
	}

	return t, nil
}

func (c Context) Arange(start, stop, step float32, dtype ml.DType) ml.Tensor {
	switch dtype {
	case ml.DTypeF32:
		// ggml_arange creates a float32 tensor
		return &Tensor{
			b: c.b,
			t: C.ggml_arange(c.ctx, C.float(start), C.float(stop), C.float(step)),
		}
	case ml.DTypeI32:
		// ggml_cast does not support float32 to int32 conversion
		arange := make([]int32, 0, int((stop-start)/step))
		for i := start; i < stop; i += step {
			arange = append(arange, int32(i))
		}

		t, err := c.Input().FromIntSlice(arange, len(arange))
		if err != nil {
			panic(err)
		}

		return t
	default:
		panic("unsupported dtype for arange")
	}
}

func (c *Context) Close() {
	if c != nil {
		for _, b := range *c.allocatedBuffers {
			C.ggml_backend_buffer_free(b)
		}
		*c.allocatedBuffers = nil

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

func (t *Tensor) Neg(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_neg(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) Add(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_add(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

func (t *Tensor) Repeat(ctx ml.Context, dim, n int) ml.Tensor {
	if dim < 0 || dim >= C.GGML_MAX_DIMS {
		panic("invalid dimension")
	}

	shape := make([]C.int64_t, C.GGML_MAX_DIMS)
	for i := range C.GGML_MAX_DIMS {
		if i == dim {
			shape[i] = C.int64_t(t.Dim(i) * n)
		} else {
			shape[i] = C.int64_t(t.Dim(i))
		}
	}

	tmpl := C.ggml_new_tensor(ctx.(*Context).ctx, t.t._type, C.int(len(shape)), unsafe.SliceData(shape))
	return &Tensor{
		b: t.b,
		t: C.ggml_repeat(ctx.(*Context).ctx, t.t, tmpl),
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

func (t *Tensor) Div(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_div(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
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

func (t *Tensor) MulmatID(ctx ml.Context, t2, ids ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_mul_mat_id(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, ids.(*Tensor).t),
	}
}

func (t *Tensor) LayerNorm(ctx ml.Context, w, b ml.Tensor, eps float32) ml.Tensor {
	tt := C.ggml_norm(ctx.(*Context).ctx, t.t, C.float(eps))
	if w != nil {
		tt = C.ggml_mul(ctx.(*Context).ctx, tt, w.(*Tensor).t)
		if b != nil {
			tt = C.ggml_add(ctx.(*Context).ctx, tt, b.(*Tensor).t)
		}
	}

	return &Tensor{b: t.b, t: tt}
}

func (t *Tensor) RMSNorm(ctx ml.Context, w ml.Tensor, eps float32) ml.Tensor {
	tt := C.ggml_rms_norm(ctx.(*Context).ctx, t.t, C.float(eps))
	if w != nil {
		tt = C.ggml_mul(ctx.(*Context).ctx, tt, w.(*Tensor).t)
	}

	return &Tensor{b: t.b, t: tt}
}

func (t *Tensor) Pad(ctx ml.Context, shape ...int) ml.Tensor {
	if len(shape) != 4 {
		panic("expected 4 dimensions")
	} else if shape[3] != 0 {
		panic("cuda does not support 4d tensors")
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

func (t *Tensor) SumRows(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_sum_rows(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) Softmax(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_soft_max(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) Sin(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_sin(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) Cos(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_cos(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) Tanh(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_tanh_inplace(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) Sigmoid(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_sigmoid_inplace(ctx.(*Context).ctx, t.t),
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
	ropeTypeNorm   C.int = 0
	ropeTypeNeox   C.int = 2
	ropeTypeMrope  C.int = 8
	ropeTypeVision C.int = 24
)

func (t *Tensor) RoPE(ctx ml.Context, positionIDs, ropeFactors ml.Tensor, ropeDim, ropeType uint32, ropeBase, ropeScale float32, options ...ml.RopeOption) ml.Tensor {
	// Default options
	opts := &ml.RopeOptions{
		OriginalContextLen: 131072,
	}

	// Apply any provided options
	for _, option := range options {
		option(opts)
	}

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
			ctx.(*Context).ctx,
			dequant,
			positionIDs.(*Tensor).t,
			ropeFactors.(*Tensor).t,
			C.int(ropeDim),
			C.int(ropeType),
			C.int(opts.OriginalContextLen),
			C.float(ropeBase),
			C.float(ropeScale),
			C.float(0.0),
			C.float(1.0),
			C.float(32.0),
			C.float(1.0),
		),
	}
}

func (t *Tensor) IM2Col(ctx ml.Context, t2 ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_im2col(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, C.int(s0), C.int(s1), C.int(p0), C.int(p1), C.int(d0), C.int(d1), true, C.GGML_TYPE_F32),
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

func (t *Tensor) AvgPool2D(ctx ml.Context, k, s int, p float32) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_pool_2d(ctx.(*Context).ctx, t.t, C.GGML_OP_POOL_AVG, C.int(k), C.int(k), C.int(s), C.int(s), C.float(p), C.float(p)),
	}
}

func (t *Tensor) Set(ctx ml.Context, t2 ml.Tensor, offset int, strides ...int) ml.Tensor {
	var tt *C.struct_ggml_tensor
	switch len(strides) {
	case 0:
		tt = C.ggml_set_1d(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, C.size_t(offset))
	case 1:
		tt = C.ggml_set_2d(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, C.size_t(offset), C.size_t(strides[0]))
	default:
		panic("unsupported number of dimensions")
	}

	return &Tensor{b: t.b, t: tt}
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

func (t *Tensor) Duplicate(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_dup(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) TopK(ctx ml.Context, k int) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_top_k(ctx.(*Context).ctx, t.t, C.int(k)),
	}
}

func (t *Tensor) Argsort(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_argsort(ctx.(*Context).ctx, t.t, C.GGML_SORT_ORDER_ASC),
	}
}

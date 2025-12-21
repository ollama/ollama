package ggml

// #cgo linux LDFLAGS: -lrt -lpthread -ldl -lstdc++ -lm
// #cgo windows LDFLAGS: -lpthread
// #cgo CPPFLAGS: -I${SRCDIR}/ggml/include
// #include <stdlib.h>
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-cpu.h"
// #include "ggml-backend.h"
import "C"

import (
	"cmp"
	"context"
	"encoding/binary"
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
	"sync"
	"sync/atomic"
	"unicode"
	"unsafe"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	ggml "github.com/ollama/ollama/ml/backend/ggml/ggml/src"
	"github.com/ollama/ollama/ml/nn/rope"
	"golang.org/x/sync/errgroup"
)

var (
	cpus, accels, gpus []C.ggml_backend_dev_t
	backends           map[C.ggml_backend_dev_t]C.ggml_backend_t
)

var initDevices = sync.OnceFunc(func() {
	ggml.OnceLoad()

	backends = make(map[C.ggml_backend_dev_t]C.ggml_backend_t)
	for i := range C.ggml_backend_dev_count() {
		d := C.ggml_backend_dev_get(i)

		switch C.ggml_backend_dev_type(d) {
		case C.GGML_BACKEND_DEVICE_TYPE_CPU:
			if len(cpus) == 0 {
				// only the first cpu device should be used
				cpus = append(cpus, d)
			}
		case C.GGML_BACKEND_DEVICE_TYPE_ACCEL:
			accels = append(accels, d)
		case C.GGML_BACKEND_DEVICE_TYPE_GPU,
			C.GGML_BACKEND_DEVICE_TYPE_IGPU:
			gpus = append(gpus, d)
		}

		backends[d] = C.ggml_backend_dev_init(d, nil)
	}
})

type layerDevice struct {
	d  C.ggml_backend_dev_t
	bt C.ggml_backend_buffer_type_t
}

type Backend struct {
	// modelPath is the location of the model data
	modelPath string

	meta *fsggml.GGML

	// allocMemory means that memory should be allocated for tensors and not
	// just a dry run
	allocMemory bool

	// tensorLoadTargets maps from the name of the tensor in the file
	// to the name that is used by the model definition
	tensorLoadTargets map[string][]string

	schedMu       sync.Mutex // Only one Compute can run at a time
	sched         C.ggml_backend_sched_t
	schedBackends []C.ggml_backend_t
	schedBufts    []C.ggml_backend_buffer_type_t

	tensors map[string]*C.struct_ggml_tensor

	// input is the backend buffer type used for inputs
	input C.ggml_backend_buffer_type_t

	// output is the backend device used for outputs
	output C.ggml_backend_dev_t

	// layers is the backend used for repeating layers
	layers map[int]layerDevice

	// requiredMemory is the cumulative memory allocations needed by the backend
	requiredMemory *ml.BackendMemory

	// btDeviceMemory maps from a buffer type to the memory allocations associated with that device
	btDeviceMemory map[C.ggml_backend_buffer_type_t]*ml.DeviceMemory

	flashAttention ml.FlashAttentionType

	// maxGraphNodes is the maximum allowed number of graph nodes in this scheduler
	maxGraphNodes int

	// weightBuffers are the GGML contexts and buffers for allocating weights
	weightBuffers map[*C.struct_ggml_context]C.ggml_backend_buffer_t
}

var once sync.Once

func New(modelPath string, params ml.BackendParams) (ml.Backend, error) {
	r, err := os.Open(modelPath)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	meta, err := fsggml.Decode(r, -1)
	if err != nil {
		return nil, err
	}

	once.Do(func() {
		slog.Info(
			"",
			"architecture", meta.KV().Architecture(),
			"file_type", meta.KV().FileType(),
			"name", meta.KV().String("general.name"),
			"description", meta.KV().String("general.description"),
			"num_tensors", len(meta.Tensors().Items()),
			"num_key_values", len(meta.KV()),
		)
	})

	initDevices()

	var requiredMemory ml.BackendMemory
	btDeviceMemory := make(map[C.ggml_backend_buffer_type_t]*ml.DeviceMemory)

	type deviceBufferType struct {
		d   C.ggml_backend_dev_t
		bts []C.ggml_backend_buffer_type_t
	}

	blocks := int(meta.KV().BlockCount())

	// create list of buffer types for the cpu
	cpuDeviceBufferType := deviceBufferType{d: C.ggml_backend_dev_by_type(C.GGML_BACKEND_DEVICE_TYPE_CPU)}
	for _, d := range append(accels, append(gpus, cpus...)...) {
		switch C.ggml_backend_dev_type(d) {
		case C.GGML_BACKEND_DEVICE_TYPE_CPU,
			C.GGML_BACKEND_DEVICE_TYPE_ACCEL:
			bt := C.ggml_backend_dev_buffer_type(d)
			cpuDeviceBufferType.bts = append(cpuDeviceBufferType.bts, bt)

			btDeviceMemory[C.ggml_backend_dev_buffer_type(d)] = &requiredMemory.CPU
		}
	}

	requiredMemory.CPU.Name = C.GoString(C.ggml_backend_dev_name(cpuDeviceBufferType.d))
	var props C.struct_ggml_backend_dev_props
	C.ggml_backend_dev_get_props(cpuDeviceBufferType.d, &props)
	requiredMemory.CPU.ID = C.GoString(props.id)
	requiredMemory.CPU.Library = C.GoString(props.library)
	requiredMemory.CPU.Weights = make([]uint64, blocks+1)
	requiredMemory.CPU.Cache = make([]uint64, blocks+1)

	// create list of buffer types for each gpu
	var gpuDeviceBufferTypes []deviceBufferType
	requiredMemory.GPUs = make([]ml.DeviceMemory, len(gpus))
	for i, d := range gpus {
		bt := C.ggml_backend_dev_buffer_type(d)
		gpuDeviceBufferTypes = append(gpuDeviceBufferTypes, deviceBufferType{
			d:   d,
			bts: append([]C.ggml_backend_buffer_type_t{bt}, cpuDeviceBufferType.bts...),
		})

		btDeviceMemory[bt] = &requiredMemory.GPUs[i]
		requiredMemory.GPUs[i].Name = C.GoString(C.ggml_backend_dev_name(d))
		var props C.struct_ggml_backend_dev_props
		C.ggml_backend_dev_get_props(d, &props)
		requiredMemory.GPUs[i].ID = C.GoString(props.id)
		requiredMemory.GPUs[i].Library = C.GoString(props.library)
		requiredMemory.GPUs[i].Weights = make([]uint64, blocks+1)
		requiredMemory.GPUs[i].Cache = make([]uint64, blocks+1)
	}

	// inputs always use cpu
	input := cpuDeviceBufferType

	assignLayer := func(layer int) deviceBufferType {
		for _, p := range params.GPULayers {
			for _, l := range p.Layers {
				if l == layer {
					for i := range requiredMemory.GPUs {
						if requiredMemory.GPUs[i].DeviceID == p.DeviceID {
							return gpuDeviceBufferTypes[i]
						}
					}

					return cpuDeviceBufferType
				}
			}
		}

		return cpuDeviceBufferType
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
	ctxs := make(map[C.ggml_backend_buffer_type_t]*C.struct_ggml_context)
	createTensor := func(t tensor, bts []C.ggml_backend_buffer_type_t, layer int) *C.struct_ggml_tensor {
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

			kind := t.source.Kind
			if t.source.Kind == 4 {
				// transform raw mxfp4 stream to ggml mxfp4 format
				kind = 39
			} else if t.source.Kind == uint32(fsggml.TensorTypeBF16) && strings.HasSuffix(t.source.Name, "_exps.bias") {
				// transform "_exps.bias" from bf16 to fp32; add_ids only supports fp32 tensors
				kind = uint32(fsggml.TensorTypeF32)
			}

			tt := C.ggml_new_tensor(ctxs[bt], kind, C.int(len(t.source.Shape)), (*C.int64_t)(unsafe.Pointer(&t.source.Shape[0])))
			C.ggml_set_name(tt, cname)

			logutil.Trace("created tensor", "name", name, "shape", t.source.Shape, "dtype", t.source.Kind, "buffer_type", C.GoString(C.ggml_backend_buft_name(bt)))

			size := pad(C.ggml_backend_buft_get_alloc_size(bt, tt), C.ggml_backend_buft_get_alignment(bt))
			if layer == -1 {
				requiredMemory.InputWeights += uint64(size)
			} else {
				btDeviceMemory[bt].Weights[layer] += uint64(size)
			}

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
			createTensor(tensor{source: t}, input.bts, -1)
			if _, ok := meta.Tensors().GroupLayers()["output"]; !ok && t.Name == "token_embd.weight" {
				createTensor(tensor{source: t, target: "output.weight"}, output.bts, blocks)
			}
		case contains(t.Name, "cls", "output", "output_norm",
			"altup_proj", "altup_unembd_proj",
			"per_layer_token_embd", "per_layer_model_proj", "per_layer_proj_norm"):
			createTensor(tensor{source: t}, output.bts, blocks)
		case strings.HasPrefix(t.Name, "v.") || strings.HasPrefix(t.Name, "mm.") || strings.HasPrefix(t.Name, "s."):
			// TODO: assign vision tensors to the gpu if possible
			createTensor(tensor{source: t}, output.bts, blocks)
		case contains(t.Name, "rope_freqs", "rope_factors_long", "rope_factors_short"):
			// these tensors should be repeated per layer
			for i, layer := range layers {
				createTensor(tensor{
					source: t,
					target: "blk." + strconv.Itoa(i) + "." + t.Name,
				}, layer.bts, i)
			}
		default:
			layerIndex := -1
			if fields := strings.FieldsFunc(t.Name, func(r rune) bool { return !unicode.IsNumber(r) }); len(fields) > 0 {
				if i, err := strconv.Atoi(fields[0]); err == nil {
					layerIndex = i
				}
			}

			if layerIndex >= 0 {
				createTensor(tensor{source: t}, layers[layerIndex].bts, layerIndex)
			} else {
				// load all other tensors on the cpu
				createTensor(tensor{source: t}, input.bts, -1)
			}
		}
	}

	// map tensor names to tensors for easy lookup later
	tensors := make(map[string]*C.struct_ggml_tensor)
	for _, c := range ctxs {
		for t := C.ggml_get_first_tensor(c); t != nil; t = C.ggml_get_next_tensor(c, t) {
			tensors[C.GoString(C.ggml_get_name(t))] = t
		}
	}

	// map devices to backend buffer types so new tensors can be assigned to the correct device
	deviceBufferTypes := make(map[C.ggml_backend_dev_t]C.ggml_backend_buffer_type_t)

	// create backends and buffer types used for the compute graph scheduler
	var schedBackends []C.ggml_backend_t
	var schedBufts []C.ggml_backend_buffer_type_t
	for _, d := range append(gpus, append(accels, cpus...)...) {
		b := backends[d]
		bt := C.ggml_backend_get_default_buffer_type(b)

		// Always include CPU as a fallback but otherwise, just use the devices where we assigned layers
		if !slices.Contains(cpuDeviceBufferType.bts, bt) {
			if c, ok := ctxs[bt]; !ok || C.ggml_get_first_tensor(c) == nil {
				continue
			}
		}

		deviceBufferTypes[d] = bt

		schedBackends = append(schedBackends, b)
		schedBufts = append(schedBufts, bt)

		if C.ggml_backend_is_cpu(b) {
			// set number of threads for cpu backend
			C.ggml_backend_cpu_set_n_threads(b, C.int(Threads(params.NumThreads)))
		}
	}

	maxGraphNodes := max(1024, len(meta.Tensors().Items())*8)

	sched := C.ggml_backend_sched_new_ext(
		(*C.ggml_backend_t)(unsafe.Pointer(&schedBackends[0])),
		(*C.ggml_backend_buffer_type_t)(unsafe.Pointer(&schedBufts[0])),
		C.int(len(schedBackends)),
		C.size_t(maxGraphNodes),
		C._Bool(false),
		C._Bool(true),
		C._Bool(params.AllocMemory),
	)

	// allocate buffers for each context
	bbs := make(map[*C.struct_ggml_context]C.ggml_backend_buffer_t, len(ctxs))
	for bt, c := range ctxs {
		if C.ggml_get_first_tensor(c) == nil {
			continue
		}

		b := C.ggml_backend_alloc_ctx_tensors_from_buft(c, bt)
		if b == nil {
			for _, b := range bbs {
				C.ggml_backend_buffer_free(b)
			}

			for _, ctx := range ctxs {
				C.ggml_free(ctx)
			}

			panic(ml.ErrNoMem{BackendMemory: requiredMemory})
		}

		C.ggml_backend_buffer_set_usage(b, C.GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
		bbs[c] = b
	}

	for bs := range maps.Values(bbs) {
		logutil.Trace("model weights", "buffer", C.GoString(C.ggml_backend_buffer_name(bs)),
			"size", format.HumanBytes2(uint64(C.ggml_backend_buffer_get_size(bs))))
	}

	return &Backend{
		modelPath:         modelPath,
		allocMemory:       params.AllocMemory,
		flashAttention:    params.FlashAttention,
		meta:              meta,
		tensorLoadTargets: targets,
		tensors:           tensors,
		sched:             sched,
		schedBackends:     schedBackends,
		schedBufts:        schedBufts,
		input:             deviceBufferTypes[input.d],
		output:            output.d,
		layers: func() map[int]layerDevice {
			m := make(map[int]layerDevice)
			for i, layer := range layers {
				m[i] = layerDevice{
					d:  layer.d,
					bt: deviceBufferTypes[layer.d],
				}
			}
			return m
		}(),
		requiredMemory: &requiredMemory,
		btDeviceMemory: btDeviceMemory,
		maxGraphNodes:  maxGraphNodes,
		weightBuffers:  bbs,
	}, nil
}

func init() {
	ml.RegisterBackend("ggml", New)
}

func (b *Backend) Close() {
	if b == nil {
		return
	}

	for ctx, b := range b.weightBuffers {
		C.ggml_backend_buffer_free(b)
		C.ggml_free(ctx)
	}

	C.ggml_backend_sched_free(b.sched)
}

func (b *Backend) Load(ctx context.Context, progress func(float32)) error {
	if !b.allocMemory {
		return errors.New("cannot load model without memory allocation")
	}

	// Mimic llama runner logs summarizing layers and memory
	gpuLayers := 0
	for layer := range maps.Values(b.layers) {
		switch C.ggml_backend_dev_type(layer.d) {
		case C.GGML_BACKEND_DEVICE_TYPE_GPU,
			C.GGML_BACKEND_DEVICE_TYPE_IGPU:
			gpuLayers++
		}
	}
	slog.Info(fmt.Sprintf("offloading %d repeating layers to GPU", gpuLayers))

	switch C.ggml_backend_dev_type(b.output) {
	case C.GGML_BACKEND_DEVICE_TYPE_CPU:
		slog.Info("offloading output layer to CPU")
	case C.GGML_BACKEND_DEVICE_TYPE_GPU,
		C.GGML_BACKEND_DEVICE_TYPE_IGPU:
		slog.Info("offloading output layer to GPU")
		gpuLayers++
	case C.GGML_BACKEND_DEVICE_TYPE_ACCEL:
		slog.Info("offloading output layer to ACCEL")
	}
	slog.Info(fmt.Sprintf("offloaded %d/%d layers to GPU", gpuLayers, len(b.layers)+1))

	var doneBytes atomic.Uint64
	totalBytes := uint64(b.meta.Length) - b.meta.Tensors().Offset

	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(runtime.GOMAXPROCS(0))
	for _, t := range b.meta.Tensors().Items() {
		g.Go(func() error {
			tts := make([]*C.struct_ggml_tensor, max(1, len(b.tensorLoadTargets[t.Name])))
			for i := range tts {
				target := b.tensorLoadTargets[t.Name][i]
				if target == "" {
					target = t.Name
				}

				tt, ok := b.tensors[target]
				if !ok {
					return fmt.Errorf("unassigned tensor: %s", t.Name)
				}

				tts[i] = tt
			}

			// Create a new FD for each goroutine so that each FD is read sequentially, rather than
			// seeking around within an FD shared between all goroutines.
			file, err := os.Open(b.modelPath)
			if err != nil {
				slog.Warn("file open error", "file", b.modelPath, "error", err)
				return err
			}
			defer file.Close()
			sr := io.NewSectionReader(file, int64(b.meta.Tensors().Offset+t.Offset), int64(t.Size()))

			if t.Kind == 4 && tts[0]._type == 39 {
				// source is mxfp4, target is ggml mxfp4

				const BS = 17                             // MXFP4 block size
				bts := make([]byte, 8*BS*format.KibiByte) // ~128k block aligned
				var s uint64
				var tmp [16]byte
				for s < t.Size() {
					// Stop if either the parent context has been canceled or if any of the other tensors returned an error
					if err := ctx.Err(); err != nil {
						return err
					}
					n, err := io.ReadFull(sr, bts[:min(len(bts), int(t.Size()-s))])
					if err != nil {
						slog.Warn("file read error", "file", b.modelPath, "error", err)
						return err
					}
					for j := range n / BS {
						for i := 1; i < 9; i++ {
							// transform a1b2c3 ... x7y8z9 -> 71xa82yb93zc
							a, b := bts[j*BS+i], bts[j*BS+i+8]
							tmp[2*(i-1)] = (a & 0x0F) | (b << 4)
							tmp[2*(i-1)+1] = (a >> 4) | (b & 0xF0)
						}
						copy(bts[j*BS+1:j*BS+17], tmp[:])
					}

					for _, tt := range tts {
						C.ggml_backend_tensor_set(tt, unsafe.Pointer(&bts[0]), C.size_t(s), C.size_t(n))
					}

					s += uint64(n)

					if progress != nil {
						done := doneBytes.Add(uint64(n))
						progress(float32(done) / float32(totalBytes))
					}
				}
				return nil
			} else if strings.HasSuffix(t.Name, "_exps.bias") && t.Kind == 30 && tts[0]._type == 0 {
				// source is bf16, target is ggml fp32

				// data is bf16 but we need to convert to fp32
				bts := make([]byte, 128*format.KibiByte)
				var e uint64
				for e < t.Elements() {
					// Stop if either the parent context has been canceled or if any of the other tensors returned an error
					if err := ctx.Err(); err != nil {
						return err
					}
					n, err := io.ReadFull(sr, bts[:min(len(bts), int(t.Elements()-e)*2)])
					if err != nil {
						slog.Warn("file read error", "file", b.modelPath, "error", err)
						return err
					}
					fp32 := ConvertToF32(bts, uint32(fsggml.TensorTypeBF16), uint64(n/2))

					for _, tt := range tts {
						C.ggml_backend_tensor_set(tt, unsafe.Pointer(&fp32[0]), C.size_t(e*4), C.size_t(n*2))
					}
					e += uint64(n / 2)
					if progress != nil {
						done := doneBytes.Add(uint64(n))
						progress(float32(done) / float32(totalBytes))
					}
				}
				return nil
			}

			bts := make([]byte, 128*format.KibiByte)

			var s uint64
			for s < t.Size() {
				// Stop if either the parent context has been canceled or if any of the other tensors returned an error
				if err := ctx.Err(); err != nil {
					return err
				}

				n, err := io.ReadFull(sr, bts[:min(len(bts), int(t.Size()-s))])
				if err != nil {
					slog.Warn("file read error", "file", b.modelPath, "error", err)
					return err
				}

				for _, tt := range tts {
					C.ggml_backend_tensor_set(tt, unsafe.Pointer(&bts[0]), C.size_t(s), C.size_t(n))
				}

				s += uint64(n)

				if progress != nil {
					done := doneBytes.Add(uint64(n))
					progress(float32(done) / float32(totalBytes))
				}
			}

			return nil
		})
	}

	// Cleanup any backend state from devices that we didn't end up using
nextDevice:
	for _, d := range append(gpus, append(accels, cpus...)...) {
		for _, backend := range b.schedBackends {
			if d == C.ggml_backend_get_device(backend) {
				continue nextDevice
			}
		}

		C.ggml_backend_dev_reset(d)
	}

	if err := g.Wait(); err != nil {
		return err
	}

	return nil
}

func (b *Backend) BackendMemory() ml.BackendMemory {
	return *b.requiredMemory
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

	var allocatedBuffers []C.ggml_backend_buffer_t

	return &Context{
		b:             b,
		maxGraphNodes: n,
		ctx: C.ggml_init(C.struct_ggml_init_params{
			mem_size: C.size_t(n)*C.ggml_tensor_overhead() + C.ggml_graph_overhead_custom(C.size_t(n), false),
			no_alloc: true,
		}),
		allocatedBuffers: &allocatedBuffers,
		layer:            -1,
	}
}

func (b *Backend) CacheConfig() ml.CacheConfig {
	if b.flashAttention == ml.FlashAttentionEnabled {
		return ml.CacheConfig{CachePadding: 256, MaskDType: ml.DTypeF16}
	} else {
		return ml.CacheConfig{CachePadding: 256, PermutedV: true}
	}
}

func (b *Backend) BackendDevices() []ml.DeviceInfo {
	deviceInfos := []ml.DeviceInfo{}
	for _, dev := range gpus {
		// If we have a model loaded, and it's only loaded on a subset of the devices
		// skip idle/unused devices to avoid initializing them and causing VRAM allocations
		if b.allocMemory {
			idleDev := true
			for _, backend := range b.schedBackends {
				if dev == C.ggml_backend_get_device(backend) {
					idleDev = false
					break
				}
			}
			if idleDev {
				slog.Debug("skipping unused backend device", "description", C.GoString(C.ggml_backend_dev_description(dev)))
				continue
			}
		}

		info := ml.DeviceInfo{}
		props := C.struct_ggml_backend_dev_props{}
		C.ggml_backend_dev_get_props(dev, &props)
		info.Name = C.GoString(props.name)
		info.Description = C.GoString(props.description)
		info.ID = C.GoString(props.id)
		info.Library = C.GoString(props.library)
		info.ComputeMajor = (int)(props.compute_major)
		info.ComputeMinor = (int)(props.compute_minor)
		info.DriverMajor = (int)(props.driver_major)
		info.DriverMinor = (int)(props.driver_minor)
		info.Integrated = props.integrated != 0
		if props.library != nil {
			info.Library = C.GoString(props.library)
		}
		if props.device_id != nil {
			info.PCIID = C.GoString(props.device_id)
		}
		info.LibraryPath = ggml.LibPaths()
		C.ggml_backend_dev_memory(dev, &props.memory_free, &props.memory_total)
		info.TotalMemory = (uint64)(props.memory_total)
		info.FreeMemory = (uint64)(props.memory_free)

		deviceInfos = append(deviceInfos, info)
	}
	return deviceInfos
}

type Context struct {
	b *Backend

	ctx   *C.struct_ggml_context
	graph *C.struct_ggml_cgraph

	// batchSize is a hint to optimize processing
	batchSize int

	// buft is the buffer type used for new tensors
	buft C.ggml_backend_buffer_type_t

	// allocatedBuffers are buffers for tensors that we have allocated in this context
	// so that we can free them when we close the context
	allocatedBuffers *[]C.ggml_backend_buffer_t

	// maxGraphNodes is the maximum allowed number of graph nodes in this context
	maxGraphNodes int

	// layer is the graph layer that this context is allocating for - assumed to be cache
	layer int
}

func (c *Context) Input() ml.Context {
	if c.b.input != nil {
		return &Context{
			b:                c.b,
			ctx:              c.ctx,
			buft:             c.b.input,
			allocatedBuffers: c.allocatedBuffers,
			maxGraphNodes:    c.maxGraphNodes,
			layer:            -1,
		}
	}

	return c
}

func (c *Context) Layer(i int) ml.Context {
	if layer, ok := c.b.layers[i]; ok {
		return &Context{
			b:                c.b,
			ctx:              c.ctx,
			buft:             layer.bt,
			allocatedBuffers: c.allocatedBuffers,
			maxGraphNodes:    c.maxGraphNodes,
			layer:            i,
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

func (c *Context) SetBatchSize(batchSize int) {
	c.batchSize = batchSize
}

func (c *Context) Compute(tensors ...ml.Tensor) {
	c.ComputeWithNotify(nil, tensors...)
}

func (c *Context) ComputeWithNotify(cb func(), tensors ...ml.Tensor) {
	c.b.schedMu.Lock()
	defer c.b.schedMu.Unlock()
	if cb != nil {
		go cb()
	}

	if c.batchSize > 0 {
		C.ggml_backend_sched_set_batch_size(c.b.sched, C.int(c.batchSize))
	}

	if status := C.ggml_backend_sched_graph_compute_async(c.b.sched, c.graph); status != C.GGML_STATUS_SUCCESS {
		panic(fmt.Errorf("error computing ggml graph: %v", status))
	}
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

func (c *Context) Reserve() {
	if c.batchSize > 0 {
		C.ggml_backend_sched_set_batch_size(c.b.sched, C.int(c.batchSize))
	}

	reserved := C.ggml_backend_sched_reserve(c.b.sched, c.graph)

	slog.Debug("compute graph", "nodes", C.ggml_graph_n_nodes(c.graph), "splits", C.ggml_backend_sched_get_n_splits(c.b.sched))

	// Reserve may get called multiple times for different graphs - we just want the last run, which will contain the max allocations
	for _, bt := range c.b.schedBufts {
		c.b.btDeviceMemory[bt].Graph = 0
	}

	for i := range c.b.schedBackends {
		bufferSize := C.ggml_backend_sched_get_attempted_buffer_size(c.b.sched, c.b.schedBackends[i])
		c.b.btDeviceMemory[c.b.schedBufts[i]].Graph += uint64(bufferSize)

		logutil.Trace("compute graph", "backend", C.GoString(C.ggml_backend_name(c.b.schedBackends[i])),
			"buffer_type", C.GoString(C.ggml_backend_buft_name(c.b.schedBufts[i])), "size", format.HumanBytes2(uint64(bufferSize)))
	}

	if !reserved {
		panic(ml.ErrNoMem{BackendMemory: *c.b.requiredMemory})
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

func pad(length, pad C.size_t) C.size_t {
	return ((length + pad - 1) / pad) * pad
}

func (c *Context) newTensor(dtype ml.DType, shape []int) *Tensor {
	if c.buft == nil {
		panic("set Input or Layer before creating tensors")
	}

	cdtype := ggmlDType(dtype)

	if len(shape) < 1 || shape[0] == 0 {
		var shape C.int64_t = 0
		return &Tensor{b: c.b, t: C.ggml_new_tensor(c.ctx, cdtype, 1, &shape)}
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
	if c.layer >= 0 {
		c.b.btDeviceMemory[c.buft].Cache[c.layer] += uint64(size)
	}

	if b == nil {
		panic(ml.ErrNoMem{BackendMemory: *c.b.requiredMemory})
	}

	*c.allocatedBuffers = append(*c.allocatedBuffers, b)
	C.ggml_backend_tensor_alloc(b, t, C.ggml_backend_buffer_get_base(b))
	return &Tensor{b: c.b, t: t}
}

func (c *Context) Empty(dtype ml.DType, shape ...int) ml.Tensor {
	return c.newTensor(dtype, shape)
}

func (c *Context) Zeros(dtype ml.DType, shape ...int) ml.Tensor {
	t := c.newTensor(dtype, shape)
	if c.b.allocMemory {
		C.ggml_set_zero(t.t)
	}
	return t
}

func checkShape[S ~[]E, E any](s S, shape ...int) {
	n := len(s)

	if n == 0 {
		return
	}

	for _, v := range shape {
		n /= v
	}

	if n != 1 {
		panic(fmt.Errorf("invalid shape: %v", shape))
	}
}

func (c Context) FromBytes(dtype ml.DType, s []uint8, shape ...int) ml.Tensor {
	// Unchecked to handle quantized types
	t := c.newTensor(dtype, shape)
	if c.b.allocMemory {
		t.FromBytes(s)
	}

	return t
}

func (c *Context) FromFloats(s []float32, shape ...int) ml.Tensor {
	checkShape(s, shape...)

	t := c.newTensor(ml.DTypeF32, shape)

	if c.b.allocMemory {
		t.FromFloats(s)
	}

	return t
}

func (c *Context) FromInts(s []int32, shape ...int) ml.Tensor {
	checkShape(s, shape...)

	t := c.newTensor(ml.DTypeI32, shape)
	if c.b.allocMemory {
		t.FromInts(s)
	}

	return t
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

		return c.Input().FromInts(arange, len(arange))
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

func tensorSet[S ~[]E, E byte | float32 | int32](t *Tensor, s S) {
	if len(s) == 0 {
		return
	}
	if int(C.ggml_nbytes(t.t)) != len(s)*binary.Size(s[0]) {
		panic("data size does not match tensor size")
	}
	C.ggml_backend_tensor_set(t.t, unsafe.Pointer(&s[0]), 0, C.ggml_nbytes(t.t))
}

func (t *Tensor) FromBytes(s []byte) {
	tensorSet(t, s)
}

func (t *Tensor) FromFloats(s []float32) {
	tensorSet(t, s)
}

func (t *Tensor) FromInts(s []int32) {
	tensorSet(t, s)
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
	case C.GGML_TYPE_MXFP4:
		return ml.DTypeMXFP4
	default:
		return ml.DTypeOther
	}
}

func ggmlDType(dtype ml.DType) uint32 {
	switch dtype {
	case ml.DTypeF32:
		return C.GGML_TYPE_F32
	case ml.DTypeF16:
		return C.GGML_TYPE_F16
	case ml.DTypeQ80:
		return C.GGML_TYPE_Q8_0
	case ml.DTypeQ40:
		return C.GGML_TYPE_Q4_0
	case ml.DTypeI32:
		return C.GGML_TYPE_I32
	case ml.DTypeMXFP4:
		return C.GGML_TYPE_MXFP4
	default:
		panic("unsupported dtype")
	}
}

func (t *Tensor) Cast(ctx ml.Context, dtype ml.DType) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_cast(ctx.(*Context).ctx, t.t, ggmlDType(dtype)),
	}
}

func (t *Tensor) Add(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_add(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

func (t *Tensor) Sub(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_sub(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
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

func (t *Tensor) Contiguous(ctx ml.Context, shape ...int) ml.Tensor {
	if slices.Contains(shape, -1) {
		inferShape(t, shape)
	}

	switch len(shape) {
	case 0:
		return &Tensor{
			b: t.b,
			t: C.ggml_cont(ctx.(*Context).ctx, t.t),
		}
	case 1:
		return &Tensor{
			b: t.b,
			t: C.ggml_cont_1d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0])),
		}
	case 2:
		return &Tensor{
			b: t.b,
			t: C.ggml_cont_2d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), C.int64_t(shape[1])),
		}
	case 3:
		return &Tensor{
			b: t.b,
			t: C.ggml_cont_3d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), C.int64_t(shape[1]), C.int64_t(shape[2])),
		}
	case 4:
		return &Tensor{
			b: t.b,
			t: C.ggml_cont_4d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), C.int64_t(shape[1]), C.int64_t(shape[2]), C.int64_t(shape[3])),
		}
	default:
		panic("unsupported number of dimensions")
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

// Mulmat performs matrix multiplication between two tensors.
// If t has shape [m, p, ...] and t2 has shape [m, n, ...],
// Mulmat returns a new Tensor with shape [p, n, ...].
//
// Note: this is similar to matmul(t2, t.tranpose(-1, -2)) in other libraries.
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

func (t *Tensor) AddID(ctx ml.Context, t2, ids ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_add_id(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, ids.(*Tensor).t),
	}
}

func (t *Tensor) L2Norm(ctx ml.Context, eps float32) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_l2_norm(ctx.(*Context).ctx, t.t, C.float(eps)),
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

// Permute permutes t according to order. Permute panics if the number of dimensions
// in order does not match the number of dimensions in t.
func (t *Tensor) Permute(ctx ml.Context, order ...int) ml.Tensor {
	if len(order) != len(t.Shape()) && len(order) != 4 {
		panic("invalid number of dimensions for permute")
	}

	// ggml_permute requires 4 dimensions so fill in the rest
	for i := len(order); i < 4; i++ {
		order = append(order, i)
	}

	return &Tensor{
		b: t.b,
		t: C.ggml_permute(ctx.(*Context).ctx, t.t, C.int(order[0]), C.int(order[1]), C.int(order[2]), C.int(order[3])),
	}
}

func (t *Tensor) Rows(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_get_rows(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

func (t *Tensor) SetRows(ctx ml.Context, src ml.Tensor, idxs ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_set_rows(ctx.(*Context).ctx, t.t, src.(*Tensor).t, idxs.(*Tensor).t),
	}
}

func (t *Tensor) Copy(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_cpy(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

// inferShape updates shape in place to automatically set a single -1 dimesion
// based on the input tensor and the other dimensions
func inferShape(t *Tensor, shape []int) {
	total := 1
	for _, dim := range t.Shape() {
		total *= dim
	}

	dim := -1
	for i := range shape {
		switch shape[i] {
		case -1:
			if dim != -1 {
				panic("only one dimension can be inferred")
			}
			dim = i
		case 0:
			panic("dimension cannot be zero")
		default:
			if total%shape[i] != 0 {
				panic("cannot infer dimension")
			}

			total /= shape[i]
		}
	}

	if dim != -1 {
		shape[dim] = total
	}
}

func (t *Tensor) Reshape(ctx ml.Context, shape ...int) ml.Tensor {
	if !C.ggml_is_contiguous(t.t) {
		return t.Contiguous(ctx, shape...)
	}

	if slices.Contains(shape, -1) {
		inferShape(t, shape)
	}

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

func (t *Tensor) RoPE(ctx ml.Context, positions ml.Tensor, ropeDim int, ropeBase, ropeScale float32, options ...func(*rope.Options)) ml.Tensor {
	// Default options
	opts := rope.Options{Factors: &Tensor{}}

	// Apply any provided options
	for _, option := range options {
		option(&opts)
	}

	dequant := t.t
	if C.ggml_is_quantized(t.t._type) {
		dequant = C.ggml_cast(ctx.(*Context).ctx, t.t, C.GGML_TYPE_F32)
	}

	var tt *C.struct_ggml_tensor
	if len(opts.MRoPE.Sections) > 0 {
		mropeSections := make([]C.int32_t, 4)
		for i, section := range opts.MRoPE.Sections {
			mropeSections[i] = C.int32_t(section)
		}

		tt = C.ggml_rope_multi(
			ctx.(*Context).ctx,
			dequant,
			positions.(*Tensor).t,
			opts.Factors.(*Tensor).t,
			C.int(ropeDim),
			unsafe.SliceData(mropeSections),
			C.int(opts.Type),
			cmp.Or(C.int(opts.YaRN.OriginalContextLength), 128<<10),
			C.float(ropeBase),
			C.float(ropeScale),
			C.float(opts.YaRN.ExtrapolationFactor),
			cmp.Or(C.float(opts.YaRN.AttentionFactor), 1),
			cmp.Or(C.float(opts.YaRN.BetaFast), 32),
			cmp.Or(C.float(opts.YaRN.BetaSlow), 1),
		)
	} else {
		tt = C.ggml_rope_ext(
			ctx.(*Context).ctx,
			dequant,
			positions.(*Tensor).t,
			opts.Factors.(*Tensor).t,
			C.int(ropeDim),
			C.int(opts.Type),
			cmp.Or(C.int(opts.YaRN.OriginalContextLength), 128<<10),
			C.float(ropeBase),
			C.float(ropeScale),
			C.float(opts.YaRN.ExtrapolationFactor),
			cmp.Or(C.float(opts.YaRN.AttentionFactor), 1),
			cmp.Or(C.float(opts.YaRN.BetaFast), 32),
			cmp.Or(C.float(opts.YaRN.BetaSlow), 1),
		)
	}
	return &Tensor{b: t.b, t: tt}
}

func (t *Tensor) IM2Col(ctx ml.Context, t2 ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_im2col(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, C.int(s0), C.int(s1), C.int(p0), C.int(p1), C.int(d0), C.int(d1), true, C.GGML_TYPE_F32),
	}
}

func (t *Tensor) GELU(ctx ml.Context, t2 ...ml.Tensor) ml.Tensor {
	if len(t2) > 0 {
		return &Tensor{
			b: t.b,
			t: C.ggml_geglu_split(ctx.(*Context).ctx, t.t, t2[0].(*Tensor).t),
		}
	}
	return &Tensor{
		b: t.b,
		t: C.ggml_gelu_inplace(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) QuickGELU(ctx ml.Context, t2 ...ml.Tensor) ml.Tensor {
	var tt *C.struct_ggml_tensor
	if len(t2) > 0 {
		tt = C.ggml_geglu_quick_split(ctx.(*Context).ctx, t.t, t2[0].(*Tensor).t)
	} else {
		tt = C.ggml_gelu_quick_inplace(ctx.(*Context).ctx, t.t)
	}
	return &Tensor{b: t.b, t: tt}
}

func (t *Tensor) SILU(ctx ml.Context, t2 ...ml.Tensor) ml.Tensor {
	if len(t2) > 0 {
		return &Tensor{
			b: t.b,
			t: C.ggml_swiglu_split(ctx.(*Context).ctx, t.t, t2[0].(*Tensor).t),
		}
	}
	return &Tensor{
		b: t.b,
		t: C.ggml_silu_inplace(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) RELU(ctx ml.Context, t2 ...ml.Tensor) ml.Tensor {
	if len(t2) > 0 {
		return &Tensor{
			b: t.b,
			t: C.ggml_reglu_split(ctx.(*Context).ctx, t.t, t2[0].(*Tensor).t),
		}
	}
	return &Tensor{
		b: t.b,
		t: C.ggml_relu_inplace(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) SILUAlphaLimit(ctx ml.Context, up ml.Tensor, alpha, limit float32) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_swiglu_oai(ctx.(*Context).ctx, t.t, up.(*Tensor).t, C.float(alpha), C.float(limit)),
	}
}

func (t *Tensor) Conv2D(ctx ml.Context, t2 ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_conv_2d(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, C.int(s0), C.int(s1), C.int(p0), C.int(p1), C.int(d0), C.int(d1)),
	}
}

func (t *Tensor) Conv3D(ctx ml.Context, t2 ml.Tensor, c, s0, s1, s2, p0, p1, p2, d0, d1, d2 int) ml.Tensor {
	var tt ml.Tensor = &Tensor{
		b: t.b,
		t: C.ggml_conv_3d(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, C.int64_t(c), C.int(s0), C.int(s1), C.int(s2), C.int(p0), C.int(p1), C.int(p2), C.int(d0), C.int(d1), C.int(d2)),
	}

	tt = tt.Reshape(ctx, t.Dim(3)/c, t2.Dim(3)/c)
	return tt
}

func (t *Tensor) AvgPool2D(ctx ml.Context, k, s int, p float32) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_pool_2d(ctx.(*Context).ctx, t.t, C.GGML_OP_POOL_AVG, C.int(k), C.int(k), C.int(s), C.int(s), C.float(p), C.float(p)),
	}
}

func (t *Tensor) ScaledDotProductAttention(ctx ml.Context, key, value, mask, sinks ml.Tensor, vmla ml.Tensor, scale float64, cacheConfigApplied bool) ml.Tensor {
	// If the cache didn't help us with required transformations, do them here
	if !cacheConfigApplied {
		cacheConfig := t.b.CacheConfig()

		// Padding key and value to CachePadding is a performance optimization, not a requirement, so we don't do it if it wasn't done by the caller

		if cacheConfig.PermutedV {
			value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)
		}

		if mask != nil {
			if mask.DType() != cacheConfig.MaskDType {
				mask = mask.Cast(ctx, cacheConfig.MaskDType)
			}
		}
	}

	var kqMask *C.struct_ggml_tensor
	if mask != nil {
		kqMask = mask.(*Tensor).t
	}

	query := t.Permute(ctx, 0, 2, 1, 3)
	key = key.Permute(ctx, 0, 2, 1, 3)

	if t.b.flashAttention == ml.FlashAttentionEnabled {
		value = value.Permute(ctx, 0, 2, 1, 3)

		kqv := C.ggml_flash_attn_ext(ctx.(*Context).ctx, query.(*Tensor).t, key.(*Tensor).t, value.(*Tensor).t, kqMask, C.float(scale), 0, 0)
		if sinks != nil {
			C.ggml_flash_attn_ext_add_sinks(kqv, sinks.(*Tensor).t)
		}
		C.ggml_flash_attn_ext_set_prec(kqv, C.GGML_PREC_F32)

		if vmla != nil {
			var cur ml.Tensor = &Tensor{b: t.b, t: kqv}
			cur = cur.Permute(ctx, 0, 2, 1, 3)
			cur = vmla.Mulmat(ctx, cur)
			cur = cur.Permute(ctx, 0, 2, 1, 3)
			cur = cur.Contiguous(ctx)
			kqv = cur.(*Tensor).t
		}

		return &Tensor{b: t.b, t: kqv}
	} else {
		kq := key.MulmatFullPrec(ctx, query)
		kq = &Tensor{
			b: t.b,
			t: C.ggml_soft_max_ext(ctx.(*Context).ctx, kq.(*Tensor).t, kqMask, C.float(scale), 0),
		}
		if sinks != nil {
			C.ggml_soft_max_add_sinks(kq.(*Tensor).t, sinks.(*Tensor).t)
		}

		kqv := value.Mulmat(ctx, kq)
		if vmla != nil {
			kqv = vmla.Mulmat(ctx, kqv)
		}

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
		t: C.ggml_argsort_top_k(ctx.(*Context).ctx, t.t, C.int(k)),
	}
}

func (t *Tensor) Argsort(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_argsort(ctx.(*Context).ctx, t.t, C.GGML_SORT_ORDER_ASC),
	}
}

func (t *Tensor) Mean(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_mean(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) Variance(ctx ml.Context) ml.Tensor {
	return t.Add(ctx, t.Mean(ctx).Scale(ctx, -1)).
		Sqr(ctx).
		SumRows(ctx).
		Scale(ctx, 1/float64(t.Dim(0)))
}

func (t *Tensor) Stddev(ctx ml.Context) ml.Tensor {
	return t.Variance(ctx).Sqrt(ctx)
}

func (t *Tensor) Sqr(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_sqr(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) Sqrt(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_sqrt(ctx.(*Context).ctx, t.t),
	}
}

func (t *Tensor) Interpolate(ctx ml.Context, dims [4]int, samplingMode ml.SamplingMode) ml.Tensor {
	var mode C.uint32_t
	switch samplingMode {
	case ml.SamplingModeNearest:
		mode = C.GGML_SCALE_MODE_NEAREST
	case ml.SamplingModeBilinear:
		mode = C.GGML_SCALE_MODE_BILINEAR
	default:
		panic("unsupported interpolate mode")
	}

	return &Tensor{
		b: t.b,
		t: C.ggml_interpolate(ctx.(*Context).ctx, t.t, C.int64_t(dims[0]), C.int64_t(dims[1]), C.int64_t(dims[2]), C.int64_t(dims[3]), mode),
	}
}

// Slice returns a view of the tensor sliced along dim from low to high in step steps.
// Slice panics if the dimension is invalid or the slice parameters are out of range.
// If dim=0 and step>1, the tensor is a copy rather than a view to ensure proper shape.
func (t *Tensor) Slice(ctx ml.Context, dim int, low, high, step int) ml.Tensor {
	if dim < 0 || dim >= C.GGML_MAX_DIMS {
		panic("invalid dimension")
	} else if low < 0 || high > t.Dim(dim) || low >= high || step < 1 {
		panic("invalid slice parameters")
	}

	if dim == 0 && step > 1 {
		// dim=0,step>1 is a special case so handle it here first
		return t.View(ctx,
			low*t.Stride(0), 1,
			step*t.Stride(0), (high-low+1)/step,
			t.Stride(1), t.Dim(1),
			// preserve dim 3 by merging it into dim 2
			t.Stride(2), t.Dim(2)*t.Dim(3),
		).Contiguous(ctx, (high-low+1)/step, t.Dim(1), t.Dim(2), t.Dim(3))
	}

	args := []int{
		low * t.Stride(dim), t.Dim(0),
		t.Stride(1), t.Dim(1),
		t.Stride(2), t.Dim(2),
		t.Stride(3), t.Dim(3),
	}

	if step == 1 {
		args[dim*2+1] = high - low
		return t.View(ctx, args[0], args[1:]...)
	} else {
		args[dim*2] = step * t.Stride(dim)
		args[dim*2+1] = (high - low + 1) / step
		return t.View(ctx, args[0], args[1:]...)
	}
}

// Chunk the tensor into chunk sized tensors along dim. Each sub-tensor is a view of
// the original.
func (t *Tensor) Chunk(ctx ml.Context, dim, chunk int) []ml.Tensor {
	sections := make([]int, 0, t.Dim(dim)/chunk+1)
	for rest := t.Dim(dim); rest > 0; rest -= chunk {
		sections = append(sections, min(chunk, rest))
	}
	return t.ChunkSections(ctx, dim, sections...)
}

// ChunkSections split the tensor into section sized tensors along dim. Each sub-tensor is a
// view of the original. The size of the dim must equal the sum of sections.
func (t *Tensor) ChunkSections(ctx ml.Context, dim int, sections ...int) []ml.Tensor {
	var offset int
	s := make([]ml.Tensor, len(sections))
	for i, section := range sections {
		s[i] = t.Slice(ctx, dim, offset, offset+section, 1)
		offset += section
	}
	if offset != t.Dim(dim) {
		panic("sections do not sum to tensor dimension")
	}
	return s
}

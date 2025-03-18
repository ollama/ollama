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
	"context"
	"errors"
	"io"
	"log/slog"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	ggml "github.com/ollama/ollama/ml/backend/ggml/ggml/src"
	"golang.org/x/sync/errgroup"
)

type backend struct {
	gpus, cpus []*C.struct_ggml_backend_device
	bufts      map[*C.struct_ggml_backend_device][]*C.struct_ggml_backend_buffer_type
	ctxs       map[*C.struct_ggml_backend_buffer_type]*C.struct_ggml_context
	bbs        map[*C.struct_ggml_backend_buffer_type]*C.struct_ggml_backend_buffer
	readers    map[*C.struct_ggml_tensor]io.Reader
	reserved   map[*C.struct_ggml_context]uint64

	onceScheduler sync.Once
	scheduler     *scheduler
}

var _ ml.Backend2 = (*backend)(nil)

func New2() (ml.Backend2, error) {
	ggml.OnceLoad()

	var cpus, accels, gpus []*C.struct_ggml_backend_device
	for i := range C.ggml_backend_dev_count() {
		d := C.ggml_backend_dev_get(C.size_t(i))
		switch C.ggml_backend_dev_type(d) {
		case C.GGML_BACKEND_DEVICE_TYPE_CPU:
			// only the first cpu device should be used
			if len(cpus) > 0 {
				continue
			}

			cpus = append(cpus, d)
		case C.GGML_BACKEND_DEVICE_TYPE_ACCEL:
			accels = append(accels, d)
		case C.GGML_BACKEND_DEVICE_TYPE_GPU:
			gpus = append(gpus, d)
		}
	}

	bufts := make(map[*C.struct_ggml_backend_device][]*C.struct_ggml_backend_buffer_type)

	cpu := C.ggml_backend_dev_by_type(C.GGML_BACKEND_DEVICE_TYPE_CPU)
	for _, d := range append(accels, cpus...) {
		bufts[cpu] = append(bufts[cpu], C.ggml_backend_dev_buffer_type(d))
	}

	for _, d := range gpus {
		bufts[d] = append(bufts[d], append([]*C.struct_ggml_backend_buffer_type{C.ggml_backend_dev_buffer_type(d)}, bufts[cpu]...)...)
	}

	return &backend{
		// merge accels and cpus
		gpus:     gpus,
		cpus:     append(accels, cpus...),
		bufts:    bufts,
		ctxs:     make(map[*C.struct_ggml_backend_buffer_type]*C.struct_ggml_context, len(bufts)),
		bbs:      make(map[*C.struct_ggml_backend_buffer_type]*C.struct_ggml_backend_buffer, len(bufts)),
		readers:  make(map[*C.struct_ggml_tensor]io.Reader),
		reserved: make(map[*C.struct_ggml_context]uint64),
	}, nil
}

func (b *backend) Close() {
}

func (b *backend) NewContext() ml.Context {
	return &Context{
		b: &Backend{
			input:  b.bufts[b.cpus[0]][0],
			output: b.bufts[b.cpus[0]][0],
			layers: func() map[int]*C.struct_ggml_backend_buffer_type {
				m := make(map[int]*C.struct_ggml_backend_buffer_type)
				for i := range 100 {
					m[i] = b.bufts[b.gpus[0]][0]
				}
				return m
			}(),
			sched: func() *C.struct_ggml_backend_sched {
				return b.Scheduler().(*scheduler).s
			}(),
			maxGraphNodes: 8192,
		},
		ctx: C.ggml_init(C.struct_ggml_init_params{
			mem_size: C.ggml_tensor_overhead() * C.size_t(4000),
			no_alloc: true,
		}),
		buft:          b.bufts[b.cpus[0]][0],
		maxGraphNodes: 8192,
	}
}

func (b *backend) Get(tensorReader fs.TensorReader, preferredDevice ml.Device) ml.Tensor {
	var ctx *C.struct_ggml_context

	var devices []*C.struct_ggml_backend_device
	if preferredDevice == ml.GPU {
		devices = b.gpus
	}

	for _, d := range append(devices, b.cpus...) {
		var free, total C.size_t
		C.ggml_backend_dev_memory(d, &free, &total)

		for _, buft := range b.bufts[d] {
			if _, ok := b.ctxs[buft]; !ok {
				b.ctxs[buft] = C.ggml_init(C.struct_ggml_init_params{
					mem_size: C.ggml_tensor_overhead() * C.size_t(1000),
					no_alloc: true,
				})
			}

			ctx = b.ctxs[buft]
			if free > 0 && b.reserved[ctx]+uint64(tensorReader.Size()) >= uint64(free) {
				slog.Info("no space available", "device", C.GoString(C.ggml_backend_dev_name(d)), "free", format.HumanBytes2(uint64(free)), "total", format.HumanBytes2(uint64(total)), "reserve", format.HumanBytes2(b.reserved[ctx]), "size", format.HumanBytes2(uint64(tensorReader.Size())))
				continue
			}

			cname := C.CString(tensorReader.Name())
			defer C.free(unsafe.Pointer(cname))

			if t := C.ggml_get_tensor(ctx, cname); t != nil {
				slog.Info("using existing tensor in buffer type", "name", tensorReader.Name(), "buffer_type", C.GoString(C.ggml_backend_buft_name(buft)))
				return &Tensor{t: t}
			}

			shape := make([]C.int64_t, len(tensorReader.Shape()))
			for i, s := range tensorReader.Shape() {
				shape[i] = C.int64_t(s)
			}

			t := C.ggml_new_tensor(ctx, uint32(tensorReader.DType()), C.int(len(tensorReader.Shape())), unsafe.SliceData(shape))
			C.ggml_set_name(t, cname)

			b.readers[t] = tensorReader
			b.reserved[ctx] += uint64(tensorReader.Size())

			slog.Info("creating new tensor in buffer type", "name", tensorReader.Name(), "buffer_type", C.GoString(C.ggml_backend_buft_name(buft)), "reserve", format.HumanBytes2(b.reserved[ctx]))
			return &Tensor{t: t}
		}
	}

	panic("no device available")
}

func (b *backend) LoadAll(ctx context.Context) error {
	// allocate buffers for each context
	for buft, ctx := range b.ctxs {
		if C.ggml_get_first_tensor(ctx) == nil {
			continue
		}

		bb := C.ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)
		C.ggml_backend_buffer_set_usage(bb, C.GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
		b.bbs[buft] = bb
	}

	for _, bb := range b.bbs {
		slog.Info("", "buffer.size", C.ggml_backend_buffer_get_size(bb), "buffer.usage", C.ggml_backend_buffer_get_usage(bb))
	}

	pool := sync.Pool{
		New: func() any {
			return new(bytes.Buffer)
		},
	}

	g, ctx := errgroup.WithContext(context.Background())
	g.SetLimit(runtime.GOMAXPROCS(0))
	for t, r := range b.readers {
		g.Go(func() error {
			var s uint64

			for {
				b := pool.Get().(*bytes.Buffer)
				b.Reset()

				n, err := io.CopyN(b, r, 32*format.KibiByte)
				if n > 0 {
				} else if errors.Is(err, io.EOF) {
					break
				} else if err != nil {
					return err
				}

				C.ggml_backend_tensor_set(t, unsafe.Pointer(&b.Bytes()[0]), C.size_t(s), C.size_t(n))
				pool.Put(b)
			}

			return nil
		})
	}

	go func() {
		<-ctx.Done()
		g.Go(func() error {
			return ctx.Err()
		})
	}()

	return g.Wait()
}

type scheduler struct {
	s *C.struct_ggml_backend_sched
}

var (
	_ ml.Scheduler = (*scheduler)(nil)
	_ ml.Reserver  = (*scheduler)(nil)
)

func (b *backend) Scheduler() ml.Scheduler {
	b.onceScheduler.Do(func() {
		devices := append(b.gpus, b.cpus...)
		backends := make([]C.ggml_backend_t, len(devices))
		bufts := make([]C.ggml_backend_buffer_type_t, len(devices))
		for i, device := range devices {
			backend := C.ggml_backend_dev_init(device, nil)
			buft := C.ggml_backend_get_default_buffer_type(backend)
			if d := C.ggml_backend_get_device(backend); C.ggml_backend_dev_type(d) == C.GGML_BACKEND_DEVICE_TYPE_CPU && len(b.gpus) > 0 {
				if hbt := C.ggml_backend_dev_host_buffer_type(b.gpus[0]); hbt != nil {
					buft = hbt
				}
			}

			slog.Info("scheduler", "backend", C.GoString(C.ggml_backend_name(backend)), "buffer_type", C.GoString(C.ggml_backend_buft_name(buft)))
			backends[i] = backend
			bufts[i] = buft
		}

		maxGraphNodes := max(8192, 1)
		b.scheduler = &scheduler{
			s: C.ggml_backend_sched_new(
				unsafe.SliceData(backends),
				unsafe.SliceData(bufts),
				C.int(len(backends)),
				C.size_t(maxGraphNodes),
				C._Bool(len(b.gpus) > 1),
			),
		}
	})

	return b.scheduler
}

func (s scheduler) Schedule() {
}

func (s scheduler) Reserve() {
}

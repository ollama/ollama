package mlx

// #include "generated.h"
import "C"

import (
	"log/slog"
	"sync"
)

type Device struct {
	ctx C.mlx_device
}

func (d Device) LogValue() slog.Value {
	str := C.mlx_string_new()
	defer C.mlx_string_free(str)
	C.mlx_device_tostring(&str, d.ctx)
	return slog.StringValue(C.GoString(C.mlx_string_data(str)))
}

var (
	// defaultDeviceMu guards defaultDevice/defaultDeviceSet, which are
	// touched by SetDefaultDeviceGPU from every thread that initializes MLX.
	defaultDeviceMu  sync.Mutex
	defaultDevice    Device
	defaultDeviceSet bool

	// defaultStreams caches the per-thread default streams. MLX default
	// streams are thread-local (since MLX 0.31.2), so a single process-wide
	// cache would hand a stream created on one thread to ops evaluated on
	// another — which throws "There is no Stream(gpu, N) in current thread".
	defaultStreams sync.Map // map[uint64]Stream, keyed by thread ID
)

func resetDefaultStreamCache() {
	defaultDeviceMu.Lock()
	defaultDeviceSet = false
	defaultDeviceMu.Unlock()
	defaultStreams.Range(func(k, _ any) bool {
		defaultStreams.Delete(k)
		return true
	})
}

func DefaultDevice() Device {
	defaultDeviceMu.Lock()
	defer defaultDeviceMu.Unlock()
	if !defaultDeviceSet {
		d := C.mlx_device_new()
		C.mlx_get_default_device(&d)
		defaultDevice = Device{d}
		defaultDeviceSet = true
	}

	return defaultDevice
}

// GPUIsAvailable returns true if a GPU device is available.
func GPUIsAvailable() bool {
	dev := C.mlx_device_new_type(C.MLX_GPU, 0)
	defer C.mlx_device_free(dev)
	var avail C.bool
	C.mlx_device_is_available(&avail, dev)
	return bool(avail)
}

// SetDefaultDeviceGPU sets the default MLX device to GPU.
func SetDefaultDeviceGPU() {
	dev := C.mlx_device_new_type(C.MLX_GPU, 0)
	C.mlx_set_default_device(dev)
	C.mlx_device_free(dev)
	resetDefaultStreamCache()
}

type Stream struct {
	ctx C.mlx_stream
}

func (s Stream) LogValue() slog.Value {
	str := C.mlx_string_new()
	defer C.mlx_string_free(str)
	C.mlx_stream_tostring(&str, s.ctx)
	return slog.StringValue(C.GoString(C.mlx_string_data(str)))
}

// DefaultStream returns the calling thread's default stream. The result is
// cached per thread: resolving a stream on the wrong thread records it into
// arrays that can then only be evaluated on the stream's owning thread.
func DefaultStream() Stream {
	tid := currentThreadID()
	if s, ok := defaultStreams.Load(tid); ok {
		return s.(Stream)
	}

	s := C.mlx_stream_new()
	C.mlx_get_default_stream(&s, DefaultDevice().ctx)
	stream := Stream{s}
	defaultStreams.Store(tid, stream)
	return stream
}

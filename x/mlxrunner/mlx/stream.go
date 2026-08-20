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
	defaultDeviceMu  sync.Mutex
	defaultDevice    Device
	defaultDeviceSet bool
	defaultStreams   sync.Map // map[uint64]Stream, keyed by thread ID
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

func DefaultStream() Stream {
	tid := currentThreadID()
	if s, ok := defaultStreams.Load(tid); ok {
		return s.(Stream)
	}

	dev := DefaultDevice()
	s := C.mlx_stream_new()
	C.mlx_get_default_stream(&s, dev.ctx)
	stream := Stream{s}
	defaultStreams.Store(tid, stream)
	return stream
}

package mlx

// #include "generated.h"
//
// // MLX default streams are thread-local, so cache the C handle per thread.
// static __thread mlx_stream ollama_default_stream;
// static __thread int ollama_default_stream_set;
//
// static mlx_stream ollama_get_default_stream(void) {
//     if (!ollama_default_stream_set) {
//         mlx_device device = mlx_device_new();
//         mlx_get_default_device(&device);
//         ollama_default_stream = mlx_stream_new();
//         mlx_get_default_stream(&ollama_default_stream, device);
//         mlx_device_free(device);
//         ollama_default_stream_set = 1;
//     }
//     return ollama_default_stream;
// }
//
// static void ollama_reset_default_stream(void) {
//     if (ollama_default_stream_set) {
//         mlx_stream_free(ollama_default_stream);
//         ollama_default_stream_set = 0;
//     }
// }
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
)

func resetDefaultStreamCache() {
	C.ollama_reset_default_stream()

	defaultDeviceMu.Lock()
	defaultDeviceSet = false
	defaultDeviceMu.Unlock()
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
	defaultDeviceMu.Lock()
	defer defaultDeviceMu.Unlock()

	dev := C.mlx_device_new_type(C.MLX_GPU, 0)
	C.mlx_set_default_device(dev)
	C.mlx_device_free(dev)
	C.ollama_reset_default_stream()
	defaultDeviceSet = false
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

// DefaultStream returns the calling thread's default stream.
func DefaultStream() Stream {
	return Stream{C.ollama_get_default_stream()}
}

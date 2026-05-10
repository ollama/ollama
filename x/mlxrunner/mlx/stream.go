package mlx

// #include "generated.h"
//
// static __thread mlx_stream _mlx_default_stream;
// static __thread int _mlx_default_stream_set = 0;
// static __thread mlx_stream _mlx_default_cpu_stream;
// static __thread int _mlx_default_cpu_stream_set = 0;
//
// static void mlx_reset_thread_stream_cache(void) {
//     if (_mlx_default_stream_set) {
//         mlx_stream_free(_mlx_default_stream);
//         _mlx_default_stream_set = 0;
//     }
//     if (_mlx_default_cpu_stream_set) {
//         mlx_stream_free(_mlx_default_cpu_stream);
//         _mlx_default_cpu_stream_set = 0;
//     }
// }
//
// static void mlx_get_thread_default_stream(mlx_stream* out, mlx_device dev) {
//     if (!_mlx_default_stream_set) {
//         _mlx_default_stream = mlx_stream_new();
//         mlx_get_default_stream(&_mlx_default_stream, dev);
//         _mlx_default_stream_set = 1;
//     }
//     *out = _mlx_default_stream;
// }
//
// static void mlx_get_thread_default_cpu_stream(mlx_stream* out) {
//     if (!_mlx_default_cpu_stream_set) {
//         _mlx_default_cpu_stream = mlx_default_cpu_stream_new();
//         _mlx_default_cpu_stream_set = 1;
//     }
//     *out = _mlx_default_cpu_stream;
// }
import "C"

import "log/slog"

type Device struct {
	ctx C.mlx_device
}

func (d Device) LogValue() slog.Value {
	str := C.mlx_string_new()
	defer C.mlx_string_free(str)
	C.mlx_device_tostring(&str, d.ctx)
	return slog.StringValue(C.GoString(C.mlx_string_data(str)))
}

func resetDefaultStreamCache() {
	C.mlx_reset_thread_stream_cache()
}

func DefaultDevice() Device {
	d := C.mlx_device_new()
	C.mlx_get_default_device(&d)
	return Device{d}
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
	requireBoundThread("DefaultStream")
	s := C.mlx_stream_new()
	C.mlx_get_thread_default_stream(&s, DefaultDevice().ctx)
	return Stream{s}
}

// DefaultCPUStream returns a cached CPU stream for load operations.
func DefaultCPUStream() Stream {
	requireBoundThread("DefaultCPUStream")
	s := C.mlx_stream_new()
	C.mlx_get_thread_default_cpu_stream(&s)
	return Stream{s}
}

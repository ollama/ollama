package mlx

// #include "generated.h"
import "C"

import (
	"log/slog"
	"runtime"
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
	defaultDevice    Device
	defaultDeviceSet bool
	defaultDeviceGPU bool
	defaultStream    Stream
	defaultStreamSet bool
)

func resetDefaultStreamCache() {
	defaultDeviceSet = false
	defaultDeviceGPU = false
	defaultStreamSet = false
}

func DefaultDevice() Device {
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
	if C.mlx_device_is_available(&avail, dev) != 0 || !bool(avail) {
		return false
	}
	if runtime.GOOS == "darwin" && (!MetalIsAvailable() || !canUseGPUDevice(dev)) {
		return false
	}
	return true
}

func canUseGPUDevice(dev C.mlx_device) bool {
	previous := C.mlx_device_new()
	previousDeviceGPU := defaultDeviceGPU
	havePrevious := C.mlx_get_default_device(&previous) == 0
	defer C.mlx_device_free(previous)
	defer func() {
		if havePrevious {
			_ = C.mlx_set_default_device(previous)
		} else {
			cpu := C.mlx_device_new_type(C.MLX_CPU, 0)
			_ = C.mlx_set_default_device(cpu)
			C.mlx_device_free(cpu)
		}
		resetDefaultStreamCache()
		defaultDeviceGPU = previousDeviceGPU
		DefaultStream()
	}()

	if err := mlxErr("set default GPU device failed", func() C.int {
		return C.mlx_set_default_device(dev)
	}); err != nil {
		return false
	}
	resetDefaultStreamCache()
	defaultDeviceGPU = true
	DefaultStream()

	arr := C.mlx_array_new()
	defer C.mlx_array_free(arr)
	if err := mlxErr("GPU device probe failed", func() C.int {
		return C.mlx_array_set_bool(&arr, C.bool(true))
	}); err != nil {
		return false
	}

	vector := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vector)
	C.mlx_vector_array_append_value(vector, arr)
	return mlxErr("GPU device eval probe failed", func() C.int {
		return C.mlx_eval(vector)
	}) == nil
}

// SetDefaultDeviceGPU sets the default MLX device to GPU.
func SetDefaultDeviceGPU() {
	setDefaultDevice(C.MLX_GPU, true)
}

func setDefaultDevice(deviceType C.mlx_device_type, gpu bool) {
	dev := C.mlx_device_new_type(deviceType, 0)
	defer C.mlx_device_free(dev)
	mlxCheck("set default MLX device failed", func() C.int {
		return C.mlx_set_default_device(dev)
	})
	resetDefaultStreamCache()
	defaultDeviceGPU = gpu
	DefaultStream()
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
	if !defaultStreamSet {
		s := defaultStreamForDevice(defaultDeviceGPU)
		if s.ctx == nil {
			s = C.mlx_stream_new_device(DefaultDevice().ctx)
		}
		C.mlx_set_default_stream(s)
		defaultStream = Stream{s}
		defaultStreamSet = true
	}

	return defaultStream
}

func defaultStreamForDevice(gpu bool) C.mlx_stream {
	if gpu {
		return C.mlx_default_gpu_stream_new()
	}
	return C.mlx_default_cpu_stream_new()
}

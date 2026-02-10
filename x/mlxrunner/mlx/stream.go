//go:build mlx

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

var DefaultDevice = sync.OnceValue(func() Device {
	d := C.mlx_device_new()
	C.mlx_get_default_device(&d)
	return Device{d}
})

type Stream struct {
	ctx C.mlx_stream
}

func (s Stream) LogValue() slog.Value {
	str := C.mlx_string_new()
	defer C.mlx_string_free(str)
	C.mlx_stream_tostring(&str, s.ctx)
	return slog.StringValue(C.GoString(C.mlx_string_data(str)))
}

var DefaultStream = sync.OnceValue(func() Stream {
	s := C.mlx_stream_new()
	C.mlx_get_default_stream(&s, DefaultDevice().ctx)
	return Stream{s}
})

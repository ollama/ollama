package mlx

// #include "generated.h"
// #include <stdlib.h>
import "C"

import (
	"fmt"
	"log/slog"
	"strconv"
	"unsafe"
)

func (b Byte) String() string {
	return strconv.FormatInt(int64(b), 10) + " B"
}

func (b KibiByte) String() string {
	return strconv.FormatFloat(float64(b)/(1<<10), 'f', 2, 64) + " KiB"
}

func (b MebiByte) String() string {
	return strconv.FormatFloat(float64(b)/(1<<(2*10)), 'f', 2, 64) + " MiB"
}

func (b GibiByte) String() string {
	return strconv.FormatFloat(float64(b)/(1<<(3*10)), 'f', 2, 64) + " GiB"
}

func (b TebiByte) String() string {
	return strconv.FormatFloat(float64(b)/(1<<(4*10)), 'f', 2, 64) + " TiB"
}

func PrettyBytes(n int) fmt.Stringer {
	switch {
	case n < 1<<10:
		return Byte(n)
	case n < 1<<(2*10):
		return KibiByte(n)
	case n < 1<<(3*10):
		return MebiByte(n)
	case n < 1<<(4*10):
		return GibiByte(n)
	default:
		return TebiByte(n)
	}
}

func ActiveMemory() int {
	var active C.size_t
	mlxCheck(C.mlx_get_active_memory(&active))
	return int(active)
}

func CacheMemory() int {
	var cache C.size_t
	mlxCheck(C.mlx_get_cache_memory(&cache))
	return int(cache)
}

func PeakMemory() int {
	var peak C.size_t
	mlxCheck(C.mlx_get_peak_memory(&peak))
	return int(peak)
}

func ResetPeakMemory() {
	mlxCheck(C.mlx_reset_peak_memory())
}

// MaxRecommendedWorkingSetSize returns the device's recommended upper bound
// for resident Metal allocations.
func MaxRecommendedWorkingSetSize() (int, error) {
	size, ok, err := deviceInfoSize("max_recommended_working_set_size")
	if err != nil {
		return 0, err
	}
	if !ok {
		// mlx-c reports a missing key with a non-zero return and no message.
		return 0, fmt.Errorf("mlx: no max_recommended_working_set_size in device info")
	}
	return size, nil
}

// DeviceMemory returns the selected GPU device's total and currently
// driver-free memory from device info. It reports ok=false on backends that
// do not publish those keys.
func DeviceMemory() (total, free int, ok bool) {
	total, totalOK, err := deviceInfoSize("total_memory")
	if err != nil || !totalOK {
		return 0, 0, false
	}
	free, freeOK, err := deviceInfoSize("free_memory")
	if err != nil {
		return 0, 0, false
	}
	return total, free, freeOK
}

// deviceInfoSize reads a size_t key from the default device's info snapshot,
// reporting ok=false when the backend does not publish the key.
func deviceInfoSize(key string) (int, bool, error) {
	info := mlxCheck(C.mlx_device_info_new())
	if err := mlxError(C.mlx_device_info_get(&info, DefaultDevice().ctx)); err != nil {
		return 0, false, err
	}
	defer freeDeviceInfo(info)

	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	var size C.size_t
	rc := C.mlx_device_info_get_size(&size, info, cKey)
	if err := lastError(); err != nil {
		return 0, false, err
	}
	return int(size), rc == 0, nil
}

// SetMemoryLimit sets the GPU allocator's maximum memory use and returns the
// previous limit. Contrary to what the name suggests, it does not constrain
// Metal memory use; mlx_set_wired_limit is the Metal governor.
func SetMemoryLimit(limit int) (int, error) {
	if limit < 0 {
		return 0, fmt.Errorf("mlx: memory limit must be non-negative")
	}

	var previous C.size_t
	if err := mlxError(C.mlx_set_memory_limit(&previous, C.size_t(limit))); err != nil {
		return 0, err
	}
	return int(previous), nil
}

// SetWiredLimit sets the maximum amount of Metal memory MLX keeps resident and
// returns the previous limit.
func SetWiredLimit(limit int) (int, error) {
	if limit < 0 {
		return 0, fmt.Errorf("mlx: wired limit must be non-negative")
	}

	var previous C.size_t
	if err := mlxError(C.mlx_set_wired_limit(&previous, C.size_t(limit))); err != nil {
		return 0, err
	}
	return int(previous), nil
}

type Memory struct{}

func (Memory) LogValue() slog.Value {
	return slog.GroupValue(
		slog.Any("active", PrettyBytes(ActiveMemory())),
		slog.Any("cache", PrettyBytes(CacheMemory())),
		slog.Any("peak", PrettyBytes(PeakMemory())),
	)
}

type (
	Byte     int
	KibiByte int
	MebiByte int
	GibiByte int
	TebiByte int
)

func ClearCache() {
	mlxCheck(C.mlx_clear_cache())
}

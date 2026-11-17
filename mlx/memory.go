package mlx

// #include "generated.h"
// #include <stdlib.h>
import "C"

import (
	"fmt"
	"log/slog"
	"math"
	"runtime"
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

// priorPeak preserves the reporting interval while measuring smaller operations.
// Like other MLX state, it is accessed only on the MLX thread.
var priorPeak int

func PeakMemory() int {
	return max(priorPeak, nativePeakMemory())
}

func nativePeakMemory() int {
	var peak C.size_t
	mlxCheck(C.mlx_get_peak_memory(&peak))
	return int(peak)
}

func ResetPeakMemory() {
	priorPeak = 0
	mlxCheck(C.mlx_reset_peak_memory())
}

// MeasurePeakMemory returns the allocation peak during f while preserving the
// peak since ResetPeakMemory for request logs and other cumulative reporting.
func MeasurePeakMemory(f func()) int {
	priorPeak = PeakMemory()
	mlxCheck(C.mlx_reset_peak_memory())
	f()
	return nativePeakMemory()
}

// MaxRecommendedWorkingSetSize returns the device's recommended upper bound
// for resident Metal allocations.
func MaxRecommendedWorkingSetSize() (int, error) {
	info := mlxCheck(C.mlx_device_info_new())
	defer func() { freeDeviceInfo(info) }() // get populates the initially empty handle.
	if err := mlxError(C.mlx_device_info_get(&info, DefaultDevice().ctx)); err != nil {
		return 0, err
	}
	return deviceInfoSize(info, "max_recommended_working_set_size")
}

func deviceInfoSize(info C.mlx_device_info, name string) (int, error) {
	key := C.CString(name)
	defer C.free(unsafe.Pointer(key))

	var size C.size_t
	if err := mlxError(C.mlx_device_info_get_size(&size, info, key)); err != nil {
		return 0, fmt.Errorf("MLX device info %q: %w", name, err)
	}
	if uint64(size) > math.MaxInt {
		return 0, fmt.Errorf("MLX device info %q exceeds signed byte range", name)
	}
	return int(size), nil
}

// MemoryLimit returns MLX's current allocation budget, a guideline for graph
// evaluation rather than a hard limit. An unlimited budget saturates at MaxInt.
func MemoryLimit() (int, error) {
	var limit C.size_t
	if err := mlxError(C.mlx_get_memory_limit(&limit)); err != nil {
		return 0, err
	}
	return int(min(uint64(limit), uint64(math.MaxInt))), nil
}

// IntegratedGPU reports whether the default GPU shares system memory. Older
// native libraries without this property return an error, not a guess.
func IntegratedGPU() (bool, error) {
	info := mlxCheck(C.mlx_device_info_new())
	defer func() { freeDeviceInfo(info) }() // get populates the initially empty handle.
	if err := mlxError(C.mlx_device_info_get(&info, DefaultDevice().ctx)); err != nil {
		return false, err
	}
	integrated, err := deviceInfoSize(info, "integrated")
	return integrated != 0, err
}

// MemoryBudget is the current process's WDDM local-segment budget and usage.
// On integrated GPUs this segment can include shared system memory.
type MemoryBudget struct {
	Budget, Usage int64
}

// DeviceMemoryBudget returns the current process's WDDM budget on Windows CUDA.
// Other backends return nil. Missing or failed Windows CUDA telemetry returns
// an error; a successfully queried zero budget is valid.
func DeviceMemoryBudget() (*MemoryBudget, error) {
	if runtime.GOOS != "windows" || !CUDAIsAvailable() {
		return nil, nil
	}
	info := mlxCheck(C.mlx_device_info_new())
	defer func() { freeDeviceInfo(info) }() // get populates the initially empty handle.
	if err := mlxError(C.mlx_device_info_get(&info, DefaultDevice().ctx)); err != nil {
		return nil, err
	}
	budget, err := deviceInfoSize(info, "memory_budget")
	if err != nil {
		return nil, err
	}
	usage, err := deviceInfoSize(info, "memory_usage")
	if err != nil {
		return nil, err
	}
	return &MemoryBudget{Budget: int64(budget), Usage: int64(usage)}, nil
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

// SetCacheLimit limits the freed buffers retained by MLX's allocator.
func SetCacheLimit(limit int) int {
	if limit < 0 {
		panic("mlx: negative cache limit")
	}
	var previous C.size_t
	mlxCheck(C.mlx_set_cache_limit(&previous, C.size_t(limit)))
	return int(previous)
}

// SPDX-License-Identifier: MIT
//go:build !darwin

package discover

/*
#cgo CFLAGS: -I${SRCDIR}
#include "level_zero_info.h"
*/
import "C"

import (
	"fmt"
	"log/slog"
	"sync"
	"unsafe"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/ml"
)

// lzInitOnce guards the single call to lz_init so it is performed exactly once
// per process regardless of how many goroutines call getLevelZeroGPUInfo.
var lzInitOnce sync.Once

// lzInitErr records the result of the one-time init attempt.
var lzInitErr error

// levelZeroMaxDevices is the maximum number of L0 devices the shim can return
// per enumeration call.  16 is the recommended minimum in ze_ollama.h.
const levelZeroMaxDevices = 16

// getLevelZeroGPUInfo enumerates Intel Level Zero devices (GPU + optionally NPU)
// and returns them as []ml.DeviceInfo entries compatible with the discover
// package's device slice.
//
// Thread safety: safe to call concurrently; sync.Once protects the one-time init.
// Fallback: returns an empty slice without panicking when the shared library is
// absent (ADR-L0-006 graceful-missing-loader policy).
//
// INFO-2 COMPLIANCE: ml.DeviceInfo.Variant does not exist (verified absent via
// grep of ml/backend.go and ml/backend/backend.go).  GPU/NPU distinction is
// encoded in the Library field as "level-zero-gpu" vs "level-zero-npu".
//
// CGO BOUNDARY (ADR-L0-005):
//   unsafe.Pointer is used ONLY for the caller-allocated lz_device_info_t array
//   passed to lz_enumerate_devices.  No other unsafe usage.
//   Opaque handles would be stored as C.uintptr_t on the Go side; they are not
//   opened here (device selection happens later in the scheduler).
func getLevelZeroGPUInfo(libPath string) []ml.DeviceInfo {
	lzInitOnce.Do(func() {
		res := C.lz_init()
		if res != C.LZ_OK {
			lzInitErr = fmt.Errorf("level zero init failed: %s (code %d)",
				C.GoString(C.lz_result_str(res)), int(res))
		}
	})

	if lzInitErr != nil {
		slog.Debug("level zero unavailable", "error", lzInitErr)
		return nil
	}

	npuEnable := envconfig.L0NPUEnable()
	slog.Debug("level zero discover start", "npu_enable", npuEnable)

	// Caller-allocated device-info array; unsafe.Pointer used only here per ADR-L0-005.
	var buf [levelZeroMaxDevices]C.lz_device_info_t
	var count C.size_t

	res := C.lz_enumerate_devices(
		(*C.lz_device_info_t)(unsafe.Pointer(&buf[0])),
		C.size_t(levelZeroMaxDevices),
		&count,
	)
	if res != C.LZ_OK {
		slog.Warn("level zero enumeration failed",
			"code", int(res),
			"detail", C.GoString(C.lz_result_str(res)),
			"npu_enable", npuEnable)
		return nil
	}

	n := int(count)
	if n == 0 {
		slog.Debug("level zero: no devices found", "npu_enable", npuEnable)
		return nil
	}

	// Build the LibraryPath slice.  libPath is the directory containing
	// libggml-level-zero.so / ggml-level-zero.dll, typically
	// build/lib/ollama/level_zero on Linux.  If empty we use ml.LibOllamaPath
	// so that the scheduler can still locate the backend.
	libraryDirs := []string{ml.LibOllamaPath}
	if libPath != "" && libPath != ml.LibOllamaPath {
		libraryDirs = append(libraryDirs, libPath)
	}

	devices := make([]ml.DeviceInfo, 0, n)
	for i := 0; i < n; i++ {
		d := &buf[i]

		// Encode GPU/NPU distinction into the Library field.  The Variant field
		// does not exist in the frozen ml.DeviceInfo ABI (INFO-2 compliant).
		library := "level-zero-gpu"
		if d.device_kind == C.LZ_DEV_NPU {
			library = "level-zero-npu"
		}

		info := ml.DeviceInfo{
			DeviceID: ml.DeviceID{
				// Numeric ID matching enumeration index; postFilteredID in
				// runner.go may reassign it after deduplication.
				ID:      fmt.Sprintf("%d", i),
				Library: library,
			},
			Name: C.GoString(&d.name[0]),
			Description: fmt.Sprintf("Intel Level Zero %s (compute_units=%d clock_mhz=%d uuid=%s)",
				library, uint32(d.compute_units), uint32(d.clock_mhz), C.GoString(&d.uuid[0])),
			TotalMemory: uint64(d.total_memory),
			FreeMemory:  uint64(d.free_memory),
			// ComputeMajor/Minor not defined for L0 — set to zero (unsupported).
			ComputeMajor: 0,
			ComputeMinor: 0,
			LibraryPath:  libraryDirs,
		}
		devices = append(devices, info)
	}

	slog.Info("level zero devices discovered", "count", len(devices), "npu_enable", npuEnable)
	return devices
}

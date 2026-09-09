//go:build linux

package discover

/*
#cgo linux LDFLAGS: -ldl

#include <stdint.h>
#include <stdlib.h>

#define ZE_MAX_DEVICE_NAME 256

// Minimal Level Zero structure definitions mirroring the layouts in ze_api.h
// and zes_api.h so the loader library can be called through dlsym without
// vendoring the oneAPI headers. All enums are C ints (4 bytes) and the
// structures are naturally aligned on LP64 targets.
typedef struct _ze_pci_address {
	uint32_t domain;
	uint32_t bus;
	uint32_t device;
	uint32_t function;
} ze_pci_address;

typedef struct _ze_device_properties {
	uint32_t stype;
	const void *pNext;
	uint32_t type;
	uint32_t vendorId;
	uint32_t deviceId;
	uint32_t flags;
	uint32_t subdeviceId;
	uint32_t coreClockRate;
	uint64_t maxMemAllocSize;
	uint32_t maxHardwareContexts;
	uint32_t maxCommandQueuePriority;
	uint32_t numThreadsPerEU;
	uint32_t physicalEUSimdWidth;
	uint32_t numEUsPerSubslice;
	uint32_t numSubslicesPerSlice;
	uint32_t numSlices;
	uint64_t timerResolution;
	uint32_t timestampValidBits;
	uint32_t kernelTimestampValidBits;
	uint64_t kernelTimestamp;
	char name[ZE_MAX_DEVICE_NAME];
	uint32_t uuidSize;
	uint8_t uuid[16];
} ze_device_properties;

typedef struct _ze_device_memory_properties {
	uint32_t stype;
	const void *pNext;
	uint32_t flags;
	uint32_t maxClockRate;
	uint64_t totalSize;
	char name[ZE_MAX_DEVICE_NAME];
} ze_device_memory_properties;

typedef struct _ze_pci_ext_properties {
	uint32_t stype;
	const void *pNext;
	ze_pci_address address;
	uint32_t maxSpeed;
} ze_pci_ext_properties;

typedef struct _zes_mem_state {
	uint32_t stype;
	void *pNext;
	uint32_t type;
	uint64_t physicalSize;
	uint64_t free;
	uint64_t size;
	int32_t health;
} zes_mem_state;

typedef int (*ze_init_fn)(unsigned int);
typedef int (*ze_driver_get_fn)(unsigned int *, void **);
typedef int (*ze_device_get_fn)(void *, unsigned int *, void **);
typedef int (*ze_device_get_properties_fn)(void *, ze_device_properties *);
typedef int (*ze_device_get_memory_properties_fn)(void *, unsigned int *, ze_device_memory_properties *);
typedef int (*ze_device_pci_get_properties_fn)(void *, ze_pci_ext_properties *);
typedef int (*zes_init_fn)(unsigned int);
typedef int (*zes_driver_get_fn)(unsigned int *, void **);
typedef int (*zes_device_get_fn)(void *, unsigned int *, void **);
typedef int (*zes_device_enum_memory_modules_fn)(void *, unsigned int *, void **);
typedef int (*zes_device_get_memory_state_fn)(void *, zes_mem_state *);

static int ollama_call_ze_init(void *fn) {
	return ((ze_init_fn) fn)(0);
}

static int ollama_call_ze_driver_get(void *fn, unsigned int *count, void **drivers) {
	return ((ze_driver_get_fn) fn)(count, drivers);
}

static int ollama_call_ze_device_get(void *fn, void *driver, unsigned int *count, void **devices) {
	return ((ze_device_get_fn) fn)(driver, count, devices);
}

static int ollama_call_ze_device_get_properties(void *fn, void *device, ze_device_properties *props) {
	return ((ze_device_get_properties_fn) fn)(device, props);
}

static int ollama_call_ze_device_get_memory_properties(void *fn, void *device, unsigned int *count, ze_device_memory_properties *props) {
	return ((ze_device_get_memory_properties_fn) fn)(device, count, props);
}

static int ollama_call_ze_device_pci_get_properties(void *fn, void *device, ze_pci_ext_properties *props) {
	return ((ze_device_pci_get_properties_fn) fn)(device, props);
}

static int ollama_call_zes_init(void *fn) {
	return ((zes_init_fn) fn)(0);
}

static int ollama_call_zes_driver_get(void *fn, unsigned int *count, void **drivers) {
	return ((zes_driver_get_fn) fn)(count, drivers);
}

static int ollama_call_zes_device_get(void *fn, void *driver, unsigned int *count, void **devices) {
	return ((zes_device_get_fn) fn)(driver, count, devices);
}

static int ollama_call_zes_device_enum_memory_modules(void *fn, void *device, unsigned int *count, void **modules) {
	return ((zes_device_enum_memory_modules_fn) fn)(device, count, modules);
}

static int ollama_call_zes_device_get_memory_state(void *fn, void *module, zes_mem_state *state) {
	return ((zes_device_get_memory_state_fn) fn)(module, state);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"github.com/ollama/ollama/ml"
)

// Level Zero constants used by the probe (values from ze_api.h / zes_api.h).
const (
	zeResultSuccess                       = 0
	zeResultErrorUninitialized            = 0x78000001
	zeDeviceTypeGPU                       = 1
	zeStructureTypeDeviceProperties       = 2
	zeStructureTypeDeviceMemoryProperties = 6
	zeStructureTypePCIExtProperties       = 13
	zexStructureTypeMemState              = 4
	zeVendorIDIntel                       = 0x8086
)

func init() {
	probeIntelLevelZeroDevices = probeLevelZeroLinux
}

// dlopenIntelLibrary opens a Level Zero library, searching intelLibPaths before
// falling back to the dynamic linker's own search path. found reports whether
// the library was located on the filesystem (as opposed to resolved by the
// dynamic linker's cache).
func dlopenIntelLibrary(name string) (handle dlHandle, found bool, err error) {
	candidates := make([]string, 0, len(intelLibPaths)+1)
	for _, dir := range intelLibPaths {
		candidates = append(candidates, filepath.Join(dir, name))
	}
	candidates = append(candidates, name)

	var errs []string
	for _, path := range candidates {
		if filepath.IsAbs(path) {
			if _, statErr := os.Stat(path); statErr != nil {
				continue
			}
			found = true
		}
		handle, err = dlopen(path, false)
		if err == nil {
			return handle, found, nil
		}
		errs = append(errs, err.Error())
	}
	return dlHandle{}, found, errors.New(strings.Join(errs, "; "))
}

func probeLevelZeroLinux() ([]intelLevelZeroDevice, error) {
	// The sysman (zes*) API used for free VRAM is gated behind this env var on
	// current drivers. Set it before any Level Zero library is initialized.
	oldSysman, hadSysman := os.LookupEnv("ZES_ENABLE_SYSMAN")
	os.Setenv("ZES_ENABLE_SYSMAN", "1")
	defer func() {
		if hadSysman {
			os.Setenv("ZES_ENABLE_SYSMAN", oldSysman)
		} else {
			os.Unsetenv("ZES_ENABLE_SYSMAN")
		}
	}()

	// Pre-load the Intel GPU driver so the loader can bind to it even when it
	// lives outside the loader's default search path. Best effort: if this
	// fails we let zeInit report the authoritative error.
	if _, _, err := dlopenIntelLibrary(intelGpuLibName); err != nil {
		slog.Debug("Intel Level Zero GPU driver library not loadable", "library", intelGpuLibName, "error", err)
	}

	loader, _, err := dlopenIntelLibrary(oneapiLibName)
	if err != nil {
		loader, _, err = dlopenIntelLibrary(oneapiLoaderLibName)
	}
	if err != nil {
		return nil, fmt.Errorf("no Level Zero loader library found (%s or %s): %w", oneapiLibName, oneapiLoaderLibName, err)
	}

	zeInit, err := dlsym(loader, "zeInit")
	if err != nil {
		return nil, err
	}
	zeDriverGet, err := dlsym(loader, "zeDriverGet")
	if err != nil {
		return nil, err
	}
	zeDeviceGet, err := dlsym(loader, "zeDeviceGet")
	if err != nil {
		return nil, err
	}
	zeDeviceGetProperties, err := dlsym(loader, "zeDeviceGetProperties")
	if err != nil {
		return nil, err
	}
	zeDeviceGetMemoryProperties, err := dlsym(loader, "zeDeviceGetMemoryProperties")
	if err != nil {
		return nil, err
	}
	zeDevicePciGetProperties, _ := dlsym(loader, "zeDevicePciGetProperties")

	// The sysman API is optional - without it we can still report total VRAM.
	var zesDriverGet, zesDeviceGet, zesDeviceEnumMemoryModules, zesDeviceGetMemoryState unsafe.Pointer
	if zesInit, err := dlsym(loader, "zesInit"); err == nil {
		if ret := C.ollama_call_zes_init(zesInit); ret == zeResultSuccess {
			zesDriverGet, _ = dlsym(loader, "zesDriverGet")
			zesDeviceGet, _ = dlsym(loader, "zesDeviceGet")
			zesDeviceEnumMemoryModules, _ = dlsym(loader, "zesDeviceEnumMemoryModules")
			zesDeviceGetMemoryState, _ = dlsym(loader, "zesDeviceGetMemoryState")
		}
	}
	hasSysman := zesDriverGet != nil && zesDeviceGet != nil &&
		zesDeviceEnumMemoryModules != nil && zesDeviceGetMemoryState != nil

	if ret := C.ollama_call_ze_init(zeInit); ret != zeResultSuccess {
		if uint32(ret) == zeResultErrorUninitialized {
			return nil, fmt.Errorf("zeInit returned ZE_RESULT_ERROR_UNINITIALIZED - no Intel GPU driver found by the Level Zero loader (is %s installed?)", intelGpuLibName)
		}
		return nil, fmt.Errorf("zeInit failed: result=0x%x", uint32(ret))
	}

	driverCount, err := zeDriverCount(zeDriverGet)
	if err != nil {
		return nil, err
	}
	drivers := make([]unsafe.Pointer, driverCount)
	if driverCount > 0 {
		var count C.uint = C.uint(driverCount)
		if ret := C.ollama_call_ze_driver_get(zeDriverGet, &count, &drivers[0]); ret != zeResultSuccess {
			return nil, fmt.Errorf("zeDriverGet failed: result=0x%x", uint32(ret))
		}
	}

	var lzDevices []intelLevelZeroDevice
	for driverIdx := range drivers {
		if drivers[driverIdx] == nil {
			continue
		}
		deviceCount, err := zeDeviceCount(zeDeviceGet, drivers[driverIdx])
		if err != nil {
			continue
		}
		devices := make([]unsafe.Pointer, deviceCount)
		if deviceCount > 0 {
			var count C.uint = C.uint(deviceCount)
			if ret := C.ollama_call_ze_device_get(zeDeviceGet, drivers[driverIdx], &count, &devices[0]); ret != zeResultSuccess {
				continue
			}
		}
		for deviceIdx, device := range devices {
			if device == nil {
				continue
			}
			lz, ok := levelZeroDevice(zeDeviceGetProperties, zeDeviceGetMemoryProperties, zeDevicePciGetProperties, device)
			if !ok {
				continue
			}
			if hasSysman {
				// The sysman and core APIs enumerate drivers and devices in
				// the same order, so correlate by index.
				free, total := zesDeviceMemory(zesDriverGet, zesDeviceGet, zesDeviceEnumMemoryModules, zesDeviceGetMemoryState, driverIdx, deviceIdx)
				if total > 0 && (lz.TotalMemory == 0 || ml.SimilarDeviceMemory(total, lz.TotalMemory)) {
					lz.FreeMemory = free
				} else if free > 0 && (lz.TotalMemory == 0 || free <= lz.TotalMemory) {
					lz.FreeMemory = free
				}
			}
			lzDevices = append(lzDevices, lz)
		}
	}

	return lzDevices, nil
}

func zeDriverCount(zeDriverGet unsafe.Pointer) (int, error) {
	var count C.uint
	if ret := C.ollama_call_ze_driver_get(zeDriverGet, &count, nil); ret != zeResultSuccess {
		return 0, fmt.Errorf("zeDriverGet failed: result=0x%x", uint32(ret))
	}
	return int(count), nil
}

func zeDeviceCount(zeDeviceGet unsafe.Pointer, driver unsafe.Pointer) (int, error) {
	var count C.uint
	if ret := C.ollama_call_ze_device_get(zeDeviceGet, driver, &count, nil); ret != zeResultSuccess {
		return 0, fmt.Errorf("zeDeviceGet failed: result=0x%x", uint32(ret))
	}
	return int(count), nil
}

func levelZeroDevice(
	zeDeviceGetProperties unsafe.Pointer,
	zeDeviceGetMemoryProperties unsafe.Pointer,
	zeDevicePciGetProperties unsafe.Pointer,
	device unsafe.Pointer,
) (intelLevelZeroDevice, bool) {
	var props C.ze_device_properties
	props.stype = zeStructureTypeDeviceProperties
	if ret := C.ollama_call_ze_device_get_properties(zeDeviceGetProperties, device, &props); ret != zeResultSuccess {
		return intelLevelZeroDevice{}, false
	}
	if props._type != zeDeviceTypeGPU || props.vendorId != zeVendorIDIntel {
		return intelLevelZeroDevice{}, false
	}

	lz := intelLevelZeroDevice{
		Name:     C.GoString(&props.name[0]),
		VendorID: uint32(props.vendorId),
		DeviceID: uint32(props.deviceId),
	}

	var memCount C.uint
	if ret := C.ollama_call_ze_device_get_memory_properties(zeDeviceGetMemoryProperties, device, &memCount, nil); ret == zeResultSuccess && memCount > 0 {
		memProps := make([]C.ze_device_memory_properties, int(memCount))
		for i := range memProps {
			memProps[i].stype = zeStructureTypeDeviceMemoryProperties
		}
		if ret := C.ollama_call_ze_device_get_memory_properties(zeDeviceGetMemoryProperties, device, &memCount, &memProps[0]); ret == zeResultSuccess {
			// Pick the largest memory module as the device's usable VRAM.
			for i := range memProps {
				if total := uint64(memProps[i].totalSize); total > lz.TotalMemory {
					lz.TotalMemory = total
				}
			}
		}
	}

	if zeDevicePciGetProperties != nil {
		var pci C.ze_pci_ext_properties
		pci.stype = zeStructureTypePCIExtProperties
		if ret := C.ollama_call_ze_device_pci_get_properties(zeDevicePciGetProperties, device, &pci); ret == zeResultSuccess {
			lz.PCIAddress = fmt.Sprintf("%04x:%02x:%02x.%d",
				pci.address.domain, pci.address.bus, pci.address.device, pci.address.function)
		}
	}

	return lz, lz.Name != ""
}

// zesDeviceMemory returns the summed free and total memory of a sysman device
// correlated by driver/device index with the core API enumeration.
func zesDeviceMemory(zesDriverGet, zesDeviceGet, zesDeviceEnumMemoryModules, zesDeviceGetMemoryState unsafe.Pointer, driverIdx, deviceIdx int) (uint64, uint64) {
	var driverCount C.uint
	if ret := C.ollama_call_zes_driver_get(zesDriverGet, &driverCount, nil); ret != zeResultSuccess || int(driverCount) <= driverIdx {
		return 0, 0
	}
	drivers := make([]unsafe.Pointer, int(driverCount))
	if ret := C.ollama_call_zes_driver_get(zesDriverGet, &driverCount, &drivers[0]); ret != zeResultSuccess || drivers[driverIdx] == nil {
		return 0, 0
	}

	var deviceCount C.uint
	if ret := C.ollama_call_zes_device_get(zesDeviceGet, drivers[driverIdx], &deviceCount, nil); ret != zeResultSuccess || int(deviceCount) <= deviceIdx {
		return 0, 0
	}
	devices := make([]unsafe.Pointer, int(deviceCount))
	if ret := C.ollama_call_zes_device_get(zesDeviceGet, drivers[driverIdx], &deviceCount, &devices[0]); ret != zeResultSuccess || devices[deviceIdx] == nil {
		return 0, 0
	}

	var moduleCount C.uint
	if ret := C.ollama_call_zes_device_enum_memory_modules(zesDeviceEnumMemoryModules, devices[deviceIdx], &moduleCount, nil); ret != zeResultSuccess || moduleCount == 0 {
		return 0, 0
	}
	modules := make([]unsafe.Pointer, int(moduleCount))
	if ret := C.ollama_call_zes_device_enum_memory_modules(zesDeviceEnumMemoryModules, devices[deviceIdx], &moduleCount, &modules[0]); ret != zeResultSuccess {
		return 0, 0
	}

	var free, total uint64
	for _, module := range modules {
		if module == nil {
			continue
		}
		var state C.zes_mem_state
		state.stype = zexStructureTypeMemState
		if ret := C.ollama_call_zes_device_get_memory_state(zesDeviceGetMemoryState, module, &state); ret != zeResultSuccess {
			continue
		}
		free += uint64(state.free)
		total += uint64(state.size)
	}
	return free, total
}

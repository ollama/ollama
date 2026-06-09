//go:build linux && cgo

package discover

/*
#cgo linux LDFLAGS: -ldl

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef int (*nvmlInit_fn)(void);
typedef int (*nvmlShutdown_fn)(void);
typedef int (*nvmlDeviceGetCount_fn)(uint32_t*);
typedef int (*nvmlDeviceGetHandleByIndex_fn)(uint32_t, void**);
typedef int (*nvmlDeviceGetPowerManagementLimitConstraints_fn)(void*, uint32_t*, uint32_t*);
typedef int (*nvmlDeviceSetPowerManagementLimit_fn)(void*, uint32_t);
typedef int (*nvmlDeviceGetPowerManagementLimit_fn)(void*, uint32_t*);

static int call_nvml_init(void* fn) {
    return ((nvmlInit_fn)fn)();
}

static int call_nvml_shutdown(void* fn) {
    return ((nvmlShutdown_fn)fn)();
}

static int call_nvml_device_get_count(void* fn, uint32_t* count) {
    return ((nvmlDeviceGetCount_fn)fn)(count);
}

static int call_nvml_device_get_handle_by_index(void* fn, uint32_t index, void** device) {
    return ((nvmlDeviceGetHandleByIndex_fn)fn)(index, device);
}

static int call_nvml_device_get_power_limit_constraints(void* fn, void* device, uint32_t* minLimit, uint32_t* maxLimit) {
    return ((nvmlDeviceGetPowerManagementLimitConstraints_fn)fn)(device, minLimit, maxLimit);
}

static int call_nvml_device_set_power_limit(void* fn, void* device, uint32_t limit) {
    return ((nvmlDeviceSetPowerManagementLimit_fn)fn)(device, limit);
}

static int call_nvml_device_get_power_limit(void* fn, void* device, uint32_t* limit) {
    return ((nvmlDeviceGetPowerManagementLimit_fn)fn)(device, limit);
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// PowerLimitManager handles GPU power limit operations via NVML
type PowerLimitManager struct {
	nvmlHandle         unsafe.Pointer
	initFn             unsafe.Pointer
	shutdownFn         unsafe.Pointer
	deviceGetCountFn   unsafe.Pointer
	deviceGetHandleFn  unsafe.Pointer
	getConstraintsFn   unsafe.Pointer
	setPowerLimitFn    unsafe.Pointer
	getPowerLimitFn    unsafe.Pointer
	initialized        bool
}

// NVML error codes
const (
	nvmlSuccess = 0
)

// NewPowerLimitManager creates a new power limit manager
func NewPowerLimitManager() (*PowerLimitManager, error) {
	pm := &PowerLimitManager{}
	if err := pm.initialize(); err != nil {
		return nil, err
	}
	return pm, nil
}

func (pm *PowerLimitManager) initialize() error {
	var err error

	// Load NVML library
	var nvmlHandle unsafe.Pointer
	if nvmlHandle, err = pm.loadNVML(); err != nil {
		return fmt.Errorf("failed to load NVML: %w", err)
	}
	pm.nvmlHandle = nvmlHandle

	// Load required functions
	pm.initFn, err = dlsym(dlHandle{ptr: nvmlHandle}, "nvmlInit_v2")
	if err != nil {
		pm.initFn, err = dlsym(dlHandle{ptr: nvmlHandle}, "nvmlInit")
		if err != nil {
			return fmt.Errorf("nvmlInit not found: %w", err)
		}
	}

	pm.shutdownFn, err = dlsym(dlHandle{ptr: nvmlHandle}, "nvmlShutdown")
	if err != nil {
		return fmt.Errorf("nvmlShutdown not found: %w", err)
	}

	pm.deviceGetCountFn, err = dlsym(dlHandle{ptr: nvmlHandle}, "nvmlDeviceGetCount")
	if err != nil {
		return fmt.Errorf("nvmlDeviceGetCount not found: %w", err)
	}

	pm.deviceGetHandleFn, err = dlsym(dlHandle{ptr: nvmlHandle}, "nvmlDeviceGetHandleByIndex")
	if err != nil {
		return fmt.Errorf("nvmlDeviceGetHandleByIndex not found: %w", err)
	}

	pm.getConstraintsFn, err = dlsym(dlHandle{ptr: nvmlHandle}, "nvmlDeviceGetPowerManagementLimitConstraints")
	if err != nil {
		// This function might not be available in older NVML versions
		pm.getConstraintsFn = nil
	}

	pm.setPowerLimitFn, err = dlsym(dlHandle{ptr: nvmlHandle}, "nvmlDeviceSetPowerManagementLimit")
	if err != nil {
		return fmt.Errorf("nvmlDeviceSetPowerManagementLimit not found: %w", err)
	}

	pm.getPowerLimitFn, err = dlsym(dlHandle{ptr: nvmlHandle}, "nvmlDeviceGetPowerManagementLimit")
	if err != nil {
		return fmt.Errorf("nvmlDeviceGetPowerManagementLimit not found: %w", err)
	}

	// Initialize NVML
	ret := C.call_nvml_init(pm.initFn)
	if ret != nvmlSuccess {
		return fmt.Errorf("nvmlInit failed: %d", ret)
	}

	pm.initialized = true
	return nil
}

func (pm *PowerLimitManager) loadNVML() (unsafe.Pointer, error) {
	handle, err := dlopenFirst([]string{"libnvidia-ml.so.1", "libnvidia-ml.so"}, false)
	if err != nil {
		return nil, err
	}
	return handle.ptr, nil
}

// Close shuts down NVML
func (pm *PowerLimitManager) Close() error {
	if !pm.initialized || pm.shutdownFn == nil {
		return nil
	}
	ret := C.call_nvml_shutdown(pm.shutdownFn)
	if ret != nvmlSuccess {
		return fmt.Errorf("nvmlShutdown failed: %d", ret)
	}
	pm.initialized = false
	return nil
}

// GetDeviceCount returns the number of NVIDIA GPUs
func (pm *PowerLimitManager) GetDeviceCount() (uint32, error) {
	if !pm.initialized {
		return 0, fmt.Errorf("NVML not initialized")
	}

	var count C.uint32_t
	ret := C.call_nvml_device_get_count(pm.deviceGetCountFn, &count)
	if ret != nvmlSuccess {
		return 0, fmt.Errorf("nvmlDeviceGetCount failed: %d", ret)
	}
	return uint32(count), nil
}

// SetPowerLimit sets the power limit for a specific GPU (in milliwatts)
func (pm *PowerLimitManager) SetPowerLimit(deviceIndex uint32, powerLimitMW uint32) error {
	if !pm.initialized {
		return fmt.Errorf("NVML not initialized")
	}

	// Get device handle
	var device unsafe.Pointer
	ret := C.call_nvml_device_get_handle_by_index(pm.deviceGetHandleFn, C.uint32_t(deviceIndex), &device)
	if ret != nvmlSuccess {
		return fmt.Errorf("nvmlDeviceGetHandleByIndex failed: %d", ret)
	}

	// Set power limit
	ret = C.call_nvml_device_set_power_limit(pm.setPowerLimitFn, device, C.uint32_t(powerLimitMW))
	if ret != nvmlSuccess {
		return fmt.Errorf("nvmlDeviceSetPowerManagementLimit failed: %d", ret)
	}

	return nil
}

// GetPowerLimit gets the current power limit for a specific GPU (in milliwatts)
func (pm *PowerLimitManager) GetPowerLimit(deviceIndex uint32) (uint32, error) {
	if !pm.initialized {
		return 0, fmt.Errorf("NVML not initialized")
	}

	// Get device handle
	var device unsafe.Pointer
	ret := C.call_nvml_device_get_handle_by_index(pm.deviceGetHandleFn, C.uint32_t(deviceIndex), &device)
	if ret != nvmlSuccess {
		return 0, fmt.Errorf("nvmlDeviceGetHandleByIndex failed: %d", ret)
	}

	// Get power limit
	var limit C.uint32_t
	ret = C.call_nvml_device_get_power_limit(pm.getPowerLimitFn, device, &limit)
	if ret != nvmlSuccess {
		return 0, fmt.Errorf("nvmlDeviceGetPowerManagementLimit failed: %d", ret)
	}

	return uint32(limit), nil
}

// GetPowerLimitConstraints gets the min/max power limits for a specific GPU (in milliwatts)
func (pm *PowerLimitManager) GetPowerLimitConstraints(deviceIndex uint32) (minLimit, maxLimit uint32, err error) {
	if !pm.initialized {
		return 0, 0, fmt.Errorf("NVML not initialized")
	}

	if pm.getConstraintsFn == nil {
		return 0, 0, fmt.Errorf("nvmlDeviceGetPowerManagementLimitConstraints not available")
	}

	// Get device handle
	var device unsafe.Pointer
	ret := C.call_nvml_device_get_handle_by_index(pm.deviceGetHandleFn, C.uint32_t(deviceIndex), &device)
	if ret != nvmlSuccess {
		return 0, 0, fmt.Errorf("nvmlDeviceGetHandleByIndex failed: %d", ret)
	}

	// Get constraints
	var minC, maxC C.uint32_t
	ret = C.call_nvml_device_get_power_limit_constraints(pm.getConstraintsFn, device, &minC, &maxC)
	if ret != nvmlSuccess {
		return 0, 0, fmt.Errorf("nvmlDeviceGetPowerManagementLimitConstraints failed: %d", ret)
	}

	return uint32(minC), uint32(maxC), nil
}

// SetPowerLimitForAll sets the power limit for all NVIDIA GPUs (in watts)
func (pm *PowerLimitManager) SetPowerLimitForAll(powerLimitWatts int) error {
	if powerLimitWatts <= 0 {
		return fmt.Errorf("invalid power limit: %d watts", powerLimitWatts)
	}

	count, err := pm.GetDeviceCount()
	if err != nil {
		return err
	}

	// Convert watts to milliwatts
	powerLimitMW := uint32(powerLimitWatts * 1000)

	for i := uint32(0); i < count; i++ {
		if err := pm.SetPowerLimit(i, powerLimitMW); err != nil {
			return fmt.Errorf("failed to set power limit for GPU %d: %w", i, err)
		}
	}

	return nil
}

// GetPowerLimitsForAll gets the current power limits for all NVIDIA GPUs
func (pm *PowerLimitManager) GetPowerLimitsForAll() ([]uint32, error) {
	count, err := pm.GetDeviceCount()
	if err != nil {
		return nil, err
	}

	limits := make([]uint32, count)
	for i := uint32(0); i < count; i++ {
		limit, err := pm.GetPowerLimit(i)
		if err != nil {
			return nil, fmt.Errorf("failed to get power limit for GPU %d: %w", i, err)
		}
		limits[i] = limit
	}

	return limits, nil
}

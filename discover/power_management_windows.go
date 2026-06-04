//go:build windows

package discover

import (
	"fmt"
	"unsafe"

	"golang.org/x/sys/windows"
)

// PowerLimitManager handles GPU power limit operations via NVML
type PowerLimitManager struct {
	nvmlDLL            *windows.DLL
	initFn             *windows.Proc
	shutdownFn         *windows.Proc
	deviceGetCountFn   *windows.Proc
	deviceGetHandleFn  *windows.Proc
	getConstraintsFn   *windows.Proc
	setPowerLimitFn    *windows.Proc
	getPowerLimitFn    *windows.Proc
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
	// Load NVML library
	nvml, err := pm.loadNVML()
	if err != nil {
		return fmt.Errorf("failed to load NVML: %w", err)
	}
	pm.nvmlDLL = nvml

	// Load required functions
	pm.initFn, err = findProc(nvml, "nvmlInit_v2")
	if err != nil {
		pm.initFn, err = findProc(nvml, "nvmlInit")
		if err != nil {
			return fmt.Errorf("nvmlInit not found: %w", err)
		}
	}

	pm.shutdownFn, err = findProc(nvml, "nvmlShutdown")
	if err != nil {
		return fmt.Errorf("nvmlShutdown not found: %w", err)
	}

	pm.deviceGetCountFn, err = findProc(nvml, "nvmlDeviceGetCount")
	if err != nil {
		return fmt.Errorf("nvmlDeviceGetCount not found: %w", err)
	}

	pm.deviceGetHandleFn, err = findProc(nvml, "nvmlDeviceGetHandleByIndex")
	if err != nil {
		return fmt.Errorf("nvmlDeviceGetHandleByIndex not found: %w", err)
	}

	pm.getConstraintsFn, err = findProc(nvml, "nvmlDeviceGetPowerManagementLimitConstraints")
	if err != nil {
		// This function might not be available in older NVML versions
		pm.getConstraintsFn = nil
	}

	pm.setPowerLimitFn, err = findProc(nvml, "nvmlDeviceSetPowerManagementLimit")
	if err != nil {
		return fmt.Errorf("nvmlDeviceSetPowerManagementLimit not found: %w", err)
	}

	pm.getPowerLimitFn, err = findProc(nvml, "nvmlDeviceGetPowerManagementLimit")
	if err != nil {
		return fmt.Errorf("nvmlDeviceGetPowerManagementLimit not found: %w", err)
	}

	// Initialize NVML
	ret, _, _ := pm.initFn.Call()
	if ret != nvmlSuccess {
		return fmt.Errorf("nvmlInit failed: %d", ret)
	}

	pm.initialized = true
	return nil
}

func (pm *PowerLimitManager) loadNVML() (*windows.DLL, error) {
	// Try to load from System32 first
	nvml, err := loadDLLFromSystem32("nvml.dll")
	if err == nil {
		return nvml, nil
	}

	// Try to load from NVIDIA's NVML directories
	nvml, err = loadDLLFromDirs([]string{"nvml.dll"}, nvidiaNVMLDirsWindows())
	if err == nil {
		return nvml, nil
	}

	return nil, fmt.Errorf("nvml.dll not found")
}

// Close shuts down NVML
func (pm *PowerLimitManager) Close() error {
	if !pm.initialized || pm.shutdownFn == nil {
		return nil
	}
	ret, _, _ := pm.shutdownFn.Call()
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

	var count uint32
	ret, _, _ := pm.deviceGetCountFn.Call(uintptr(unsafe.Pointer(&count)))
	if ret != nvmlSuccess {
		return 0, fmt.Errorf("nvmlDeviceGetCount failed: %d", ret)
	}
	return count, nil
}

// SetPowerLimit sets the power limit for a specific GPU (in milliwatts)
func (pm *PowerLimitManager) SetPowerLimit(deviceIndex uint32, powerLimitMW uint32) error {
	if !pm.initialized {
		return fmt.Errorf("NVML not initialized")
	}

	// Get device handle
	var device unsafe.Pointer
	ret, _, _ := pm.deviceGetHandleFn.Call(uintptr(deviceIndex), uintptr(unsafe.Pointer(&device)))
	if ret != nvmlSuccess {
		return fmt.Errorf("nvmlDeviceGetHandleByIndex failed: %d", ret)
	}

	// Set power limit
	ret, _, _ = pm.setPowerLimitFn.Call(uintptr(device), uintptr(powerLimitMW))
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
	ret, _, _ := pm.deviceGetHandleFn.Call(uintptr(deviceIndex), uintptr(unsafe.Pointer(&device)))
	if ret != nvmlSuccess {
		return 0, fmt.Errorf("nvmlDeviceGetHandleByIndex failed: %d", ret)
	}

	// Get power limit
	var limit uint32
	ret, _, _ = pm.getPowerLimitFn.Call(uintptr(device), uintptr(unsafe.Pointer(&limit)))
	if ret != nvmlSuccess {
		return 0, fmt.Errorf("nvmlDeviceGetPowerManagementLimit failed: %d", ret)
	}

	return limit, nil
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
	ret, _, _ := pm.deviceGetHandleFn.Call(uintptr(deviceIndex), uintptr(unsafe.Pointer(&device)))
	if ret != nvmlSuccess {
		return 0, 0, fmt.Errorf("nvmlDeviceGetHandleByIndex failed: %d", ret)
	}

	// Get constraints
	var minC, maxC uint32
	ret, _, _ = pm.getConstraintsFn.Call(uintptr(device), uintptr(unsafe.Pointer(&minC)), uintptr(unsafe.Pointer(&maxC)))
	if ret != nvmlSuccess {
		return 0, 0, fmt.Errorf("nvmlDeviceGetPowerManagementLimitConstraints failed: %d", ret)
	}

	return minC, maxC, nil
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

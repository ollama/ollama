package discover

import (
	"errors"
	"fmt"
	"log/slog"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	hipSuccess       = 0
	hipErrorNoDevice = 100
)

type hipDevicePropMinimal struct {
	Name        [256]byte
	unused1     [140]byte
	GcnArchName [256]byte // gfx####
	iGPU        int       // Doesn't seem to actually report correctly
	unused2     [128]byte
}

// Wrap the amdhip64.dll library for GPU discovery
type HipLib struct {
	dll                    windows.Handle
	hipGetDeviceCount      uintptr
	hipGetDeviceProperties uintptr
	hipMemGetInfo          uintptr
	hipSetDevice           uintptr
	hipDriverGetVersion    uintptr
}

func NewHipLib() (*HipLib, error) {
	// At runtime we depend on v6, so discover GPUs with the same library for a consistent set of GPUs
	h, err := windows.LoadLibrary("amdhip64_6.dll")
	if err != nil {
		return nil, fmt.Errorf("unable to load amdhip64_6.dll, please make sure to upgrade to the latest amd driver: %w", err)
	}
	hl := &HipLib{}
	hl.dll = h
	hl.hipGetDeviceCount, err = windows.GetProcAddress(hl.dll, "hipGetDeviceCount")
	if err != nil {
		return nil, err
	}
	hl.hipGetDeviceProperties, err = windows.GetProcAddress(hl.dll, "hipGetDeviceProperties")
	if err != nil {
		return nil, err
	}
	hl.hipMemGetInfo, err = windows.GetProcAddress(hl.dll, "hipMemGetInfo")
	if err != nil {
		return nil, err
	}
	hl.hipSetDevice, err = windows.GetProcAddress(hl.dll, "hipSetDevice")
	if err != nil {
		return nil, err
	}
	hl.hipDriverGetVersion, err = windows.GetProcAddress(hl.dll, "hipDriverGetVersion")
	if err != nil {
		return nil, err
	}
	return hl, nil
}

// The hip library only evaluates the HIP_VISIBLE_DEVICES variable at startup
// so we have to unload/reset the library after we do our initial discovery
// to make sure our updates to that variable are processed by llama.cpp
func (hl *HipLib) Release() {
	err := windows.FreeLibrary(hl.dll)
	if err != nil {
		slog.Warn("failed to unload amdhip64.dll", "error", err)
	}
	hl.dll = 0
}

func (hl *HipLib) AMDDriverVersion() (driverMajor, driverMinor int, err error) {
	if hl.dll == 0 {
		return 0, 0, errors.New("dll has been unloaded")
	}
	var version int
	status, _, err := syscall.SyscallN(hl.hipDriverGetVersion, uintptr(unsafe.Pointer(&version)))
	if status != hipSuccess {
		return 0, 0, fmt.Errorf("failed call to hipDriverGetVersion: %d %s", status, err)
	}

	slog.Debug("hipDriverGetVersion", "version", version)
	driverMajor = version / 10000000
	driverMinor = (version - (driverMajor * 10000000)) / 100000

	return driverMajor, driverMinor, nil
}

func (hl *HipLib) HipGetDeviceCount() int {
	if hl.dll == 0 {
		slog.Error("dll has been unloaded")
		return 0
	}
	var count int
	status, _, err := syscall.SyscallN(hl.hipGetDeviceCount, uintptr(unsafe.Pointer(&count)))
	if status == hipErrorNoDevice {
		slog.Info("AMD ROCm reports no devices found")
		return 0
	}
	if status != hipSuccess {
		slog.Warn("failed call to hipGetDeviceCount", "status", status, "error", err)
	}
	return count
}

func (hl *HipLib) HipSetDevice(device int) error {
	if hl.dll == 0 {
		return errors.New("dll has been unloaded")
	}
	status, _, err := syscall.SyscallN(hl.hipSetDevice, uintptr(device))
	if status != hipSuccess {
		return fmt.Errorf("failed call to hipSetDevice: %d %s", status, err)
	}
	return nil
}

func (hl *HipLib) HipGetDeviceProperties(device int) (*hipDevicePropMinimal, error) {
	if hl.dll == 0 {
		return nil, errors.New("dll has been unloaded")
	}
	var props hipDevicePropMinimal
	status, _, err := syscall.SyscallN(hl.hipGetDeviceProperties, uintptr(unsafe.Pointer(&props)), uintptr(device))
	if status != hipSuccess {
		return nil, fmt.Errorf("failed call to hipGetDeviceProperties: %d %s", status, err)
	}
	return &props, nil
}

// free, total, err
func (hl *HipLib) HipMemGetInfo() (uint64, uint64, error) {
	if hl.dll == 0 {
		return 0, 0, errors.New("dll has been unloaded")
	}
	var totalMemory uint64
	var freeMemory uint64
	status, _, err := syscall.SyscallN(hl.hipMemGetInfo, uintptr(unsafe.Pointer(&freeMemory)), uintptr(unsafe.Pointer(&totalMemory)))
	if status != hipSuccess {
		return 0, 0, fmt.Errorf("failed call to hipMemGetInfo: %d %s", status, err)
	}
	return freeMemory, totalMemory, nil
}

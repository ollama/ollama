package discover

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>

int _hipGetDeviceCount(void* fn, int* count) {
	return ((int (*)(int*)) fn)(count);
}

int _hipGetDeviceProperties(void* fn, void* prop, int deviceId) {
	return ((int (*)(void*, int)) fn)(prop, deviceId);
}

int _hipMemGetInfo(void* fn, size_t* free, size_t* total) {
	return ((int (*)(size_t*, size_t*)) fn)(free, total);
}

int _hipSetDevice(void* fn, int deviceId) {
	return ((int (*)(int)) fn)(deviceId);
}

int _hipDriverGetVersion(void* fn, int* deviceVersion) {
	return ((int (*)(int*)) fn)(deviceVersion);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"log/slog"
	"unsafe"
)

// Wrap the libamdhip64.so library for GPU discovery
type HipLibImpl struct {
	dll                    unsafe.Pointer
	hipGetDeviceCount      unsafe.Pointer
	hipGetDeviceProperties unsafe.Pointer
	hipMemGetInfo          unsafe.Pointer
	hipSetDevice           unsafe.Pointer
	hipDriverGetVersion    unsafe.Pointer
}

func NewHipLib() (HipLib, error) {
	libDir, err := AMDValidateLibDir()
	if err != nil {
		return nil, fmt.Errorf("unable to verify rocm library, will use cpu %w", err)
	}

	h := C.dlopen(C.CString(libDir+"/libamdhip64.so"), C.RTLD_LAZY)

	if h == nil {
		return nil, fmt.Errorf("unable to load libamdhip64.so")
	}
	hl := &HipLibImpl{}
	hl.dll = h
	hl.hipGetDeviceCount = C.dlsym(hl.dll, C.CString("hipGetDeviceCount"))
	if hl.hipGetDeviceCount == nil {
		return nil, fmt.Errorf("unable to load hipGetDeviceCount")
	}
	hl.hipGetDeviceProperties = C.dlsym(hl.dll, C.CString("hipGetDeviceProperties"))
	if hl.hipGetDeviceProperties == nil {
		return nil, fmt.Errorf("unable to load hipGetDeviceProperties")
	}
	hl.hipMemGetInfo = C.dlsym(hl.dll, C.CString("hipMemGetInfo"))
	if hl.hipMemGetInfo == nil {
		return nil, fmt.Errorf("unable to load hipMemGetInfo")
	}
	hl.hipSetDevice = C.dlsym(hl.dll, C.CString("hipSetDevice"))
	if hl.hipSetDevice == nil {
		return nil, fmt.Errorf("unable to load hipSetDevice")
	}
	hl.hipDriverGetVersion = C.dlsym(hl.dll, C.CString("hipDriverGetVersion"))
	if hl.hipDriverGetVersion == nil {
		return nil, fmt.Errorf("unable to load hipDriverGetVersion")
	}
	return hl, nil
}

// The hip library only evaluates the HIP_VISIBLE_DEVICES variable at startup
// so we have to unload/reset the library after we do our initial discovery
// to make sure our updates to that variable are processed by llama.cpp
func (hl *HipLibImpl) Release() {
	C.dlclose(hl.dll)
	hl.dll = nil
}

func (hl *HipLibImpl) AMDDriverVersion() (driverMajor, driverMinor int, err error) {
	if hl.dll == nil {
		return 0, 0, errors.New("dll has been unloaded")
	}
	var version C.int
	status := uintptr(C._hipDriverGetVersion(hl.hipDriverGetVersion, &version))
	if status != hipSuccess {
		return 0, 0, fmt.Errorf("failed call to hipDriverGetVersion: %d", status)
	}

	slog.Debug("hipDriverGetVersion", "version", version)
	driverMajor = int(version / 10000000)
	driverMinor = int((version - C.int(driverMajor*10000000)) / 100000)

	return driverMajor, driverMinor, nil
}

func (hl *HipLibImpl) HipGetDeviceCount() int {
	if hl.dll == nil {
		slog.Error("dll has been unloaded")
		return 0
	}
	var count C.int
	status := C._hipGetDeviceCount(hl.hipGetDeviceCount, &count)
	if status == hipErrorNoDevice {
		slog.Info("AMD ROCm reports no devices found")
		return 0
	}
	if status != hipSuccess {
		slog.Warn("failed call to hipGetDeviceCount", "status", status)
	}
	return int(count)
}

func (hl *HipLibImpl) HipSetDevice(device int) error {
	if hl.dll == nil {
		return errors.New("dll has been unloaded")
	}
	status := C._hipSetDevice(hl.hipSetDevice, C.int(device))
	if status != hipSuccess {
		return fmt.Errorf("failed call to hipSetDevice: %d", status)
	}
	return nil
}

func (hl *HipLibImpl) HipGetDeviceProperties(device int) (*hipDevicePropMinimal, error) {
	if hl.dll == nil {
		return nil, errors.New("dll has been unloaded")
	}
	var props hipDevicePropMinimal
	status := C._hipGetDeviceProperties(hl.hipGetDeviceProperties, unsafe.Pointer(&props), C.int(device))
	if status != hipSuccess {
		return nil, fmt.Errorf("failed call to hipGetDeviceProperties: %d", status)
	}
	return &props, nil
}

// free, total, err
func (hl *HipLibImpl) HipMemGetInfo() (uint64, uint64, error) {
	if hl.dll == nil {
		return 0, 0, errors.New("dll has been unloaded")
	}
	var totalMemory C.size_t
	var freeMemory C.size_t
	status := C._hipMemGetInfo(hl.hipMemGetInfo, &freeMemory, &totalMemory)
	if status != hipSuccess {
		return 0, 0, fmt.Errorf("failed call to hipMemGetInfo: %d", status)
	}
	return uint64(freeMemory), uint64(totalMemory), nil
}

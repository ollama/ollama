//go:build linux

package discover

/*
#cgo linux LDFLAGS: -ldl

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static void * ollama_dlopen(const char * path, int global) {
	return dlopen(path, RTLD_NOW | (global ? RTLD_GLOBAL : RTLD_LOCAL));
}

static void * ollama_dlsym(void * handle, const char * name) {
	return dlsym(handle, name);
}

static const char * ollama_dlerror(void) {
	const char * err = dlerror();
	return err ? err : "";
}

typedef void * (*ollama_ggml_backend_load_fn)(const char *);
typedef size_t (*ollama_ggml_backend_reg_dev_count_fn)(void *);
typedef void * (*ollama_ggml_backend_reg_dev_get_fn)(void *, size_t);
typedef const char * (*ollama_ggml_backend_reg_name_fn)(void *);
typedef void (*ollama_ggml_backend_dev_get_props_fn)(void *, void *);

static void * ollama_call_ggml_backend_load(void * fn, const char * path) {
	return ((ollama_ggml_backend_load_fn) fn)(path);
}

static size_t ollama_call_ggml_backend_reg_dev_count(void * fn, void * reg) {
	return ((ollama_ggml_backend_reg_dev_count_fn) fn)(reg);
}

static void * ollama_call_ggml_backend_reg_dev_get(void * fn, void * reg, size_t index) {
	return ((ollama_ggml_backend_reg_dev_get_fn) fn)(reg, index);
}

static const char * ollama_call_ggml_backend_reg_name(void * fn, void * reg) {
	return ((ollama_ggml_backend_reg_name_fn) fn)(reg);
}

static void ollama_call_ggml_backend_dev_get_props(void * fn, void * dev, void * props) {
	((ollama_ggml_backend_dev_get_props_fn) fn)(dev, props);
}

static const char * ollama_cstr_from_uintptr(uintptr_t ptr) {
	return (const char *) ptr;
}

typedef int (*ollama_cu_init_fn)(unsigned int);
typedef int (*ollama_cu_driver_get_version_fn)(int *);
typedef int (*ollama_cu_device_get_count_fn)(int *);
typedef int (*ollama_cu_device_get_fn)(int *, int);
typedef int (*ollama_cu_device_get_attribute_fn)(int *, int, int);
typedef int (*ollama_cu_device_get_name_fn)(char *, int, int);
typedef int (*ollama_cu_device_total_mem_fn)(size_t *, int);
typedef int (*ollama_cu_device_get_pci_bus_id_fn)(char *, int, int);

static int ollama_call_cu_init(void * fn) {
	return ((ollama_cu_init_fn) fn)(0);
}

static int ollama_call_cu_driver_get_version(void * fn, int * version) {
	return ((ollama_cu_driver_get_version_fn) fn)(version);
}

static int ollama_call_cu_device_get_count(void * fn, int * count) {
	return ((ollama_cu_device_get_count_fn) fn)(count);
}

static int ollama_call_cu_device_get(void * fn, int * device, int index) {
	return ((ollama_cu_device_get_fn) fn)(device, index);
}

static int ollama_call_cu_device_get_attribute(void * fn, int * value, int attr, int device) {
	return ((ollama_cu_device_get_attribute_fn) fn)(value, attr, device);
}

static int ollama_call_cu_device_get_name(void * fn, char * name, int len, int device) {
	return ((ollama_cu_device_get_name_fn) fn)(name, len, device);
}

static int ollama_call_cu_device_total_mem(void * fn, size_t * total, int device) {
	return ((ollama_cu_device_total_mem_fn) fn)(total, device);
}

static int ollama_call_cu_device_get_pci_bus_id(void * fn, char * pci, int len, int device) {
	return ((ollama_cu_device_get_pci_bus_id_fn) fn)(pci, len, device);
}

typedef int (*ollama_nvml_init_fn)(void);
typedef int (*ollama_nvml_shutdown_fn)(void);
typedef int (*ollama_nvml_system_get_driver_version_fn)(char *, unsigned int);

static int ollama_call_nvml_init(void * fn) {
	return ((ollama_nvml_init_fn) fn)();
}

static int ollama_call_nvml_shutdown(void * fn) {
	return ((ollama_nvml_shutdown_fn) fn)();
}

static int ollama_call_nvml_system_get_driver_version(void * fn, char * version, unsigned int len) {
	return ((ollama_nvml_system_get_driver_version_fn) fn)(version, len);
}

*/
import "C"

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"unsafe"
)

const (
	cuSuccess                               = 0
	cuDeviceAttributeComputeCapabilityMajor = 75
	cuDeviceAttributeComputeCapabilityMinor = 76
	cuDeviceAttributeIntegrated             = 18
)

type dlHandle struct {
	ptr unsafe.Pointer
}

func runPlatformNativeProbe(ctx context.Context, libDirs []string) ([]nativeProbeDevice, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	ggmlDevices, ggmlErr := probeGGMLDevicesLinux(libDirs)
	var cudaDevices []nativeProbeDevice
	var cudaErr error
	if nativeProbeHasCUDA(libDirs) {
		cudaDevices, cudaErr = probeCUDADriverLinux()
	}
	var rocmDevices []nativeProbeDevice
	var rocmErr error
	if nativeProbeHasROCm(libDirs) {
		rocmDevices, rocmErr = probeROCmSysfsLinux()
	}

	devices := mergeNativeProbeDevices(mergeNativeProbeDevices(ggmlDevices, cudaDevices), rocmDevices)
	if len(devices) > 0 {
		return devices, nil
	}

	if ggmlErr != nil {
		return nil, ggmlErr
	}
	if rocmErr != nil {
		return nil, rocmErr
	}
	return nil, cudaErr
}

func probeGGMLDevicesLinux(libDirs []string) ([]nativeProbeDevice, error) {
	if len(libDirs) == 0 {
		return nil, errors.New("no library directories provided")
	}

	baseDir := libDirs[0]
	if baseDir == "" {
		return nil, errors.New("empty GGML library directory")
	}

	base, err := dlopen(ggmlLibraryFile(baseDir, "ggml-base"), true)
	if err != nil {
		return nil, err
	}

	ggml, err := dlopen(ggmlLibraryFile(baseDir, "ggml"), true)
	if err != nil {
		return nil, err
	}

	backendLoad, err := dlsym(ggml, "ggml_backend_load")
	if err != nil {
		return nil, err
	}
	regDevCount, err := dlsym(base, "ggml_backend_reg_dev_count")
	if err != nil {
		return nil, err
	}
	regDevGet, err := dlsym(base, "ggml_backend_reg_dev_get")
	if err != nil {
		return nil, err
	}
	regName, err := dlsym(base, "ggml_backend_reg_name")
	if err != nil {
		return nil, err
	}
	devGetProps, err := dlsym(base, "ggml_backend_dev_get_props")
	if err != nil {
		return nil, err
	}

	var devices []nativeProbeDevice
	for _, backendPath := range nativeProbeBackendFiles(libDirs) {
		reg := callGGMLBackendLoad(backendLoad, backendPath)
		if reg == nil {
			continue
		}

		library := ggmlProbeLibraryName(callGGMLRegName(regName, reg))
		count := int(callGGMLRegDevCount(regDevCount, reg))
		for i := range count {
			dev := callGGMLRegDevGet(regDevGet, reg, i)
			if dev == nil {
				continue
			}
			props := callGGMLDeviceProps(devGetProps, dev)
			if props.MemoryTotal == 0 {
				continue
			}
			devices = append(devices, nativeProbeDevice{
				Library:             library,
				Index:               i,
				IndexMatchesBackend: true,
				Name:                cString(props.Name),
				Description:         cString(props.Description),
				DeviceID:            cString(props.DeviceID),
				Integrated:          ggmlDeviceTypeIntegrated(props.Type),
				IntegratedKnown: props.Type == ggmlBackendDeviceTypeGPU ||
					props.Type == ggmlBackendDeviceTypeIGPU,
				TotalMemory: uint64(props.MemoryTotal),
				FreeMemory:  uint64(props.MemoryFree),
			})
			slog.Debug("GGML GPU device type", "library", library, "index", i, "ggml_type", props.Type, "integrated", ggmlDeviceTypeIntegrated(props.Type))
		}
	}

	return devices, nil
}

func probeCUDADriverLinux() ([]nativeProbeDevice, error) {
	cuda, err := dlopenFirst([]string{"libcuda.so.1", "libcuda.so"}, false)
	if err != nil {
		return nil, err
	}

	cuInit, err := dlsym(cuda, "cuInit")
	if err != nil {
		return nil, err
	}
	cuDriverGetVersion, err := dlsym(cuda, "cuDriverGetVersion")
	if err != nil {
		return nil, err
	}
	cuDeviceGetCount, err := dlsym(cuda, "cuDeviceGetCount")
	if err != nil {
		return nil, err
	}
	cuDeviceGet, err := dlsym(cuda, "cuDeviceGet")
	if err != nil {
		return nil, err
	}
	cuDeviceGetAttribute, err := dlsym(cuda, "cuDeviceGetAttribute")
	if err != nil {
		return nil, err
	}
	cuDeviceGetName, err := dlsym(cuda, "cuDeviceGetName")
	if err != nil {
		return nil, err
	}
	cuDeviceTotalMem, err := dlsymAny(cuda, "cuDeviceTotalMem_v2", "cuDeviceTotalMem")
	if err != nil {
		return nil, err
	}
	cuDeviceGetPCIBusID, _ := dlsym(cuda, "cuDeviceGetPCIBusId")

	if ret := C.ollama_call_cu_init(cuInit); ret != cuSuccess {
		return nil, fmt.Errorf("cuInit failed: %d", int(ret))
	}

	var driverVersion C.int
	driverMajor, driverMinor := 0, 0
	if ret := C.ollama_call_cu_driver_get_version(cuDriverGetVersion, &driverVersion); ret == cuSuccess {
		version := int(driverVersion)
		driverMajor = version / 1000
		driverMinor = (version - driverMajor*1000) / 10
	}

	nvidiaDriverMajor := 0
	if driver, err := probeNVIDIADriverMajorLinux(); err == nil {
		nvidiaDriverMajor = driver
	}

	var count C.int
	if ret := C.ollama_call_cu_device_get_count(cuDeviceGetCount, &count); ret != cuSuccess {
		return nil, fmt.Errorf("cuDeviceGetCount failed: %d", int(ret))
	}

	deviceCount := int(count)
	devices := make([]nativeProbeDevice, 0, deviceCount)
	for i := range deviceCount {
		var device C.int
		if ret := C.ollama_call_cu_device_get(cuDeviceGet, &device, C.int(i)); ret != cuSuccess {
			continue
		}

		major := cudaDeviceAttribute(cuDeviceGetAttribute, cuDeviceAttributeComputeCapabilityMajor, device)
		minor := cudaDeviceAttribute(cuDeviceGetAttribute, cuDeviceAttributeComputeCapabilityMinor, device)
		integrated := cudaDeviceAttribute(cuDeviceGetAttribute, cuDeviceAttributeIntegrated, device) == 1

		var name [128]C.char
		_ = C.ollama_call_cu_device_get_name(cuDeviceGetName, &name[0], C.int(len(name)), device)

		var total C.size_t
		_ = C.ollama_call_cu_device_total_mem(cuDeviceTotalMem, &total, device)

		pci := ""
		if cuDeviceGetPCIBusID != nil {
			var pciBuf [32]C.char
			if ret := C.ollama_call_cu_device_get_pci_bus_id(cuDeviceGetPCIBusID, &pciBuf[0], C.int(len(pciBuf)), device); ret == cuSuccess {
				pci = strings.ToLower(C.GoString(&pciBuf[0]))
			}
		}

		devices = append(devices, nativeProbeDevice{
			Library:             "CUDA",
			Index:               i,
			IndexMatchesBackend: true,
			Description:         C.GoString(&name[0]),
			DeviceID:            pci,
			Integrated:          integrated,
			IntegratedKnown:     true,
			TotalMemory:         uint64(total),
			ComputeMajor:        major,
			ComputeMinor:        minor,
			CUDADriverMajor:     driverMajor,
			CUDADriverMinor:     driverMinor,
			NVIDIADriverMajor:   nvidiaDriverMajor,
		})
	}

	return devices, nil
}

func probeROCmSysfsLinux() ([]nativeProbeDevice, error) {
	sysfsDevices, err := readROCmLinuxSysfsDevices("/sys")
	if err != nil {
		return nil, err
	}

	override := hsaOverrideGFXTarget()
	// Sysfs stays in physical KFD order; ROCm visibility envs can reindex the
	// backend device list, so filtered sysfs data must merge by PCI ID only.
	backendIndex := !rocmVisibleDevicesEnvSet()
	devices := make([]nativeProbeDevice, 0, len(sysfsDevices))
	for i, sysfsDevice := range sysfsDevices {
		gfxTarget := sysfsDevice.gfxTarget
		if override != "" {
			gfxTarget = override
		}
		devices = append(devices, nativeProbeDevice{
			Library:             "ROCm",
			Index:               i,
			IndexMatchesBackend: backendIndex,
			DeviceID:            sysfsDevice.pciID,
			Integrated:          sysfsDevice.integrated,
			IntegratedKnown:     sysfsDevice.known,
			GFXTarget:           gfxTarget,
		})
	}
	return devices, nil
}

func rocmVisibleDevicesEnvSet() bool {
	for _, name := range []string{"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"} {
		if os.Getenv(name) != "" {
			return true
		}
	}
	return false
}

func probeNVIDIADriverMajorLinux() (int, error) {
	nvml, err := dlopenFirst([]string{"libnvidia-ml.so.1", "libnvidia-ml.so"}, false)
	if err != nil {
		return 0, err
	}
	initFn, err := dlsym(nvml, "nvmlInit_v2")
	if err != nil {
		return 0, err
	}
	shutdownFn, err := dlsym(nvml, "nvmlShutdown")
	if err != nil {
		return 0, err
	}
	driverFn, err := dlsym(nvml, "nvmlSystemGetDriverVersion")
	if err != nil {
		return 0, err
	}
	if ret := C.ollama_call_nvml_init(initFn); ret != 0 {
		return 0, fmt.Errorf("nvmlInit_v2 failed: %d", int(ret))
	}
	defer C.ollama_call_nvml_shutdown(shutdownFn)

	var version [80]C.char
	if ret := C.ollama_call_nvml_system_get_driver_version(driverFn, &version[0], C.uint(len(version))); ret != 0 {
		return 0, fmt.Errorf("nvmlSystemGetDriverVersion failed: %d", int(ret))
	}
	return parseNVIDIADriverMajor(C.GoString(&version[0]))
}

func cudaDeviceAttribute(fn unsafe.Pointer, attr int, device C.int) int {
	var value C.int
	if ret := C.ollama_call_cu_device_get_attribute(fn, &value, C.int(attr), device); ret != cuSuccess {
		return 0
	}
	return int(value)
}

func dlopenFirst(names []string, global bool) (dlHandle, error) {
	var errs []string
	for _, name := range names {
		handle, err := dlopen(name, global)
		if err == nil {
			return handle, nil
		}
		errs = append(errs, err.Error())
	}
	return dlHandle{}, errors.New(strings.Join(errs, "; "))
}

func dlopen(path string, global bool) (dlHandle, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	handle := C.ollama_dlopen(cpath, boolToCInt(global))
	if handle == nil {
		return dlHandle{}, fmt.Errorf("dlopen %s: %s", path, C.GoString(C.ollama_dlerror()))
	}
	return dlHandle{ptr: handle}, nil
}

func dlsym(handle dlHandle, name string) (unsafe.Pointer, error) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	sym := C.ollama_dlsym(handle.ptr, cname)
	if sym == nil {
		return nil, fmt.Errorf("dlsym %s: %s", name, C.GoString(C.ollama_dlerror()))
	}
	return sym, nil
}

func dlsymAny(handle dlHandle, names ...string) (unsafe.Pointer, error) {
	var errs []string
	for _, name := range names {
		sym, err := dlsym(handle, name)
		if err == nil {
			return sym, nil
		}
		errs = append(errs, err.Error())
	}
	return nil, errors.New(strings.Join(errs, "; "))
}

func callGGMLBackendLoad(fn unsafe.Pointer, path string) unsafe.Pointer {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	return C.ollama_call_ggml_backend_load(fn, cpath)
}

func callGGMLRegDevCount(fn unsafe.Pointer, reg unsafe.Pointer) uintptr {
	return uintptr(C.ollama_call_ggml_backend_reg_dev_count(fn, reg))
}

func callGGMLRegDevGet(fn unsafe.Pointer, reg unsafe.Pointer, index int) unsafe.Pointer {
	return C.ollama_call_ggml_backend_reg_dev_get(fn, reg, C.size_t(index))
}

func callGGMLRegName(fn unsafe.Pointer, reg unsafe.Pointer) string {
	return C.GoString(C.ollama_call_ggml_backend_reg_name(fn, reg))
}

func callGGMLDeviceProps(fn unsafe.Pointer, dev unsafe.Pointer) ggmlBackendDevProps {
	var props ggmlBackendDevProps
	C.ollama_call_ggml_backend_dev_get_props(fn, dev, unsafe.Pointer(&props))
	return props
}

func cString(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}
	return C.GoString(C.ollama_cstr_from_uintptr(C.uintptr_t(ptr)))
}

func boolToCInt(v bool) C.int {
	if v {
		return 1
	}
	return 0
}

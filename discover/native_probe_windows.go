//go:build windows

package discover

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"github.com/ollama/ollama/llm"
	"golang.org/x/sys/windows"
)

const (
	cuSuccessWindows                         = 0
	cuDeviceAttributeComputeCapabilityMajorW = 75
	cuDeviceAttributeComputeCapabilityMinorW = 76
	cuDeviceAttributeIntegratedW             = 18
	hipSuccessWindows                        = 0
	hipDeviceAttributeIntegratedWindows      = 16
)

func runPlatformNativeProbe(ctx context.Context, libDirs []string) ([]nativeProbeDevice, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	ggmlDevices, ggmlErr := probeGGMLDevicesWindows(libDirs)
	var cudaDevices []nativeProbeDevice
	var cudaErr error
	if nativeProbeHasCUDA(libDirs) {
		cudaDevices, cudaErr = probeCUDADriverWindows()
	}
	var rocmDevices []nativeProbeDevice
	var rocmErr error
	if nativeProbeHasROCm(libDirs) {
		rocmDevices, rocmErr = probeHIPRuntimeWindows(libDirs)
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

func probeGGMLDevicesWindows(libDirs []string) ([]nativeProbeDevice, error) {
	if len(libDirs) == 0 || libDirs[0] == "" {
		return nil, errors.New("empty GGML library directory")
	}

	base, err := loadDLLFromPath(ggmlLibraryFile(libDirs[0], "ggml-base"))
	if err != nil {
		return nil, err
	}
	ggml, err := loadDLLFromPath(ggmlLibraryFile(libDirs[0], "ggml"))
	if err != nil {
		return nil, err
	}

	backendLoad, err := findProc(ggml, "ggml_backend_load")
	if err != nil {
		return nil, err
	}
	regDevCount, err := findProc(base, "ggml_backend_reg_dev_count")
	if err != nil {
		return nil, err
	}
	regDevGet, err := findProc(base, "ggml_backend_reg_dev_get")
	if err != nil {
		return nil, err
	}
	regName, err := findProc(base, "ggml_backend_reg_name")
	if err != nil {
		return nil, err
	}
	devGetProps, err := findProc(base, "ggml_backend_dev_get_props")
	if err != nil {
		return nil, err
	}

	var devices []nativeProbeDevice
	for _, backendPath := range nativeProbeBackendFiles(libDirs) {
		cpath, err := windows.BytePtrFromString(backendPath)
		if err != nil {
			return nil, err
		}
		reg, _, _ := backendLoad.Call(uintptr(unsafe.Pointer(cpath)))
		if reg == 0 {
			continue
		}

		regNamePtr, _, _ := regName.Call(reg)
		library := ggmlProbeLibraryName(windowsCString(regNamePtr))
		count, _, _ := regDevCount.Call(reg)
		for i := uintptr(0); i < count; i++ {
			dev, _, _ := regDevGet.Call(reg, i)
			if dev == 0 {
				continue
			}
			var props ggmlBackendDevProps
			devGetProps.Call(dev, uintptr(unsafe.Pointer(&props)))
			if props.MemoryTotal == 0 {
				continue
			}
			devices = append(devices, nativeProbeDevice{
				Library:             library,
				Index:               int(i),
				IndexMatchesBackend: true,
				Name:                windowsCString(props.Name),
				Description:         windowsCString(props.Description),
				DeviceID:            windowsCString(props.DeviceID),
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

func probeCUDADriverWindows() ([]nativeProbeDevice, error) {
	cuda, err := loadDLLFromSystem32("nvcuda.dll")
	if err != nil {
		return nil, err
	}
	cuInit, err := findProc(cuda, "cuInit")
	if err != nil {
		return nil, err
	}
	cuDriverGetVersion, err := findProc(cuda, "cuDriverGetVersion")
	if err != nil {
		return nil, err
	}
	cuDeviceGetCount, err := findProc(cuda, "cuDeviceGetCount")
	if err != nil {
		return nil, err
	}
	cuDeviceGet, err := findProc(cuda, "cuDeviceGet")
	if err != nil {
		return nil, err
	}
	cuDeviceGetAttribute, err := findProc(cuda, "cuDeviceGetAttribute")
	if err != nil {
		return nil, err
	}
	cuDeviceGetName, err := findProc(cuda, "cuDeviceGetName")
	if err != nil {
		return nil, err
	}
	cuDeviceTotalMem, err := procAny(cuda, "cuDeviceTotalMem_v2", "cuDeviceTotalMem")
	if err != nil {
		return nil, err
	}
	cuDeviceGetPCIBusID, _ := findProc(cuda, "cuDeviceGetPCIBusId")

	if ret, _, _ := cuInit.Call(0); ret != cuSuccessWindows {
		return nil, fmt.Errorf("cuInit failed: %d", ret)
	}

	driverMajor, driverMinor := 0, 0
	var driverVersion int32
	if ret, _, _ := cuDriverGetVersion.Call(uintptr(unsafe.Pointer(&driverVersion))); ret == cuSuccessWindows {
		version := int(driverVersion)
		driverMajor = version / 1000
		driverMinor = (version - driverMajor*1000) / 10
	}

	nvidiaDriverMajor := 0
	if driver, err := probeNVIDIADriverMajorWindows(); err == nil {
		nvidiaDriverMajor = driver
	}

	var count int32
	if ret, _, _ := cuDeviceGetCount.Call(uintptr(unsafe.Pointer(&count))); ret != cuSuccessWindows {
		return nil, fmt.Errorf("cuDeviceGetCount failed: %d", ret)
	}

	devices := make([]nativeProbeDevice, 0, int(count))
	for i := range int(count) {
		var device int32
		if ret, _, _ := cuDeviceGet.Call(uintptr(unsafe.Pointer(&device)), uintptr(i)); ret != cuSuccessWindows {
			continue
		}

		major := cudaDeviceAttributeWindows(cuDeviceGetAttribute, cuDeviceAttributeComputeCapabilityMajorW, device)
		minor := cudaDeviceAttributeWindows(cuDeviceGetAttribute, cuDeviceAttributeComputeCapabilityMinorW, device)
		integrated := cudaDeviceAttributeWindows(cuDeviceGetAttribute, cuDeviceAttributeIntegratedW, device) == 1

		name := make([]byte, 128)
		cuDeviceGetName.Call(uintptr(unsafe.Pointer(&name[0])), uintptr(len(name)), uintptr(device))

		var total uintptr
		cuDeviceTotalMem.Call(uintptr(unsafe.Pointer(&total)), uintptr(device))

		pci := ""
		if cuDeviceGetPCIBusID != nil {
			pciBuf := make([]byte, 32)
			if ret, _, _ := cuDeviceGetPCIBusID.Call(uintptr(unsafe.Pointer(&pciBuf[0])), uintptr(len(pciBuf)), uintptr(device)); ret == cuSuccessWindows {
				pci = strings.ToLower(byteCString(pciBuf))
			}
		}

		devices = append(devices, nativeProbeDevice{
			Library:             "CUDA",
			Index:               i,
			IndexMatchesBackend: true,
			Description:         byteCString(name),
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

func probeHIPRuntimeWindows(libDirs []string) ([]nativeProbeDevice, error) {
	hipPath, err := llm.WindowsROCmRuntimeDLLPath(libDirs)
	if err != nil {
		return nil, err
	}
	hip, err := loadDLLFromPath(hipPath)
	if err != nil {
		return nil, err
	}
	hipGetDeviceCount, err := findProc(hip, "hipGetDeviceCount")
	if err != nil {
		return nil, err
	}
	hipDeviceGetName, err := findProc(hip, "hipDeviceGetName")
	if err != nil {
		return nil, err
	}
	hipDeviceTotalMem, err := findProc(hip, "hipDeviceTotalMem")
	if err != nil {
		return nil, err
	}
	hipDeviceGetPCIBusID, _ := findProc(hip, "hipDeviceGetPCIBusId")
	hipDeviceGetAttribute, _ := findProc(hip, "hipDeviceGetAttribute")

	var count int32
	if ret, _, _ := hipGetDeviceCount.Call(uintptr(unsafe.Pointer(&count))); ret != hipSuccessWindows {
		return nil, fmt.Errorf("hipGetDeviceCount failed: %d", ret)
	}

	devices := make([]nativeProbeDevice, 0, int(count))
	for i := range int(count) {
		name := make([]byte, 128)
		hipDeviceGetName.Call(uintptr(unsafe.Pointer(&name[0])), uintptr(len(name)), uintptr(i))

		var total uintptr
		hipDeviceTotalMem.Call(uintptr(unsafe.Pointer(&total)), uintptr(i))

		pci := ""
		if hipDeviceGetPCIBusID != nil {
			pciBuf := make([]byte, 32)
			if ret, _, _ := hipDeviceGetPCIBusID.Call(uintptr(unsafe.Pointer(&pciBuf[0])), uintptr(len(pciBuf)), uintptr(i)); ret == hipSuccessWindows {
				pci = strings.ToLower(byteCString(pciBuf))
			}
		}

		integrated, integratedKnown := false, false
		if hipDeviceGetAttribute != nil {
			integrated = hipDeviceAttributeWindows(hipDeviceGetAttribute, hipDeviceAttributeIntegratedWindows, int32(i)) == 1
			integratedKnown = true
		}

		devices = append(devices, nativeProbeDevice{
			Library:             "ROCm",
			Index:               i,
			IndexMatchesBackend: true,
			Description:         byteCString(name),
			DeviceID:            pci,
			Integrated:          integrated,
			IntegratedKnown:     integratedKnown,
			TotalMemory:         uint64(total),
		})
	}

	return devices, nil
}

func probeNVIDIADriverMajorWindows() (int, error) {
	nvml, err := loadDLLFromSystem32("nvml.dll")
	if err != nil {
		nvml, err = loadDLLFromDirs([]string{"nvml.dll"}, nvidiaNVMLDirsWindows())
	}
	if err != nil {
		return 0, err
	}
	initFn, err := findProc(nvml, "nvmlInit_v2")
	if err != nil {
		return 0, err
	}
	shutdownFn, err := findProc(nvml, "nvmlShutdown")
	if err != nil {
		return 0, err
	}
	driverFn, err := findProc(nvml, "nvmlSystemGetDriverVersion")
	if err != nil {
		return 0, err
	}
	if ret, _, _ := initFn.Call(); ret != 0 {
		return 0, fmt.Errorf("nvmlInit_v2 failed: %d", ret)
	}
	defer shutdownFn.Call()

	version := make([]byte, 80)
	if ret, _, _ := driverFn.Call(uintptr(unsafe.Pointer(&version[0])), uintptr(len(version))); ret != 0 {
		return 0, fmt.Errorf("nvmlSystemGetDriverVersion failed: %d", ret)
	}
	return parseNVIDIADriverMajor(byteCString(version))
}

func cudaDeviceAttributeWindows(fn *windows.Proc, attr int, device int32) int {
	var value int32
	if ret, _, _ := fn.Call(uintptr(unsafe.Pointer(&value)), uintptr(attr), uintptr(device)); ret != cuSuccessWindows {
		return 0
	}
	return int(value)
}

func hipDeviceAttributeWindows(fn *windows.Proc, attr int, device int32) int {
	var value int32
	if ret, _, _ := fn.Call(uintptr(unsafe.Pointer(&value)), uintptr(attr), uintptr(device)); ret != hipSuccessWindows {
		return 0
	}
	return int(value)
}

func findProc(dll *windows.DLL, name string) (*windows.Proc, error) {
	return dll.FindProc(name)
}

// Use LoadLibraryEx so GPU discovery does not honor the current directory or PATH for DLL resolution.
func loadDLLFromSystem32(name string) (*windows.DLL, error) {
	return loadDLLWithFlags(name, windows.LOAD_LIBRARY_SEARCH_SYSTEM32)
}

func loadDLLFromPath(path string) (*windows.DLL, error) {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	return loadDLLWithFlags(absPath, windows.LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|windows.LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)
}

func loadDLLWithFlags(name string, flags uintptr) (*windows.DLL, error) {
	handle, err := windows.LoadLibraryEx(name, 0, flags)
	if err != nil {
		return nil, fmt.Errorf("failed to load %s: %w", name, err)
	}
	return &windows.DLL{Name: name, Handle: handle}, nil
}

func loadDLLFromDirs(names, dirs []string) (*windows.DLL, error) {
	var errs []string
	for _, name := range names {
		for _, dir := range dirs {
			path := filepath.Join(dir, name)
			if _, err := os.Stat(path); err != nil {
				continue
			}
			dll, err := loadDLLFromPath(path)
			if err == nil {
				return dll, nil
			}
			errs = append(errs, err.Error())
		}
	}
	if len(errs) == 0 {
		return nil, fmt.Errorf("no matching DLL found: %s", strings.Join(names, ", "))
	}
	return nil, errors.New(strings.Join(errs, "; "))
}

func nvidiaNVMLDirsWindows() []string {
	var dirs []string
	for _, root := range windowsProgramFilesDirs() {
		dirs = append(dirs, filepath.Join(root, "NVIDIA Corporation", "NVSMI"))
	}
	return uniqueAbsDirs(dirs)
}

func windowsProgramFilesDirs() []string {
	return uniqueAbsDirs([]string{
		os.Getenv("ProgramW6432"),
		os.Getenv("ProgramFiles"),
	})
}

func uniqueAbsDirs(dirs []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, dir := range dirs {
		if dir == "" {
			continue
		}
		absDir, err := filepath.Abs(dir)
		if err != nil {
			continue
		}
		absDir = filepath.Clean(absDir)
		key := strings.ToLower(absDir)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, absDir)
	}
	return out
}

func procAny(dll *windows.DLL, names ...string) (*windows.Proc, error) {
	var errs []string
	for _, name := range names {
		proc, err := dll.FindProc(name)
		if err == nil {
			return proc, nil
		}
		errs = append(errs, err.Error())
	}
	return nil, errors.New(strings.Join(errs, "; "))
}

//nolint:govet // Windows Proc.Call returns C string pointers as uintptr.
func windowsCString(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}
	return windows.BytePtrToString((*byte)(unsafe.Pointer(ptr)))
}

func byteCString(data []byte) string {
	for i, b := range data {
		if b == 0 {
			return string(data[:i])
		}
	}
	return string(data)
}

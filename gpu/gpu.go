//go:build linux || windows

package gpu

/*
#cgo linux LDFLAGS: -lrt -lpthread -ldl -lstdc++ -lm
#cgo windows LDFLAGS: -lpthread

#include "gpu_info.h"

*/
import "C"
import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
)

type cudaHandles struct {
	deviceCount int
	cudart      *C.cudart_handle_t
	nvcuda      *C.nvcuda_handle_t
	nvml        *C.nvml_handle_t
}

type oneapiHandles struct {
	oneapi      *C.oneapi_handle_t
	deviceCount int
}

const (
	cudaMinimumMemory = 457 * format.MebiByte
	rocmMinimumMemory = 457 * format.MebiByte
	// TODO OneAPI minimum memory
)

var (
	gpuMutex      sync.Mutex
	bootstrapped  bool
	cpuCapability CPUCapability
	cpus          []CPUInfo
	cudaGPUs      []CudaGPUInfo
	nvcudaLibPath string
	cudartLibPath string
	oneapiLibPath string
	nvmlLibPath   string
	rocmGPUs      []RocmGPUInfo
	oneapiGPUs    []OneapiGPUInfo
)

// With our current CUDA compile flags, older than 5.0 will not work properly
var CudaComputeMin = [2]C.int{5, 0}

var RocmComputeMin = 9

// TODO find a better way to detect iGPU instead of minimum memory
const IGPUMemLimit = 1 * format.GibiByte // 512G is what they typically report, so anything less than 1G must be iGPU

var CudartLinuxGlobs = []string{
	"/usr/local/cuda/lib64/libcudart.so*",
	"/usr/lib/x86_64-linux-gnu/nvidia/current/libcudart.so*",
	"/usr/lib/x86_64-linux-gnu/libcudart.so*",
	"/usr/lib/wsl/lib/libcudart.so*",
	"/usr/lib/wsl/drivers/*/libcudart.so*",
	"/opt/cuda/lib64/libcudart.so*",
	"/usr/local/cuda*/targets/aarch64-linux/lib/libcudart.so*",
	"/usr/lib/aarch64-linux-gnu/nvidia/current/libcudart.so*",
	"/usr/lib/aarch64-linux-gnu/libcudart.so*",
	"/usr/local/cuda/lib*/libcudart.so*",
	"/usr/lib*/libcudart.so*",
	"/usr/local/lib*/libcudart.so*",
}

var CudartWindowsGlobs = []string{
	"c:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*\\bin\\cudart64_*.dll",
}

var NvmlWindowsGlobs = []string{
	"c:\\Windows\\System32\\nvml.dll",
}

var NvcudaLinuxGlobs = []string{
	"/usr/local/cuda*/targets/*/lib/libcuda.so*",
	"/usr/lib/*-linux-gnu/nvidia/current/libcuda.so*",
	"/usr/lib/*-linux-gnu/libcuda.so*",
	"/usr/lib/wsl/lib/libcuda.so*",
	"/usr/lib/wsl/drivers/*/libcuda.so*",
	"/opt/cuda/lib*/libcuda.so*",
	"/usr/local/cuda/lib*/libcuda.so*",
	"/usr/lib*/libcuda.so*",
	"/usr/local/lib*/libcuda.so*",
}

var NvcudaWindowsGlobs = []string{
	"c:\\windows\\system*\\nvcuda.dll",
}

var OneapiWindowsGlobs = []string{
	"c:\\Windows\\System32\\DriverStore\\FileRepository\\*\\ze_intel_gpu64.dll",
}

var OneapiLinuxGlobs = []string{
	"/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so*",
	"/usr/lib*/libze_intel_gpu.so*",
}

// Jetson devices have JETSON_JETPACK="x.y.z" factory set to the Jetpack version installed.
// Included to drive logic for reducing Ollama-allocated overhead on L4T/Jetson devices.
var CudaTegra string = os.Getenv("JETSON_JETPACK")

// Note: gpuMutex must already be held
func initCudaHandles() *cudaHandles {

	// TODO - if the ollama build is CPU only, don't do these checks as they're irrelevant and confusing

	cHandles := &cudaHandles{}
	// Short Circuit if we already know which library to use
	if nvmlLibPath != "" {
		cHandles.nvml, _ = LoadNVMLMgmt([]string{nvmlLibPath})
		return cHandles
	}
	if nvcudaLibPath != "" {
		cHandles.deviceCount, cHandles.nvcuda, _ = LoadNVCUDAMgmt([]string{nvcudaLibPath})
		return cHandles
	}
	if cudartLibPath != "" {
		cHandles.deviceCount, cHandles.cudart, _ = LoadCUDARTMgmt([]string{cudartLibPath})
		return cHandles
	}

	slog.Debug("searching for GPU discovery libraries for NVIDIA")
	var cudartMgmtName string
	var cudartMgmtPatterns []string
	var nvcudaMgmtName string
	var nvcudaMgmtPatterns []string
	var nvmlMgmtName string
	var nvmlMgmtPatterns []string

	tmpDir, _ := PayloadsDir()
	switch runtime.GOOS {
	case "windows":
		cudartMgmtName = "cudart64_*.dll"
		localAppData := os.Getenv("LOCALAPPDATA")
		cudartMgmtPatterns = []string{filepath.Join(localAppData, "Programs", "Ollama", cudartMgmtName)}
		cudartMgmtPatterns = append(cudartMgmtPatterns, CudartWindowsGlobs...)
		// Aligned with driver, we can't carry as payloads
		nvcudaMgmtName = "nvcuda.dll"
		nvcudaMgmtPatterns = NvcudaWindowsGlobs

		// Use nvml to refresh free memory on windows only
		nvmlMgmtName = "nvml.dll"
		nvmlMgmtPatterns = make([]string, len(NvmlWindowsGlobs))
		copy(nvmlMgmtPatterns, NvmlWindowsGlobs)

	case "linux":
		cudartMgmtName = "libcudart.so*"
		if tmpDir != "" {
			// TODO - add "payloads" for subprocess
			cudartMgmtPatterns = []string{filepath.Join(tmpDir, "cuda*", cudartMgmtName)}
		}
		cudartMgmtPatterns = append(cudartMgmtPatterns, CudartLinuxGlobs...)
		// Aligned with driver, we can't carry as payloads
		nvcudaMgmtName = "libcuda.so*"
		nvcudaMgmtPatterns = NvcudaLinuxGlobs

		// nvml omitted on linux
	default:
		return cHandles
	}

	if len(nvmlMgmtPatterns) > 0 {
		nvmlLibPaths := FindGPULibs(nvmlMgmtName, nvmlMgmtPatterns)
		if len(nvmlLibPaths) > 0 {
			nvml, libPath := LoadNVMLMgmt(nvmlLibPaths)
			if nvml != nil {
				slog.Debug("nvidia-ml loaded", "library", libPath)
				cHandles.nvml = nvml
				nvmlLibPath = libPath
			}
		}
	}

	nvcudaLibPaths := FindGPULibs(nvcudaMgmtName, nvcudaMgmtPatterns)
	if len(nvcudaLibPaths) > 0 {
		deviceCount, nvcuda, libPath := LoadNVCUDAMgmt(nvcudaLibPaths)
		if nvcuda != nil {
			slog.Debug("detected GPUs", "count", deviceCount, "library", libPath)
			cHandles.nvcuda = nvcuda
			cHandles.deviceCount = deviceCount
			nvcudaLibPath = libPath
			return cHandles
		}
	}

	cudartLibPaths := FindGPULibs(cudartMgmtName, cudartMgmtPatterns)
	if len(cudartLibPaths) > 0 {
		deviceCount, cudart, libPath := LoadCUDARTMgmt(cudartLibPaths)
		if cudart != nil {
			slog.Debug("detected GPUs", "library", libPath, "count", deviceCount)
			cHandles.cudart = cudart
			cHandles.deviceCount = deviceCount
			cudartLibPath = libPath
			return cHandles
		}
	}

	return cHandles
}

// Note: gpuMutex must already be held
func initOneAPIHandles() *oneapiHandles {
	oHandles := &oneapiHandles{}
	var oneapiMgmtName string
	var oneapiMgmtPatterns []string

	// Short Circuit if we already know which library to use
	if oneapiLibPath != "" {
		oHandles.deviceCount, oHandles.oneapi, _ = LoadOneapiMgmt([]string{oneapiLibPath})
		return oHandles
	}

	switch runtime.GOOS {
	case "windows":
		oneapiMgmtName = "ze_intel_gpu64.dll"
		oneapiMgmtPatterns = OneapiWindowsGlobs
	case "linux":
		oneapiMgmtName = "libze_intel_gpu.so"
		oneapiMgmtPatterns = OneapiLinuxGlobs
	default:
		return oHandles
	}

	oneapiLibPaths := FindGPULibs(oneapiMgmtName, oneapiMgmtPatterns)
	if len(oneapiLibPaths) > 0 {
		oHandles.deviceCount, oHandles.oneapi, oneapiLibPath = LoadOneapiMgmt(oneapiLibPaths)
	}

	return oHandles
}

func GetGPUInfo() GpuInfoList {
	// TODO - consider exploring lspci (and equivalent on windows) to check for
	// GPUs so we can report warnings if we see Nvidia/AMD but fail to load the libraries
	gpuMutex.Lock()
	defer gpuMutex.Unlock()
	needRefresh := true
	var cHandles *cudaHandles
	var oHandles *oneapiHandles
	defer func() {
		if cHandles != nil {
			if cHandles.cudart != nil {
				C.cudart_release(*cHandles.cudart)
			}
			if cHandles.nvcuda != nil {
				C.nvcuda_release(*cHandles.nvcuda)
			}
			if cHandles.nvml != nil {
				C.nvml_release(*cHandles.nvml)
			}
		}
		if oHandles != nil {
			if oHandles.oneapi != nil {
				// TODO - is this needed?
				C.oneapi_release(*oHandles.oneapi)
			}
		}
	}()

	if !bootstrapped {
		slog.Debug("Detecting GPUs")
		needRefresh = false
		cpuCapability = getCPUCapability()
		var memInfo C.mem_info_t
		C.cpu_check_ram(&memInfo)
		if memInfo.err != nil {
			slog.Info("error looking up CPU memory", "error", C.GoString(memInfo.err))
			C.free(unsafe.Pointer(memInfo.err))
			return []GpuInfo{}
		}
		cpuInfo := CPUInfo{
			GpuInfo: GpuInfo{
				Library: "cpu",
				Variant: cpuCapability.ToVariant(),
			},
		}
		cpuInfo.TotalMemory = uint64(memInfo.total)
		cpuInfo.FreeMemory = uint64(memInfo.free)
		cpuInfo.ID = C.GoString(&memInfo.gpu_id[0])
		cpus = []CPUInfo{cpuInfo}

		// Fallback to CPU mode if we're lacking required vector extensions on x86
		if cpuCapability < GPURunnerCPUCapability && runtime.GOARCH == "amd64" {
			slog.Warn("CPU does not have minimum vector extensions, GPU inference disabled", "required", GPURunnerCPUCapability.ToString(), "detected", cpuCapability.ToString())
			bootstrapped = true
			// No need to do any GPU discovery, since we can't run on them
			return GpuInfoList{cpus[0].GpuInfo}
		}

		// On windows we bundle the nvidia library one level above the runner dir
		depPath := ""
		if runtime.GOOS == "windows" && envconfig.RunnersDir != "" {
			depPath = filepath.Dir(envconfig.RunnersDir)
		}

		// Load ALL libraries
		cHandles = initCudaHandles()

		// NVIDIA
		for i := range cHandles.deviceCount {
			if cHandles.cudart != nil || cHandles.nvcuda != nil {
				gpuInfo := CudaGPUInfo{
					GpuInfo: GpuInfo{
						Library: "cuda",
					},
					index: i,
				}
				var driverMajor int
				var driverMinor int
				if cHandles.cudart != nil {
					C.cudart_bootstrap(*cHandles.cudart, C.int(i), &memInfo)
				} else {
					C.nvcuda_bootstrap(*cHandles.nvcuda, C.int(i), &memInfo)
					driverMajor = int(cHandles.nvcuda.driver_major)
					driverMinor = int(cHandles.nvcuda.driver_minor)
				}
				if memInfo.err != nil {
					slog.Info("error looking up nvidia GPU memory", "error", C.GoString(memInfo.err))
					C.free(unsafe.Pointer(memInfo.err))
					continue
				}
				if memInfo.major < CudaComputeMin[0] || (memInfo.major == CudaComputeMin[0] && memInfo.minor < CudaComputeMin[1]) {
					slog.Info(fmt.Sprintf("[%d] CUDA GPU is too old. Compute Capability detected: %d.%d", i, memInfo.major, memInfo.minor))
					continue
				}
				gpuInfo.TotalMemory = uint64(memInfo.total)
				gpuInfo.FreeMemory = uint64(memInfo.free)
				gpuInfo.ID = C.GoString(&memInfo.gpu_id[0])
				gpuInfo.Compute = fmt.Sprintf("%d.%d", memInfo.major, memInfo.minor)
				gpuInfo.MinimumMemory = cudaMinimumMemory
				gpuInfo.DependencyPath = depPath
				gpuInfo.Name = C.GoString(&memInfo.gpu_name[0])
				gpuInfo.DriverMajor = int(driverMajor)
				gpuInfo.DriverMinor = int(driverMinor)

				// TODO potentially sort on our own algorithm instead of what the underlying GPU library does...
				cudaGPUs = append(cudaGPUs, gpuInfo)
			}
		}

		// Intel
		oHandles = initOneAPIHandles()
		for d := 0; oHandles.oneapi != nil && d < int(oHandles.oneapi.num_drivers); d++ {
			if oHandles.oneapi == nil {
				// shouldn't happen
				slog.Warn("nil oneapi handle with driver count", "count", int(oHandles.oneapi.num_drivers))
				continue
			}
			devCount := C.oneapi_get_device_count(*oHandles.oneapi, C.int(d))
			for i := 0; i < int(devCount); i++ {
				gpuInfo := OneapiGPUInfo{
					GpuInfo: GpuInfo{
						Library: "oneapi",
					},
					driverIndex: d,
					gpuIndex:    i,
				}
				// TODO - split bootstrapping from updating free memory
				C.oneapi_check_vram(*oHandles.oneapi, C.int(d), C.int(i), &memInfo)
				// TODO - convert this to MinimumMemory based on testing...
				var totalFreeMem float64 = float64(memInfo.free) * 0.95 // work-around: leave some reserve vram for mkl lib used in ggml-sycl backend.
				memInfo.free = C.uint64_t(totalFreeMem)
				gpuInfo.TotalMemory = uint64(memInfo.total)
				gpuInfo.FreeMemory = uint64(memInfo.free)
				gpuInfo.ID = C.GoString(&memInfo.gpu_id[0])
				gpuInfo.Name = C.GoString(&memInfo.gpu_name[0])
				// TODO dependency path?
				oneapiGPUs = append(oneapiGPUs, gpuInfo)
			}
		}

		rocmGPUs = AMDGetGPUInfo()
		bootstrapped = true
	}

	// For detected GPUs, load library if not loaded

	// Refresh free memory usage
	if needRefresh {
		// TODO - CPU system memory tracking/refresh
		var memInfo C.mem_info_t
		if cHandles == nil && len(cudaGPUs) > 0 {
			cHandles = initCudaHandles()
		}
		for i, gpu := range cudaGPUs {
			if cHandles.nvml != nil {
				C.nvml_get_free(*cHandles.nvml, C.int(gpu.index), &memInfo.free, &memInfo.total, &memInfo.used)
			} else if cHandles.cudart != nil {
				C.cudart_bootstrap(*cHandles.cudart, C.int(gpu.index), &memInfo)
			} else if cHandles.nvcuda != nil {
				C.nvcuda_get_free(*cHandles.nvcuda, C.int(gpu.index), &memInfo.free, &memInfo.total)
				memInfo.used = memInfo.total - memInfo.free
			} else {
				// shouldn't happen
				slog.Warn("no valid cuda library loaded to refresh vram usage")
				break
			}
			if memInfo.err != nil {
				slog.Warn("error looking up nvidia GPU memory", "error", C.GoString(memInfo.err))
				C.free(unsafe.Pointer(memInfo.err))
				continue
			}
			if memInfo.free == 0 {
				slog.Warn("error looking up nvidia GPU memory")
				continue
			}
			slog.Debug("updating cuda memory data",
				"gpu", gpu.ID,
				"name", gpu.Name,
				slog.Group(
					"before",
					"total", format.HumanBytes2(gpu.TotalMemory),
					"free", format.HumanBytes2(gpu.FreeMemory),
				),
				slog.Group(
					"now",
					"total", format.HumanBytes2(uint64(memInfo.total)),
					"free", format.HumanBytes2(uint64(memInfo.free)),
					"used", format.HumanBytes2(uint64(memInfo.used)),
				),
			)
			cudaGPUs[i].FreeMemory = uint64(memInfo.free)
		}

		if oHandles == nil && len(oneapiGPUs) > 0 {
			oHandles = initOneAPIHandles()
		}
		for i, gpu := range oneapiGPUs {
			if oHandles.oneapi == nil {
				// shouldn't happen
				slog.Warn("nil oneapi handle with device count", "count", oHandles.deviceCount)
				continue
			}
			C.oneapi_check_vram(*oHandles.oneapi, C.int(gpu.driverIndex), C.int(gpu.gpuIndex), &memInfo)
			// TODO - convert this to MinimumMemory based on testing...
			var totalFreeMem float64 = float64(memInfo.free) * 0.95 // work-around: leave some reserve vram for mkl lib used in ggml-sycl backend.
			memInfo.free = C.uint64_t(totalFreeMem)
			oneapiGPUs[i].FreeMemory = uint64(memInfo.free)
		}

		err := RocmGPUInfoList(rocmGPUs).RefreshFreeMemory()
		if err != nil {
			slog.Debug("problem refreshing ROCm free memory", "error", err)
		}
	}

	resp := []GpuInfo{}
	for _, gpu := range cudaGPUs {
		resp = append(resp, gpu.GpuInfo)
	}
	for _, gpu := range rocmGPUs {
		resp = append(resp, gpu.GpuInfo)
	}
	for _, gpu := range oneapiGPUs {
		resp = append(resp, gpu.GpuInfo)
	}
	if len(resp) == 0 {
		resp = append(resp, cpus[0].GpuInfo)
	}
	return resp
}

func GetCPUMem() (memInfo, error) {
	var ret memInfo
	var info C.mem_info_t
	C.cpu_check_ram(&info)
	if info.err != nil {
		defer C.free(unsafe.Pointer(info.err))
		return ret, fmt.Errorf(C.GoString(info.err))
	}
	ret.FreeMemory = uint64(info.free)
	ret.TotalMemory = uint64(info.total)
	return ret, nil
}

func FindGPULibs(baseLibName string, defaultPatterns []string) []string {
	// Multiple GPU libraries may exist, and some may not work, so keep trying until we exhaust them
	var ldPaths []string
	var patterns []string
	gpuLibPaths := []string{}
	slog.Debug("Searching for GPU library", "name", baseLibName)

	switch runtime.GOOS {
	case "windows":
		ldPaths = strings.Split(os.Getenv("PATH"), ";")
	case "linux":
		ldPaths = strings.Split(os.Getenv("LD_LIBRARY_PATH"), ":")
	default:
		return gpuLibPaths
	}
	// Start with whatever we find in the PATH/LD_LIBRARY_PATH
	for _, ldPath := range ldPaths {
		d, err := filepath.Abs(ldPath)
		if err != nil {
			continue
		}
		patterns = append(patterns, filepath.Join(d, baseLibName+"*"))
	}
	patterns = append(patterns, defaultPatterns...)
	slog.Debug("gpu library search", "globs", patterns)
	for _, pattern := range patterns {

		// Nvidia PhysX known to return bogus results
		if strings.Contains(pattern, "PhysX") {
			slog.Debug("skipping PhysX cuda library path", "path", pattern)
			continue
		}
		// Ignore glob discovery errors
		matches, _ := filepath.Glob(pattern)
		for _, match := range matches {
			// Resolve any links so we don't try the same lib multiple times
			// and weed out any dups across globs
			libPath := match
			tmp := match
			var err error
			for ; err == nil; tmp, err = os.Readlink(libPath) {
				if !filepath.IsAbs(tmp) {
					tmp = filepath.Join(filepath.Dir(libPath), tmp)
				}
				libPath = tmp
			}
			new := true
			for _, cmp := range gpuLibPaths {
				if cmp == libPath {
					new = false
					break
				}
			}
			if new {
				gpuLibPaths = append(gpuLibPaths, libPath)
			}
		}
	}
	slog.Debug("discovered GPU libraries", "paths", gpuLibPaths)
	return gpuLibPaths
}

func LoadCUDARTMgmt(cudartLibPaths []string) (int, *C.cudart_handle_t, string) {
	var resp C.cudart_init_resp_t
	resp.ch.verbose = getVerboseState()
	for _, libPath := range cudartLibPaths {
		lib := C.CString(libPath)
		defer C.free(unsafe.Pointer(lib))
		C.cudart_init(lib, &resp)
		if resp.err != nil {
			slog.Debug("Unable to load cudart", "library", libPath, "error", C.GoString(resp.err))
			C.free(unsafe.Pointer(resp.err))
		} else {
			return int(resp.num_devices), &resp.ch, libPath
		}
	}
	return 0, nil, ""
}

func LoadNVCUDAMgmt(nvcudaLibPaths []string) (int, *C.nvcuda_handle_t, string) {
	var resp C.nvcuda_init_resp_t
	resp.ch.verbose = getVerboseState()
	for _, libPath := range nvcudaLibPaths {
		lib := C.CString(libPath)
		defer C.free(unsafe.Pointer(lib))
		C.nvcuda_init(lib, &resp)
		if resp.err != nil {
			slog.Debug("Unable to load nvcuda", "library", libPath, "error", C.GoString(resp.err))
			C.free(unsafe.Pointer(resp.err))
		} else {
			return int(resp.num_devices), &resp.ch, libPath
		}
	}
	return 0, nil, ""
}

func LoadNVMLMgmt(nvmlLibPaths []string) (*C.nvml_handle_t, string) {
	var resp C.nvml_init_resp_t
	resp.ch.verbose = getVerboseState()
	for _, libPath := range nvmlLibPaths {
		lib := C.CString(libPath)
		defer C.free(unsafe.Pointer(lib))
		C.nvml_init(lib, &resp)
		if resp.err != nil {
			slog.Info(fmt.Sprintf("Unable to load NVML management library %s: %s", libPath, C.GoString(resp.err)))
			C.free(unsafe.Pointer(resp.err))
		} else {
			return &resp.ch, libPath
		}
	}
	return nil, ""
}

func LoadOneapiMgmt(oneapiLibPaths []string) (int, *C.oneapi_handle_t, string) {
	var resp C.oneapi_init_resp_t
	num_devices := 0
	resp.oh.verbose = getVerboseState()
	for _, libPath := range oneapiLibPaths {
		lib := C.CString(libPath)
		defer C.free(unsafe.Pointer(lib))
		C.oneapi_init(lib, &resp)
		if resp.err != nil {
			slog.Debug("Unable to load oneAPI management library", "library", libPath, "error", C.GoString(resp.err))
			C.free(unsafe.Pointer(resp.err))
		} else {
			for i := 0; i < int(resp.oh.num_drivers); i++ {
				num_devices += int(C.oneapi_get_device_count(resp.oh, C.int(i)))
			}
			return num_devices, &resp.oh, libPath
		}
	}
	return 0, nil, ""
}

func getVerboseState() C.uint16_t {
	if envconfig.Debug {
		return C.uint16_t(1)
	}
	return C.uint16_t(0)
}

// Given the list of GPUs this instantiation is targeted for,
// figure out the visible devices environment variable
//
// If different libraries are detected, the first one is what we use
func (l GpuInfoList) GetVisibleDevicesEnv() (string, string) {
	if len(l) == 0 {
		return "", ""
	}
	switch l[0].Library {
	case "cuda":
		return cudaGetVisibleDevicesEnv(l)
	case "rocm":
		return rocmGetVisibleDevicesEnv(l)
	case "oneapi":
		return oneapiGetVisibleDevicesEnv(l)
	default:
		slog.Debug("no filter required for library " + l[0].Library)
		return "", ""
	}
}

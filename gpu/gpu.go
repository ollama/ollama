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
	"strconv"
	"strings"
	"sync"
	"unsafe"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
)

type handles struct {
	deviceCount int
	cudart      *C.cudart_handle_t
	nvcuda      *C.nvcuda_handle_t
	oneapi      *C.oneapi_handle_t
}

const (
	cudaMinimumMemory = 457 * format.MebiByte
	rocmMinimumMemory = 457 * format.MebiByte
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
func initCudaHandles() *handles {

	// TODO - if the ollama build is CPU only, don't do these checks as they're irrelevant and confusing

	gpuHandles := &handles{}
	// Short Circuit if we already know which library to use
	if nvcudaLibPath != "" {
		gpuHandles.deviceCount, gpuHandles.nvcuda, _ = LoadNVCUDAMgmt([]string{nvcudaLibPath})
		return gpuHandles
	}
	if cudartLibPath != "" {
		gpuHandles.deviceCount, gpuHandles.cudart, _ = LoadCUDARTMgmt([]string{cudartLibPath})
		return gpuHandles
	}

	slog.Debug("searching for GPU discovery libraries for NVIDIA")
	var cudartMgmtName string
	var cudartMgmtPatterns []string
	var nvcudaMgmtName string
	var nvcudaMgmtPatterns []string
	var oneapiMgmtName string
	var oneapiMgmtPatterns []string

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
		oneapiMgmtName = "ze_intel_gpu64.dll"
		oneapiMgmtPatterns = OneapiWindowsGlobs
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
		oneapiMgmtName = "libze_intel_gpu.so"
		oneapiMgmtPatterns = OneapiLinuxGlobs
	default:
		return gpuHandles
	}

	nvcudaLibPaths := FindGPULibs(nvcudaMgmtName, nvcudaMgmtPatterns)
	if len(nvcudaLibPaths) > 0 {
		deviceCount, nvcuda, libPath := LoadNVCUDAMgmt(nvcudaLibPaths)
		if nvcuda != nil {
			slog.Debug("detected GPUs", "count", deviceCount, "library", libPath)
			gpuHandles.nvcuda = nvcuda
			gpuHandles.deviceCount = deviceCount
			nvcudaLibPath = libPath
			return gpuHandles
		}
	}

	cudartLibPaths := FindGPULibs(cudartMgmtName, cudartMgmtPatterns)
	if len(cudartLibPaths) > 0 {
		deviceCount, cudart, libPath := LoadCUDARTMgmt(cudartLibPaths)
		if cudart != nil {
			slog.Debug("detected GPUs", "library", libPath, "count", deviceCount)
			gpuHandles.cudart = cudart
			gpuHandles.deviceCount = deviceCount
			cudartLibPath = libPath
			return gpuHandles
		}
	}

	oneapiLibPaths := FindGPULibs(oneapiMgmtName, oneapiMgmtPatterns)
	if len(oneapiLibPaths) > 0 {
		deviceCount, oneapi, libPath := LoadOneapiMgmt(oneapiLibPaths)
		if oneapi != nil {
			slog.Debug("detected Intel GPUs", "library", libPath, "count", deviceCount)
			gpuHandles.oneapi = oneapi
			gpuHandles.deviceCount = deviceCount
			oneapiLibPath = libPath
			return gpuHandles
		}
	}

	return gpuHandles
}

func GetGPUInfo() GpuInfoList {
	// TODO - consider exploring lspci (and equivalent on windows) to check for
	// GPUs so we can report warnings if we see Nvidia/AMD but fail to load the libraries
	gpuMutex.Lock()
	defer gpuMutex.Unlock()
	needRefresh := true
	var gpuHandles *handles
	defer func() {
		if gpuHandles == nil {
			return
		}
		if gpuHandles.cudart != nil {
			C.cudart_release(*gpuHandles.cudart)
		}
		if gpuHandles.nvcuda != nil {
			C.nvcuda_release(*gpuHandles.nvcuda)
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

		// TODO - implement

		// TODO refine the discovery to only gather total memory

		// On windows we bundle the nvidia library one level above the runner dir
		depPath := ""
		if runtime.GOOS == "windows" && envconfig.RunnersDir != "" {
			depPath = filepath.Dir(envconfig.RunnersDir)
		}

		// Load ALL libraries
		gpuHandles = initCudaHandles()

		// TODO needs a refactoring pass to init oneapi handles

		// NVIDIA
		for i := range gpuHandles.deviceCount {
			if gpuHandles.cudart != nil || gpuHandles.nvcuda != nil {
				gpuInfo := CudaGPUInfo{
					GpuInfo: GpuInfo{
						Library: "cuda",
					},
					index: i,
				}
				var driverMajor int
				var driverMinor int
				if gpuHandles.cudart != nil {
					C.cudart_bootstrap(*gpuHandles.cudart, C.int(i), &memInfo)
				} else {
					C.nvcuda_bootstrap(*gpuHandles.nvcuda, C.int(i), &memInfo)
					driverMajor = int(gpuHandles.nvcuda.driver_major)
					driverMinor = int(gpuHandles.nvcuda.driver_minor)
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
			if gpuHandles.oneapi != nil {
				gpuInfo := OneapiGPUInfo{
					GpuInfo: GpuInfo{
						Library: "oneapi",
					},
					index: i,
				}
				// TODO - split bootstrapping from updating free memory
				C.oneapi_check_vram(*gpuHandles.oneapi, &memInfo)
				var totalFreeMem float64 = float64(memInfo.free) * 0.95 // work-around: leave some reserve vram for mkl lib used in ggml-sycl backend.
				memInfo.free = C.uint64_t(totalFreeMem)
				gpuInfo.TotalMemory = uint64(memInfo.total)
				gpuInfo.FreeMemory = uint64(memInfo.free)
				gpuInfo.ID = strconv.Itoa(i)
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
		if gpuHandles == nil && len(cudaGPUs) > 0 {
			gpuHandles = initCudaHandles()
		}
		for i, gpu := range cudaGPUs {
			if gpuHandles.cudart != nil {
				C.cudart_bootstrap(*gpuHandles.cudart, C.int(gpu.index), &memInfo)
			} else {
				C.nvcuda_get_free(*gpuHandles.nvcuda, C.int(gpu.index), &memInfo.free)
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
			slog.Debug("updating cuda free memory", "gpu", gpu.ID, "name", gpu.Name, "before", format.HumanBytes2(gpu.FreeMemory), "now", format.HumanBytes2(uint64(memInfo.free)))
			cudaGPUs[i].FreeMemory = uint64(memInfo.free)
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

func LoadOneapiMgmt(oneapiLibPaths []string) (int, *C.oneapi_handle_t, string) {
	var resp C.oneapi_init_resp_t
	resp.oh.verbose = getVerboseState()
	for _, libPath := range oneapiLibPaths {
		lib := C.CString(libPath)
		defer C.free(unsafe.Pointer(lib))
		C.oneapi_init(lib, &resp)
		if resp.err != nil {
			slog.Debug("Unable to load oneAPI management library", "library", libPath, "error", C.GoString(resp.err))
			C.free(unsafe.Pointer(resp.err))
		} else {
			return int(resp.num_devices), &resp.oh, libPath
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

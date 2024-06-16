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
	var cudartMgmtPatterns []string

	// Aligned with driver, we can't carry as payloads
	nvcudaMgmtPatterns := NvcudaGlobs

	if runtime.GOOS == "windows" {
		localAppData := os.Getenv("LOCALAPPDATA")
		cudartMgmtPatterns = []string{filepath.Join(localAppData, "Programs", "Ollama", CudartMgmtName)}
	}
	tmpDir, _ := PayloadsDir()
	if tmpDir != "" {
		// TODO - add "payloads" for subprocess
		cudartMgmtPatterns = []string{filepath.Join(tmpDir, "cuda*", CudartMgmtName)}
	}
	cudartMgmtPatterns = append(cudartMgmtPatterns, CudartGlobs...)

	if len(NvmlGlobs) > 0 {
		nvmlLibPaths := FindGPULibs(NvmlMgmtName, NvmlGlobs)
		if len(nvmlLibPaths) > 0 {
			nvml, libPath := LoadNVMLMgmt(nvmlLibPaths)
			if nvml != nil {
				slog.Debug("nvidia-ml loaded", "library", libPath)
				cHandles.nvml = nvml
				nvmlLibPath = libPath
			}
		}
	}

	nvcudaLibPaths := FindGPULibs(NvcudaMgmtName, nvcudaMgmtPatterns)
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

	cudartLibPaths := FindGPULibs(CudartMgmtName, cudartMgmtPatterns)
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

	// Short Circuit if we already know which library to use
	if oneapiLibPath != "" {
		oHandles.deviceCount, oHandles.oneapi, _ = LoadOneapiMgmt([]string{oneapiLibPath})
		return oHandles
	}

	oneapiLibPaths := FindGPULibs(OneapiMgmtName, OneapiGlobs)
	if len(oneapiLibPaths) > 0 {
		oHandles.deviceCount, oHandles.oneapi, oneapiLibPath = LoadOneapiMgmt(oneapiLibPaths)
	}

	return oHandles
}

func GetCPUInfo() GpuInfoList {
	gpuMutex.Lock()
	if !bootstrapped {
		gpuMutex.Unlock()
		GetGPUInfo()
	} else {
		gpuMutex.Unlock()
	}
	return GpuInfoList{cpus[0].GpuInfo}
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
		cpuCapability = GetCPUCapability()
		var memInfo C.mem_info_t

		mem, err := GetCPUMem()
		if err != nil {
			slog.Warn("error looking up system memory", "error", err)
		}
		cpus = []CPUInfo{CPUInfo{
			GpuInfo: GpuInfo{
				memInfo: mem,
				Library: "cpu",
				Variant: cpuCapability,
				ID:      "0",
			},
		}}

		// Fallback to CPU mode if we're lacking required vector extensions on x86
		if cpuCapability < GPURunnerCPUCapability && runtime.GOARCH == "amd64" {
			slog.Warn("CPU does not have minimum vector extensions, GPU inference disabled", "required", GPURunnerCPUCapability, "detected", cpuCapability)
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
				gpuInfo.DriverMajor = driverMajor
				gpuInfo.DriverMinor = driverMinor

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
			for i := range devCount {
				gpuInfo := OneapiGPUInfo{
					GpuInfo: GpuInfo{
						Library: "oneapi",
					},
					driverIndex: d,
					gpuIndex:    int(i),
				}
				// TODO - split bootstrapping from updating free memory
				C.oneapi_check_vram(*oHandles.oneapi, C.int(d), i, &memInfo)
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
		mem, err := GetCPUMem()
		if err != nil {
			slog.Warn("error looking up system memory", "error", err)
		} else {
			slog.Debug("updating system memory data",
				slog.Group(
					"before",
					"total", format.HumanBytes2(cpus[0].TotalMemory),
					"free", format.HumanBytes2(cpus[0].FreeMemory),
				),
				slog.Group(
					"now",
					"total", format.HumanBytes2(mem.TotalMemory),
					"free", format.HumanBytes2(mem.FreeMemory),
				),
			)
			cpus[0].FreeMemory = mem.FreeMemory
		}

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

		err = RocmGPUInfoList(rocmGPUs).RefreshFreeMemory()
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
			for i := range resp.oh.num_drivers {
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

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

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/server/envconfig"
)

type handles struct {
	deviceCount int
	cudart      *C.cudart_handle_t
	nvcuda      *C.nvcuda_handle_t
}

const (
	cudaMinimumMemory = 457 * format.MebiByte
	rocmMinimumMemory = 457 * format.MebiByte
)

var gpuMutex sync.Mutex

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

// Jetson devices have JETSON_JETPACK="x.y.z" factory set to the Jetpack version installed.
// Included to drive logic for reducing Ollama-allocated overhead on L4T/Jetson devices.
var CudaTegra string = os.Getenv("JETSON_JETPACK")

// Note: gpuMutex must already be held
func initGPUHandles() *handles {

	// TODO - if the ollama build is CPU only, don't do these checks as they're irrelevant and confusing

	gpuHandles := &handles{}
	var cudartMgmtName string
	var cudartMgmtPatterns []string
	var nvcudaMgmtName string
	var nvcudaMgmtPatterns []string

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
	default:
		return gpuHandles
	}

	slog.Debug("Detecting GPUs")
	nvcudaLibPaths := FindGPULibs(nvcudaMgmtName, nvcudaMgmtPatterns)
	if len(nvcudaLibPaths) > 0 {
		deviceCount, nvcuda, libPath := LoadNVCUDAMgmt(nvcudaLibPaths)
		if nvcuda != nil {
			slog.Debug("detected GPUs", "count", deviceCount, "library", libPath)
			gpuHandles.nvcuda = nvcuda
			gpuHandles.deviceCount = deviceCount
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

	gpuHandles := initGPUHandles()
	defer func() {
		if gpuHandles.cudart != nil {
			C.cudart_release(*gpuHandles.cudart)
		}
		if gpuHandles.nvcuda != nil {
			C.nvcuda_release(*gpuHandles.nvcuda)
		}
	}()

	// All our GPU builds on x86 have AVX enabled, so fallback to CPU if we don't detect at least AVX
	cpuVariant := GetCPUVariant()
	if cpuVariant == "" && runtime.GOARCH == "amd64" {
		slog.Warn("CPU does not have AVX or AVX2, disabling GPU support.")
	}

	// On windows we bundle the nvidia library one level above the runner dir
	depPath := ""
	if runtime.GOOS == "windows" && envconfig.RunnersDir != "" {
		depPath = filepath.Dir(envconfig.RunnersDir)
	}

	var memInfo C.mem_info_t
	resp := []GpuInfo{}

	// NVIDIA first
	for i := 0; i < gpuHandles.deviceCount; i++ {
		// TODO once we support CPU compilation variants of GPU libraries refine this...
		if cpuVariant == "" && runtime.GOARCH == "amd64" {
			continue
		}
		gpuInfo := GpuInfo{
			Library: "cuda",
		}
		var driverMajor int
		var driverMinor int
		if gpuHandles.cudart != nil {
			C.cudart_check_vram(*gpuHandles.cudart, C.int(i), &memInfo)
		} else {
			C.nvcuda_check_vram(*gpuHandles.nvcuda, C.int(i), &memInfo)
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
		resp = append(resp, gpuInfo)
	}

	// Then AMD
	resp = append(resp, AMDGetGPUInfo()...)

	if len(resp) == 0 {
		C.cpu_check_ram(&memInfo)
		if memInfo.err != nil {
			slog.Info("error looking up CPU memory", "error", C.GoString(memInfo.err))
			C.free(unsafe.Pointer(memInfo.err))
			return resp
		}
		gpuInfo := GpuInfo{
			Library: "cpu",
			Variant: cpuVariant,
		}
		gpuInfo.TotalMemory = uint64(memInfo.total)
		gpuInfo.FreeMemory = uint64(memInfo.free)
		gpuInfo.ID = C.GoString(&memInfo.gpu_id[0])

		resp = append(resp, gpuInfo)
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
	default:
		slog.Debug("no filter required for library " + l[0].Library)
		return "", ""
	}
}

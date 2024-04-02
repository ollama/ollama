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

	"github.com/ollama/ollama/format"
)

type handles struct {
	nvml   *C.nvml_handle_t
	cudart *C.cudart_handle_t
}

const (
	cudaMinimumMemory = 457 * format.MebiByte
	rocmMinimumMemory = 457 * format.MebiByte
)

var gpuMutex sync.Mutex

// With our current CUDA compile flags, older than 5.0 will not work properly
var CudaComputeMin = [2]C.int{5, 0}

// Possible locations for the nvidia-ml library
var NvmlLinuxGlobs = []string{
	"/usr/local/cuda/lib64/libnvidia-ml.so*",
	"/usr/lib/x86_64-linux-gnu/nvidia/current/libnvidia-ml.so*",
	"/usr/lib/x86_64-linux-gnu/libnvidia-ml.so*",
	"/usr/lib/wsl/lib/libnvidia-ml.so*",
	"/usr/lib/wsl/drivers/*/libnvidia-ml.so*",
	"/opt/cuda/lib64/libnvidia-ml.so*",
	"/usr/lib*/libnvidia-ml.so*",
	"/usr/lib/aarch64-linux-gnu/nvidia/current/libnvidia-ml.so*",
	"/usr/lib/aarch64-linux-gnu/libnvidia-ml.so*",
	"/usr/local/lib*/libnvidia-ml.so*",

	// TODO: are these stubs ever valid?
	"/opt/cuda/targets/x86_64-linux/lib/stubs/libnvidia-ml.so*",
}

var NvmlWindowsGlobs = []string{
	"c:\\Windows\\System32\\nvml.dll",
}

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

// Jetson devices have JETSON_JETPACK="x.y.z" factory set to the Jetpack version installed.
// Included to drive logic for reducing Ollama-allocated overhead on L4T/Jetson devices.
var CudaTegra string = os.Getenv("JETSON_JETPACK")

// Note: gpuMutex must already be held
func initGPUHandles() *handles {

	// TODO - if the ollama build is CPU only, don't do these checks as they're irrelevant and confusing

	gpuHandles := &handles{nil, nil}
	var nvmlMgmtName string
	var nvmlMgmtPatterns []string
	var cudartMgmtName string
	var cudartMgmtPatterns []string

	tmpDir, _ := PayloadsDir()
	switch runtime.GOOS {
	case "windows":
		nvmlMgmtName = "nvml.dll"
		nvmlMgmtPatterns = make([]string, len(NvmlWindowsGlobs))
		copy(nvmlMgmtPatterns, NvmlWindowsGlobs)
		cudartMgmtName = "cudart64_*.dll"
		localAppData := os.Getenv("LOCALAPPDATA")
		cudartMgmtPatterns = []string{filepath.Join(localAppData, "Programs", "Ollama", cudartMgmtName)}
		cudartMgmtPatterns = append(cudartMgmtPatterns, CudartWindowsGlobs...)
	case "linux":
		nvmlMgmtName = "libnvidia-ml.so"
		nvmlMgmtPatterns = make([]string, len(NvmlLinuxGlobs))
		copy(nvmlMgmtPatterns, NvmlLinuxGlobs)
		cudartMgmtName = "libcudart.so*"
		if tmpDir != "" {
			// TODO - add "payloads" for subprocess
			cudartMgmtPatterns = []string{filepath.Join(tmpDir, "cuda*", cudartMgmtName)}
		}
		cudartMgmtPatterns = append(cudartMgmtPatterns, CudartLinuxGlobs...)
	default:
		return gpuHandles
	}

	slog.Info("Detecting GPU type")
	cudartLibPaths := FindGPULibs(cudartMgmtName, cudartMgmtPatterns)
	if len(cudartLibPaths) > 0 {
		cudart := LoadCUDARTMgmt(cudartLibPaths)
		if cudart != nil {
			slog.Info("Nvidia GPU detected via cudart")
			gpuHandles.cudart = cudart
			return gpuHandles
		}
	}

	// TODO once we build confidence, remove this and the gpu_info_nvml.[ch] files
	nvmlLibPaths := FindGPULibs(nvmlMgmtName, nvmlMgmtPatterns)
	if len(nvmlLibPaths) > 0 {
		nvml := LoadNVMLMgmt(nvmlLibPaths)
		if nvml != nil {
			slog.Info("Nvidia GPU detected via nvidia-ml")
			gpuHandles.nvml = nvml
			return gpuHandles
		}
	}
	return gpuHandles
}

func GetGPUInfo() GpuInfo {
	// TODO - consider exploring lspci (and equivalent on windows) to check for
	// GPUs so we can report warnings if we see Nvidia/AMD but fail to load the libraries
	gpuMutex.Lock()
	defer gpuMutex.Unlock()

	gpuHandles := initGPUHandles()
	defer func() {
		if gpuHandles.nvml != nil {
			C.nvml_release(*gpuHandles.nvml)
		}
		if gpuHandles.cudart != nil {
			C.cudart_release(*gpuHandles.cudart)
		}
	}()

	// All our GPU builds on x86 have AVX enabled, so fallback to CPU if we don't detect at least AVX
	cpuVariant := GetCPUVariant()
	if cpuVariant == "" && runtime.GOARCH == "amd64" {
		slog.Warn("CPU does not have AVX or AVX2, disabling GPU support.")
	}

	var memInfo C.mem_info_t
	resp := GpuInfo{}
	if gpuHandles.nvml != nil && (cpuVariant != "" || runtime.GOARCH != "amd64") {
		C.nvml_check_vram(*gpuHandles.nvml, &memInfo)
		if memInfo.err != nil {
			slog.Info(fmt.Sprintf("[nvidia-ml] error looking up NVML GPU memory: %s", C.GoString(memInfo.err)))
			C.free(unsafe.Pointer(memInfo.err))
		} else if memInfo.count > 0 {
			// Verify minimum compute capability
			var cc C.nvml_compute_capability_t
			C.nvml_compute_capability(*gpuHandles.nvml, &cc)
			if cc.err != nil {
				slog.Info(fmt.Sprintf("[nvidia-ml] error looking up NVML GPU compute capability: %s", C.GoString(cc.err)))
				C.free(unsafe.Pointer(cc.err))
			} else if cc.major > CudaComputeMin[0] || (cc.major == CudaComputeMin[0] && cc.minor >= CudaComputeMin[1]) {
				slog.Info(fmt.Sprintf("[nvidia-ml] NVML CUDA Compute Capability detected: %d.%d", cc.major, cc.minor))
				resp.Library = "cuda"
				resp.MinimumMemory = cudaMinimumMemory
			} else {
				slog.Info(fmt.Sprintf("[nvidia-ml] CUDA GPU is too old. Falling back to CPU mode. Compute Capability detected: %d.%d", cc.major, cc.minor))
			}
		}
	} else if gpuHandles.cudart != nil && (cpuVariant != "" || runtime.GOARCH != "amd64") {
		C.cudart_check_vram(*gpuHandles.cudart, &memInfo)
		if memInfo.err != nil {
			slog.Info(fmt.Sprintf("[cudart] error looking up CUDART GPU memory: %s", C.GoString(memInfo.err)))
			C.free(unsafe.Pointer(memInfo.err))
		} else if memInfo.count > 0 {
			// Verify minimum compute capability
			var cc C.cudart_compute_capability_t
			C.cudart_compute_capability(*gpuHandles.cudart, &cc)
			if cc.err != nil {
				slog.Info(fmt.Sprintf("[cudart] error looking up CUDA compute capability: %s", C.GoString(cc.err)))
				C.free(unsafe.Pointer(cc.err))
			} else if cc.major > CudaComputeMin[0] || (cc.major == CudaComputeMin[0] && cc.minor >= CudaComputeMin[1]) {
				slog.Info(fmt.Sprintf("[cudart] CUDART CUDA Compute Capability detected: %d.%d", cc.major, cc.minor))
				resp.Library = "cuda"
				resp.MinimumMemory = cudaMinimumMemory
			} else {
				slog.Info(fmt.Sprintf("[cudart] CUDA GPU is too old. Falling back to CPU mode. Compute Capability detected: %d.%d", cc.major, cc.minor))
			}
		}
	} else {
		AMDGetGPUInfo(&resp)
		if resp.Library != "" {
			resp.MinimumMemory = rocmMinimumMemory
			return resp
		}
	}
	if resp.Library == "" {
		C.cpu_check_ram(&memInfo)
		resp.Library = "cpu"
		resp.Variant = cpuVariant
	}
	if memInfo.err != nil {
		slog.Info(fmt.Sprintf("error looking up CPU memory: %s", C.GoString(memInfo.err)))
		C.free(unsafe.Pointer(memInfo.err))
		return resp
	}

	resp.DeviceCount = uint32(memInfo.count)
	resp.FreeMemory = uint64(memInfo.free)
	resp.TotalMemory = uint64(memInfo.total)
	return resp
}

func getCPUMem() (memInfo, error) {
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

func CheckVRAM() (int64, error) {
	userLimit := os.Getenv("OLLAMA_MAX_VRAM")
	if userLimit != "" {
		avail, err := strconv.ParseInt(userLimit, 10, 64)
		if err != nil {
			return 0, fmt.Errorf("Invalid OLLAMA_MAX_VRAM setting %s: %s", userLimit, err)
		}
		slog.Info(fmt.Sprintf("user override OLLAMA_MAX_VRAM=%d", avail))
		return avail, nil
	}
	gpuInfo := GetGPUInfo()
	if gpuInfo.FreeMemory > 0 && (gpuInfo.Library == "cuda" || gpuInfo.Library == "rocm") {
		return int64(gpuInfo.FreeMemory), nil
	}

	return 0, fmt.Errorf("no GPU detected") // TODO - better handling of CPU based memory determiniation
}

func FindGPULibs(baseLibName string, patterns []string) []string {
	// Multiple GPU libraries may exist, and some may not work, so keep trying until we exhaust them
	var ldPaths []string
	gpuLibPaths := []string{}
	slog.Info(fmt.Sprintf("Searching for GPU management library %s", baseLibName))

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
	slog.Debug(fmt.Sprintf("gpu management search paths: %v", patterns))
	for _, pattern := range patterns {
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
	slog.Info(fmt.Sprintf("Discovered GPU libraries: %v", gpuLibPaths))
	return gpuLibPaths
}

func LoadNVMLMgmt(nvmlLibPaths []string) *C.nvml_handle_t {
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
			return &resp.ch
		}
	}
	return nil
}

func LoadCUDARTMgmt(cudartLibPaths []string) *C.cudart_handle_t {
	var resp C.cudart_init_resp_t
	resp.ch.verbose = getVerboseState()
	for _, libPath := range cudartLibPaths {
		lib := C.CString(libPath)
		defer C.free(unsafe.Pointer(lib))
		C.cudart_init(lib, &resp)
		if resp.err != nil {
			slog.Info(fmt.Sprintf("Unable to load cudart CUDA management library %s: %s", libPath, C.GoString(resp.err)))
			C.free(unsafe.Pointer(resp.err))
		} else {
			return &resp.ch
		}
	}
	return nil
}

func getVerboseState() C.uint16_t {
	if debug := os.Getenv("OLLAMA_DEBUG"); debug != "" {
		return C.uint16_t(1)
	}
	return C.uint16_t(0)
}

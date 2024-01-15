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
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"unsafe"
)

type handles struct {
	cuda *C.cuda_handle_t
	rocm *C.rocm_handle_t
}

var gpuMutex sync.Mutex
var gpuHandles *handles = nil

// With our current CUDA compile flags, 5.2 and older will not work properly
const CudaComputeMajorMin = 6

// Possible locations for the nvidia-ml library
var CudaLinuxGlobs = []string{
	"/usr/local/cuda/lib64/libnvidia-ml.so*",
	"/usr/lib/x86_64-linux-gnu/nvidia/current/libnvidia-ml.so*",
	"/usr/lib/x86_64-linux-gnu/libnvidia-ml.so*",
	"/usr/lib/wsl/lib/libnvidia-ml.so*",
	"/opt/cuda/lib64/libnvidia-ml.so*",
	"/usr/lib*/libnvidia-ml.so*",
	"/usr/local/lib*/libnvidia-ml.so*",
	"/usr/lib/aarch64-linux-gnu/nvidia/current/libnvidia-ml.so*",
	"/usr/lib/aarch64-linux-gnu/libnvidia-ml.so*",
}

var CudaWindowsGlobs = []string{
	"c:\\Windows\\System32\\nvml.dll",
}

var RocmLinuxGlobs = []string{
	"/opt/rocm*/lib*/librocm_smi64.so*",
}

var RocmWindowsGlobs = []string{
	"c:\\Windows\\System32\\rocm_smi64.dll",
}

// Note: gpuMutex must already be held
func initGPUHandles() {

	// TODO - if the ollama build is CPU only, don't do these checks as they're irrelevant and confusing

	var cudaMgmtName string
	var cudaMgmtPatterns []string
	var rocmMgmtName string
	var rocmMgmtPatterns []string
	switch runtime.GOOS {
	case "windows":
		cudaMgmtName = "nvml.dll"
		cudaMgmtPatterns = make([]string, len(CudaWindowsGlobs))
		copy(cudaMgmtPatterns, CudaWindowsGlobs)
		rocmMgmtName = "rocm_smi64.dll"
		rocmMgmtPatterns = make([]string, len(RocmWindowsGlobs))
		copy(rocmMgmtPatterns, RocmWindowsGlobs)
	case "linux":
		cudaMgmtName = "libnvidia-ml.so"
		cudaMgmtPatterns = make([]string, len(CudaLinuxGlobs))
		copy(cudaMgmtPatterns, CudaLinuxGlobs)
		rocmMgmtName = "librocm_smi64.so"
		rocmMgmtPatterns = make([]string, len(RocmLinuxGlobs))
		copy(rocmMgmtPatterns, RocmLinuxGlobs)
	default:
		return
	}

	log.Printf("Detecting GPU type")
	gpuHandles = &handles{nil, nil}
	cudaLibPaths := FindGPULibs(cudaMgmtName, cudaMgmtPatterns)
	if len(cudaLibPaths) > 0 {
		cuda := LoadCUDAMgmt(cudaLibPaths)
		if cuda != nil {
			log.Printf("Nvidia GPU detected")
			gpuHandles.cuda = cuda
			return
		}
	}

	rocmLibPaths := FindGPULibs(rocmMgmtName, rocmMgmtPatterns)
	if len(rocmLibPaths) > 0 {
		rocm := LoadROCMMgmt(rocmLibPaths)
		if rocm != nil {
			log.Printf("Radeon GPU detected")
			gpuHandles.rocm = rocm
			return
		}
	}
}

func GetGPUInfo() GpuInfo {
	// TODO - consider exploring lspci (and equivalent on windows) to check for
	// GPUs so we can report warnings if we see Nvidia/AMD but fail to load the libraries
	gpuMutex.Lock()
	defer gpuMutex.Unlock()
	if gpuHandles == nil {
		initGPUHandles()
	}

	var memInfo C.mem_info_t
	resp := GpuInfo{}
	if gpuHandles.cuda != nil {
		C.cuda_check_vram(*gpuHandles.cuda, &memInfo)
		if memInfo.err != nil {
			log.Printf("error looking up CUDA GPU memory: %s", C.GoString(memInfo.err))
			C.free(unsafe.Pointer(memInfo.err))
		} else {
			// Verify minimum compute capability
			var cc C.cuda_compute_capability_t
			C.cuda_compute_capability(*gpuHandles.cuda, &cc)
			if cc.err != nil {
				log.Printf("error looking up CUDA GPU compute capability: %s", C.GoString(cc.err))
				C.free(unsafe.Pointer(cc.err))
			} else if cc.major >= CudaComputeMajorMin {
				log.Printf("CUDA Compute Capability detected: %d.%d", cc.major, cc.minor)
				resp.Library = "cuda"
			} else {
				log.Printf("CUDA GPU is too old. Falling back to CPU mode. Compute Capability detected: %d.%d", cc.major, cc.minor)
			}
		}
	} else if gpuHandles.rocm != nil {
		C.rocm_check_vram(*gpuHandles.rocm, &memInfo)
		if memInfo.err != nil {
			log.Printf("error looking up ROCm GPU memory: %s", C.GoString(memInfo.err))
			C.free(unsafe.Pointer(memInfo.err))
		} else {
			resp.Library = "rocm"
			var version C.rocm_version_resp_t
			C.rocm_get_version(*gpuHandles.rocm, &version)
			verString := C.GoString(version.str)
			if version.status == 0 {
				resp.Variant = "v" + verString
			} else {
				log.Printf("failed to look up ROCm version: %s", verString)
			}
			C.free(unsafe.Pointer(version.str))
		}
	}
	if resp.Library == "" {
		C.cpu_check_ram(&memInfo)
		resp.Library = "cpu"
		resp.Variant = GetCPUVariant()
	}
	if memInfo.err != nil {
		log.Printf("error looking up CPU memory: %s", C.GoString(memInfo.err))
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
	gpuInfo := GetGPUInfo()
	if gpuInfo.FreeMemory > 0 && (gpuInfo.Library == "cuda" || gpuInfo.Library == "rocm") {
		// leave 10% or 512MiB of VRAM free per GPU to handle unaccounted for overhead
		overhead := gpuInfo.FreeMemory / 10
		gpus := uint64(gpuInfo.DeviceCount)
		if overhead < gpus*512*1024*1024 {
			overhead = gpus * 512 * 1024 * 1024
		}
		return int64(gpuInfo.FreeMemory - overhead), nil
	}

	return 0, fmt.Errorf("no GPU detected") // TODO - better handling of CPU based memory determiniation
}

func FindGPULibs(baseLibName string, patterns []string) []string {
	// Multiple GPU libraries may exist, and some may not work, so keep trying until we exhaust them
	var ldPaths []string
	gpuLibPaths := []string{}
	log.Printf("Searching for GPU management library %s", baseLibName)

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
	log.Printf("Discovered GPU libraries: %v", gpuLibPaths)
	return gpuLibPaths
}

func LoadCUDAMgmt(cudaLibPaths []string) *C.cuda_handle_t {
	var resp C.cuda_init_resp_t
	for _, libPath := range cudaLibPaths {
		lib := C.CString(libPath)
		defer C.free(unsafe.Pointer(lib))
		C.cuda_init(lib, &resp)
		if resp.err != nil {
			log.Printf("Unable to load CUDA management library %s: %s", libPath, C.GoString(resp.err))
			C.free(unsafe.Pointer(resp.err))
		} else {
			return &resp.ch
		}
	}
	return nil
}

func LoadROCMMgmt(rocmLibPaths []string) *C.rocm_handle_t {
	var resp C.rocm_init_resp_t
	for _, libPath := range rocmLibPaths {
		lib := C.CString(libPath)
		defer C.free(unsafe.Pointer(lib))
		C.rocm_init(lib, &resp)
		if resp.err != nil {
			log.Printf("Unable to load ROCm management library %s: %s", libPath, C.GoString(resp.err))
			C.free(unsafe.Pointer(resp.err))
		} else {
			return &resp.rh
		}
	}
	return nil
}

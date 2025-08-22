package discover

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ollama/ollama/format"
)

// Jetson devices have JETSON_JETPACK="x.y.z" factory set to the Jetpack version installed.
// Included to drive logic for reducing Ollama-allocated overhead on L4T/Jetson devices.
var CudaTegra string = os.Getenv("JETSON_JETPACK")

func GetCPUInfo() GpuInfo {
	mem, err := GetCPUMem()
	if err != nil {
		slog.Warn("error looking up system memory", "error", err)
	}

	return GpuInfo{
		memInfo: mem,
		Library: "cpu",
		ID:      "0",
	}
}

func GetGPUInfo() GpuInfoList {
	resp := []GpuInfo{}

	// Our current packaging model places ggml-hip in the main directory
	// but keeps rocm in an isolated directory.  We have to add it to
	// the [LD_LIBRARY_]PATH so ggml-hip will load properly
	rocmDir := filepath.Join(LibOllamaPath, "rocm")
	if _, err := os.Stat(rocmDir); err != nil {
		rocmDir = ""
	}

	for _, dev := range GPUDevices() {
		info := GpuInfo{
			ID:   dev.ID,
			Name: dev.Description,
			memInfo: memInfo{
				TotalMemory: dev.TotalMemory,
				FreeMemory:  dev.FreeMemory,
			},
			Library: dev.Library,
			// TODO can we avoid variant
			DependencyPath: dev.LibraryPath,
			DriverMajor:    dev.DriverMajor,
			DriverMinor:    dev.DriverMinor,
		}
		if dev.Library == "CUDA" || dev.Library == "HIP" {
			info.MinimumMemory = 457 * format.MebiByte
		}
		if dev.Library == "HIP" {
			info.Compute = fmt.Sprintf("gfx%x%02x", dev.ComputeMajor, dev.ComputeMinor)
			if rocmDir != "" {
				info.DependencyPath = append(info.DependencyPath, rocmDir)
			}
		} else {
			info.Compute = fmt.Sprintf("%d.%d", dev.ComputeMajor, dev.ComputeMinor)
		}
		resp = append(resp, info)
	}
	if len(resp) == 0 {
		mem, err := GetCPUMem()
		if err != nil {
			slog.Warn("error looking up system memory", "error", err)
		}

		resp = append(resp, GpuInfo{
			memInfo: mem,
			Library: "cpu",
			ID:      "0",
		})
	}
	return resp
}

// Given the list of GPUs this instantiation is targeted for,
// figure out the visible devices environment variable
//
// # If different libraries are detected, the first one is what we use
//
// TODO once we're purely running on the new runner, this level of device
// filtering will no longer be necessary.  Instead the runner can be told which
// of the set of GPUs to utilize and handle filtering itself, instead of relying
// on the env var to hide devices from the underlying GPU libraries
func (l GpuInfoList) GetVisibleDevicesEnv() (string, string) {
	if len(l) == 0 {
		return nil
	}
	switch l[0].Library {
	case "cuda", "CUDA":
		return cudaGetVisibleDevicesEnv(l)
	case "rocm", "HIP":
		return rocmGetVisibleDevicesEnv(l)
	default:
		slog.Debug("no filter required for library " + l[0].Library)
		return "", ""
	}
	return vd
}

func rocmGetVisibleDevicesEnv(gpuInfo []GpuInfo) (string, string) {
	ids := []string{}
	for _, info := range gpuInfo {
		if info.Library != "rocm" {
			// TODO shouldn't happen if things are wired correctly...
			slog.Debug("rocmGetVisibleDevicesEnv skipping over non-rocm device", "library", info.Library)
			continue
		}
		ids = append(ids, info.ID)
	}
	var envVar string
	if runtime.GOOS != "linux" {
		envVar = "HIP_VISIBLE_DEVICES"
	} else {
		envVar = "ROCR_VISIBLE_DEVICES"
	}
	// There are 3 potential env vars to use to select GPUs.
	// ROCR_VISIBLE_DEVICES supports UUID or numeric but does not work on Windows
	// HIP_VISIBLE_DEVICES supports numeric IDs only
	// GPU_DEVICE_ORDINAL supports numeric IDs only
	return envVar, strings.Join(ids, ",")
}

func cudaGetVisibleDevicesEnv(gpuInfo []GpuInfo) (string, string) {
	ids := []string{}
	for _, info := range gpuInfo {
		if info.Library != "cuda" {
			// TODO shouldn't happen if things are wired correctly...
			slog.Debug("cudaGetVisibleDevicesEnv skipping over non-cuda device", "library", info.Library)
			continue
		}
		ids = append(ids, info.ID)
	}
	return "CUDA_VISIBLE_DEVICES", strings.Join(ids, ",")
}

func GetSystemInfo() SystemInfo {
	gpus := GetGPUInfo()
	if len(gpus) == 1 && gpus[0].Library == "cpu" {
		gpus = []GpuInfo{}
	}
	details, err := GetCPUDetails()
	if err != nil {
		slog.Warn("failed to look up CPU details", "error", err)
	}

	sys := CPUInfo{
		CPUs:    details,
		GpuInfo: GetCPUInfo(),
	}

	return SystemInfo{
		System: sys,
		GPUs:   gpus,
	}
}

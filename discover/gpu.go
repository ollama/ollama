package discover

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
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
		DeviceID: ml.DeviceID{
			Library: "cpu",
			ID:      "0",
		},
	}
}

func GetGPUInfo(ctx context.Context, runners []FilteredRunnerDiscovery) GpuInfoList {
	devs := GPUDevices(ctx, runners)
	return devInfoToInfoList(devs)
}

func devInfoToInfoList(devs []ml.DeviceInfo) GpuInfoList {
	resp := []GpuInfo{}
	// Our current packaging model places ggml-hip in the main directory
	// but keeps rocm in an isolated directory.  We have to add it to
	// the [LD_LIBRARY_]PATH so ggml-hip will load properly
	rocmDir := filepath.Join(LibOllamaPath, "rocm")
	if _, err := os.Stat(rocmDir); err != nil {
		rocmDir = ""
	}

	for _, dev := range devs {
		info := GpuInfo{
			DeviceID: dev.DeviceID,
			filterID: dev.FilteredID,
			Name:     dev.Description,
			memInfo: memInfo{
				TotalMemory: dev.TotalMemory,
				FreeMemory:  dev.FreeMemory,
			},
			// TODO can we avoid variant
			DependencyPath: dev.LibraryPath,
			DriverMajor:    dev.DriverMajor,
			DriverMinor:    dev.DriverMinor,
		}
		if dev.Library == "CUDA" || dev.Library == "ROCm" {
			info.MinimumMemory = 457 * format.MebiByte
		}
		if dev.Library == "ROCm" {
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
			DeviceID: ml.DeviceID{
				Library: "cpu",
				ID:      "0",
			},
		})
	}
	return resp
}

// Given the list of GPUs this instantiation is targeted for,
// figure out the visible devices environment variable
//
// If different libraries are detected, the first one is what we use
func (l GpuInfoList) GetVisibleDevicesEnv() []string {
	if len(l) == 0 {
		return nil
	}
	return []string{rocmGetVisibleDevicesEnv(l)}
}

func rocmGetVisibleDevicesEnv(gpuInfo []GpuInfo) string {
	ids := []string{}
	for _, info := range gpuInfo {
		if info.Library != "ROCm" {
			continue
		}
		// If the devices requires a numeric ID, for filtering purposes, we use the unfiltered ID number
		if info.filterID != "" {
			ids = append(ids, info.filterID)
		} else {
			ids = append(ids, info.ID)
		}
	}
	if len(ids) == 0 {
		return ""
	}
	envVar := "ROCR_VISIBLE_DEVICES="
	if runtime.GOOS != "linux" {
		envVar = "HIP_VISIBLE_DEVICES="
	}
	// There are 3 potential env vars to use to select GPUs.
	// ROCR_VISIBLE_DEVICES supports UUID or numeric but does not work on Windows
	// HIP_VISIBLE_DEVICES supports numeric IDs only
	// GPU_DEVICE_ORDINAL supports numeric IDs only
	return envVar + strings.Join(ids, ",")
}

// GetSystemInfo returns the last cached state of the GPUs on the system
func GetSystemInfo() SystemInfo {
	deviceMu.Lock()
	defer deviceMu.Unlock()
	gpus := devInfoToInfoList(devices)
	if len(gpus) == 1 && gpus[0].Library == "cpu" {
		gpus = []GpuInfo{}
	}

	return SystemInfo{
		System: CPUInfo{
			CPUs:    GetCPUDetails(),
			GpuInfo: GetCPUInfo(),
		},
		GPUs: gpus,
	}
}

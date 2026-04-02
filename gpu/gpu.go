package gpu

import (
	"fmt"
	"log/slog"
	"runtime"
	"strconv"
	"strings"
)

type GpuInfo struct {
	Library string `json:"library,omitempty"`
	ID      string `json:"id"`
	Name    string `json:"name"`
	Compute string `json:"compute,omitempty"`
	Drivers []string `json:"drivers,omitempty"`
	Meminfo
}

type Meminfo struct {
	TotalMemory uint64 `json:"total_memory,omitempty"`
	FreeMemory  uint64 `json:"free_memory,omitempty"`
}

type GpuInfoList []GpuInfo

type MemoryInfo struct {
	TotalMemory uint64
	FreeMemory  uint64
}

// GetGPUInfo returns GPU information
func GetGPUInfo() GpuInfoList {
	var gpuInfo GpuInfoList

	switch runtime.GOOS {
	case "linux":
		gpuInfo = append(gpuInfo, getCudaGPUs()...)
		gpuInfo = append(gpuInfo, getRocmGPUs()...)
	case "windows":
		gpuInfo = append(gpuInfo, getCudaGPUs()...)
	case "darwin":
		gpuInfo = append(gpuInfo, getMetalGPUs()...)
	}

	return gpuInfo
}

// CheckVRAM returns the available VRAM in bytes
func CheckVRAM() (memInfo MemoryInfo, err error) {
	gpuInfo := GetGPUInfo()
	if len(gpuInfo) == 0 {
		return memInfo, fmt.Errorf("no GPU detected")
	}

	// Calculate total memory across all GPUs
	var totalMemory, freeMemory uint64
	for _, gpu := range gpuInfo {
		totalMemory += gpu.TotalMemory
		freeMemory += gpu.FreeMemory
	}

	// Apply conservative memory calculation to prevent OOM
	// Reserve 10% of total memory for system overhead
	reservedMemory := totalMemory / 10
	if freeMemory > reservedMemory {
		freeMemory -= reservedMemory
	} else {
		freeMemory = 0
	}

	memInfo.TotalMemory = totalMemory
	memInfo.FreeMemory = freeMemory

	slog.Debug("GPU memory info", "total", formatBytes(totalMemory), "free", formatBytes(freeMemory), "reserved", formatBytes(reservedMemory))

	return memInfo, nil
}

// EstimateGPULayers estimates how many layers can fit in GPU memory
func EstimateGPULayers(gpuMemory, layerSize uint64, numLayers int) int {
	if layerSize == 0 {
		return 0
	}

	// Calculate how many layers can fit
	layersInGPU := int(gpuMemory / layerSize)

	// Don't exceed the total number of layers
	if layersInGPU > numLayers {
		layersInGPU = numLayers
	}

	return layersInGPU
}

func formatBytes(bytes uint64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := uint64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// parseMemoryString parses memory strings like "24576 MiB" into bytes
func parseMemoryString(memStr string) (uint64, error) {
	parts := strings.Fields(strings.TrimSpace(memStr))
	if len(parts) != 2 {
		return 0, fmt.Errorf("invalid memory format: %s", memStr)
	}

	value, err := strconv.ParseFloat(parts[0], 64)
	if err != nil {
		return 0, fmt.Errorf("invalid memory value: %s", parts[0])
	}

	unit := strings.ToUpper(parts[1])
	var multiplier uint64

	switch unit {
	case "B":
		multiplier = 1
	case "KB", "KIB":
		multiplier = 1024
	case "MB", "MIB":
		multiplier = 1024 * 1024
	case "GB", "GIB":
		multiplier = 1024 * 1024 * 1024
	case "TB", "TIB":
		multiplier = 1024 * 1024 * 1024 * 1024
	default:
		return 0, fmt.Errorf("unknown memory unit: %s", unit)
	}

	return uint64(value * float64(multiplier)), nil
}
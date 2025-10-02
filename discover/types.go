package discover

import (
	"context"
	"log/slog"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

type memInfo struct {
	TotalMemory uint64 `json:"total_memory,omitempty"`
	FreeMemory  uint64 `json:"free_memory,omitempty"`
	FreeSwap    uint64 `json:"free_swap,omitempty"` // TODO split this out for system only
}

// Beginning of an `ollama info` command
type GpuInfo struct { // TODO better name maybe "InferenceProcessor"?
	ml.DeviceID
	memInfo

	// Optional variant to select (e.g. versions, cpu feature flags)
	Variant string `json:"variant"`

	// MinimumMemory represents the minimum memory required to use the GPU
	MinimumMemory uint64 `json:"-"`

	// Any extra PATH/LD_LIBRARY_PATH dependencies required for the Library to operate properly
	DependencyPath []string `json:"lib_path,omitempty"`

	// Set to true if we can NOT reliably discover FreeMemory.  A value of true indicates
	// the FreeMemory is best effort, and may over or under report actual memory usage
	// False indicates FreeMemory can generally be trusted on this GPU
	UnreliableFreeMemory bool

	// GPU information
	filterID string // AMD Workaround: The numeric ID of the device used to filter out other devices
	Name     string `json:"name"`    // user friendly name if available
	Compute  string `json:"compute"` // Compute Capability or gfx

	// Driver Information - TODO no need to put this on each GPU
	DriverMajor int `json:"driver_major,omitempty"`
	DriverMinor int `json:"driver_minor,omitempty"`

	// TODO other performance capability info to help in scheduling decisions
}

func (gpu GpuInfo) RunnerName() string {
	if gpu.Variant != "" {
		return gpu.Library + "_" + gpu.Variant
	}
	return gpu.Library
}

type CPUInfo struct {
	GpuInfo
	CPUs []CPU
}

// CPU type represents a CPU Package occupying a socket
type CPU struct {
	ID                  string `cpuinfo:"processor"`
	VendorID            string `cpuinfo:"vendor_id"`
	ModelName           string `cpuinfo:"model name"`
	CoreCount           int
	EfficiencyCoreCount int // Performance = CoreCount - Efficiency
	ThreadCount         int
}

type GpuInfoList []GpuInfo

func (l GpuInfoList) ByLibrary() []GpuInfoList {
	resp := []GpuInfoList{}
	libs := []string{}
	for _, info := range l {
		found := false
		requested := info.Library
		if info.Variant != "" {
			requested += "_" + info.Variant
		}
		for i, lib := range libs {
			if lib == requested {
				resp[i] = append(resp[i], info)
				found = true
				break
			}
		}
		if !found {
			libs = append(libs, requested)
			resp = append(resp, []GpuInfo{info})
		}
	}
	return resp
}

func LogDetails(devices []ml.DeviceInfo) {
	for _, dev := range devices {
		var libs []string
		for _, dir := range dev.LibraryPath {
			if strings.Contains(dir, filepath.Join("lib", "ollama")) {
				libs = append(libs, filepath.Base(dir))
			}
		}
		typeStr := "discrete"
		if dev.Integrated {
			typeStr = "iGPU"
		}
		slog.Info("inference compute",
			"id", dev.ID,
			"library", dev.Library,
			"compute", dev.Compute(),
			"name", dev.Name,
			"description", dev.Description,
			"libdirs", strings.Join(libs, ","),
			"driver", dev.Driver(),
			"pci_id", dev.PCIID,
			"type", typeStr,
			"total", format.HumanBytes2(dev.TotalMemory),
			"available", format.HumanBytes2(dev.FreeMemory),
		)
	}
	// CPU inference
	if len(devices) == 0 {
		dev, _ := GetCPUMem()
		slog.Info("inference compute",
			"id", "cpu",
			"library", "cpu",
			"compute", "",
			"name", "cpu",
			"description", "cpu",
			"libdirs", "ollama",
			"driver", "",
			"pci_id", "",
			"type", "",
			"total", format.HumanBytes2(dev.TotalMemory),
			"available", format.HumanBytes2(dev.FreeMemory),
		)
	}
}

// Sort by Free Space
type ByFreeMemory []GpuInfo

func (a ByFreeMemory) Len() int           { return len(a) }
func (a ByFreeMemory) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByFreeMemory) Less(i, j int) bool { return a[i].FreeMemory < a[j].FreeMemory }

type SystemInfo struct {
	System CPUInfo   `json:"system"`
	GPUs   []GpuInfo `json:"gpus"`
}

// Return the optimal number of threads to use for inference
func (si SystemInfo) GetOptimalThreadCount() int {
	if len(si.System.CPUs) == 0 {
		// Fall back to Go's num CPU
		return runtime.NumCPU()
	}

	coreCount := 0
	for _, c := range si.System.CPUs {
		coreCount += c.CoreCount - c.EfficiencyCoreCount
	}

	return coreCount
}

// For each GPU, check if it does NOT support flash attention
func (l GpuInfoList) FlashAttentionSupported() bool {
	for _, gpu := range l {
		supportsFA := gpu.Library == "cpu" ||
			gpu.Name == "Metal" || gpu.Library == "Metal" ||
			(gpu.Library == "CUDA" && gpu.DriverMajor >= 7) ||
			gpu.Library == "ROCm"

		if !supportsFA {
			return false
		}
	}
	return true
}

type BaseRunner interface {
	// GetPort returns the localhost port number the runner is running on
	GetPort() int

	// HasExited indicates if the runner is no longer running.  This can be used during
	// bootstrap to detect if a given filtered device is incompatible and triggered an assert
	HasExited() bool
}

type RunnerDiscovery interface {
	BaseRunner

	// GetDeviceInfos will perform a query of the underlying device libraries
	// for device identification and free VRAM information
	// During bootstrap scenarios, this routine may take seconds to complete
	GetDeviceInfos(ctx context.Context) []ml.DeviceInfo
}

type FilteredRunnerDiscovery interface {
	RunnerDiscovery

	// GetActiveDeviceIDs returns the filtered set of devices actively in
	// use by this runner for running models.  If the runner is a bootstrap runner, no devices
	// will be active yet so no device IDs are returned.
	// This routine will not query the underlying device and will return immediately
	GetActiveDeviceIDs() []ml.DeviceID
}

package discover

import (
	"fmt"
	"log/slog"
	"sort"
	"strings"

	"github.com/ollama/ollama/format"
)

type memInfo struct {
	TotalMemory uint64 `json:"total_memory,omitempty"`
	FreeMemory  uint64 `json:"free_memory,omitempty"`
	FreeSwap    uint64 `json:"free_swap,omitempty"` // TODO split this out for system only
}

// Beginning of an `ollama info` command
type GpuInfo struct { // TODO better name maybe "InferenceProcessor"?
	memInfo
	Library string `json:"library,omitempty"`

	// Optional variant to select (e.g. versions, cpu feature flags)
	Variant string `json:"variant"`

	// MinimumMemory represents the minimum memory required to use the GPU
	MinimumMemory uint64 `json:"-"`

	// Any extra PATH/LD_LIBRARY_PATH dependencies required for the Library to operate properly
	DependencyPath []string `json:"lib_path,omitempty"`

	// Extra environment variables specific to the GPU as list of [key,value]
	EnvWorkarounds [][2]string `json:"envs,omitempty"`

	// Set to true if we can NOT reliably discover FreeMemory.  A value of true indicates
	// the FreeMemory is best effort, and may over or under report actual memory usage
	// False indicates FreeMemory can generally be trusted on this GPU
	UnreliableFreeMemory bool

	// GPU information
	ID      string `json:"gpu_id"`  // string to use for selection of this specific GPU
	Name    string `json:"name"`    // user friendly name if available
	Compute string `json:"compute"` // Compute Capability or gfx

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

type CudaGPUInfo struct {
	GpuInfo
	OSOverhead   uint64 // Memory overhead between the driver library and management library
	index        int    //nolint:unused,nolintlint
	computeMajor int    //nolint:unused,nolintlint
	computeMinor int    //nolint:unused,nolintlint
}
type CudaGPUInfoList []CudaGPUInfo

type RocmGPUInfo struct {
	GpuInfo
	usedFilepath string //nolint:unused,nolintlint
	index        int    //nolint:unused,nolintlint
}
type RocmGPUInfoList []RocmGPUInfo

type OneapiGPUInfo struct {
	GpuInfo
	driverIndex int //nolint:unused,nolintlint
	gpuIndex    int //nolint:unused,nolintlint
}
type OneapiGPUInfoList []OneapiGPUInfo

type GpuInfoList []GpuInfo

type UnsupportedGPUInfo struct {
	GpuInfo
	Reason string `json:"reason"`
}

// Split up the set of gpu info's by Library
// This assumes the oldest version is compatible with the newest card, which may
// not be the case if the user has a very new and very old GPU
func (l GpuInfoList) ByLibrary() []GpuInfoList {
	resp := []GpuInfoList{}
	libs := []string{}
	for _, info := range l {
		found := false
		for i, lib := range libs {
			if lib == info.Library {
				resp[i] = append(resp[i], info)
				found = true
				break
			}
		}
		if !found {
			libs = append(libs, info.Library)
			resp = append(resp, []GpuInfo{info})
		}
	}
	return resp
}

// Report the GPU information into the log an Info level
func (l GpuInfoList) LogDetails() {
	for _, g := range l {
		slog.Info("inference compute",
			"id", g.ID,
			"library", g.Library,
			"variant", g.Variant,
			"compute", g.Compute,
			"driver", fmt.Sprintf("%d.%d", g.DriverMajor, g.DriverMinor),
			"name", g.Name,
			"total", format.HumanBytes2(g.TotalMemory),
			"available", format.HumanBytes2(g.FreeMemory),
		)
	}
}

// Sort by Free Space
type ByFreeMemory []GpuInfo

func (a ByFreeMemory) Len() int           { return len(a) }
func (a ByFreeMemory) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByFreeMemory) Less(i, j int) bool { return a[i].FreeMemory < a[j].FreeMemory }

// Sort by Variant
type ByVariant []GpuInfo

func (a ByVariant) Len() int           { return len(a) }
func (a ByVariant) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByVariant) Less(i, j int) bool { return strings.Compare(a[i].Variant, a[j].Variant) < 0 } // TODO do better than alpha sort

type SystemInfo struct {
	System          CPUInfo              `json:"system"`
	GPUs            []GpuInfo            `json:"gpus"`
	UnsupportedGPUs []UnsupportedGPUInfo `json:"unsupported_gpus"`
	DiscoveryErrors []string             `json:"discovery_errors"`
}

// Return the optimal number of threads to use for inference
func (si SystemInfo) GetOptimalThreadCount() int {
	if len(si.System.CPUs) == 0 {
		return 0
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
		supportsFA := gpu.Library == "metal" ||
			(gpu.Library == "cuda" && gpu.DriverMajor >= 7) ||
			gpu.Library == "rocm"

		if !supportsFA {
			return false
		}
	}
	return true
}

func (l GpuInfoList) BestRunnerName() string {
	if len(l) == 0 {
		return ""
	}
	// Sort by variant, which will yield the oldest variant first
	sgl := append(make(GpuInfoList, 0, len(l)), l...)
	sort.Sort(ByVariant(sgl))
	info := sgl[0]

	requested := info.Library
	if info.Variant != "" {
		requested += "_" + info.Variant
	}
	return requested
}

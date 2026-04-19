package autotune

import (
	"fmt"
	"runtime"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/ml"
)

// HardwareProfile is the result of probing system hardware.
type HardwareProfile struct {
	CPUs     []discover.CPU
	GPUs     []ml.DeviceInfo
	System   ml.SystemInfo
	Platform PlatformInfo
}

// PlatformInfo captures OS-level details relevant to tuning.
type PlatformInfo struct {
	OS   string // "windows", "linux", "darwin"
	Arch string // "amd64", "arm64"
}

// DetectHardware builds a HardwareProfile from live system data.
// gpus should be the result of a previous discover.GPUDevices call;
// pass nil to skip GPU info (CPU-only mode).
func DetectHardware(gpus []ml.DeviceInfo) HardwareProfile {
	cpus := discover.GetCPUDetails()
	sysInfo := discover.GetSystemInfo()

	return HardwareProfile{
		CPUs:   cpus,
		GPUs:   gpus,
		System: sysInfo,
		Platform: PlatformInfo{
			OS:   runtime.GOOS,
			Arch: runtime.GOARCH,
		},
	}
}

// ---------- Derived helpers ----------

// TotalVRAM returns the sum of TotalMemory across all discrete GPUs.
func (h *HardwareProfile) TotalVRAM() uint64 {
	var total uint64
	for _, g := range h.GPUs {
		if !g.Integrated {
			total += g.TotalMemory
		}
	}
	return total
}

// FreeVRAM returns the sum of FreeMemory across all discrete GPUs.
func (h *HardwareProfile) FreeVRAM() uint64 {
	var total uint64
	for _, g := range h.GPUs {
		if !g.Integrated {
			total += g.FreeMemory
		}
	}
	return total
}

// DiscreteGPUCount returns the number of discrete GPUs.
func (h *HardwareProfile) DiscreteGPUCount() int {
	n := 0
	for _, g := range h.GPUs {
		if !g.Integrated {
			n++
		}
	}
	return n
}

// HasGPU returns true if at least one discrete GPU is available.
func (h *HardwareProfile) HasGPU() bool {
	return h.DiscreteGPUCount() > 0
}

// FlashAttentionCapable returns true if all GPUs support flash attention.
func (h *HardwareProfile) FlashAttentionCapable() bool {
	if len(h.GPUs) == 0 {
		return false
	}
	return ml.FlashAttentionSupported(h.GPUs)
}

// SmallestGPUVRAM returns the VRAM of the smallest discrete GPU.
func (h *HardwareProfile) SmallestGPUVRAM() uint64 {
	var smallest uint64
	for _, g := range h.GPUs {
		if g.Integrated {
			continue
		}
		if smallest == 0 || g.TotalMemory < smallest {
			smallest = g.TotalMemory
		}
	}
	return smallest
}

// Summary returns a human-readable one-liner.
func (h *HardwareProfile) Summary() string {
	cpuName := "unknown CPU"
	if len(h.CPUs) > 0 {
		cpuName = h.CPUs[0].ModelName
	}
	gpuStr := "no GPU"
	if n := h.DiscreteGPUCount(); n > 0 {
		gpuStr = fmt.Sprintf("%d GPU(s), %.1f GB VRAM",
			n, float64(h.TotalVRAM())/(1024*1024*1024))
	}
	return fmt.Sprintf("%s | %s | %.1f GB RAM | %s/%s",
		cpuName, gpuStr,
		float64(h.System.TotalMemory)/(1024*1024*1024),
		h.Platform.OS, h.Platform.Arch)
}

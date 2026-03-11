package fitcheck

import (
	"context"
	"runtime"
	"sort"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/ml"
)

// HardwareProfile is the consolidated view of this machine's capabilities.
type HardwareProfile struct {
	// GPUs is the list of detected GPU devices.
	GPUs []ml.DeviceInfo `json:"gpus"`

	// BestGPU is the GPU with the highest FreeMemory; nil if no GPU detected.
	BestGPU *ml.DeviceInfo `json:"best_gpu,omitempty"`

	// RAMTotalBytes is the total system RAM in bytes.
	RAMTotalBytes uint64 `json:"ram_total_bytes"`

	// RAMAvailableBytes is the currently available system RAM in bytes.
	RAMAvailableBytes uint64 `json:"ram_available_bytes"`

	// DiskModelPathBytes is the total capacity of the filesystem hosting the models directory.
	DiskModelPathBytes uint64 `json:"disk_model_path_bytes"`

	// DiskModelAvailBytes is the available space on the filesystem hosting the models directory.
	DiskModelAvailBytes uint64 `json:"disk_model_avail_bytes"`

	// ModelsDir is the path to the Ollama models directory.
	ModelsDir string `json:"models_dir"`

	// OS is the operating system (runtime.GOOS).
	OS string `json:"os"`

	// Arch is the CPU architecture (runtime.GOARCH).
	Arch string `json:"arch"`
}

// Collect gathers hardware facts. Completes in under 3 seconds.
// modelsDir should be the path returned by envconfig.Models().
func Collect(modelsDir string) (HardwareProfile, error) {
	var hw HardwareProfile
	hw.ModelsDir = modelsDir
	hw.OS = runtime.GOOS
	hw.Arch = runtime.GOARCH

	// RAM via discover.GetSystemInfo() which returns ml.SystemInfo
	sysInfo := discover.GetSystemInfo()
	hw.RAMTotalBytes = sysInfo.TotalMemory
	hw.RAMAvailableBytes = sysInfo.FreeMemory

	// GPU discovery — pass a background context and no active runners (discovery mode only).
	gpus := discover.GPUDevices(context.Background(), nil)
	hw.GPUs = gpus

	// Pick the best GPU (highest FreeMemory among non-integrated GPUs, falling
	// back to any GPU if all are integrated).
	sort.Slice(hw.GPUs, func(i, j int) bool {
		return hw.GPUs[i].FreeMemory > hw.GPUs[j].FreeMemory
	})
	for i := range hw.GPUs {
		if hw.GPUs[i].FreeMemory > 0 {
			hw.BestGPU = &hw.GPUs[i]
			break
		}
	}
	if hw.BestGPU == nil && len(hw.GPUs) > 0 {
		hw.BestGPU = &hw.GPUs[0]
	}

	// Disk stats — platform-specific helper defined in disk_unix.go / disk_windows.go
	total, avail, err := diskStats(modelsDir)
	if err == nil {
		hw.DiskModelPathBytes = total
		hw.DiskModelAvailBytes = avail
	}

	return hw, nil
}

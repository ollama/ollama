//go:build darwin

package gpu

import "C"
import (
	"runtime"

	"github.com/jmorganca/ollama/api"
)

// CheckVRAM returns the free VRAM in bytes on Linux machines with NVIDIA GPUs
func CheckVRAM() (int64, error) {
	// TODO - assume metal, and return free memory?
	return 0, nil

}

func GetGPUInfo() GpuInfo {
	// TODO - Metal vs. x86 macs...
	mem, _ := getCPUMem()
	return GpuInfo{
		Library: "default",
		memInfo: mem,
	}
}

func getCPUMem() (memInfo, error) {
	return memInfo{
		TotalMemory: 0,
		FreeMemory:  0,
	}, nil
}

func NumGPU(numLayer, fileSizeBytes int64, opts api.Options) int {
	if opts.NumGPU != -1 {
		return opts.NumGPU
	}

	// metal only supported on arm64
	if runtime.GOARCH == "arm64" {
		return 1
	}

	return 0
}

func nativeInit() error {
	return nil
}

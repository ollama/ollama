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

	return GpuInfo{
		Driver:      "METAL",
		Library:     "default",
		TotalMemory: 0,
		FreeMemory:  0,
	}
}

func NumGPU(numLayer, fileSizeBytes int64, opts api.Options) int {
	if runtime.GOARCH == "arm64" {
		return 1
	}

	// metal only supported on arm64
	return 0
}

func nativeInit() error {
	return nil
}

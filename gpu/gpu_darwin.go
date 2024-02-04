//go:build darwin

package gpu

import "C"
import (
	"runtime"
	"github.com/pbnjay/memory"
	"os/exec"
	"strconv"
	"strings"
	"fmt"
	"log/slog"
)

// CheckVRAM returns the free VRAM in bytes on Linux machines with NVIDIA GPUs
func CheckVRAM() (int64, error) {
	if runtime.GOARCH == "amd64" {
		// gpu not supported, this may not be metal
		return 0, nil
	}
	
	var wiredLimit int64
 
	// on macOS the user can set the upper limit for allocated vram by the
	// kernel variable iogpu.wired_limit_mb
	sysctl := exec.Command("sysctl", "-n", "iogpu.wired_limit_mb")
    sysctlOut, err := sysctl.Output()
    if err != nil {
		slog.Info(fmt.Sprintf("error reading iogpu.wired_limit_mb: %s", err.Error()))
		wiredLimit = 0
	}else{
		sysctlOutString := strings.TrimSpace(string(sysctlOut))
		sysctlOutInt, err := strconv.ParseInt(sysctlOutString, 10, 64)
		if err != nil {
			slog.Info(fmt.Sprintf("error parsing sysctl output: %s", err.Error()))
			wiredLimit = 0
		}else{
			// convert mb to byte
			wiredLimit = 1024 * 1024 * sysctlOutInt
		}
	}
	
	slog.Info(fmt.Sprintf("iogpu wired limit in bytes: %d", wiredLimit))
	
	if  wiredLimit == 0 {
		// use the system default limits

		// on macOS, there's already buffer for available vram (see below) so just return the total
		systemMemory := int64(memory.TotalMemory())

		// macOS limits how much memory is available to the GPU based on the amount of system memory
		if systemMemory <= 36*1024*1024*1024 {
			wiredLimit = systemMemory * 2 / 3
		} else {
			wiredLimit = systemMemory * 3 / 4
		}
	}
	
	slog.Info(fmt.Sprintf("vram limit in bytes: %d", wiredLimit))

	return wiredLimit, nil
}

func GetGPUInfo() GpuInfo {
	mem, _ := getCPUMem()
	if runtime.GOARCH == "amd64" {
		return GpuInfo{
			Library: "cpu",
			Variant: GetCPUVariant(),
			memInfo: mem,
		}
	}
	return GpuInfo{
		Library: "metal",
		memInfo: mem,
	}
}

func getCPUMem() (memInfo, error) {
	return memInfo{
		TotalMemory: 0,
		FreeMemory:  0,
		DeviceCount: 0,
	}, nil
}

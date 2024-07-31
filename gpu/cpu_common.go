<<<<<<< HEAD
package gpu

import (
	"log/slog"

	"golang.org/x/sys/cpu"
)

func GetCPUVariant() string {
	if cpu.X86.HasAVX2 {
		slog.Debug("CPU has AVX2")
		return "avx2"
	}
	if cpu.X86.HasAVX {
		slog.Debug("CPU has AVX")
		return "avx"
	}
	slog.Debug("CPU does not have vector extensions")
	// else LCD
	return ""
}
=======
package gpu

import (
	"golang.org/x/sys/cpu"
)

func GetCPUCapability() CPUCapability {
	if cpu.X86.HasAVX2 {
		return CPUCapabilityAVX2
	}
	if cpu.X86.HasAVX {
		return CPUCapabilityAVX
	}
	// else LCD
	return CPUCapabilityNone
}
>>>>>>> 0b01490f7a487eae06890c5eabcd2270e32605a5

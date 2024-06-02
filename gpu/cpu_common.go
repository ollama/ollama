package gpu

import (
	"log/slog"

	"golang.org/x/sys/cpu"
)

func GetCPUVariant() string {
	// Check for x86 features
	if cpu.X86.HasAVX2 {
		slog.Debug("CPU has AVX2")
		return "avx2"
	}
	if cpu.X86.HasAVX {
		slog.Debug("CPU has AVX")
		return "avx"
	}

	// Check for ARM64 features
	if cpu.ARM64.HasASIMD {
		slog.Debug("CPU has NEON (ASIMD)")
		return "asimd"
	}
	if cpu.ARM64.HasSVE {
		slog.Debug("CPU has SVE")
		return "sve"
	}

	slog.Debug("CPU does not have recognized vector extensions")
	// else LCD
	return ""
}

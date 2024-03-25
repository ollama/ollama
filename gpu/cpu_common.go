package gpu

import (
	"log/slog"

	"golang.org/x/sys/cpu"
)

func GetCPUVariant() string {
	if cpu.X86.HasAVX2 {
		slog.Info("CPU has AVX2")
		return "avx2"
	}
	if cpu.X86.HasAVX {
		slog.Info("CPU has AVX")
		return "avx"
	}
	slog.Info("CPU does not have vector extensions")
	// else LCD
	return ""
}

package gpu

import (
	"golang.org/x/sys/cpu"
)

func GetCPUVariant() string {
	return getCPUCapability().ToVariant()
}

func getCPUCapability() CPUCapability {
	if cpu.X86.HasAVX2 {
		return CPUCapabilityAVX2
	}
	if cpu.X86.HasAVX {
		return CPUCapabilityAVX
	}
	// else LCD
	return CPUCapabilityBase
}

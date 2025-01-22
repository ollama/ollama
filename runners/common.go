package runners

import (
	"golang.org/x/sys/cpu"
)

type CPUCapability uint32

// Override at build time when building base GPU runners
// var GPURunnerCPUCapability = CPUCapabilityAVX

const (
	CPUCapabilityNone CPUCapability = iota
	CPUCapabilityAVX
	CPUCapabilityAVX2
	// TODO AVX512
)

func (c CPUCapability) String() string {
	switch c {
	case CPUCapabilityAVX:
		return "avx"
	case CPUCapabilityAVX2:
		return "avx2"
	default:
		return "no vector extensions"
	}
}

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

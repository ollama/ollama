package gpu

import (
	"log"

	"golang.org/x/sys/cpu"
)

func GetCPUVariant() string {
	if cpu.X86.HasAVX2 {
		log.Printf("CPU has AVX2")
		return "avx2"
	}
	if cpu.X86.HasAVX {
		log.Printf("CPU has AVX")
		return "avx"
	}
	log.Printf("CPU does not have vector extensions")
	// else LCD
	return ""
}

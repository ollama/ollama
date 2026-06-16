package server

import (
	"testing"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

// availableMemoryForGPU must not clamp a dedicated BIOS carveout (e.g. Strix Halo,
// where the GPU carveout is larger than host RAM) down to host free memory, while
// still clamping a genuine host-shared iGPU whose pool is carved from host RAM.
func TestAvailableMemoryForGPUCarveout(t *testing.T) {
	cases := []struct {
		name string
		sys  ml.SystemInfo
		gpu  ml.DeviceInfo
		want uint64
	}{
		{
			name: "large carveout APU not clamped to host RAM",
			sys:  ml.SystemInfo{TotalMemory: 32 * format.GigaByte, FreeMemory: 4 * format.GigaByte},
			gpu:  ml.DeviceInfo{Integrated: true, TotalMemory: 96 * format.GigaByte, FreeMemory: 72 * format.GigaByte},
			want: 72 * format.GigaByte,
		},
		{
			name: "small UMA iGPU still clamped to host free",
			sys:  ml.SystemInfo{TotalMemory: 16 * format.GigaByte, FreeMemory: 6 * format.GigaByte},
			gpu:  ml.DeviceInfo{Integrated: true, TotalMemory: 16 * format.GigaByte, FreeMemory: 12 * format.GigaByte},
			want: 6 * format.GigaByte,
		},
		{
			name: "discrete GPU ignores host free",
			sys:  ml.SystemInfo{TotalMemory: 16 * format.GigaByte, FreeMemory: 2 * format.GigaByte},
			gpu:  ml.DeviceInfo{Integrated: false, TotalMemory: 24 * format.GigaByte, FreeMemory: 20 * format.GigaByte},
			want: 20 * format.GigaByte,
		},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if got := availableMemoryForGPU(tt.sys, tt.gpu); got != tt.want {
				t.Fatalf("availableMemoryForGPU = %d, want %d", got, tt.want)
			}
		})
	}
}

// availableMemoryForLoad must treat a dedicated carveout APU like discrete VRAM so
// the shared-memory host-free bound does not collapse its budget.
func TestAvailableMemoryForLoadCarveout(t *testing.T) {
	sys := ml.SystemInfo{TotalMemory: 32 * format.GigaByte, FreeMemory: 4 * format.GigaByte}
	gpus := []ml.DeviceInfo{{Integrated: true, TotalMemory: 96 * format.GigaByte, FreeMemory: 72 * format.GigaByte}}

	available, gpuFree, systemLimited := availableMemoryForLoad(sys, gpus)
	if available != 72*format.GigaByte {
		t.Fatalf("available = %d, want %d", available, 72*format.GigaByte)
	}
	if gpuFree != 72*format.GigaByte {
		t.Fatalf("gpuFree = %d, want %d", gpuFree, 72*format.GigaByte)
	}
	if systemLimited {
		t.Fatalf("systemLimited = true, want false for a dedicated carveout")
	}
}

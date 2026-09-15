package discover

import (
	"testing"

	"github.com/ollama/ollama/ml"
)

func vulkanDevicesForIntelTest() []ml.DeviceInfo {
	return []ml.DeviceInfo{
		{
			DeviceID:    ml.DeviceID{ID: "0", Library: "Vulkan"},
			Name:        "Vulkan0",
			Description: "Intel(R) Arc(TM) B70 Graphics",
			TotalMemory: 26 << 30, // Vulkan heap typically excludes reserved regions
			FreeMemory:  24 << 30,
		},
		{
			DeviceID:    ml.DeviceID{ID: "1", Library: "CUDA"},
			Name:        "CUDA0",
			Description: "NVIDIA GeForce RTX 4060 Ti",
			TotalMemory: 16 << 30,
			FreeMemory:  14 << 30,
		},
	}
}

func TestInferLibrarySYCL(t *testing.T) {
	tests := []struct {
		label, name, description, want string
	}{
		{"sycl device", "SYCL0", "Intel(R) Arc(TM) B70 Graphics", "SYCL"},
		{"vulkan device", "Vulkan0", "Intel(R) Arc(TM) B70 Graphics", "Vulkan"},
		{"cuda device", "CUDA0", "NVIDIA GeForce RTX 4060 Ti", "CUDA"},
	}

	for _, tt := range tests {
		if got := inferLibrary(tt.name, tt.description); got != tt.want {
			t.Errorf("inferLibrary(%q, %q) = %q, want %q", tt.name, tt.description, got, tt.want)
		}
	}
}

func TestApplyIntelLevelZeroRefinementMatchesByName(t *testing.T) {
	devices := vulkanDevicesForIntelTest()
	lzDevices := []intelLevelZeroDevice{
		{
			Name:        "Intel(R) Arc(TM) B70 Graphics",
			VendorID:    0x8086,
			DeviceID:    0xe20b,
			PCIAddress:  "0000:03:00.0",
			TotalMemory: 32 << 30, // Level Zero reports the full physical VRAM
			FreeMemory:  31 << 30,
		},
	}

	refined := applyIntelLevelZeroRefinement(devices, []int{0}, lzDevices)
	if refined != 1 {
		t.Fatalf("refined = %d, want 1", refined)
	}

	dev := devices[0]
	if dev.PCIID != "0000:03:00.0" {
		t.Errorf("PCIID = %q, want %q", dev.PCIID, "0000:03:00.0")
	}
	if dev.TotalMemory != 32<<30 {
		t.Errorf("TotalMemory = %d, want %d", dev.TotalMemory, 32<<30)
	}
	if dev.FreeMemory != 31<<30 {
		t.Errorf("FreeMemory = %d, want %d", dev.FreeMemory, 31<<30)
	}
	if dev.Description != "Intel(R) Arc(TM) B70 Graphics" {
		t.Errorf("Description = %q", dev.Description)
	}
	if dev.Library != "Vulkan" {
		t.Errorf("Library = %q, want Vulkan (must not change backend)", dev.Library)
	}

	// The non-Vulkan device must be untouched.
	if devices[1].TotalMemory != 16<<30 || devices[1].PCIID != "" {
		t.Errorf("non-Vulkan device was modified: %+v", devices[1])
	}
}

func TestApplyIntelLevelZeroRefinementMatchesByPCIID(t *testing.T) {
	devices := vulkanDevicesForIntelTest()
	devices[0].PCIID = "0000:03:00.0"
	lzDevices := []intelLevelZeroDevice{
		{
			Name:        "Intel(R) Arc(TM) B70 Graphics",
			PCIAddress:  "0000:03:00.0",
			TotalMemory: 32 << 30,
			FreeMemory:  20 << 30,
		},
	}

	if refined := applyIntelLevelZeroRefinement(devices, []int{0}, lzDevices); refined != 1 {
		t.Fatalf("refined = %d, want 1", refined)
	}
	if devices[0].FreeMemory != 20<<30 {
		t.Errorf("FreeMemory = %d, want %d", devices[0].FreeMemory, 20<<30)
	}
}

func TestApplyIntelLevelZeroRefinementKeepsSimilarVRAM(t *testing.T) {
	devices := vulkanDevicesForIntelTest()
	lzDevices := []intelLevelZeroDevice{
		{
			Name:        "Intel(R) Arc(TM) B70 Graphics",
			TotalMemory: 26 << 30, // within SimilarDeviceMemory tolerance
			FreeMemory:  22 << 30,
		},
	}

	if refined := applyIntelLevelZeroRefinement(devices, []int{0}, lzDevices); refined != 1 {
		t.Fatalf("refined = %d, want 1", refined)
	}
	if devices[0].TotalMemory != 26<<30 {
		t.Errorf("TotalMemory = %d, want unchanged %d", devices[0].TotalMemory, 26<<30)
	}
}

func TestApplyIntelLevelZeroRefinementSkipsAmbiguousNames(t *testing.T) {
	devices := vulkanDevicesForIntelTest()
	lzDevices := []intelLevelZeroDevice{
		{Name: "Intel(R) Arc(TM) B70 Graphics", TotalMemory: 32 << 30},
		{Name: "Intel(R) Arc(TM) B70 Graphics", TotalMemory: 32 << 30},
	}

	if refined := applyIntelLevelZeroRefinement(devices, []int{0}, lzDevices); refined != 0 {
		t.Fatalf("refined = %d, want 0", refined)
	}
	if devices[0].TotalMemory != 26<<30 {
		t.Errorf("TotalMemory = %d, want unchanged", devices[0].TotalMemory)
	}
}

func TestApplyIntelLevelZeroRefinementNoMatch(t *testing.T) {
	devices := vulkanDevicesForIntelTest()
	lzDevices := []intelLevelZeroDevice{
		{Name: "Intel(R) Arc(TM) A770 Graphics", TotalMemory: 16 << 30},
	}

	if refined := applyIntelLevelZeroRefinement(devices, []int{0}, lzDevices); refined != 0 {
		t.Fatalf("refined = %d, want 0", refined)
	}
}

func TestApplyIntelLevelZeroRefinementEmptyProbe(t *testing.T) {
	devices := vulkanDevicesForIntelTest()
	if refined := applyIntelLevelZeroRefinement(devices, []int{0}, nil); refined != 0 {
		t.Fatalf("refined = %d, want 0", refined)
	}
}

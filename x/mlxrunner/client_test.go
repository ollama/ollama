package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

func TestMLXGPUIDsUsesFirstDevice(t *testing.T) {
	gpus := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{ID: "0", Library: "Metal"}},
		{DeviceID: ml.DeviceID{ID: "1", Library: "Metal"}},
	}

	got := mlxGPUIDs(gpus)
	if len(got) != 1 || got[0] != gpus[0].DeviceID {
		t.Fatalf("mlxGPUIDs() = %#v, want first GPU %v", got, gpus[0].DeviceID)
	}
}

func TestMLXGPUIDsNoDevices(t *testing.T) {
	if got := mlxGPUIDs(nil); len(got) != 0 {
		t.Fatalf("mlxGPUIDs(nil) = %#v, want no devices", got)
	}
}

func TestEstimateLoadMemoryUsesFirstDevice(t *testing.T) {
	gpus := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{ID: "0", Library: "Metal"}, FreeMemory: 8 * format.GibiByte},
		{DeviceID: ml.DeviceID{ID: "1", Library: "Metal"}, FreeMemory: 64 * format.GibiByte},
	}

	estimate := estimateLoadMemory(7*format.GibiByte, 32768, gpus)
	if !estimate.HasGPU {
		t.Fatal("estimate.HasGPU = false, want true")
	}
	if estimate.ContextLength != 32768 {
		t.Fatalf("estimate.ContextLength = %d, want 32768", estimate.ContextLength)
	}
	if !estimate.Fits() {
		t.Fatal("estimate.Fits() = false, want true")
	}
	if estimate.GPUFree != gpus[0].FreeMemory {
		t.Fatalf("estimate.GPUFree = %d, want %d", estimate.GPUFree, gpus[0].FreeMemory)
	}
	if estimate.Available >= gpus[1].FreeMemory {
		t.Fatalf("estimate.Available = %d, want first GPU availability", estimate.Available)
	}
}

func TestEstimateLoadMemoryDetectsOversize(t *testing.T) {
	gpus := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{ID: "0", Library: "Metal"}, FreeMemory: 8 * format.GibiByte},
	}

	estimate := estimateLoadMemory(9*format.GibiByte, 0, gpus)
	if estimate.Fits() {
		t.Fatal("estimate.Fits() = true, want false")
	}
}

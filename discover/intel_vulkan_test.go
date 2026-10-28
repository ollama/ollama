package discover

import (
	"runtime"
	"testing"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

func setSystemTotalMemory(t *testing.T, total uint64) {
	t.Helper()
	orig := systemTotalMemory
	systemTotalMemory = func() uint64 { return total }
	t.Cleanup(func() { systemTotalMemory = orig })
}

func TestFormatIntelIPVersion(t *testing.T) {
	tests := map[uint32]string{
		0:          "unknown",
		0x03118000: "12.70.0",
		0x03128004: "12.74.4",
		0x05010004: "20.4.4",
	}
	for in, want := range tests {
		if got := formatIntelIPVersion(in); got != want {
			t.Errorf("formatIntelIPVersion(%#x) = %q, want %q", in, got, want)
		}
	}
}

func TestIntelIGPUAllowedByDefault(t *testing.T) {
	lnl := ml.DeviceInfo{
		DeviceID:       ml.DeviceID{Library: "Vulkan", ID: "0"},
		Description:    "Intel(R) Arc(TM) 140V GPU (16GB)",
		Integrated:     true,
		IntelIPVersion: 0x05010004, // LNL_B0 20.4.4
	}
	with := func(f func(*ml.DeviceInfo)) ml.DeviceInfo {
		d := lnl
		f(&d)
		return d
	}

	tests := []struct {
		name        string
		device      ml.DeviceInfo
		hasDiscrete bool
		memory      uint64
		want        bool
	}{
		{"lunar lake 32GB", lnl, false, 32 * format.GibiByte, true},
		{"mtl baseline", with(func(d *ml.DeviceInfo) { d.IntelIPVersion = 0x03118000 }), false, 32 * format.GibiByte, true},
		{"arl-h", with(func(d *ml.DeviceInfo) { d.IntelIPVersion = 0x03128000 }), false, 32 * format.GibiByte, true},
		{"adl-n below baseline", with(func(d *ml.DeviceInfo) { d.IntelIPVersion = 0x03010000 }), false, 32 * format.GibiByte, false},
		{"pre-mtl revision", with(func(d *ml.DeviceInfo) { d.IntelIPVersion = 0x03117fff }), false, 32 * format.GibiByte, false},
		{"extension unavailable", with(func(d *ml.DeviceInfo) { d.IntelIPVersion = 0 }), false, 32 * format.GibiByte, false},
		{"discrete present", lnl, true, 32 * format.GibiByte, false},
		{"16GB system reporting 15.6GiB", lnl, false, 15*format.GibiByte + 600*format.MebiByte, true},
		{"memory exactly threshold", lnl, false, 15 * format.GibiByte, true},
		{"8GB system", lnl, false, 8 * format.GibiByte, false},
		{"memory unknown", lnl, false, 0, false},
		{"not integrated", with(func(d *ml.DeviceInfo) { d.Integrated = false }), false, 32 * format.GibiByte, false},
		{"not vulkan", with(func(d *ml.DeviceInfo) { d.Library = "CUDA" }), false, 32 * format.GibiByte, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			setSystemTotalMemory(t, tt.memory)
			if got := intelIGPUAllowedByDefault(tt.device, tt.hasDiscrete); got != tt.want {
				t.Fatalf("intelIGPUAllowedByDefault() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFilterIntegratedGPUsIntelVulkan(t *testing.T) {
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		t.Skip("integrated GPU filtering is bypassed on Apple Silicon")
	}

	intel := ml.DeviceInfo{
		DeviceID:       ml.DeviceID{Library: "Vulkan", ID: "0"},
		Description:    "Intel(R) Arc(TM) 140V GPU (16GB)",
		Integrated:     true,
		IntelIPVersion: 0x05010004,
	}
	discrete := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Vulkan", ID: "1"}, Description: "Intel(R) Arc(TM) B580 Graphics"}
	setSystemTotalMemory(t, 32*format.GibiByte)

	t.Run("auto admits supported intel iGPU without dGPU", func(t *testing.T) {
		got := filterIntegratedGPUs([]ml.DeviceInfo{intel})
		assertDeviceIDs(t, got, []ml.DeviceID{intel.DeviceID})
	})

	t.Run("auto keeps dGPU only when present", func(t *testing.T) {
		got := filterIntegratedGPUs([]ml.DeviceInfo{intel, discrete})
		assertDeviceIDs(t, got, []ml.DeviceID{discrete.DeviceID})
	})

	t.Run("explicit false drops supported intel iGPU", func(t *testing.T) {
		t.Setenv("OLLAMA_IGPU_ENABLE", "false")
		got := filterIntegratedGPUs([]ml.DeviceInfo{intel})
		assertDeviceIDs(t, got, nil)
	})

	t.Run("explicit true admits iGPU alongside dGPU", func(t *testing.T) {
		t.Setenv("OLLAMA_IGPU_ENABLE", "true")
		got := filterIntegratedGPUs([]ml.DeviceInfo{intel, discrete})
		assertDeviceIDs(t, got, []ml.DeviceID{intel.DeviceID, discrete.DeviceID})
	})

	t.Run("auto drops iGPU with insufficient memory", func(t *testing.T) {
		setSystemTotalMemory(t, 8*format.GibiByte)
		got := filterIntegratedGPUs([]ml.DeviceInfo{intel})
		assertDeviceIDs(t, got, nil)
	})
}

//go:build (linux && cgo) || windows

package discover

import "testing"

func TestPackIntelIPVersion(t *testing.T) {
	tests := []struct {
		arch, release, revision uint32
		want                    uint32
	}{
		{12, 70, 0, 0x03118000}, // MTL_U_A0
		{12, 74, 4, 0x03128004}, // ARL_H_B0
		{20, 4, 4, 0x05010004},  // LNL_B0
		{35, 10, 4, 0x08c28004}, // NVL_P_B0
		{0, 0, 0, 0},            // not reported
		{12, 256, 0, 0},         // out of range
		{12, 70, 0x4000, 0},     // out of range
	}
	for _, tt := range tests {
		if got := packIntelIPVersion(tt.arch, tt.release, tt.revision); got != tt.want {
			t.Errorf("packIntelIPVersion(%d, %d, %d) = %#x, want %#x", tt.arch, tt.release, tt.revision, got, tt.want)
		}
	}
}

func TestVulkanPCIBusID(t *testing.T) {
	if got, want := vulkanPCIBusID(0, 0, 2, 0), "0000:00:02.0"; got != want {
		t.Fatalf("vulkanPCIBusID() = %q, want %q", got, want)
	}
}

func TestApplyIntelVulkanDeviceInfo(t *testing.T) {
	const lnl = 0x05010004

	t.Run("matches by PCI ID", func(t *testing.T) {
		devices := []nativeProbeDevice{
			{Library: "Vulkan", Description: "Intel(R) Arc(TM) 140V GPU (16GB)", DeviceID: "0000:00:02.0", Integrated: true},
		}
		applyIntelVulkanDeviceInfo(devices, []intelVulkanDevice{
			{Name: "Intel(R) Arc(TM) 140V GPU (16GB)", PCIID: "0000:00:02.0", Integrated: true, IPVersion: lnl},
		})
		if devices[0].IntelIPVersion != lnl {
			t.Fatalf("IntelIPVersion = %#x, want %#x", devices[0].IntelIPVersion, lnl)
		}
	})

	t.Run("falls back to unique name match", func(t *testing.T) {
		devices := []nativeProbeDevice{
			{Library: "Vulkan", Description: "Intel(R) Arc(TM) 140V GPU (16GB)", Integrated: true},
		}
		applyIntelVulkanDeviceInfo(devices, []intelVulkanDevice{
			{Name: "Intel(R) Arc(TM) 140V GPU (16GB)", Integrated: true, IPVersion: lnl},
		})
		if devices[0].IntelIPVersion != lnl {
			t.Fatalf("IntelIPVersion = %#x, want %#x", devices[0].IntelIPVersion, lnl)
		}
	})

	t.Run("mismatched PCI ID is not matched by name", func(t *testing.T) {
		devices := []nativeProbeDevice{
			{Library: "Vulkan", Description: "Intel(R) Graphics", DeviceID: "0000:00:02.0", Integrated: true},
		}
		applyIntelVulkanDeviceInfo(devices, []intelVulkanDevice{
			{Name: "Intel(R) Graphics", PCIID: "0000:03:00.0", Integrated: true, IPVersion: lnl},
		})
		if devices[0].IntelIPVersion != 0 {
			t.Fatalf("IntelIPVersion = %#x, want 0", devices[0].IntelIPVersion)
		}
	})

	t.Run("duplicate PCI ID is skipped", func(t *testing.T) {
		devices := []nativeProbeDevice{
			{Library: "Vulkan", Description: "Intel(R) Graphics", DeviceID: "0000:00:00.0", Integrated: true},
		}
		applyIntelVulkanDeviceInfo(devices, []intelVulkanDevice{
			{Name: "Intel(R) Graphics", PCIID: "0000:00:00.0", Integrated: true, IPVersion: lnl},
			{Name: "Intel(R) Arc(TM) B580 Graphics", PCIID: "0000:00:00.0", IPVersion: 0x05004000},
		})
		if devices[0].IntelIPVersion != 0 {
			t.Fatalf("IntelIPVersion = %#x, want 0", devices[0].IntelIPVersion)
		}
	})

	t.Run("ambiguous name match is skipped", func(t *testing.T) {
		devices := []nativeProbeDevice{
			{Library: "Vulkan", Description: "Intel(R) Graphics", Integrated: true},
		}
		applyIntelVulkanDeviceInfo(devices, []intelVulkanDevice{
			{Name: "Intel(R) Graphics", Integrated: true, IPVersion: lnl},
			{Name: "Intel(R) Graphics", Integrated: true, IPVersion: lnl},
		})
		if devices[0].IntelIPVersion != 0 {
			t.Fatalf("IntelIPVersion = %#x, want 0", devices[0].IntelIPVersion)
		}
	})

	t.Run("discrete and non-vulkan devices are ignored", func(t *testing.T) {
		devices := []nativeProbeDevice{
			{Library: "Vulkan", Description: "Intel(R) Arc(TM) B580 Graphics", DeviceID: "0000:03:00.0"},
			{Library: "CUDA", Description: "Intel(R) Graphics", DeviceID: "0000:00:02.0", Integrated: true},
		}
		applyIntelVulkanDeviceInfo(devices, []intelVulkanDevice{
			{Name: "Intel(R) Arc(TM) B580 Graphics", PCIID: "0000:03:00.0", IPVersion: 0x05004000},
			{Name: "Intel(R) Graphics", PCIID: "0000:00:02.0", Integrated: true, IPVersion: lnl},
		})
		for i, dev := range devices {
			if dev.IntelIPVersion != 0 {
				t.Fatalf("device %d IntelIPVersion = %#x, want 0", i, dev.IntelIPVersion)
			}
		}
	})
}

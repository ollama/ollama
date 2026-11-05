package discover

import (
	"fmt"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

const (
	// MTL_U_A0 (12.70.0) from intel/compute-runtime third_party/aot_config_headers/platforms.h.
	intelIPVersionMTL = 0x03118000

	// 16 GB systems report slightly less than 16 GiB of usable memory.
	intelIGPUMinSystemMemory = 15 * format.GibiByte
)

// systemTotalMemory is a variable so tests can override it.
var systemTotalMemory = func() uint64 {
	return GetSystemInfo().TotalMemory
}

// formatIntelIPVersion decodes the platforms.h arch<<22 | release<<14 | revision encoding.
func formatIntelIPVersion(v uint32) string {
	if v == 0 {
		return "unknown"
	}
	return fmt.Sprintf("%d.%d.%d", v>>22, (v>>14)&0xff, v&0x3fff)
}

type intelVulkanDevice struct {
	Name       string
	PCIID      string
	Integrated bool
	IPVersion  uint32
}

// packIntelIPVersion encodes an IP version like the AOT::PRODUCT_CONFIG values
// in platforms.h, returning 0 for unreported or out-of-range values.
func packIntelIPVersion(arch, release, revision uint32) uint32 {
	if arch > 0x3ff || release > 0xff || revision > 0x3fff {
		return 0
	}
	return arch<<22 | release<<14 | revision
}

// vulkanPCIBusID matches the device_id format ggml-vulkan reports.
func vulkanPCIBusID(domain, bus, device, function uint32) string {
	return fmt.Sprintf("%04x:%02x:%02x.%x", domain, bus, device, uint8(function))
}

// applyIntelVulkanDeviceInfo attaches probed Intel IP versions to integrated
// Vulkan devices, matching by PCI bus ID or, failing that, a unique name match.
func applyIntelVulkanDeviceInfo(devices []nativeProbeDevice, probed []intelVulkanDevice) {
	for i := range devices {
		dev := &devices[i]
		if dev.Library != "Vulkan" || !dev.Integrated {
			continue
		}

		match := -1
		if dev.DeviceID != "" {
			for j, p := range probed {
				if p.PCIID != "" && strings.EqualFold(p.PCIID, dev.DeviceID) {
					if match >= 0 {
						slog.Debug("intel vulkan device info skipped: ambiguous PCI ID", "pci_id", dev.DeviceID)
						match = -1
						break
					}
					match = j
				}
			}
		}
		if match < 0 {
			description := dev.Description
			if description == "" {
				description = dev.Name
			}
			for j, p := range probed {
				if dev.DeviceID != "" && p.PCIID != "" {
					continue
				}
				if !p.Integrated || !ml.SimilarDeviceDescription(description, p.Name) {
					continue
				}
				if match >= 0 {
					slog.Debug("intel vulkan device info skipped: ambiguous device name match", "name", description)
					match = -1
					break
				}
				match = j
			}
		}
		if match < 0 || !probed[match].Integrated {
			continue
		}

		dev.IntelIPVersion = probed[match].IPVersion
		slog.Debug("intel vulkan device info", "name", probed[match].Name, "pci_id", probed[match].PCIID,
			"ip_version", formatIntelIPVersion(dev.IntelIPVersion))
	}
}

// intelIGPUAllowedByDefault admits an Intel Vulkan iGPU when the driver reports
// a Meteor Lake (Ultra Series 1) or newer IP via VK_INTEL_device_info, no discrete GPU is
// present, and the system has enough memory.
func intelIGPUAllowedByDefault(device ml.DeviceInfo, hasDiscrete bool) bool {
	if device.Library != "Vulkan" || !device.Integrated || device.IntelIPVersion == 0 {
		return false
	}

	ipVersion := formatIntelIPVersion(device.IntelIPVersion)
	if device.IntelIPVersion < intelIPVersionMTL {
		slog.Debug("intel integrated GPU below default-enable baseline", "name", device.Description, "ip_version", ipVersion)
		return false
	}
	if hasDiscrete {
		slog.Debug("intel integrated GPU not enabled by default: discrete GPU present", "name", device.Description)
		return false
	}
	total := systemTotalMemory()
	if total < intelIGPUMinSystemMemory {
		slog.Info("intel integrated GPU not enabled by default: insufficient system memory",
			"name", device.Description, "total", format.HumanBytes2(total), "required", format.HumanBytes2(intelIGPUMinSystemMemory))
		return false
	}

	slog.Info("enabling intel integrated GPU by default", "name", device.Description, "ip_version", ipVersion,
		"system_memory", format.HumanBytes2(total))
	return true
}

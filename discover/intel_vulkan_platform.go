//go:build (linux && cgo) || windows

package discover

import (
	"fmt"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/ml"
)

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

func nativeProbeHasVulkan(libDirs []string) bool {
	for _, dir := range libDirs {
		if strings.Contains(strings.ToLower(dir), "vulkan") {
			return true
		}
	}
	for _, file := range nativeProbeBackendFiles(libDirs) {
		if strings.Contains(strings.ToLower(file), "vulkan") {
			return true
		}
	}
	return false
}

func addIntelVulkanDeviceInfo(devices []nativeProbeDevice, libDirs []string) {
	hasIntegratedVulkan := false
	for _, dev := range devices {
		if dev.Library == "Vulkan" && dev.Integrated {
			hasIntegratedVulkan = true
			break
		}
	}
	if !hasIntegratedVulkan || !nativeProbeHasVulkan(libDirs) {
		return
	}

	probed, err := probeIntelVulkanDevices(libDirs)
	if err != nil {
		slog.Debug("intel vulkan device info unavailable", "error", err)
		return
	}
	applyIntelVulkanDeviceInfo(devices, probed)
}

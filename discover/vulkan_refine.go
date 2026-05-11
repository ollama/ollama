package discover

import (
	"errors"
	"log/slog"
	"runtime"
	"strings"

	"github.com/ollama/ollama/ml"
)

var errWindowsVulkanProbeUnsupported = errors.New("windows vulkan probe unsupported")

type vulkanPhysicalDevice struct {
	Name       string
	Integrated bool
}

var probeLlamaServerVulkanDevices = func() ([]vulkanPhysicalDevice, error) {
	return nil, errWindowsVulkanProbeUnsupported
}

func refineLlamaServerDevices(devices []ml.DeviceInfo) []ml.DeviceInfo {
	if runtime.GOOS != "windows" {
		return devices
	}

	var vulkanIndexes []int
	hasIntegratedVulkan := false
	for i, device := range devices {
		if device.Library != "Vulkan" {
			continue
		}
		vulkanIndexes = append(vulkanIndexes, i)
		hasIntegratedVulkan = hasIntegratedVulkan || device.Integrated
	}

	if len(vulkanIndexes) == 0 || hasIntegratedVulkan {
		return devices
	}

	probed, err := probeLlamaServerVulkanDevices()
	if err != nil {
		if !errors.Is(err, errWindowsVulkanProbeUnsupported) {
			slog.Debug("windows vulkan device refinement unavailable", "error", err)
		}
		return devices
	}

	if !applyWindowsVulkanRefinement(devices, probed) {
		return devices
	}

	return devices
}

func applyWindowsVulkanRefinement(devices []ml.DeviceInfo, probed []vulkanPhysicalDevice) bool {
	var vulkanIndexes []int
	for i, device := range devices {
		if device.Library == "Vulkan" {
			if device.Integrated {
				return false
			}
			vulkanIndexes = append(vulkanIndexes, i)
		}
	}

	if len(probed) != len(vulkanIndexes) {
		slog.Debug("windows vulkan device refinement skipped: device count mismatch",
			"llama_server_count", len(vulkanIndexes), "vulkan_count", len(probed))
		return false
	}

	for i, probedDevice := range probed {
		description := devices[vulkanIndexes[i]].Description
		if !sameVulkanDeviceName(description, probedDevice.Name) {
			slog.Debug("windows vulkan device refinement skipped: device name mismatch",
				"index", i, "llama_server_name", description, "vulkan_name", probedDevice.Name)
			return false
		}
	}

	for i, probedDevice := range probed {
		devices[vulkanIndexes[i]].Integrated = probedDevice.Integrated
	}

	slog.Debug("windows vulkan device refinement applied", "devices", len(vulkanIndexes))
	return true
}

func sameVulkanDeviceName(a, b string) bool {
	return strings.EqualFold(strings.TrimSpace(a), strings.TrimSpace(b))
}

// Vulkan discovery needs a small amount of normalization around device type.
// llama-server discovery output does not currently expose a stable structured
// backend type field, so we use explicit Vulkan UMA metadata when it is
// present and, on Windows, refine the result with a direct Vulkan API query.
// The goal is to preserve correct integrated-vs-discrete scheduling decisions
// without relying on device-name heuristics.
package discover

import (
	"bufio"
	"errors"
	"log/slog"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	"github.com/ollama/ollama/ml"
)

// vulkanUMARegex matches Vulkan debug lines like:
//
//	ggml_vulkan: 0 = Intel(R) Graphics (...) | uma: 1 | fp16: 1 |
var vulkanUMARegex = regexp.MustCompile(
	`ggml_vulkan:\s+(\d+)\s+=.*\|\s+uma:\s+([01])\s+\|`,
)

func parseVulkanUMA(output string) map[int]bool {
	integratedByIndex := make(map[int]bool)

	scanner := bufio.NewScanner(strings.NewReader(output))
	for scanner.Scan() {
		if matches := vulkanUMARegex.FindStringSubmatch(scanner.Text()); matches != nil {
			idx, _ := strconv.Atoi(matches[1])
			integratedByIndex[idx] = matches[2] == "1"
		}
	}

	return integratedByIndex
}

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

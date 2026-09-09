// Intel discovery needs a small amount of backend-specific handling beyond the
// generic llama-server device list. On Linux, the Vulkan backend enumerates
// Intel Arc GPUs but can report generic descriptions and imprecise VRAM sizes,
// and it never exposes the PCI address. The Intel Level Zero management API
// (libze.so.1 + libze_intel_gpu.so.1) provides the canonical device name,
// memory sizes, free VRAM (via the sysman API) and PCI BDF, which we use to
// refine the Vulkan device entries in place. Intel GPUs can also run through
// the native SYCL backend when the sycl runner payload is installed; this
// refinement only touches Vulkan entries so SYCL metadata stays authoritative.
package discover

import (
	"errors"
	"log/slog"
	"runtime"
	"strings"

	"github.com/ollama/ollama/ml"
)

const (
	// oneapiLibName is the Intel Level Zero loader library.
	oneapiLibName = "libze.so.1"

	// oneapiLoaderLibName is the canonical name the Level Zero loader is
	// packaged under on most distributions. Some installs only provide this
	// name, so it is probed as a fallback for oneapiLibName.
	oneapiLoaderLibName = "libze_loader.so.1"

	// intelGpuLibName is the Intel Level Zero GPU driver library the loader
	// needs to enumerate Intel devices.
	intelGpuLibName = "libze_intel_gpu.so.1"
)

// intelLibPaths are common library directories on Ubuntu/Debian systems where
// the Level Zero loader and Intel GPU driver libraries are installed.
var intelLibPaths = []string{
	"/usr/lib/x86_64-linux-gnu",
	"/usr/local/lib",
	"/usr/lib",
}

// intelLevelZeroDevice is the subset of Level Zero device information used to
// refine Vulkan device entries.
type intelLevelZeroDevice struct {
	Name        string
	VendorID    uint32
	DeviceID    uint32
	PCIAddress  string
	TotalMemory uint64
	FreeMemory  uint64
}

var errIntelLevelZeroProbeUnsupported = errors.New("intel level zero probe unsupported on this platform")

// probeIntelLevelZeroDevices is overridden on Linux (cgo builds) to query the
// Level Zero management API. It stays a variable to allow tests to swap it out.
var probeIntelLevelZeroDevices = func() ([]intelLevelZeroDevice, error) {
	return nil, errIntelLevelZeroProbeUnsupported
}

// refineIntelVulkanDevices refines Vulkan device entries that correspond to
// Intel GPUs with precise Level Zero metadata. Devices are modified in place.
func refineIntelVulkanDevices(devices []ml.DeviceInfo) []ml.DeviceInfo {
	if runtime.GOOS != "linux" {
		return devices
	}

	var vulkanIndexes []int
	for i, device := range devices {
		if device.Library == "Vulkan" {
			vulkanIndexes = append(vulkanIndexes, i)
		}
	}
	if len(vulkanIndexes) == 0 {
		return devices
	}

	lzDevices, err := probeIntelLevelZeroDevices()
	if err != nil {
		slog.Debug("Intel Level Zero device refinement unavailable", "error", err)
		return devices
	}

	if refined := applyIntelLevelZeroRefinement(devices, vulkanIndexes, lzDevices); refined > 0 {
		slog.Info("Intel GPU discovery refined via Level Zero", "devices", refined)
	}

	return devices
}

// applyIntelLevelZeroRefinement matches Vulkan devices against Level Zero
// devices (by PCI address first, then by device name) and refines the device
// metadata. It returns the number of devices refined.
func applyIntelLevelZeroRefinement(devices []ml.DeviceInfo, vulkanIndexes []int, lzDevices []intelLevelZeroDevice) int {
	if len(lzDevices) == 0 {
		return 0
	}

	used := make([]bool, len(lzDevices))
	refined := 0

	for _, index := range vulkanIndexes {
		device := &devices[index]

		match := -1
		if device.PCIID != "" {
			for j, lz := range lzDevices {
				if !used[j] && lz.PCIAddress != "" && strings.EqualFold(lz.PCIAddress, device.PCIID) {
					match = j
					break
				}
			}
		}
		if match < 0 {
			for j, lz := range lzDevices {
				if used[j] || lz.Name == "" || !ml.SimilarDeviceDescription(device.Description, lz.Name) {
					continue
				}
				if match >= 0 {
					// Ambiguous name match - more than one Level Zero device
					// could be this Vulkan device. Skip to stay conservative.
					slog.Debug("Intel Level Zero refinement skipped: ambiguous device name match",
						"index", index, "vulkan_name", device.Description)
					match = -1
					break
				}
				match = j
			}
		}
		if match < 0 {
			continue
		}
		used[match] = true

		lz := lzDevices[match]
		if device.PCIID == "" && lz.PCIAddress != "" {
			device.PCIID = lz.PCIAddress
		}
		if lz.Name != "" && lz.Name != device.Description {
			slog.Debug("updating Intel device description from Level Zero",
				"index", index, "vulkan_name", device.Description, "level_zero_name", lz.Name)
			device.Description = lz.Name
		}
		if lz.TotalMemory > 0 && (device.TotalMemory == 0 || !ml.SimilarDeviceMemory(device.TotalMemory, lz.TotalMemory)) {
			slog.Debug("updating Intel device VRAM from Level Zero",
				"index", index, "name", lz.Name,
				"vulkan_total", device.TotalMemory, "level_zero_total", lz.TotalMemory)
			device.TotalMemory = lz.TotalMemory
		}
		if lz.FreeMemory > 0 {
			free := lz.FreeMemory
			if device.TotalMemory > 0 && free > device.TotalMemory {
				free = device.TotalMemory
			}
			device.FreeMemory = free
		}

		refined++
	}

	return refined
}

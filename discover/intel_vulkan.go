package discover

import (
	"fmt"
	"log/slog"

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

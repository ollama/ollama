//go:build linux || windows

package discover

import (
	"fmt"
	"log/slog"
	"strings"
)

func vulkanGetVisibleDevicesEnv(gpuInfo []GpuInfo) (string, string) {
	ids := []string{}
	for _, info := range gpuInfo {
		if info.Library != "vulkan" {
			// TODO shouldn't happen if things are wired correctly...
			slog.Debug("vulkanGetVisibleDevicesEnv skipping over non-vulkan device", "library", info.Library)
			continue
		}
		ids = append(ids, info.ID)
	}
	return "GGML_VK_VISIBLE_DEVICES", strings.Join(ids, ",")
}

func vulkanVariant(gpuInfo VulkanGPUInfo) string {
	return fmt.Sprintf("v%d", gpuInfo.DriverMajor)
}

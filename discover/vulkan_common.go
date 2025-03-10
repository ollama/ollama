package discover

import (
	"log/slog"
	"strings"
)

func vkGetVisibleDevicesEnv(gpuInfo []GpuInfo) (string, string) {
	ids := []string{}
	for _, info := range gpuInfo {
		if info.Library != "vulkan" {
			// TODO shouldn't happen if things are wired correctly...
			slog.Debug("vkGetVisibleDevicesEnv skipping over non-vulkan device", "library", info.Library)
			continue
		}
		ids = append(ids, info.ID)
	}
	return "GGML_VK_VISIBLE_DEVICES", strings.Join(ids, ",")
}

//go:build linux || windows

package discover

import (
	"log/slog"
	"strings"
)

func musaGetVisibleDevicesEnv(gpuInfo []GpuInfo) (string, string) {
	ids := []string{}
	for _, info := range gpuInfo {
		if info.Library != "musa" {
			// TODO shouldn't happen if things are wired correctly...
			slog.Debug("musaGetVisibleDevicesEnv skipping over non-musa device", "library", info.Library)
			continue
		}
		ids = append(ids, info.ID)
	}
	return "MUSA_VISIBLE_DEVICES", strings.Join(ids, ",")
}

func musaVariant(gpuInfo MusaGPUInfo) string {
	return "v1"
}

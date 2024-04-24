//go:build linux || windows

package gpu

import (
	"log/slog"
	"strings"
)

func oneApiGetVisibleDevicesEnv(gpuInfo []GpuInfo) (string, string) {
	ids := []string{}
	for _, info := range gpuInfo {
		if info.Library != "oneapi" {
			slog.Debug("oneApiGetVisibleDevicesEnv skipping over oneapi device", "library", info.Library)
			continue
		}
		ids = append(ids, info.ID)
	}
	return "ONEAPI_VISIBLE_DEVICES", strings.Join(ids, ",")

}

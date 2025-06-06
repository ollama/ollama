//go:build linux || windows

package discover

import (
	"log/slog"
	"strings"
)

func syclGetVisibleDevicesEnv(gpuInfo []GpuInfo) (string, string) {
	ids := []string{}
	for _, info := range gpuInfo {
		if info.Library != "sycl" {
			// TODO shouldn't happen if things are wired correctly...
			slog.Debug("syclGetVisibleDevicesEnv skipping over non-sycl device", "library", info.Library)
			continue
		}
		ids = append(ids, info.ID)
	}
	key := "ONEAPI_DEVICE_SELECTOR"
	value := "level_zero:" + strings.Join(ids, ",")
	slog.Debug("syclGetVisibleDevicesEnv debug details",
		"ids", ids,
		"gpuInfo", gpuInfo,
		"gpuCount", len(gpuInfo))
	slog.Debug("syclGetVisibleDevicesEnv returning", "key", key, "value", value)
	return key, value
}

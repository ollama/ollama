package discover

import (
	"context"
	"log/slog"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/ml"
)

func filterOldCUDADriver(ctx context.Context, devices []ml.DeviceInfo) []ml.DeviceInfo {
	oldCUDA := func(dev ml.DeviceInfo) bool {
		return dev.Library == "CUDA" && dev.ComputeMajor > 0 && dev.ComputeMajor < 7
	}

	needsCheck := false
	for _, dev := range devices {
		if oldCUDA(dev) {
			needsCheck = true
			break
		}
	}
	if !needsCheck {
		return devices
	}

	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()

	output, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits").Output()
	if err != nil {
		slog.Warn("could not run nvidia-smi to verify CUDA driver compatibility for an older NVIDIA GPU", "error", err)
		return devices
	}

	line := strings.TrimSpace(strings.Split(string(output), "\n")[0])
	major, _, _ := strings.Cut(line, ".")
	driver, err := strconv.Atoi(major)
	if err != nil {
		slog.Warn("could not parse nvidia-smi driver version for an older NVIDIA GPU", "version", line, "error", err)
		return devices
	}
	if driver >= 570 {
		return devices
	}

	filtered := devices[:0]
	for _, dev := range devices {
		if oldCUDA(dev) {
			slog.Warn("NVIDIA driver too old",
				"device", dev.Description, "compute", dev.Compute(), "driver", driver, "required_driver", "570 or newer")
			continue
		}
		filtered = append(filtered, dev)
	}
	return filtered
}

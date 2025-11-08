package discover

import (
	"log/slog"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

type memInfo struct {
	TotalMemory uint64 `json:"total_memory,omitempty"`
	FreeMemory  uint64 `json:"free_memory,omitempty"`
	FreeSwap    uint64 `json:"free_swap,omitempty"` // TODO split this out for system only
}

// CPU type represents a CPU Package occupying a socket
type CPU struct {
	ID                  string `cpuinfo:"processor"`
	VendorID            string `cpuinfo:"vendor_id"`
	ModelName           string `cpuinfo:"model name"`
	CoreCount           int
	EfficiencyCoreCount int // Performance = CoreCount - Efficiency
	ThreadCount         int
}

func LogDetails(devices []ml.DeviceInfo) {
	sort.Sort(sort.Reverse(ml.ByFreeMemory(devices))) // Report devices in order of scheduling preference
	for _, dev := range devices {
		var libs []string
		for _, dir := range dev.LibraryPath {
			if strings.Contains(dir, filepath.Join("lib", "ollama")) {
				libs = append(libs, filepath.Base(dir))
			}
		}
		typeStr := "discrete"
		if dev.Integrated {
			typeStr = "iGPU"
		}
		slog.Info("inference compute",
			"id", dev.ID,
			"filter_id", dev.FilterID,
			"library", dev.Library,
			"compute", dev.Compute(),
			"name", dev.Name,
			"description", dev.Description,
			"libdirs", strings.Join(libs, ","),
			"driver", dev.Driver(),
			"pci_id", dev.PCIID,
			"type", typeStr,
			"total", format.HumanBytes2(dev.TotalMemory),
			"available", format.HumanBytes2(dev.FreeMemory),
		)
	}
	// CPU inference
	if len(devices) == 0 {
		dev, _ := GetCPUMem()
		slog.Info("inference compute",
			"id", "cpu",
			"library", "cpu",
			"compute", "",
			"name", "cpu",
			"description", "cpu",
			"libdirs", "ollama",
			"driver", "",
			"pci_id", "",
			"type", "",
			"total", format.HumanBytes2(dev.TotalMemory),
			"available", format.HumanBytes2(dev.FreeMemory),
		)
	}
}

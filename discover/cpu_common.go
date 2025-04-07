package discover

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

func IsNUMA() bool {
	if runtime.GOOS != "linux" {
		// numa support in llama.cpp is linux only
		return false
	}
	ids := map[string]any{}
	packageIds, _ := filepath.Glob("/sys/devices/system/cpu/cpu*/topology/physical_package_id")
	for _, packageId := range packageIds {
		id, err := os.ReadFile(packageId)
		if err == nil {
			ids[strings.TrimSpace(string(id))] = struct{}{}
		}
	}
	return len(ids) > 1
}

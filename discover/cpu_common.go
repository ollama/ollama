package discover

import (
    "os"
    "path/filepath"
    "runtime"
    "strings"
)

var (
    osReadFileFunc   = os.ReadFile
    filepathGlobFunc = filepath.Glob
    runtimeGOOS      = runtime.GOOS
)

func IsNUMA() bool {
    if runtimeGOOS != "linux" {
        // NUMA support in llama.cpp is Linux only
        return false
    }
    ids := map[string]interface{}{}
    packageIds, _ := filepathGlobFunc("/sys/devices/system/cpu/cpu*/topology/physical_package_id")
    for _, packageId := range packageIds {
        id, err := osReadFileFunc(packageId)
        if err == nil {
            ids[strings.TrimSpace(string(id))] = struct{}{}
        }
    }
    return len(ids) > 1
}

package discover

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ollama/ollama/format"
)

func GetCPUMem() (memInfo, error) {
	mem, err := getCPUMem()
	if err != nil {
		return memInfo{}, err
	}
	return getCPUMemByCgroups(mem), nil
}

func getCPUMem() (memInfo, error) {
	var mem memInfo
	var total, available, free, buffers, cached, freeSwap uint64
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return mem, err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := s.Text()
		switch {
		case strings.HasPrefix(line, "MemTotal:"):
			_, err = fmt.Sscanf(line, "MemTotal:%d", &total)
		case strings.HasPrefix(line, "MemAvailable:"):
			_, err = fmt.Sscanf(line, "MemAvailable:%d", &available)
		case strings.HasPrefix(line, "MemFree:"):
			_, err = fmt.Sscanf(line, "MemFree:%d", &free)
		case strings.HasPrefix(line, "Buffers:"):
			_, err = fmt.Sscanf(line, "Buffers:%d", &buffers)
		case strings.HasPrefix(line, "Cached:"):
			_, err = fmt.Sscanf(line, "Cached:%d", &cached)
		case strings.HasPrefix(line, "SwapFree:"):
			_, err = fmt.Sscanf(line, "SwapFree:%d", &freeSwap)
		default:
			continue
		}
		if err != nil {
			return mem, err
		}
	}
	mem.TotalMemory = total * format.KibiByte
	mem.FreeSwap = freeSwap * format.KibiByte
	if available > 0 {
		mem.FreeMemory = available * format.KibiByte
	} else {
		mem.FreeMemory = (free + buffers + cached) * format.KibiByte
	}
	return mem, nil
}

func getCPUMemByCgroups(mem memInfo) memInfo {
	total, err := getUint64ValueFromFile("/sys/fs/cgroup/memory.max")
	if err == nil {
		mem.TotalMemory = total
	}
	used, err := getUint64ValueFromFile("/sys/fs/cgroup/memory.current")
	if err == nil {
		mem.FreeMemory = mem.TotalMemory - used
	}
	return mem
}

func getUint64ValueFromFile(path string) (uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := s.Text()
		return strconv.ParseUint(line, 10, 64)
	}
	return 0, errors.New("empty file content")
}

func IsNUMA() bool {
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

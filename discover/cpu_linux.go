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

// parseCgroupMemoryStat parses the cgroup v2 memory.stat file for reclaimable memory values.
// format of the file is each line has name of a stat, a space, then the stat in bytes
// eg `inactive_file 15653130240`
func parseCgroupMemoryStat(path string) (map[string]uint64, error) {
	stats := map[string]uint64{}

	f, err := os.Open(path)
	if err != nil {
		return stats, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)
		if len(fields) != 2 {
			continue
		}
		key, valueStr := fields[0], fields[1]
		value, err := strconv.ParseUint(valueStr, 10, 64)
		if err != nil {
			continue
		}

		stats[key] = value
	}
	return stats, scanner.Err()
}

// getCPUMemByCgroups adjusts memory info based on cgroup v2 limits.
// It calculates MemAvailable using the same logic as lxcfs:
//
//	MemAvailable = memlimit - memusage + (active_file + inactive_file + slab_reclaimable)
//
// This accounts for reclaimable file cache that can be freed under memory pressure.
// see https://github.com/lxc/lxcfs/blob/c67f5f88e39ab1603a1106857b5fc78516437714/src/proc_fuse.c#L1484-L1485
func getCPUMemByCgroups(mem memInfo) memInfo {
	total, err := getUint64ValueFromFile("/sys/fs/cgroup/memory.max")
	if err == nil && total > 0 && total < mem.TotalMemory {
		mem.TotalMemory = total
	}

	used, err := getUint64ValueFromFile("/sys/fs/cgroup/memory.current")
	if err != nil {
		slog.Warn("Was able to read cgroup memory.max but not memory.current")
		return mem
	}

	stat, err := parseCgroupMemoryStat("/sys/fs/cgroup/memory.stat")
	if err != nil {
		slog.Warn("Was able to read cgroup memory.max and memory.current but not memory.stat")
		mem.FreeMemory = mem.TotalMemory - used
		return mem
	}

	activeFileMem, activeFileMemExists := stat["active_file"]
	inactiveFileMem, inactiveFileMemExists := stat["inactive_file"]
	slabReclaimableMem, slabReclaimableMemExists := stat["slab_reclaimable"]

	if !activeFileMemExists || !inactiveFileMemExists || !slabReclaimableMemExists {
		slog.Warn("Cgroup memory.stat exists but didn't have expected fields")
		mem.FreeMemory = mem.TotalMemory - used
		return mem
	}

	reclaimable := activeFileMem + inactiveFileMem + slabReclaimableMem
	if used > reclaimable {
		mem.FreeMemory = mem.TotalMemory - used + reclaimable
	} else {
		// hopefully this branch never triggers?
		slog.Warn("Cgroup reclaimable memory was greater than cgroup used memory",
			"reclaimable", reclaimable,
			"used", used)
		mem.FreeMemory = mem.TotalMemory
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

package discover

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
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

const CpuInfoFilename = "/proc/cpuinfo"

type linuxCpuInfo struct {
	ID         string `cpuinfo:"processor"`
	VendorID   string `cpuinfo:"vendor_id"`
	ModelName  string `cpuinfo:"model name"`
	PhysicalID string `cpuinfo:"physical id"`
	Siblings   string `cpuinfo:"siblings"`
	CoreID     string `cpuinfo:"core id"`
}

func GetCPUDetails() []CPU {
	file, err := os.Open(CpuInfoFilename)
	if err != nil {
		slog.Warn("failed to get CPU details", "error", err)
		return nil
	}
	defer file.Close()
	cpus := linuxCPUDetails(file)
	return overwriteThreadCountByLinuxCgroups(cpus)
}

func overwriteThreadCountByLinuxCgroups(cpus []CPU) []CPU {
	file, err := os.Open("/sys/fs/cgroup/cpu.max")
	if err != nil {
		return cpus
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if sl := strings.Split(line, " "); len(sl) == 2 {
			allowdUs, err := strconv.ParseInt(sl[0], 10, 64)
			if err != nil {
				slog.Warn("failed to parse CPU allowed micro secs", "error", err)
				return cpus
			}
			unitUs, err := strconv.ParseInt(sl[1], 10, 64)
			if err != nil {
				slog.Warn("failed to parse CPU unit micro secs", "error", err)
				return cpus
			}

			threads := int(max(allowdUs/unitUs, 1))

			cpu := cpus[0]
			cpu.CoreCount = threads
			cpu.ThreadCount = threads
			return []CPU{cpu}
		}
	}
	return cpus
}

func linuxCPUDetails(file io.Reader) []CPU {
	reColumns := regexp.MustCompile("\t+: ")
	scanner := bufio.NewScanner(file)
	cpuInfos := []linuxCpuInfo{}
	cpu := &linuxCpuInfo{}
	for scanner.Scan() {
		line := scanner.Text()
		if sl := reColumns.Split(line, 2); len(sl) > 1 {
			t := reflect.TypeOf(cpu).Elem()
			s := reflect.ValueOf(cpu).Elem()
			for i := range t.NumField() {
				field := t.Field(i)
				tag := field.Tag.Get("cpuinfo")
				if tag == sl[0] {
					s.FieldByName(field.Name).SetString(sl[1])
					break
				}
			}
		} else if strings.TrimSpace(line) == "" && cpu.ID != "" {
			cpuInfos = append(cpuInfos, *cpu)
			cpu = &linuxCpuInfo{}
		}
	}
	if cpu.ID != "" {
		cpuInfos = append(cpuInfos, *cpu)
	}

	// Process the sockets/cores/threads
	socketByID := map[string]*CPU{}
	coreBySocket := map[string]map[string]struct{}{}
	threadsByCoreBySocket := map[string]map[string]int{}
	for _, c := range cpuInfos {
		if _, found := socketByID[c.PhysicalID]; !found {
			socketByID[c.PhysicalID] = &CPU{
				ID:        c.PhysicalID,
				VendorID:  c.VendorID,
				ModelName: c.ModelName,
			}
			coreBySocket[c.PhysicalID] = map[string]struct{}{}
			threadsByCoreBySocket[c.PhysicalID] = map[string]int{}
		}
		if c.CoreID != "" {
			coreBySocket[c.PhysicalID][c.PhysicalID+":"+c.CoreID] = struct{}{}
			threadsByCoreBySocket[c.PhysicalID][c.PhysicalID+":"+c.CoreID]++
		} else {
			coreBySocket[c.PhysicalID][c.PhysicalID+":"+c.ID] = struct{}{}
			threadsByCoreBySocket[c.PhysicalID][c.PhysicalID+":"+c.ID]++
		}
	}

	// Tally up the values from the tracking maps
	for id, s := range socketByID {
		s.CoreCount = len(coreBySocket[id])
		s.ThreadCount = 0

		// This only works if HT is enabled, consider a more reliable model, maybe cache size comparisons?
		efficiencyCoreCount := 0
		for _, threads := range threadsByCoreBySocket[id] {
			s.ThreadCount += threads
			if threads == 1 {
				efficiencyCoreCount++
			}
		}
		if efficiencyCoreCount == s.CoreCount {
			// 1:1 mapping means they're not actually efficiency cores, but regular cores
			s.EfficiencyCoreCount = 0
		} else {
			s.EfficiencyCoreCount = efficiencyCoreCount
		}
	}
	keys := make([]string, 0, len(socketByID))
	result := make([]CPU, 0, len(socketByID))
	for k := range socketByID {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		result = append(result, *socketByID[k])
	}
	return result
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

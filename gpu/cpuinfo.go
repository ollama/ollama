package gpu

import (
	"bufio"
	"os"
	"reflect"
	"regexp"
	"strings"
)

const CpuInfoFilename = "/proc/cpuinfo"

type CpuInfo struct {
	ID         string `cpuinfo:"processor"`
	VendorID   string `cpuinfo:"vendor_id"`
	ModelName  string `cpuinfo:"model name"`
	PhysicalID string `cpuinfo:"physical id"`
	Siblings   string `cpuinfo:"siblings"`
	CoreID     string `cpuinfo:"core id"`
}

type CPUs []CpuInfo

func (cpus CPUs) SocketCount() int {
	ids := map[string]interface{}{}

	for _, cpu := range cpus {
		ids[cpu.PhysicalID] = struct{}{}
	}
	return len(ids)
}

func (cpus CPUs) CoreCount() int {
	cores := map[string]interface{}{}
	for _, cpu := range cpus {
		if cpu.CoreID != "" {
			cores[cpu.PhysicalID+":"+cpu.CoreID] = struct{}{}
		} else {
			cores[cpu.PhysicalID+":"+cpu.ID] = struct{}{}
		}
	}
	return len(cores)
}

func (cpus CPUs) PerformanceCoreCount() int {
	// This only works if HT is enabled, need a more reliable model...
	coreThreads := map[string]int{}
	for _, cpu := range cpus {
		if cpu.CoreID != "" {
			coreThreads[cpu.PhysicalID+":"+cpu.CoreID]++
		} else {
			coreThreads[cpu.PhysicalID+":"+cpu.ID]++
		}
	}
	efficiencyCoreCount := 0
	for _, threads := range coreThreads {
		if threads == 1 {
			efficiencyCoreCount++
		}
	}
	coreCount := cpus.CoreCount()
	if efficiencyCoreCount == coreCount {
		return coreCount
	}
	return coreCount - efficiencyCoreCount
}

func (cpus CPUs) Vendors() []string {
	vendors := map[string]interface{}{}
	for _, cpu := range cpus {
		vendors[cpu.VendorID] = struct{}{}
	}
	ret := make([]string, len(vendors))
	i := 0
	for v := range vendors {
		ret[i] = v
		i += 1
	}
	return ret
}

func (cpus CPUs) Models() []string {
	models := map[string]interface{}{}
	for _, cpu := range cpus {
		models[cpu.ModelName] = struct{}{}
	}
	ret := make([]string, len(models))
	i := 0
	for v := range models {
		ret[i] = v
		i += 1
	}
	return ret
}

func ParseCpuInfo() (CPUs, error) {
	file, err := os.Open(CpuInfoFilename)
	if err != nil {
		return nil, err
	}
	reColumns := regexp.MustCompile("\t+: ")
	scanner := bufio.NewScanner(file)
	res := []CpuInfo{}
	cpu := &CpuInfo{}
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
			res = append(res, *cpu)
			cpu = &CpuInfo{}
		}
	}
	return res, nil
}

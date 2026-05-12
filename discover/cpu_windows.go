package discover

import (
	"fmt"
	"log/slog"
	"syscall"
	"unsafe"

	"github.com/ollama/ollama/logutil"
)

type MEMORYSTATUSEX struct {
	length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

var (
	k32                              = syscall.NewLazyDLL("kernel32.dll")
	globalMemoryStatusExProc         = k32.NewProc("GlobalMemoryStatusEx")
	sizeofMemoryStatusEx             = uint32(unsafe.Sizeof(MEMORYSTATUSEX{}))
	GetLogicalProcessorInformationEx = k32.NewProc("GetLogicalProcessorInformationEx")
)

func GetCPUMem() (memInfo, error) {
	memStatus := MEMORYSTATUSEX{length: sizeofMemoryStatusEx}
	r1, _, err := globalMemoryStatusExProc.Call(uintptr(unsafe.Pointer(&memStatus)))
	if r1 == 0 {
		return memInfo{}, fmt.Errorf("GlobalMemoryStatusEx failed: %w", err)
	}
	return memInfo{TotalMemory: memStatus.TotalPhys, FreeMemory: memStatus.AvailPhys, FreeSwap: memStatus.AvailPageFile}, nil
}

type LOGICAL_PROCESSOR_RELATIONSHIP uint32

const (
	RelationProcessorCore LOGICAL_PROCESSOR_RELATIONSHIP = iota
	RelationNumaNode
	RelationCache
	RelationProcessorPackage
	RelationGroup
	RelationProcessorDie
	RelationNumaNodeEx
	RelationProcessorModule
)
const RelationAll LOGICAL_PROCESSOR_RELATIONSHIP = 0xffff

type GROUP_AFFINITY struct {
	Mask     uintptr // KAFFINITY
	Group    uint16
	Reserved [3]uint16
}

type PROCESSOR_RELATIONSHIP struct {
	Flags           byte
	EfficiencyClass byte
	Reserved        [20]byte
	GroupCount      uint16
	GroupMask       [1]GROUP_AFFINITY // len GroupCount
}

// Omitted unused structs: NUMA_NODE_RELATIONSHIP CACHE_RELATIONSHIP GROUP_RELATIONSHIP

type SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX struct {
	Relationship LOGICAL_PROCESSOR_RELATIONSHIP
	Size         uint32
	U            [1]byte // Union len Size
	// PROCESSOR_RELATIONSHIP
	// NUMA_NODE_RELATIONSHIP
	// CACHE_RELATIONSHIP
	// GROUP_RELATIONSHIP
}

func (group *GROUP_AFFINITY) IsMember(target *GROUP_AFFINITY) bool {
	if group == nil || target == nil {
		return false
	}
	return group.Mask&target.Mask != 0
}

type winPackage struct {
	groups              []*GROUP_AFFINITY
	coreCount           int // performance cores = coreCount - efficiencyCoreCount
	efficiencyCoreCount int
	threadCount         int
}

func (pkg *winPackage) IsMember(target *GROUP_AFFINITY) bool {
	for _, group := range pkg.groups {
		if group.IsMember(target) {
			return true
		}
	}
	return false
}

func getLogicalProcessorInformationEx() ([]byte, error) {
	buf := make([]byte, 1)
	bufSize := len(buf)
	ret, _, err := GetLogicalProcessorInformationEx.Call(
		uintptr(RelationAll),
		uintptr(unsafe.Pointer(&buf[0])),
		uintptr(unsafe.Pointer(&bufSize)),
	)
	if ret != 0 {
		logutil.Trace("failed to retrieve CPU payload size", "ret", ret, "size", bufSize, "error", err)
		return nil, fmt.Errorf("failed to determine size info ret:%d %w", ret, err)
	}

	buf = make([]byte, bufSize)
	ret, _, err = GetLogicalProcessorInformationEx.Call(
		uintptr(RelationAll),
		uintptr(unsafe.Pointer(&buf[0])),
		uintptr(unsafe.Pointer(&bufSize)),
	)
	if ret == 0 {
		logutil.Trace("failed to retrieve CPU information", "ret", ret, "size", len(buf), "new_size", bufSize, "error", err)
		return nil, fmt.Errorf("failed to gather processor information ret:%d buflen:%d %w", ret, bufSize, err)
	}
	return buf, nil
}

func processSystemLogicalProcessorInforationList(buf []byte) []*winPackage {
	var slpi *SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX
	// Find all the packages first
	packages := []*winPackage{}
	for bufOffset := 0; bufOffset < len(buf); bufOffset += int(slpi.Size) {
		slpi = (*SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(unsafe.Pointer(&buf[bufOffset]))
		if slpi.Relationship != RelationProcessorPackage {
			continue
		}
		pr := (*PROCESSOR_RELATIONSHIP)(unsafe.Pointer(&slpi.U[0]))
		pkg := &winPackage{}
		ga0 := unsafe.Pointer(&pr.GroupMask[0])
		for j := range pr.GroupCount {
			gm := (*GROUP_AFFINITY)(unsafe.Pointer(uintptr(ga0) + uintptr(j)*unsafe.Sizeof(GROUP_AFFINITY{})))
			pkg.groups = append(pkg.groups, gm)
		}
		packages = append(packages, pkg)
	}

	slog.Info("packages", "count", len(packages))

	// To identify efficiency cores we have to compare the relative values
	// Larger values are "less efficient" (aka, more performant)
	var maxEfficiencyClass byte
	for bufOffset := 0; bufOffset < len(buf); bufOffset += int(slpi.Size) {
		slpi = (*SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(unsafe.Pointer(&buf[bufOffset]))
		if slpi.Relationship != RelationProcessorCore {
			continue
		}
		pr := (*PROCESSOR_RELATIONSHIP)(unsafe.Pointer(&slpi.U[0]))
		if pr.EfficiencyClass > maxEfficiencyClass {
			maxEfficiencyClass = pr.EfficiencyClass
		}
	}
	if maxEfficiencyClass > 0 {
		slog.Info("efficiency cores detected", "maxEfficiencyClass", maxEfficiencyClass)
	}

	// then match up the Cores to the Packages, count up cores, threads and efficiency cores
	for bufOffset := 0; bufOffset < len(buf); bufOffset += int(slpi.Size) {
		slpi = (*SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(unsafe.Pointer(&buf[bufOffset]))
		if slpi.Relationship != RelationProcessorCore {
			continue
		}
		pr := (*PROCESSOR_RELATIONSHIP)(unsafe.Pointer(&slpi.U[0]))
		ga0 := unsafe.Pointer(&pr.GroupMask[0])
		for j := range pr.GroupCount {
			gm := (*GROUP_AFFINITY)(unsafe.Pointer(uintptr(ga0) + uintptr(j)*unsafe.Sizeof(GROUP_AFFINITY{})))
			for _, pkg := range packages {
				if pkg.IsMember(gm) {
					pkg.coreCount++
					if pr.Flags == 0 {
						pkg.threadCount++
					} else {
						pkg.threadCount += 2
					}
					if pr.EfficiencyClass < maxEfficiencyClass {
						pkg.efficiencyCoreCount++
					}
				}
			}
		}
	}

	// Summarize the results
	for i, pkg := range packages {
		slog.Info("", "package", i, "cores", pkg.coreCount, "efficiency", pkg.efficiencyCoreCount, "threads", pkg.threadCount)
	}

	return packages
}

func GetCPUDetails() []CPU {
	buf, err := getLogicalProcessorInformationEx()
	if err != nil {
		slog.Warn("failed to get CPU details", "error", err)
		return nil
	}
	packages := processSystemLogicalProcessorInforationList(buf)
	cpus := make([]CPU, len(packages))

	for i, pkg := range packages {
		cpus[i].CoreCount = pkg.coreCount
		cpus[i].EfficiencyCoreCount = pkg.efficiencyCoreCount
		cpus[i].ThreadCount = pkg.threadCount
	}
	return cpus
}

func IsNUMA() bool {
	// numa support in ggml is linux only
	return false
}

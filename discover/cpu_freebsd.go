package discover

import (
	"log/slog"
	"os/exec"
	"runtime"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"
)

func GetCPUMem() (memInfo, error) {
	var mem memInfo

	physmem, err := unix.SysctlUint64("hw.physmem")
	if err != nil {
		return mem, err
	}
	mem.TotalMemory = physmem

	pagesize, err := unix.SysctlUint32("vm.stats.vm.v_page_size")
	if err != nil {
		return mem, err
	}

	freeCount, err := unix.SysctlUint32("vm.stats.vm.v_free_count")
	if err != nil {
		return mem, err
	}

	// Inactive and cache pages can be reclaimed
	inactiveCount, err := unix.SysctlUint32("vm.stats.vm.v_inactive_count")
	if err != nil {
		inactiveCount = 0
	}
	cacheCount, err := unix.SysctlUint32("vm.stats.vm.v_cache_count")
	if err != nil {
		cacheCount = 0
	}

	mem.FreeMemory = uint64(freeCount+inactiveCount+cacheCount) * uint64(pagesize)
	mem.FreeSwap = getFreeSwap()

	return mem, nil
}

func getFreeSwap() uint64 {
	out, err := exec.Command("swapinfo", "-k").Output()
	if err != nil {
		return 0
	}

	var free uint64
	lines := strings.Split(string(out), "\n")
	for _, line := range lines[1:] { // skip header
		fields := strings.Fields(line)
		if len(fields) >= 4 {
			if avail, err := strconv.ParseUint(fields[3], 10, 64); err == nil {
				free += avail * 1024 // KB to bytes
			}
		}
	}
	return free
}

func GetCPUDetails() []CPU {
	model, err := unix.Sysctl("hw.model")
	if err != nil {
		slog.Warn("failed to get CPU model", "error", err)
		model = "Unknown"
	}

	coreCount, err := unix.SysctlUint32("kern.smp.cores")
	if err != nil {
		slog.Warn("failed to get core count", "error", err)
		coreCount = uint32(runtime.NumCPU())
	}

	threadCount := runtime.NumCPU()

	return []CPU{
		{
			ID:                  "0",
			VendorID:            "",
			ModelName:           model,
			CoreCount:           int(coreCount),
			EfficiencyCoreCount: 0,
			ThreadCount:         threadCount,
		},
	}
}

func IsNUMA() bool {
	ndomains, err := unix.SysctlUint32("vm.ndomains")
	if err != nil {
		return false
	}
	return ndomains > 1
}

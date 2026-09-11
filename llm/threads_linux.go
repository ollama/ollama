package llm

import (
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
)

// cgroupCPUMaxPath is the cgroup v2 file describing this container's CFS
// CPU quota and period, e.g. "400000 100000" for a 4-CPU budget, or "max
// 100000" when unlimited. This is the file a process sees at the root of
// its own cgroup namespace, which covers the common container case (the
// same scope as the existing memory limit read in
// discover/cpu_linux.go's getCPUMemByCgroups). It does not resolve a
// deeper cgroup path such as a systemd slice on bare metal, and cgroup v1
// (cpu.cfs_quota_us/cpu.cfs_period_us) is not read.
const cgroupCPUMaxPath = "/sys/fs/cgroup/cpu.max"

// defaultNumThreads returns the default llama-server thread count when the
// caller hasn't set one explicitly. runtime.NumCPU() already reflects the
// process's sched_getaffinity mask (e.g. a container's cpuset), but not a
// CFS quota such as Docker's `--cpus`, which throttles without narrowing
// affinity. Clamping to the cgroup quota avoids oversubscribing threads
// past the CPU budget the process actually gets, which otherwise convoys
// llama.cpp's spin-wait barriers against CFS throttling. Absent a quota,
// this returns the same logical CPU count llama-server's own auto-detect
// would have used.
func defaultNumThreads() int {
	return numThreadsForQuota(runtime.NumCPU(), cgroupCPUQuota(cgroupCPUMaxPath))
}

// cgroupCPUQuota reads a cgroup v2 cpu.max file and returns the thread
// budget ceil(quota/period), or 0 if the file is absent, unlimited
// ("max"), malformed, or the ratio doesn't fit in an int.
func cgroupCPUQuota(path string) int {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	fields := strings.Fields(string(data))
	if len(fields) != 2 || fields[0] == "max" {
		return 0
	}
	quota, err := strconv.ParseUint(fields[0], 10, 64)
	if err != nil || quota == 0 {
		return 0
	}
	period, err := strconv.ParseUint(fields[1], 10, 64)
	if err != nil || period == 0 {
		return 0
	}
	threads := quota / period
	if quota%period != 0 {
		threads++
	}
	if threads > math.MaxInt {
		return 0
	}
	return int(threads)
}

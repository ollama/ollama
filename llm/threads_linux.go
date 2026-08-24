package llm

import (
	"os"
	"runtime"
	"strconv"
	"strings"
)

// cgroupCPUMaxPath is the cgroup v2 file describing this process's CFS CPU
// quota and period, e.g. "400000 100000" for a 4-CPU budget, or "max
// 100000" when unlimited.
const cgroupCPUMaxPath = "/sys/fs/cgroup/cpu.max"

// defaultNumThreads returns the default llama-server thread count when the
// caller hasn't set one explicitly. runtime.NumCPU() already reflects the
// process's sched_getaffinity mask (e.g. a container's cpuset), but not a
// CFS quota such as Docker's `--cpus`, which throttles without narrowing
// affinity. Clamping to the cgroup quota avoids oversubscribing threads
// past the CPU budget the process actually gets, which otherwise convoys
// llama.cpp's spin-wait barriers against CFS throttling.
func defaultNumThreads() int {
	return numThreadsForQuota(runtime.NumCPU(), cgroupCPUQuota(cgroupCPUMaxPath))
}

// cgroupCPUQuota reads a cgroup v2 cpu.max file and returns the thread
// budget ceil(quota/period), or 0 if the file is absent, unlimited
// ("max"), or malformed.
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
	return int((quota + period - 1) / period)
}

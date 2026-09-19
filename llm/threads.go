package llm

// numThreadsForQuota clamps available (the host/affinity-visible CPU count)
// to quota (a cgroup CFS thread budget, or 0 if there is no limit), never
// returning less than one thread.
func numThreadsForQuota(available, quota int) int {
	n := available
	if quota > 0 && quota < n {
		n = quota
	}
	if n < 1 {
		n = 1
	}
	return n
}

//go:build !linux

package llm

import "runtime"

// defaultNumThreads returns the host core count. macOS and Windows have no
// cgroups, so there is no quota to clamp against.
func defaultNumThreads() int {
	return runtime.NumCPU()
}

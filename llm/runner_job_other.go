//go:build !windows

package llm

// AddRunnerToJob is a no-op on platforms without Windows Job Objects.
func AddRunnerToJob(int) error {
	return nil
}

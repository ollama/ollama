//go:build windows

package llm

// killProcess is a no-op on Windows: the runner lives in our Job Object with
// kill-on-job-close, so there's no stray process to chase down here.
func killProcess(pid int) {}

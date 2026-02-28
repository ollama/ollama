//go:build !windows && !darwin && !linux

package server

// platformPreventSleep is a no-op on unsupported platforms.
func platformPreventSleep() error {
	return nil
}

// platformAllowSleep is a no-op on unsupported platforms.
func platformAllowSleep() error {
	return nil
}

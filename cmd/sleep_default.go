//go:build !(windows || darwin || linux)

package cmd

// PreventSystemSleep is a no-op on unsupported platforms
func PreventSystemSleep() uint32 {
	return 0
}

// AllowSystemSleep is a no-op on unsupported platforms
func AllowSystemSleep(prevState uint32) {
	// no-op
}

// IsSystemSleepPreventionAvailable returns false on unsupported platforms
func IsSystemSleepPreventionAvailable() bool {
	return false
}

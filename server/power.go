//go:build !windows

package server

// preventSleep is a no-op on non-Windows platforms.
// On Windows, it prevents the OS from sleeping during inference.
func preventSleep() func() {
	return func() {}
}

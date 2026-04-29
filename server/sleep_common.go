//go:build !windows

package server

func preventSleep() func() {
	return func() {}
}

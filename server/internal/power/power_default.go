//go:build !darwin

package power

func preventSleep() {
	// No-op for non-darwin systems for now
}

func allowSleep() {
	// No-op for non-darwin systems for now
}

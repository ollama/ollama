//go:build !windows

package llm

func preventSleep() func() {
	return func() {}
}

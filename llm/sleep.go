//go:build !windows && !darwin

package llm

import "log/slog"

func preventSleep() (func(), error) {
	slog.Debug("sleep prevention not supported on this platform")
	return func() {}, nil
}

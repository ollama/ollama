//go:build !windows

package lifecycle

import "log/slog"

func ShowLogs() {
	slog.Warn("not implemented")
}

//go:build !windows

package lifecycle

import "log/slog"

func ShowLogs() {
	slog.Warn("ShowLogs not yet implemented")
}

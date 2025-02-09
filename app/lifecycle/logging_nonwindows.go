//go:build !windows

package lifecycle

import "log/slog"

// ShowLogs is a function that logs a warning message.
// It uses slog.Warn to log the message, making it easier to mock for testing.
func ShowLogs() {
    slog.Warn("not implemented")
}

//go:build !windows

package llm

import "log/slog"

func formatExitStatus(s ExitStatus) string {
	return decimalExitStatus(s)
}

func logExitStatus(s ExitStatus) slog.Value {
	if s == exitStatusOK || s == exitStatusUnknown {
		return slog.StringValue(s.String())
	}

	return slog.IntValue(int(s))
}
